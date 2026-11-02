# okc: problem=muse_bf16_proj
"""Impl J ThunderKittens-structure: G stack (GEMV+FI+router); TK WMMA thin-m kept compiled but off.

Keep proven G hopper_bf16_gemv (m=1) + FlashInfer allow-list + F.linear (nvjet). Thin-m TK
WMMA path disabled after profile (0.197×); kernel retained for a future SM90 WGMMA port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

ROUTER_FP32_MAX_TOKENS = 8

# Same gates as sglang.kernels.ops.gemm.hopper_bf16_gemv
_MAX_K = 17408
_MAX_N = 65536

# tk: profile submission-implementation-j.3d89ad3046f14f47: speedup 0.197×; m=1 GEMV fine.
# tk: Muse m=16 A@Wt→C shapes — qkv_gate M=16,K=8192,N=2304; pooler M=16,K=8192,N=4096;
# tk: unpooler M=16,K=4096,N=8192; o stays F.linear (nvjet, not TK).
# tk: m=16 WMMA (16,128,64) catastrophic on SM90/H200: qkv_gate 826µs (kernel 654µs) vs sglang
# tk: 13.7µs; pooler 840µs (845µs) vs 21.7µs; unpooler 420µs (418µs) vs 18.6µs — WMMA not WGMMA,
# tk: BN=128 ignores exact N divisors (2304/128=18, 4096/128=32, 8192/128=64), full-K scalar smem.
# tk: tile plan kept commented for a future WGMMA port; do not dispatch thin-m to TK.
TK_TILE_PLAN: dict[tuple[str, int], tuple[int, int, int]] = {
    # ("qkv_gate", 8): (16, 128, 64),  # tk: was pad m=8→BM=16; off — same broken WMMA path
    # ("qkv_gate", 16): (16, 128, 64),  # tk: was native BM=16; off — Muse N=2304 K=8192
    # ("qkv_gate", 32): (32, 128, 64),  # tk: was two WMMA rows; off — share WMMA catastrophe
    # ("pooler", 8): (16, 128, 64),  # tk: off — thin-m → FI/nvjet
    # ("pooler", 16): (16, 128, 64),  # tk: off — Muse N=4096 K=8192
    # ("pooler", 32): (32, 128, 64),  # tk: off — share WMMA path
    # ("unpooler", 8): (16, 128, 64),  # tk: off — thin-m → FI/nvjet
    # ("unpooler", 16): (16, 128, 64),  # tk: off — Muse N=8192 K=4096
    # ("unpooler", 32): (32, 128, 64),  # tk: off — share WMMA path
}  # tk: empty — no thin-m TK dispatch

TK_THIN_M = frozenset()  # tk: was {8,16,32}; emptied so kernel_fn falls through to FI / F.linear
TK_LAYERS = frozenset({"qkv_gate", "pooler", "unpooler"})  # tk: former TK layers; GEMV m=1 unchanged


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]; up to 8 tokens, only rows 0-1023
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [M, K]
    config: Config


# Original hopper_bf16_gemv.cuh body, torch host launch, Muse (N,K) instantiations.
# sgl_kernel / tvm deps replaced with equivalent CUDA + ATen helpers (algorithm unchanged).
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>  // tk: warp-level WMMA (SM86 stand-in for WGMMA)

#include <cstdint>
#include <utility>

using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;
using fp32_t = float;
using fp32x2_t = float2;

template <typename To, typename From>
__device__ __forceinline__ To cast(From v);

template <>
__device__ __forceinline__ fp32x2_t cast<fp32x2_t, bf16x2_t>(bf16x2_t v) {
  return __bfloat1622float2(v);
}

template <>
__device__ __forceinline__ bf16_t cast<bf16_t, fp32_t>(fp32_t v) {
  return __float2bfloat16_rn(v);
}

template <bool kUsePDL>
__device__ __forceinline__ void PDLWaitPrimary() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

template <bool kUsePDL>
__device__ __forceinline__ void PDLTriggerSecondary() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

// ---- from sglang hopper_bf16_gemv.cuh ----
constexpr uint32_t kGemvVecSize = 16 / sizeof(bf16_t);  // 8 bf16 per 16B load

__device__ __forceinline__ float dot8_f32(const float4 wv, const float4 xv) {
  const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto [w0, w1] = cast<fp32x2_t>(w2[i]);
    const auto [x0, x1] = cast<fp32x2_t>(x2[i]);
    acc = fmaf(w0, x0, acc);
    acc = fmaf(w1, x1, acc);
  }
  return acc;
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps, bool kUsePDL>
__global__ void __launch_bounds__(kNumWarps * 32)
    hopper_bf16_gemv_kernel(bf16_t* __restrict__ out, const bf16_t* __restrict__ x, const bf16_t* __restrict__ w) {
  __shared__ bf16_t sx[K];

  PDLTriggerSecondary<kUsePDL>();
  const uint32_t tid = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();
  for (uint32_t i = tid * kGemvVecSize; i < K; i += kNumWarps * 32 * kGemvVecSize) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t r0 = (blockIdx.x * kNumWarps + warp) * kRows;
  if (r0 >= N) {
    return;
  }

  float acc[kRows];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
    acc[r] = 0.0f;
  }

  constexpr uint32_t kStep = 32 * kGemvVecSize * kUnroll;
  if (r0 + kRows <= N) {
    for (uint32_t k = lane * kGemvVecSize * kUnroll; k < K; k += kStep) {
      float4 xv[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        xv[u] = *reinterpret_cast<const float4*>(sx + k + u * kGemvVecSize);
      }
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
        float4 wv[kUnroll];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          wv[u] = __ldcs(reinterpret_cast<const float4*>(wr + u * kGemvVecSize));
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          acc[r] += dot8_f32(wv[u], xv[u]);
        }
      }
    }
  } else {
    for (uint32_t k = lane * kGemvVecSize * kUnroll; k < K; k += kStep) {
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        if (r0 + r < N) {
          const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
#pragma unroll
          for (uint32_t u = 0; u < kUnroll; ++u) {
            float4 wv = __ldcs(reinterpret_cast<const float4*>(wr + u * kGemvVecSize));
            float4 xv = *reinterpret_cast<const float4*>(sx + k + u * kGemvVecSize);
            acc[r] += dot8_f32(wv, xv);
          }
        }
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
      acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (r0 + r < N) {
        out[r0 + r] = cast<bf16_t>(acc[r]);
      }
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps, bool kUsePDL>
void launch_gemv(bf16_t* out, const bf16_t* x, const bf16_t* w, cudaStream_t stream) {
  constexpr uint32_t kRowsPerBlock = kRows * kNumWarps;
  constexpr uint32_t kNumBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
  constexpr uint32_t kThreads = kNumWarps * 32;
  if constexpr (kUsePDL) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kNumBlocks);
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    cfg.dynamicSmemBytes = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(
        &cfg,
        hopper_bf16_gemv_kernel<N, K, kRows, kUnroll, kNumWarps, kUsePDL>,
        out, x, w);
  } else {
    hopper_bf16_gemv_kernel<N, K, kRows, kUnroll, kNumWarps, kUsePDL>
        <<<kNumBlocks, kThreads, 0, stream>>>(out, x, w);
  }
}

void hopper_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.size(0) == 1);
  const int N = w.size(0);
  const int K = w.size(1);
  TORCH_CHECK(out.size(0) == 1 && out.size(1) == N && w.size(1) == x.size(1));

  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  auto* wp = reinterpret_cast<const bf16_t*>(w.data_ptr());
  auto* op = reinterpret_cast<bf16_t*>(out.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();

  // Muse scored m=1 shapes + SGLang _config(n): rows,unroll,warps
  // n>=8192 -> (2,2,8); else (1,2,8)
#define DISPATCH(NN, KK, ROWS, UNROLL, WARPS)                                 \
  do {                                                                        \
    if (use_pdl)                                                              \
      launch_gemv<NN, KK, ROWS, UNROLL, WARPS, true>(op, xp, wp, stream);     \
    else                                                                      \
      launch_gemv<NN, KK, ROWS, UNROLL, WARPS, false>(op, xp, wp, stream);    \
    return;                                                                   \
  } while (0)

  if (N == 2304 && K == 8192) DISPATCH(2304, 8192, 1, 2, 8);
  if (N == 8192 && K == 1024) DISPATCH(8192, 1024, 2, 2, 8);
  if (N == 4096 && K == 8192) DISPATCH(4096, 8192, 1, 2, 8);
  if (N == 1024 && K == 4096) DISPATCH(1024, 4096, 1, 2, 8);
  if (N == 8192 && K == 4096) DISPATCH(8192, 4096, 2, 2, 8);

  TORCH_CHECK(false, "hopper_bf16_gemv: unsupported Muse shape N=", N, " K=", K);
#undef DISPATCH
}

// ---- tk: ThunderKittens-structure BF16 GEMM (SM86 WMMA ≈ H100 WGMMA roles) ----
constexpr int TK_WM = 16;  // tk: WMMA M
constexpr int TK_WN = 16;  // tk: WMMA N
constexpr int TK_WK = 16;  // tk: WMMA K

template <int BM, int BN, int BK>
__global__ void __launch_bounds__((BM / TK_WM) * (BN / TK_WN) * 32)
tk_bf16_gemm_wmma_kernel(  // tk: staged-smem + warp MMA full-K
    bf16_t* __restrict__ C, const bf16_t* __restrict__ A, const bf16_t* __restrict__ B,  // tk: C=A@B
    int M, int N, int K) {  // tk: runtime bounds for partial tiles
  constexpr int WARPS_M = BM / TK_WM;  // tk: warps along M
  constexpr int WARPS_N = BN / TK_WN;  // tk: warps along N
  constexpr int NWARPS = WARPS_M * WARPS_N;  // tk: CTA warp count
  constexpr int NTHREADS = NWARPS * 32;  // tk: exact CTA size
  const int tid = threadIdx.x;  // tk: linear tid
  const int warp = tid / 32;  // tk: warp id (= consumer role)
  const int warp_m = warp / WARPS_N;  // tk: warp row tile
  const int warp_n = warp % WARPS_N;  // tk: warp col tile
  const int block_m = blockIdx.y * BM;  // tk: CTA M origin
  const int block_n = blockIdx.x * BN;  // tk: CTA N origin

  // tk: single-stage panels + Cs (fits 48KB; local-frag store trapped on this toolchain)
  __shared__ bf16_t As[BM * BK];  // tk: staged A smem panel
  __shared__ bf16_t Bs[BK * BN];  // tk: staged B smem panel
  __shared__ float Cs[BM * BN];  // tk: FP32 C tile epilogue

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TK_WM, TK_WN, TK_WK, bf16_t, nvcuda::wmma::row_major> a_frag;  // tk: A frag
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TK_WM, TK_WN, TK_WK, bf16_t, nvcuda::wmma::row_major> b_frag;  // tk: B frag
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TK_WM, TK_WN, TK_WK, float> c_frag;  // tk: acc frag
  nvcuda::wmma::fill_fragment(c_frag, 0.0f);  // tk: zero acc

  for (int k0 = 0; k0 < K; k0 += BK) {  // tk: full-K single launch (no splitK)
    for (int i = tid; i < BM * BK; i += NTHREADS) {  // tk: producer fill A
      int r = i / BK, c = i % BK;  // tk: panel coords
      int gm = block_m + r, gk = k0 + c;  // tk: global A idx
      As[i] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : __float2bfloat16(0.f);  // tk: zero-pad
    }
    for (int i = tid; i < BK * BN; i += NTHREADS) {  // tk: producer fill B
      int r = i / BN, c = i % BN;  // tk: panel coords
      int gk = k0 + r, gn = block_n + c;  // tk: global B idx
      Bs[i] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : __float2bfloat16(0.f);  // tk: zero-pad
    }
    __syncthreads();  // tk: publish panels to consumers
#pragma unroll
    for (int kk = 0; kk < BK; kk += TK_WK) {  // tk: WMMA K slices inside BK
      nvcuda::wmma::load_matrix_sync(  // tk: warp load A
          a_frag, &As[(warp_m * TK_WM) * BK + kk], BK);  // tk: lda=BK
      nvcuda::wmma::load_matrix_sync(  // tk: warp load B
          b_frag, &Bs[kk * BN + warp_n * TK_WN], BN);  // tk: ldb=BN
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // tk: warp MMA
    }
    __syncthreads();  // tk: reuse smem for next K panel
  }

  nvcuda::wmma::store_matrix_sync(  // tk: dump warp C to smem
      &Cs[(warp_m * TK_WM) * BN + warp_n * TK_WN], c_frag, BN, nvcuda::wmma::mem_row_major);  // tk: row-major
  __syncthreads();  // tk: all warps wrote C
  for (int i = tid; i < BM * BN; i += NTHREADS) {  // tk: guarded global store
    int r = i / BN, c = i % BN;  // tk: tile coords
    int gm = block_m + r, gn = block_n + c;  // tk: global C idx
    if (gm < M && gn < N) C[(size_t)gm * N + gn] = __float2bfloat16(Cs[i]);  // tk: bf16 out
  }
}

template <int BM, int BN, int BK>
void launch_tk_gemm(bf16_t* C, const bf16_t* A, const bf16_t* B, int M, int N, int K, cudaStream_t stream) {  // tk: host launch helper
  dim3 block((BM / TK_WM) * (BN / TK_WN) * 32);  // tk: one thread per lane of tile warps
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // tk: CTA grid over N x M
  tk_bf16_gemm_wmma_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(C, A, B, M, N, K);  // tk: fire kernel
}

void tk_bf16_gemm(torch::Tensor x, torch::Tensor wt, torch::Tensor out, int bm, int bn, int bk) {  // tk: torch binding
  TORCH_CHECK(x.is_cuda() && wt.is_cuda() && out.is_cuda());  // tk: device check
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && wt.scalar_type() == at::kBFloat16);  // tk: bf16 only
  TORCH_CHECK(x.dim() == 2 && wt.dim() == 2 && out.dim() == 2);  // tk: rank-2 GEMM
  const int M = x.size(0), K = x.size(1), N = wt.size(1);  // tk: shapes
  TORCH_CHECK(wt.size(0) == K && out.size(0) == M && out.size(1) == N);  // tk: match dims
  TORCH_CHECK(x.is_contiguous() && wt.is_contiguous() && out.is_contiguous());  // tk: contig loads
  auto* Ap = reinterpret_cast<const bf16_t*>(x.data_ptr());  // tk: A ptr
  auto* Bp = reinterpret_cast<const bf16_t*>(wt.data_ptr());  // tk: B=W.T ptr
  auto* Cp = reinterpret_cast<bf16_t*>(out.data_ptr());  // tk: C ptr
  auto stream = at::cuda::getCurrentCUDAStream();  // tk: current stream
  // tk: instantiate tile plan combos used by Muse thin-m
  if (bm == 16 && bn == 128 && bk == 64) { launch_tk_gemm<16, 128, 64>(Cp, Ap, Bp, M, N, K, stream); return; }  // tk: M16 tile
  if (bm == 32 && bn == 128 && bk == 64) { launch_tk_gemm<32, 128, 64>(Cp, Ap, Bp, M, N, K, stream); return; }  // tk: M32 tile
  TORCH_CHECK(false, "tk_bf16_gemm: unsupported tile BM=", bm, " BN=", bn, " BK=", bk);  // tk: refuse unknown
}
"""

_CPP_DECL = r"""
void hopper_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl);
void tk_bf16_gemm(torch::Tensor x, torch::Tensor wt, torch::Tensor out, int bm, int bn, int bk);  // tk: export TK GEMM
"""


def _use_hopper_bf16_gemv(m: int, n: int, k: int) -> bool:
    """Mirror sglang.kernels.ops.gemm.hopper_bf16_gemv.use_hopper_bf16_gemv (SM90 path)."""
    if not (
        m == 1
        and k % 512 == 0
        and 512 <= k <= _MAX_K
        and n % 8 == 0
        and 64 <= n <= _MAX_N
    ):
        return False
    if k == 5120 and n <= 512:
        return False
    return n < 12288 or n >= 32768


def _align16(t: torch.Tensor) -> torch.Tensor:
    if t.data_ptr() % 16 == 0 and t.is_contiguous():
        return t
    out = torch.empty_like(t)
    out.copy_(t)
    return out


def _build_ext(device: torch.device):
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    build_dir = Path.home() / ".cache" / "muse_bf16_tk_j" / arch  # tk: separate cache from G
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"muse_bf16_tk_j_{arch}",  # tk: unique ext name for J
        cpp_sources=[_CPP_DECL],
        cuda_sources=[_CUDA_SRC],
        functions=["hopper_bf16_gemv", "tk_bf16_gemm"],  # tk: export both kernels
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++20",  # tk: judge ATen requires C++20 (c++17 failed okc setup)
            f"-arch={arch}",  # tk: real SASS arch so HMMA emits
            "--expt-relaxed-constexpr",
        ],
        build_directory=str(build_dir),
        verbose=False,
    )


def _probe_mm_bf16_backend(device: torch.device) -> str | None:
    # {{FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving}}{{Pick first SM90-capable mm_bf16 backend that runs on thin-m Muse shapes.}}
    try:
        from flashinfer.gemm import mm_bf16
    except ImportError:
        return None
    x = torch.zeros(8, 1024, dtype=torch.bfloat16, device=device)
    b = torch.zeros(1024, 1024, dtype=torch.bfloat16, device=device)
    for backend in ("tgv", "cute-dsl", "cutlass", "cublaslt", "auto"):
        try:
            y = mm_bf16(x, b, backend=backend)
            torch.cuda.synchronize(device)
            if y.shape == (8, 1024):
                return backend
        except Exception:
            continue
    return None


def _tk_tile_for(layer: str, m: int) -> tuple[int, int, int]:  # tk: plan lookup
    return TK_TILE_PLAN[(layer, m)]  # tk: exact allow-list key


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {
        name: _align16(w.contiguous()) for name, w in weights.items()
    }
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state[f"{name}_t"] = state[name].t().contiguous()
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()

    device = router.device
    state["ext"] = None
    state["use_pdl"] = False
    state["mm_backend"] = None
    state["mm_bf16"] = None
    state["tk_tile_plan"] = TK_TILE_PLAN  # tk: expose tile plan in state
    state["unpooler_t_1024"] = state["unpooler_t"][:, :1024].contiguous()  # tk: contig B for sliced N
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        state["use_pdl"] = major >= 9
        state["ext"] = _build_ext(device)
        state["mm_backend"] = _probe_mm_bf16_backend(device)
        if state["mm_backend"] is not None:
            from flashinfer.gemm import mm_bf16

            state["mm_bf16"] = mm_bf16
        # tk: warm TK kernels so first graph capture is cold-free
        ext = state["ext"]  # tk: compiled ext
        if ext is not None:  # tk: only when CUDA ext built
            for layer, m in TK_TILE_PLAN:  # tk: each allow-listed shape
                W = state[layer]  # tk: weight rows
                n = 1024 if (layer == "unpooler" and m <= ROUTER_FP32_MAX_TOKENS) else W.shape[0]  # tk: unpool slice
                bm, bn, bk = _tk_tile_for(layer, m)  # tk: tile from plan
                x = torch.zeros((m, W.shape[1]), dtype=torch.bfloat16, device=device)  # tk: warm A
                out = torch.empty((m, n), dtype=torch.bfloat16, device=device)  # tk: warm C
                wt = state["unpooler_t_1024"] if (layer == "unpooler" and n == 1024) else state[f"{layer}_t"][:, :n]  # tk: contig B
                ext.tk_bf16_gemm(x, wt, out, bm, bn, bk)  # tk: warm launch
            torch.cuda.synchronize(device)  # tk: finish warm
    return state


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x, layer = data["x"], data["config"]["layer"]
    if layer == "router":
        if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    n = data["config"]["n"]
    W = state[layer][:n]
    m = x.numel() // x.shape[-1]
    ext = state["ext"]
    if (
        ext is not None
        and _use_hopper_bf16_gemv(m, W.shape[0], W.shape[1])
        and x.is_contiguous()
        and W.is_contiguous()
        and x.data_ptr() % 16 == 0
        and W.data_ptr() % 16 == 0
    ):
        x1 = _align16(x.view(-1, x.shape[-1]))
        out = torch.empty((1, W.shape[0]), dtype=x.dtype, device=x.device)
        ext.hopper_bf16_gemv(x1, W, out, bool(state["use_pdl"]))
        return out.view(*x.shape[:-1], -1)

    # tk: thin-m TK GEMM disabled (TK_THIN_M empty); kernel still compiled for future WGMMA.
    # tk: was: if ext and layer in TK_LAYERS and m in {8,16,32}: tk_bf16_gemm(...(16|32),128,64)
    if ext is not None and layer in TK_LAYERS and m in TK_THIN_M:  # tk: dead until WGMMA plan refilled
        bm, bn, bk = _tk_tile_for(layer, m)  # tk: tile plan
        x2 = _align16(x.view(-1, x.shape[-1]).contiguous())  # tk: contig A
        wt = state["unpooler_t_1024"] if (layer == "unpooler" and n == 1024) else state[f"{layer}_t"][:, :n]  # tk: contig B
        out = torch.empty((x2.shape[0], n), dtype=x.dtype, device=x.device)  # tk: alloc C (graph-safe shape)
        ext.tk_bf16_gemm(x2, wt, out, int(bm), int(bn), int(bk))  # tk: single-launch full-K (unused)
        return out.view(*x.shape[:-1], n)  # tk: restore leading dims

    mm = state["mm_bf16"]
    backend = state["mm_backend"]
    # Shape allow-list from E/F profiles: keep FI only where it beat nvjet.
    use_fi = False
    if mm is not None and backend is not None:
        if layer == "pooler" and m in (8, 16, 32):
            use_fi = True
        elif layer == "unpooler" and m in (16, 32):
            use_fi = True
        elif layer == "qkv_gate" and m in (16, 32):
            use_fi = True
    if use_fi:
        return mm(x.view(-1, x.shape[-1]), state[f"{layer}_t"][:, :n], backend=backend).view(
            *x.shape[:-1], n
        )

    return F.linear(x, W, None)
