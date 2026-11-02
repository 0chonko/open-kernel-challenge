# okc: problem=muse_bf16_proj
"""Impl H: G + cublasLt no-split-K algo filter for qkv/pooler m∈{16,32}.

CUTLASS SM90 Stream-K JIT skipped (prepare ≤300s / prior art too heavy). Keeps nvjet-class
mainloop; picks first heuristic with SPLITK_NUM≤1 to kill splitKreduce. o stays F.linear.
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

# G profiles: nvjet + splitKreduce on these thin-m shapes (unpooler/o often have no reduce).
_NOSPLIT_MS = {
    "qkv_gate": (16, 32),
    "pooler": (16, 32),
}


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
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <unordered_map>
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

// {{cuBLASLt split-K filter}}{{Prefer SPLITK_NUM≤1 heuristics to drop splitKreduce while keeping nvjet mainloop.}}

static cublasLtHandle_t g_lt = nullptr;

struct NosplitEntry {
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t Ad{}, Bd{}, Cd{};
  cublasLtMatmulAlgo_t algo{};
  size_t ws = 0;
};

static std::unordered_map<uint64_t, NosplitEntry> g_nosplit;

static uint64_t nosplit_key(int64_t M, int64_t N, int64_t K) {
  return (static_cast<uint64_t>(M) << 40) | (static_cast<uint64_t>(N) << 16) | static_cast<uint64_t>(K & 0xffff);
}

static void ensure_lt() {
  if (!g_lt) {
    TORCH_CHECK(cublasLtCreate(&g_lt) == CUBLAS_STATUS_SUCCESS);
  }
}

static cublasStatus_t make_row_layout(cublasLtMatrixLayout_t* layout, cudaDataType_t dt, int rows, int cols) {
  auto st = cublasLtMatrixLayoutCreate(layout, dt, rows, cols, cols);
  if (st != CUBLAS_STATUS_SUCCESS) return st;
  cublasLtOrder_t ord = CUBLASLT_ORDER_ROW;
  return cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &ord, sizeof(ord));
}

// Returns workspace bytes (>=0) on success; -1 if no nosplit algo. Caches descs for graph-safe run.
int64_t bf16_mm_pick_nosplit(int64_t M, int64_t N, int64_t K) {
  ensure_lt();
  const uint64_t key = nosplit_key(M, N, K);
  if (g_nosplit.count(key)) {
    return static_cast<int64_t>(g_nosplit[key].ws);
  }

  NosplitEntry e;
  TORCH_CHECK(cublasLtMatmulDescCreate(&e.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS);
  cublasOperation_t t = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
  cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSB, &t, sizeof(t));
  TORCH_CHECK(make_row_layout(&e.Ad, CUDA_R_16BF, (int)M, (int)K) == CUBLAS_STATUS_SUCCESS);
  TORCH_CHECK(make_row_layout(&e.Bd, CUDA_R_16BF, (int)K, (int)N) == CUBLAS_STATUS_SUCCESS);
  TORCH_CHECK(make_row_layout(&e.Cd, CUDA_R_16BF, (int)M, (int)N) == CUBLAS_STATUS_SUCCESS);

  cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS);
  size_t ws_cap = 64ull << 20;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_cap, sizeof(ws_cap));

  cublasLtMatmulHeuristicResult_t heur[64];
  int nres = 0;
  auto st = cublasLtMatmulAlgoGetHeuristic(g_lt, e.op, e.Ad, e.Bd, e.Cd, e.Cd, pref, 64, heur, &nres);
  int64_t out_ws = -1;
  if (st == CUBLAS_STATUS_SUCCESS) {
    for (int i = 0; i < nres; ++i) {
      if (heur[i].state != CUBLAS_STATUS_SUCCESS) continue;
      int splitk = 0;
      size_t written = 0;
      cublasLtMatmulAlgoConfigGetAttribute(
          &heur[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &written);
      if (splitk <= 1) {
        e.algo = heur[i].algo;
        e.ws = heur[i].workspaceSize;
        out_ws = static_cast<int64_t>(e.ws);
        g_nosplit.emplace(key, e);
        break;
      }
    }
  }
  cublasLtMatmulPreferenceDestroy(pref);
  if (out_ws < 0) {
    cublasLtMatrixLayoutDestroy(e.Ad);
    cublasLtMatrixLayoutDestroy(e.Bd);
    cublasLtMatrixLayoutDestroy(e.Cd);
    cublasLtMatmulDescDestroy(e.op);
  }
  return out_ws;
}

void bf16_mm_run_nosplit(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor workspace) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda());
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16 && C.scalar_type() == at::kBFloat16);
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2);
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  TORCH_CHECK(B.size(0) == K && C.size(0) == M && C.size(1) == N);
  ensure_lt();
  auto it = g_nosplit.find(nosplit_key(M, N, K));
  TORCH_CHECK(it != g_nosplit.end(), "bf16_mm_run_nosplit: missing prepare pick");
  const NosplitEntry& e = it->second;
  float alpha = 1.f, beta = 0.f;
  void* ws = workspace.defined() && workspace.numel() > 0 ? workspace.data_ptr() : nullptr;
  size_t ws_bytes = workspace.defined() ? static_cast<size_t>(workspace.numel() * workspace.element_size()) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(
      cublasLtMatmul(
          g_lt, e.op, &alpha, A.data_ptr(), e.Ad, B.data_ptr(), e.Bd, &beta, C.data_ptr(), e.Cd, C.data_ptr(), e.Cd,
          &e.algo, ws, ws_bytes, stream) == CUBLAS_STATUS_SUCCESS);
}
"""

_CPP_DECL = r"""
void hopper_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl);
int64_t bf16_mm_pick_nosplit(int64_t M, int64_t N, int64_t K);
void bf16_mm_run_nosplit(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor workspace);
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


def _find_cutlass_include() -> str | None:
    candidates: list[Path] = []
    try:
        import tilelang  # noqa: F401

        candidates.append(Path(tilelang.__file__).resolve().parent / "3rdparty" / "cutlass" / "include")
    except Exception:
        pass
    for sp in __import__("site").getsitepackages():
        candidates.append(Path(sp) / "tilelang" / "3rdparty" / "cutlass" / "include")
    for c in candidates:
        if (c / "cutlass" / "cutlass.h").is_file():
            return str(c)
    return None


def _build_ext(device: torch.device):
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    gencode = f"-gencode=arch=compute_{major}{minor},code={arch}"
    build_dir = Path.home() / ".cache" / "muse_bf16_impl_h" / arch
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"muse_bf16_impl_h_{arch}",
        cpp_sources=[_CPP_DECL],
        cuda_sources=[_CUDA_SRC],
        functions=["hopper_bf16_gemv", "bf16_mm_pick_nosplit", "bf16_mm_run_nosplit"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            gencode,
        ],
        extra_ldflags=["-lcublasLt"],
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
    state["nosplit"] = {}
    state["nosplit_ws"] = None
    state["cutlass_inc"] = _find_cutlass_include()
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        state["use_pdl"] = major >= 9
        try:
            state["ext"] = _build_ext(device)
        except Exception:
            state["ext"] = None
        state["mm_backend"] = _probe_mm_bf16_backend(device)
        if state["mm_backend"] is not None:
            from flashinfer.gemm import mm_bf16

            state["mm_bf16"] = mm_bf16
        ext = state["ext"]
        if ext is not None:
            max_ws = 0
            for layer, ms in _NOSPLIT_MS.items():
                Wt = state[f"{layer}_t"]
                k, n = Wt.shape
                for m in ms:
                    ws = int(ext.bf16_mm_pick_nosplit(m, n, k))
                    if ws < 0:
                        continue
                    state["nosplit"][(layer, m)] = True
                    max_ws = max(max_ws, ws)
            if state["nosplit"]:
                state["nosplit_ws"] = torch.empty(max(max_ws, 1), dtype=torch.uint8, device=device)
                for (layer, m) in state["nosplit"]:
                    Wt = state[f"{layer}_t"]
                    xw = torch.zeros(m, Wt.shape[0], dtype=torch.bfloat16, device=device)
                    out = torch.empty(m, Wt.shape[1], dtype=torch.bfloat16, device=device)
                    ext.bf16_mm_run_nosplit(xw, Wt, out, state["nosplit_ws"])
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

    if state["nosplit"].get((layer, m)) and state["ext"] is not None:
        x2 = x.view(-1, x.shape[-1])
        out = torch.empty(x2.shape[0], n, dtype=x.dtype, device=x.device)
        state["ext"].bf16_mm_run_nosplit(x2, state[f"{layer}_t"][:, :n], out, state["nosplit_ws"])
        return out.view(*x.shape[:-1], n)

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
