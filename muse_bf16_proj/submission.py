# okc: problem=muse_bf16_proj
# streamk: docstring — Impl I Stream-K identity
"""Impl I: Stream-K.

Keep G hopper_bf16_gemv (m=1) + router + FI allow-list; thin-m qkv_gate/pooler/unpooler
m in {8,16,32} use one-launch WMMA Stream-K BF16 GEMM with in-kernel partial reduce.
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

# streamk: tile — WMMA 16x16 panels; BN wide for N coverage
BM, BN, BK = 16, 128, 16
# streamk: dispatch — thin-m shapes that skip cublasLt splitKreduce via fused Stream-K
_STREAMK_ALLOW = {
    "qkv_gate": (8, 16, 32),
    "pooler": (8, 16, 32),
    "unpooler": (8, 16, 32),
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>  // streamk: WMMA tensor cores on SM86+
#include <algorithm>  // streamk: grid = min(work, sms*2) occupancy

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

// streamk: section — one-launch WMMA Stream-K; in-kernel partial reduce (no cublasLt::splitKreduce)
namespace ns = nvcuda::wmma;  // streamk: WMMA namespace alias (not a type)
using sk_bf16 = __nv_bfloat16;  // streamk: local alias; keep G bf16_t untouched

static __device__ __forceinline__ sk_bf16 sk_f32_bf16(float v) {  // streamk: store C as bf16
  return __float2bfloat16(v);  // streamk: rn convert for C writeback
}  // streamk: end helper

constexpr int kBM = 16;  // streamk: tile M (WMMA)
constexpr int kBN = 128;  // streamk: tile N (8 warps * 16)
constexpr int kBK = 16;  // streamk: tile K (WMMA)
constexpr int kWarps = 8;  // streamk: BN/16 warps for occupancy
constexpr int kThreads = kWarps * 32;  // streamk: block threads

__device__ __forceinline__ void load_a_panel(  // streamk: A panel with zero-fill edges
    sk_bf16* __restrict__ As,  // streamk: smem [16*16]
    const sk_bf16* __restrict__ A,  // streamk: global A[M,K]
    int M, int K, int m0, int kk, int tid) {  // streamk: tile origin + k slice
  for (int i = tid; i < 16 * 16; i += kThreads) {  // streamk: coop fill
    const int r = i >> 4;  // streamk: row in panel
    const int c = i & 15;  // streamk: col in panel
    const int mr = m0 + r;  // streamk: global m
    As[i] = (mr < M) ? A[static_cast<size_t>(mr) * K + kk + c] : sk_bf16{};  // streamk: edge zero-fill
  }  // streamk: end A load loop
}  // streamk: end load_a_panel

__device__ __forceinline__ void load_b_panel(  // streamk: B panel for one warp cols
    sk_bf16* __restrict__ Bs,  // streamk: smem [16*16]
    const sk_bf16* __restrict__ B,  // streamk: global B[K,N]
    int N, int K_unused, int nw, int kk, int lane) {  // streamk: n-origin + k
  (void)K_unused;  // streamk: B stride is N not K
  for (int i = lane; i < 16 * 16; i += 32) {  // streamk: warp-coop fill
    const int r = i >> 4;  // streamk: k in panel
    const int c = i & 15;  // streamk: n in panel
    const int nc = nw + c;  // streamk: global n
    Bs[i] = (nc < N) ? B[static_cast<size_t>(kk + r) * N + nc] : sk_bf16{};  // streamk: edge zero-fill
  }  // streamk: end B load loop
}  // streamk: end load_b_panel

__global__ void streamk_bf16_kernel(  // streamk: unified full-K / Stream-K launch
    const sk_bf16* __restrict__ A,  // streamk: [M,K]
    const sk_bf16* __restrict__ B,  // streamk: [K,N] = W[:n].T
    sk_bf16* __restrict__ C,  // streamk: [M,N]
    float* __restrict__ partials,  // streamk: workspace for split partials
    int* __restrict__ flags,  // streamk: per-tile arrival counts
    int M, int N, int K,  // streamk: gemm dims
    int splits,  // streamk: K-split count (1 => full-K write)
    int tile_stride) {  // streamk: partials middle dim (=max tiles) for shared workspace
  const int tiles_m = (M + kBM - 1) / kBM;  // streamk: M tiles
  const int tiles_n = (N + kBN - 1) / kBN;  // streamk: N tiles
  const int num_tiles = tiles_m * tiles_n;  // streamk: MN tile count
  const int work = num_tiles * splits;  // streamk: total stream-k work items
  const int tid = threadIdx.x;  // streamk: linear tid
  const int warp = tid >> 5;  // streamk: warp id
  const int lane = tid & 31;  // streamk: lane id

  __shared__ sk_bf16 As[16 * 16];  // streamk: A panel smem
  __shared__ sk_bf16 Bs[kWarps * 16 * 16];  // streamk: per-warp B panels
  __shared__ float Cs[kWarps * 16 * 16];  // streamk: per-warp acc stage

  for (int wid = static_cast<int>(blockIdx.x); wid < work; wid += static_cast<int>(gridDim.x)) {  // streamk: grid-stride work
    const int tile = wid / splits;  // streamk: MN tile index
    const int split = wid - tile * splits;  // streamk: K-split index
    const int tm = tile / tiles_n;  // streamk: tile m
    const int tn = tile - tm * tiles_n;  // streamk: tile n
    const int m0 = tm * kBM;  // streamk: m origin
    const int n0 = tn * kBN;  // streamk: n origin
    const int nw = n0 + warp * 16;  // streamk: warp n origin

    const int k_iters = (K + kBK - 1) / kBK;  // streamk: K iterations
    const int iters_per = (k_iters + splits - 1) / splits;  // streamk: iters per split
    const int it0 = split * iters_per;  // streamk: split start
    const int it1 = min(it0 + iters_per, k_iters);  // streamk: split end

    ns::fragment<ns::matrix_a, 16, 16, 16, sk_bf16, ns::row_major> a_frag;  // streamk: A frag
    ns::fragment<ns::matrix_b, 16, 16, 16, sk_bf16, ns::row_major> b_frag;  // streamk: B frag
    ns::fragment<ns::accumulator, 16, 16, 16, float> c_frag;  // streamk: fp32 acc
    ns::fill_fragment(c_frag, 0.f);  // streamk: clear acc

    const bool active = (m0 < M) && (nw < N);  // streamk: skip OOB warps
    for (int it = it0; it < it1; ++it) {  // streamk: K mainloop for this split
      const int kk = it * kBK;  // streamk: k origin
      load_a_panel(As, A, M, K, m0, kk, tid);  // streamk: stage A
      __syncthreads();  // streamk: A visible
      if (active) {  // streamk: compute path
        ns::load_matrix_sync(a_frag, As, 16);  // streamk: A -> frag
        load_b_panel(Bs + warp * 16 * 16, B, N, K, nw, kk, lane);  // streamk: stage B
        __syncwarp();  // streamk: B visible in warp
        ns::load_matrix_sync(b_frag, Bs + warp * 16 * 16, 16);  // streamk: B -> frag
        ns::mma_sync(c_frag, a_frag, b_frag, c_frag);  // streamk: WMMA mma
      }  // streamk: end active
      __syncthreads();  // streamk: next iter barrier
    }  // streamk: end K loop
    if (active) {  // streamk: stage acc to smem
      ns::store_matrix_sync(Cs + warp * 16 * 16, c_frag, 16, ns::mem_row_major);  // streamk: acc -> Cs
    }  // streamk: end store
    __syncthreads();  // streamk: Cs ready for write/reduce

    if (splits == 1) {  // streamk: full-K path — direct C write
      if (m0 < M && nw < N) {  // streamk: bounds
        const float* sp = Cs + warp * 16 * 16;  // streamk: warp acc
        for (int i = lane; i < 16 * 16; i += 32) {  // streamk: coop store
          const int r = i >> 4;  // streamk: r
          const int c = i & 15;  // streamk: c
          const int mr = m0 + r;  // streamk: global m
          const int nc = nw + c;  // streamk: global n
          if (mr < M && nc < N) {  // streamk: edge guard
            C[static_cast<size_t>(mr) * N + nc] = sk_f32_bf16(sp[i]);  // streamk: write C
          }  // streamk: end guard
        }  // streamk: end store loop
      }  // streamk: end bounds
    } else {  // streamk: fused reduce path — avoid host splitKreduce
      float* part = partials + (static_cast<size_t>(split) * tile_stride + tile) * (kBM * kBN);  // streamk: partial slot (max-stride)
      if (m0 < M && nw < N) {  // streamk: write partial
        const float* sp = Cs + warp * 16 * 16;  // streamk: warp acc
        for (int i = lane; i < 16 * 16; i += 32) {  // streamk: coop partial store
          const int r = i >> 4;  // streamk: r
          const int c = i & 15;  // streamk: c
          part[r * kBN + warp * 16 + c] =  // streamk: layout [BM,BN]
              (m0 + r < M && nw + c < N) ? sp[i] : 0.f;  // streamk: zero pad
        }  // streamk: end partial store
      }  // streamk: end partial write
      __syncthreads();  // streamk: partials settled in-block
      if (tid == 0) {  // streamk: one thread reduces when last split arrives
        __threadfence();  // streamk: publish partials before arrival flag (cross-block)
        const int old = atomicAdd(flags + tile, 1);  // streamk: arrival count
        if (old == splits - 1) {  // streamk: last split does fused reduce
          for (int r = 0; r < kBM; ++r) {  // streamk: reduce rows
            const int mr = m0 + r;  // streamk: global m
            if (mr >= M) break;  // streamk: m edge
            for (int c = 0; c < kBN; ++c) {  // streamk: reduce cols
              const int nc = n0 + c;  // streamk: global n
              if (nc >= N) break;  // streamk: n edge
              float sum = 0.f;  // streamk: acc
              for (int s = 0; s < splits; ++s) {  // streamk: sum splits
                sum += partials[(static_cast<size_t>(s) * tile_stride + tile) * (kBM * kBN)  // streamk: max-stride idx
                                + r * kBN + c];  // streamk: element
              }  // streamk: end sum
              C[static_cast<size_t>(mr) * N + nc] = sk_f32_bf16(sum);  // streamk: write reduced C
            }  // streamk: end c
          }  // streamk: end r
        }  // streamk: end last-split
      }  // streamk: end tid0
      __syncthreads();  // streamk: before next work item
    }  // streamk: end split branch
  }  // streamk: end grid-stride
}  // streamk: end kernel

void streamk_bf16(  // streamk: host launch
    torch::Tensor A,  // streamk: [M,K]
    torch::Tensor B,  // streamk: [K,N]
    torch::Tensor C,  // streamk: [M,N]
    torch::Tensor partials,  // streamk: workspace
    torch::Tensor flags,  // streamk: flags
    int64_t splits,  // streamk: split count
    int64_t tile_stride) {  // streamk: partials stride (=max tiles)
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda());  // streamk: device check
  const int M = A.size(0);  // streamk: M
  const int K = A.size(1);  // streamk: K
  const int N = B.size(1);  // streamk: N
  TORCH_CHECK(B.size(0) == K);  // streamk: K match
  TORCH_CHECK(K % 16 == 0);  // streamk: WMMA K align
  const int sp = static_cast<int>(splits);  // streamk: splits
  TORCH_CHECK(sp >= 1);  // streamk: valid splits
  const int tstride = static_cast<int>(tile_stride);  // streamk: workspace stride
  TORCH_CHECK(tstride >= 1 || sp == 1);  // streamk: stride unused when full-K

  const int tiles_m = (M + kBM - 1) / kBM;  // streamk: tiles m
  const int tiles_n = (N + kBN - 1) / kBN;  // streamk: tiles n
  const int num_tiles = tiles_m * tiles_n;  // streamk: MN tiles
  const int work = num_tiles * sp;  // streamk: work items

  cudaDeviceProp prop{};  // streamk: query SMs for grid
  cudaGetDeviceProperties(&prop, A.get_device());  // streamk: device props
  int grid = std::min(work, std::max(1, prop.multiProcessorCount * 2));  // streamk: occupancy ~2 blocks/SM

  auto stream = at::cuda::getCurrentCUDAStream();  // streamk: current stream (graph-safe)
  streamk_bf16_kernel<<<grid, kThreads, 0, stream>>>(  // streamk: one launch
      reinterpret_cast<const sk_bf16*>(A.data_ptr()),  // streamk: A
      reinterpret_cast<const sk_bf16*>(B.data_ptr()),  // streamk: B
      reinterpret_cast<sk_bf16*>(C.data_ptr()),  // streamk: C
      partials.data_ptr<float>(),  // streamk: partials
      flags.data_ptr<int>(),  // streamk: flags
      M, N, K, sp, tstride);  // streamk: dims + splits + stride
}  // streamk: end host
"""

_CPP_DECL = r"""
void hopper_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl);
void streamk_bf16(torch::Tensor A, torch::Tensor B, torch::Tensor C,  // streamk: decl
                  torch::Tensor partials, torch::Tensor flags, int64_t splits,  // streamk: workspace
                  int64_t tile_stride);  // streamk: max-tiles stride
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


def _choose_splits(m: int, n: int, k: int, sms: int) -> int:  # streamk: splits heuristic
    tiles = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)  # streamk: MN tiles
    k_iters = (k + BK - 1) // BK  # streamk: K iters
    if tiles >= sms or m >= 64:  # streamk: enough MN parallelism → full-K
        return 1  # streamk: no partials
    target = sms * 2  # streamk: aim ~2 blocks/SM
    sp = (target + tiles - 1) // tiles  # streamk: splits to fill SMs
    return max(1, min(k_iters, sp, 32))  # streamk: clamp splits


def _use_streamk(layer: str, m: int) -> bool:  # streamk: allow-list gate
    return m in _STREAMK_ALLOW.get(layer, ())  # streamk: dispatch


def _out_n(layer: str, m: int, n_full: int) -> int:  # streamk: unpooler slice at m<=8
    if layer == "unpooler" and m <= 8:  # streamk: match reference out_features
        return 1024  # streamk: sliced N
    return n_full  # streamk: full N


def _build_ext(device: torch.device):
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    gencode = f"-gencode=arch=compute_{major}{minor},code={arch}"
    build_dir = Path.home() / ".cache" / "muse_bf16_impl_i_streamk" / arch  # streamk: isolate build cache
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"muse_bf16_impl_i_streamk_{arch}",  # streamk: distinct ext name
        cpp_sources=[_CPP_DECL],
        cuda_sources=[_CUDA_SRC],
        functions=["hopper_bf16_gemv", "streamk_bf16"],  # streamk: export Stream-K + G gemv
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            gencode,
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
    state["sk_splits"] = {}  # streamk: per (layer,m) split plan
    state["sk_out"] = {}  # streamk: prealloc outs (graph-safe)
    state["sk_B"] = {}  # streamk: contiguous B=W[:n].T (slice of _t may be non-contig)
    state["sk_partials"] = None  # streamk: max-size partials workspace
    state["sk_flags"] = None  # streamk: max-size flags workspace
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        state["use_pdl"] = major >= 9
        state["ext"] = _build_ext(device)
        state["mm_backend"] = _probe_mm_bf16_backend(device)
        if state["mm_backend"] is not None:
            from flashinfer.gemm import mm_bf16

            state["mm_bf16"] = mm_bf16

        sms = torch.cuda.get_device_properties(device).multi_processor_count  # streamk: for splits
        max_sp, max_tiles = 1, 0  # streamk: workspace bounds
        for layer, ms in _STREAMK_ALLOW.items():  # streamk: plan each allow-list shape
            n_full = state[layer].shape[0]  # streamk: full N
            k = state[layer].shape[1]  # streamk: K
            for m in ms:  # streamk: per-m splits
                n = _out_n(layer, m, n_full)  # streamk: effective N
                sp = _choose_splits(m, n, k, sms)  # streamk: choose splits
                key = (layer, m)  # streamk: key
                state["sk_splits"][key] = sp  # streamk: store splits
                state["sk_out"][key] = torch.empty(  # streamk: prealloc out
                    (m, n), device=device, dtype=torch.bfloat16  # streamk: graph-safe buffer
                )  # streamk: end out alloc
                bkey = (layer, n)  # streamk: pack B once per (layer,n)
                if bkey not in state["sk_B"]:  # streamk: avoid duplicate packs
                    state["sk_B"][bkey] = state[f"{layer}_t"][:, :n].contiguous()  # streamk: force contig for kernel
                state["sk_B"][key] = state["sk_B"][bkey]  # streamk: alias by (layer,m)
                tiles = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)  # streamk: tile count
                max_sp = max(max_sp, sp)  # streamk: max splits
                max_tiles = max(max_tiles, tiles)  # streamk: max tiles
        if max_tiles > 0:  # streamk: alloc once at max size
            state["sk_partials"] = torch.empty(  # streamk: partials workspace
                (max_sp, max_tiles, BM * BN), device=device, dtype=torch.float32  # streamk: [sp,tiles,BM*BN]
            )  # streamk: end partials
            state["sk_flags"] = torch.zeros(  # streamk: flags workspace
                (max_tiles,), device=device, dtype=torch.int32  # streamk: one flag/tile
            )  # streamk: end flags
            state["sk_tile_stride"] = max_tiles  # streamk: middle-dim stride for shared buf
        # streamk: optional warm launch (untimed prepare)
        ext = state["ext"]
        if ext is not None and state["sk_partials"] is not None:  # streamk: warm path
            for layer, m in (("qkv_gate", 8), ("pooler", 16), ("unpooler", 8)):  # streamk: sample warm incl sliced N
                key = (layer, m)  # streamk: key
                if key not in state["sk_out"]:  # streamk: skip missing
                    continue  # streamk: continue
                k = state[layer].shape[1]  # streamk: K
                x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)  # streamk: dummy x
                sp = int(state["sk_splits"][key])  # streamk: splits
                if sp > 1:  # streamk: reset flags for fused reduce
                    state["sk_flags"].zero_()  # streamk: clear flags
                ext.streamk_bf16(  # streamk: warm launch
                    x,  # streamk: A
                    state["sk_B"][key],  # streamk: contig B
                    state["sk_out"][key],  # streamk: C
                    state["sk_partials"],  # streamk: partials
                    state["sk_flags"],  # streamk: flags
                    sp,  # streamk: splits
                    int(state["sk_tile_stride"]),  # streamk: max-tiles stride
                )  # streamk: end warm
            torch.cuda.synchronize(device)  # streamk: settle JIT (prepare only)
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

    if (  # streamk: Stream-K dispatch before FI
        ext is not None  # streamk: ext ready
        and _use_streamk(layer, m)  # streamk: allow-list
        and state["sk_partials"] is not None  # streamk: workspace ready
    ):  # streamk: enter Stream-K
        key = (layer, m)  # streamk: lookup key
        out = state["sk_out"][key]  # streamk: prealloc out (no alloc in kernel_fn)
        sp = int(state["sk_splits"][key])  # streamk: planned splits
        x2 = x.view(-1, x.shape[-1])  # streamk: [M,K]
        if sp > 1:  # streamk: fused reduce needs clear flags (async, no host sync)
            state["sk_flags"].zero_()  # streamk: reset arrivals
        ext.streamk_bf16(  # streamk: one launch
            x2,  # streamk: A
            state["sk_B"][key],  # streamk: contig B from prepare
            out,  # streamk: C
            state["sk_partials"],  # streamk: max-size partials
            state["sk_flags"],  # streamk: max-size flags
            sp,  # streamk: splits
            int(state["sk_tile_stride"]),  # streamk: workspace stride
        )  # streamk: end launch
        return out.view(*x.shape[:-1], n)  # streamk: reshape like input batching

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
