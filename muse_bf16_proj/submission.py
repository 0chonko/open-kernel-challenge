# okc: problem=muse_bf16_proj
"""Impl D: hopper GEMV m=1 + thin-m BF16 GEMV (m=8/16/32) for qkv/pooler/unpooler.

Targets f2a7 m∈{8,16,32} nvjet+splitKreduce (~8–18%). Stages x in smem (or K-tiles),
streams each W row once. Router stays baseline (impl A scalar FP32 lost badly).
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

// {{Ladder + Stream-K intent}}{{Thin-m full-K GEMV: kill splitKreduce by one launch; x in smem (or K-tiles), W streamed once.}}
__global__ void __launch_bounds__(256)
    thin_m_bf16_gemv_kernel(bf16_t* __restrict__ out, const bf16_t* __restrict__ x,
                            const bf16_t* __restrict__ w, int M, int N, int K, int full) {
  extern __shared__ bf16_t sx[];
  const uint32_t tid = threadIdx.x;
  const uint32_t nthreads = blockDim.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t warps = nthreads / 32;
  constexpr uint32_t kRows = 2;
  const uint32_t r0 = (blockIdx.x * warps + warp) * kRows;
  if (r0 >= static_cast<uint32_t>(N)) return;
  const bool two = (r0 + 1) < static_cast<uint32_t>(N);

  constexpr int BK = 1024;

  float acc0[32];
  float acc1[32];
  // M capped at 32 by host.
  for (int mi = 0; mi < M; ++mi) {
    acc0[mi] = 0.f;
    acc1[mi] = 0.f;
  }

  if (full) {
    const size_t x_elems = static_cast<size_t>(M) * K;
    for (uint32_t i = tid * kGemvVecSize; i < x_elems; i += nthreads * kGemvVecSize) {
      *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
    }
    __syncthreads();
    for (uint32_t k = lane * kGemvVecSize; k < static_cast<uint32_t>(K); k += 32 * kGemvVecSize) {
      float4 wv0 = __ldcs(reinterpret_cast<const float4*>(w + static_cast<size_t>(r0) * K + k));
      float4 wv1;
      if (two) wv1 = __ldcs(reinterpret_cast<const float4*>(w + static_cast<size_t>(r0 + 1) * K + k));
      for (int mi = 0; mi < M; ++mi) {
        float4 xv = *reinterpret_cast<const float4*>(sx + static_cast<size_t>(mi) * K + k);
        acc0[mi] += dot8_f32(wv0, xv);
        if (two) acc1[mi] += dot8_f32(wv1, xv);
      }
    }
  } else {
    for (int k0 = 0; k0 < K; k0 += BK) {
      const int kb = min(BK, K - k0);
      const size_t tile_elems = static_cast<size_t>(M) * kb;
      for (uint32_t i = tid * kGemvVecSize; i < tile_elems; i += nthreads * kGemvVecSize) {
        const uint32_t mi = i / kb;
        const uint32_t kk = i % kb;
        *reinterpret_cast<float4*>(sx + i) =
            *reinterpret_cast<const float4*>(x + static_cast<size_t>(mi) * K + k0 + kk);
      }
      __syncthreads();
      for (uint32_t k = lane * kGemvVecSize; k < static_cast<uint32_t>(kb); k += 32 * kGemvVecSize) {
        float4 wv0 = __ldcs(reinterpret_cast<const float4*>(w + static_cast<size_t>(r0) * K + k0 + k));
        float4 wv1;
        if (two) wv1 = __ldcs(reinterpret_cast<const float4*>(w + static_cast<size_t>(r0 + 1) * K + k0 + k));
        for (int mi = 0; mi < M; ++mi) {
          float4 xv = *reinterpret_cast<const float4*>(sx + static_cast<size_t>(mi) * kb + k);
          acc0[mi] += dot8_f32(wv0, xv);
          if (two) acc1[mi] += dot8_f32(wv1, xv);
        }
      }
      __syncthreads();
    }
  }

  for (int mi = 0; mi < M; ++mi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc0[mi] += __shfl_down_sync(0xffffffff, acc0[mi], off);
      acc1[mi] += __shfl_down_sync(0xffffffff, acc1[mi], off);
    }
    if (lane == 0) {
      out[static_cast<size_t>(mi) * N + r0] = cast<bf16_t>(acc0[mi]);
      if (two) out[static_cast<size_t>(mi) * N + r0 + 1] = cast<bf16_t>(acc1[mi]);
    }
  }
}

void thin_m_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16);
  const int M = x.size(0);
  const int K = x.size(1);
  const int N = w.size(0);
  TORCH_CHECK(M >= 2 && M <= 32);
  TORCH_CHECK(w.size(1) == K && out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(K % 8 == 0 && N % 2 == 0);

  constexpr int nthreads = 256;
  const int rows_per_block = (nthreads / 32) * 2;
  const int blocks = (N + rows_per_block - 1) / rows_per_block;
  const size_t full_bytes = static_cast<size_t>(M) * K * sizeof(bf16_t);
  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, x.get_device());
  const int full = (full_bytes <= static_cast<size_t>(optin)) ? 1 : 0;
  const size_t smem = full ? full_bytes : static_cast<size_t>(M) * 1024 * sizeof(bf16_t);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* kn = thin_m_bf16_gemv_kernel;
  cudaFuncSetAttribute(kn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  kn<<<blocks, nthreads, smem, stream>>>(
      reinterpret_cast<bf16_t*>(out.data_ptr()),
      reinterpret_cast<const bf16_t*>(x.data_ptr()),
      reinterpret_cast<const bf16_t*>(w.data_ptr()),
      M, N, K, full);
}
"""

_CPP_DECL = r"""
void hopper_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl);
void thin_m_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out);
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


def _use_thin_m_bf16_gemv(m: int, n: int, k: int) -> bool:
    # Scored thin-m shapes that still show splitKreduce on f2a7 when using cuBLAS.
    if m not in (8, 16, 32):
        return False
    if not (k % 8 == 0 and 512 <= k <= _MAX_K and n % 8 == 0 and 64 <= n <= _MAX_N):
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
    gencode = f"-gencode=arch=compute_{major}{minor},code={arch}"
    build_dir = Path.home() / ".cache" / "muse_bf16_impl_d_v2_thinm_v2" / arch
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"muse_bf16_impl_d_v2_{arch}",
        cpp_sources=[_CPP_DECL],
        cuda_sources=[_CUDA_SRC],
        functions=["hopper_bf16_gemv", "thin_m_bf16_gemv"],
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


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {
        name: _align16(w.contiguous()) for name, w in weights.items()
    }
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()

    device = router.device
    state["ext"] = None
    state["use_pdl"] = False
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        state["use_pdl"] = major >= 9
        state["ext"] = _build_ext(device)
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
    aligned = (
        ext is not None
        and x.is_contiguous()
        and W.is_contiguous()
        and x.data_ptr() % 16 == 0
        and W.data_ptr() % 16 == 0
    )
    if aligned and _use_hopper_bf16_gemv(m, W.shape[0], W.shape[1]):
        x1 = _align16(x.view(-1, x.shape[-1]))
        out = torch.empty((1, W.shape[0]), dtype=x.dtype, device=x.device)
        ext.hopper_bf16_gemv(x1, W, out, bool(state["use_pdl"]))
        return out.view(*x.shape[:-1], -1)
    if aligned and _use_thin_m_bf16_gemv(m, W.shape[0], W.shape[1]):
        x2 = x.view(-1, x.shape[-1])
        out = torch.empty((x2.shape[0], W.shape[0]), dtype=x.dtype, device=x.device)
        ext.thin_m_bf16_gemv(x2, W, out)
        return out.view(*x.shape[:-1], -1)

    return F.linear(x, W, None)
