# okc: problem=muse_bf16_proj
"""m: L, plus the m=1 packed GEMV.

Same router, same hopper GEMV, same m>1 path as L. m=1 BF16 projections use the
12-bit pack. Each lane holds its 8 x values in registers for all 16 rows and
reduces once per row.
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

  if (N == 2304 && K == 8192) DISPATCH(2304, 8192, 1, 2, 8);  // l-m8-gate: unroll 2 (unroll 4 was 0.91x)
  if (N == 8192 && K == 1024) DISPATCH(8192, 1024, 2, 2, 8);
  if (N == 4096 && K == 8192) DISPATCH(4096, 8192, 1, 2, 8);
  if (N == 1024 && K == 4096) DISPATCH(1024, 4096, 1, 2, 8);
  if (N == 8192 && K == 4096) DISPATCH(8192, 4096, 2, 2, 8);

  TORCH_CHECK(false, "hopper_bf16_gemv: unsupported Muse shape N=", N, " K=", K);
#undef DISPATCH
}

// 12-bit pack: low byte kept, high byte is sign plus a 3-bit window into 8 exponents.
// Outliers stay clamped in the stream. One lane per row applies the exact residual.
constexpr int kZMaxExc = 32;
constexpr int kZWarps = 4;

__device__ __forceinline__ uint2 ldg8_cg(const void* p) {
  unsigned lo, hi;
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n" : "=r"(lo), "=r"(hi) : "l"(p));
  return uint2{lo, hi};
}

__device__ __forceinline__ unsigned ldg_u32_cg(const void* p) {
  unsigned v;
  asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ float2 cvt_bf16x2(unsigned v) {
  return cast<fp32x2_t>(*reinterpret_cast<bf16x2_t*>(&v));
}

__device__ __forceinline__ void z_decode(unsigned l0, unsigned l1, unsigned nib, int base, float w[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    unsigned b = (nib >> (8 * i)) & 0xffu;
    unsigned h0 = ((b & 8u) << 4) | (unsigned)(base + (int)(b & 7u));
    unsigned h1 = (b & 0x80u) | (unsigned)(base + (int)((b >> 4) & 7u));
    unsigned src = (i < 2) ? l0 : l1;
    unsigned sh = (i & 1) ? 16u : 0u;
    unsigned packed = ((src >> sh) & 0xffu) | (h0 << 8) | (((src >> (sh + 8)) & 0xffu) << 16) | (h1 << 24);
    float2 f = cvt_bf16x2(packed);
    w[2 * i] = f.x;
    w[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ float z_clamped(const uint8_t* rowp, int k, int base) {
  int g = k >> 3;
  int i = k & 7;
  const uint8_t* chunkp = rowp + (g >> 5) * 384;
  int lane = g & 31;
  unsigned low = chunkp[lane * 8 + i];
  unsigned nib = ldg_u32_cg(chunkp + 256 + lane * 4);
  unsigned n = (nib >> (4 * i)) & 0xfu;
  unsigned h = ((n >> 3) << 7) | (unsigned)(base + (int)(n & 7u));
  return __bfloat162float(__ushort_as_bfloat16((unsigned short)((h << 8) | low)));
}

__global__ void z_pack_kernel(const bf16_t* __restrict__ w, uint8_t* __restrict__ payload, int16_t* __restrict__ exc_k,
                              float* __restrict__ exc_w, int* __restrict__ exc_count, int N, int K, int base) {
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= N) return;
  const int nchunks = K >> 8;
  uint8_t* rowp = payload + (size_t)row * nchunks * 384;
  const uint16_t* src = reinterpret_cast<const uint16_t*>(w + (size_t)row * K);
  int slots = 0;
  bool overflow = false;
  const int groups = K >> 3;
  for (int g = 0; g < groups; ++g) {
    unsigned l0 = 0, l1 = 0, nib = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      unsigned bits = src[g * 8 + i];
      unsigned low = bits & 0xffu;
      unsigned e7 = (bits >> 8) & 0x7fu;
      unsigned sign = bits >> 15;
      int code = (int)e7 - base;
      bool bad = code < 0 || code > 7;
      if (code < 0) code = 0;
      if (code > 7) code = 7;
      if (i < 4) l0 |= low << (8 * i);
      else l1 |= low << (8 * (i - 4));
      nib |= ((sign << 3) | (unsigned)code) << (4 * i);
      if (bad) {
        if (slots < kZMaxExc) {
          size_t idx = (size_t)row * kZMaxExc + slots;
          exc_k[idx] = (int16_t)(g * 8 + i);
          exc_w[idx] = __bfloat162float(__ushort_as_bfloat16((unsigned short)bits));
          ++slots;
        } else {
          overflow = true;
        }
      }
    }
    int lane = g & 31;
    uint8_t* chunkp = rowp + (g >> 5) * 384;
    *reinterpret_cast<uint2*>(chunkp + lane * 8) = make_uint2(l0, l1);
    *reinterpret_cast<unsigned*>(chunkp + 256 + lane * 4) = nib;
  }
  exc_count[row] = overflow ? (kZMaxExc + 1) : slots;
}

template <int M, int N, int K, int SPLITS>
__global__ void __launch_bounds__((kZWarps + 1) * 32)
    z_gemv_kernel(const bf16_t* __restrict__ x, const uint8_t* __restrict__ payload, const int16_t* __restrict__ exc_k,
                  const float* __restrict__ exc_w, const int* __restrict__ exc_count, float* __restrict__ partial,
                  bf16_t* __restrict__ out, int base) {
  static_assert(M == 1, "register x is the m=1 schedule");
  constexpr int CW = kZWarps;
  constexpr int Ksplit = K / SPLITS;
  constexpr int warpK = Ksplit / CW;
  static_assert(warpK % 256 == 0, "z tile");
  constexpr int nchunks = K >> 8;

  extern __shared__ __align__(16) char smem[];
  float* part = reinterpret_cast<float*>(smem);
  float* fix = part + CW * 16 * M;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int split = (int)blockIdx.y;
  const int row0 = (int)blockIdx.x * 16;
  const int k0 = split * Ksplit;

  if (warp == CW) {
    if (lane == 0) {
      for (int r = 0; r < 16; ++r) {
        fix[r * M] = 0.f;
        int cnt = exc_count[row0 + r];
        if (cnt > kZMaxExc) cnt = kZMaxExc;
        const int16_t* ek = exc_k + (size_t)(row0 + r) * kZMaxExc;
        const float* ew = exc_w + (size_t)(row0 + r) * kZMaxExc;
        const uint8_t* rowp = payload + (size_t)(row0 + r) * nchunks * 384;
        for (int e = 0; e < cnt; ++e) {
          int kk = (int)ek[e];
          if (kk < k0 || kk >= k0 + Ksplit) continue;
          float wc = z_clamped(rowp, kk, base);
          float wt = ew[e];
          float xv = __bfloat162float(x[(size_t)kk]);
          fix[r * M] = __fmaf_rn(wt, xv, __fmaf_rn(-wc, xv, fix[r * M]));
        }
      }
    }
  } else {
    float rowacc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) rowacc[r] = 0.f;
    const int w0 = k0 + warp * warpK;
    for (int kbase = w0; kbase < w0 + warpK; kbase += 256) {
      float4 x4 = *reinterpret_cast<const float4*>(x + kbase + lane * 8);
      unsigned bits[4] = {__float_as_uint(x4.x), __float_as_uint(x4.y), __float_as_uint(x4.z), __float_as_uint(x4.w)};
      float xv[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f = cvt_bf16x2(bits[i]);
        xv[2 * i] = f.x;
        xv[2 * i + 1] = f.y;
      }
      int chunk = kbase >> 8;
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        const uint8_t* chunkp = payload + ((size_t)(row0 + r) * nchunks + chunk) * 384;
        uint2 lows = ldg8_cg(chunkp + lane * 8);
        unsigned nib = ldg_u32_cg(chunkp + 256 + lane * 4);
        float wv[8];
        z_decode(lows.x, lows.y, nib, base, wv);
#pragma unroll
        for (int j = 0; j < 8; ++j) rowacc[r] = __fmaf_rn(wv[j], xv[j], rowacc[r]);
      }
    }
    float* mypart = part + warp * 16 * M;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      float v = rowacc[r];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
      if (lane == 0) mypart[r * M] = v;
    }
  }
  __syncthreads();

  for (int idx = (int)threadIdx.x; idx < 16 * M; idx += (int)blockDim.x) {
    float s = fix[idx];
#pragma unroll
    for (int w = 0; w < CW; ++w) s += part[w * 16 * M + idx];
    int col = row0 + idx;
    if constexpr (SPLITS == 1) {
      out[col] = cast<bf16_t>(s);
    } else {
      partial[(size_t)split * N + col] = s;
    }
  }
}

template <int M, int N, int K, int SPLITS>
void launch_z(const bf16_t* x, const uint8_t* payload, const int16_t* exc_k, const float* exc_w, const int* exc_count,
              float* partial, bf16_t* out, int base, cudaStream_t stream) {
  constexpr int CW = kZWarps;
  int smem = (CW * 16 * M + 16 * M) * (int)sizeof(float);
  auto* fn = z_gemv_kernel<M, N, K, SPLITS>;
  static int ready = 0;
  if (!ready) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    ready = 1;
  }
  fn<<<dim3(N / 16, SPLITS), (CW + 1) * 32, smem, stream>>>(x, payload, exc_k, exc_w, exc_count, partial, out, base);
}

__global__ void z_reduce_kernel(bf16_t* __restrict__ out, const float* __restrict__ partial, int total, int splits) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) return;
  float s = 0.f;
  for (int sp = 0; sp < splits; ++sp) s += partial[(size_t)sp * total + idx];
  out[idx] = cast<bf16_t>(s);
}

void z_pack(torch::Tensor w, torch::Tensor payload, torch::Tensor exc_k, torch::Tensor exc_w, torch::Tensor exc_count,
            int64_t base) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kBFloat16);
  const int N = (int)w.size(0);
  const int K = (int)w.size(1);
  TORCH_CHECK(K % 256 == 0 && N % 16 == 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  z_pack_kernel<<<(N + 127) / 128, 128, 0, stream>>>(
      reinterpret_cast<const bf16_t*>(w.data_ptr()), payload.data_ptr<uint8_t>(), exc_k.data_ptr<int16_t>(),
      exc_w.data_ptr<float>(), exc_count.data_ptr<int>(), N, K, (int)base);
}

void z_gemv(torch::Tensor x, torch::Tensor payload, torch::Tensor exc_k, torch::Tensor exc_w, torch::Tensor exc_count,
            torch::Tensor out, torch::Tensor partial, int64_t base, int64_t M64, int64_t N64, int64_t K64) {
  const int M = (int)M64, N = (int)N64, K = (int)K64;
  TORCH_CHECK(x.is_cuda() && out.is_cuda() && x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.size(0) == M && x.size(1) == K && out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(payload.numel() >= (int64_t)N * (K / 256) * 384);
  TORCH_CHECK(exc_count.numel() >= N);
  const int splits = (N == 1024 && K == 4096) ? 4 : 1;
  if (splits > 1) TORCH_CHECK(partial.numel() >= (int64_t)splits * M * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  auto* pay = payload.data_ptr<uint8_t>();
  auto* ek = exc_k.data_ptr<int16_t>();
  auto* ew = exc_w.data_ptr<float>();
  auto* ec = exc_count.data_ptr<int>();
  auto* op = reinterpret_cast<bf16_t*>(out.data_ptr());
  float* pp = splits > 1 ? partial.data_ptr<float>() : nullptr;

#define LZ(MM, NN, KK, SP)                                                                                           \
  if (M == MM && N == NN && K == KK && splits == SP) {                                                               \
    launch_z<MM, NN, KK, SP>(xp, pay, ek, ew, ec, pp, op, (int)base, stream);                                       \
    if (SP > 1) {                                                                                                    \
      int total = MM * NN;                                                                                          \
      z_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(op, pp, total, SP);                                  \
    }                                                                                                                \
    return;                                                                                                          \
  }
  LZ(1, 2304, 8192, 1);
  LZ(1, 8192, 1024, 1);
  LZ(1, 4096, 8192, 1);
  LZ(1, 1024, 4096, 4);
  LZ(1, 8192, 4096, 1);
#undef LZ
  TORCH_CHECK(false, "z_gemv: unsupported M=", M, " N=", N, " K=", K);
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

// {{serving GEMV}}{{FP32 router: K-split grid (~512 CTAs), stage W tile, one FP32 reduce. N=256 cannot fill 132 SMs otherwise.}}
constexpr int kRouterN = 256;
constexpr int kRouterK = 8192;
constexpr int kRouterSplits = 8;
constexpr int kRouterWarps = 4;
constexpr int kRouterKChunk = kRouterK / kRouterSplits;  // 1024

template <int BM>
__global__ void __launch_bounds__(kRouterWarps * 32)
    router_fp32_splitk_kernel(float* __restrict__ partial, const bf16_t* __restrict__ x,
                              const float* __restrict__ w, int M) {
  constexpr int kVecIters = (kRouterKChunk / 4) / 32;  // 8
  static_assert(kRouterKChunk / 8 == kRouterWarps * 32, "one x float4 per thread");
  extern __shared__ __align__(16) char smem[];
  float* sw = reinterpret_cast<float*>(smem);
  bf16_t* sx = reinterpret_cast<bf16_t*>(sw + kRouterWarps * kRouterKChunk);

  const int split = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row = static_cast<int>(blockIdx.x) * kRouterWarps + warp;
  const int k0 = split * kRouterKChunk;

  const float4* wrow4 = reinterpret_cast<const float4*>(w + (static_cast<size_t>(row) * kRouterK + k0));
  float4* srow4 = reinterpret_cast<float4*>(sw + warp * kRouterKChunk);
  float4 wreg[kVecIters];
#pragma unroll
  for (int s = 0; s < kVecIters; ++s) wreg[s] = __ldcs(wrow4 + lane + s * 32);
#pragma unroll
  for (int s = 0; s < kVecIters; ++s) srow4[lane + s * 32] = wreg[s];
  __syncthreads();

  for (int m0 = 0; m0 < M; m0 += BM) {
#pragma unroll
    for (int t = 0; t < BM; ++t) {
      const bf16_t* src = x + (static_cast<size_t>(m0 + t) * kRouterK + k0);
      bf16_t* dst = sx + t * kRouterKChunk;
      *reinterpret_cast<float4*>(dst + tid * 8) = *reinterpret_cast<const float4*>(src + tid * 8);
    }
    __syncthreads();

    float acc[BM];
#pragma unroll
    for (int t = 0; t < BM; ++t) acc[t] = 0.f;

    const float* myw = sw + warp * kRouterKChunk;
#pragma unroll 1
    for (int s = 0; s < kVecIters; ++s) {
      int v = lane + s * 32;
      float4 wv = reinterpret_cast<const float4*>(myw)[v];
      int kk = v * 4;
#pragma unroll
      for (int t = 0; t < BM; ++t) {
        const bf16x2_t* xp = reinterpret_cast<const bf16x2_t*>(sx + t * kRouterKChunk + kk);
        float2 a = __bfloat1622float2(xp[0]);
        float2 b = __bfloat1622float2(xp[1]);
        acc[t] = __fmaf_rn(wv.x, a.x, acc[t]);
        acc[t] = __fmaf_rn(wv.y, a.y, acc[t]);
        acc[t] = __fmaf_rn(wv.z, b.x, acc[t]);
        acc[t] = __fmaf_rn(wv.w, b.y, acc[t]);
      }
    }
#pragma unroll
    for (int t = 0; t < BM; ++t) {
      float a = acc[t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
      if (lane == 0) {
        partial[(static_cast<size_t>(split) * static_cast<size_t>(M) + static_cast<size_t>(m0 + t)) * kRouterN +
                row] = a;
      }
    }
    __syncthreads();
  }
}

__global__ void router_fp32_reduce_kernel(float* __restrict__ out, const float* __restrict__ partial, int M, int splits) {
  int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = M * kRouterN;
  if (idx >= total) return;
  float s = 0.f;
  for (int sp = 0; sp < splits; ++sp) s += partial[static_cast<size_t>(sp) * total + idx];
  out[idx] = s;
}

template <int BM>
void launch_router_fp32(float* partial, const bf16_t* x, const float* w, float* out, int M, cudaStream_t stream) {
  int smem = kRouterWarps * kRouterKChunk * static_cast<int>(sizeof(float)) +
             BM * kRouterKChunk * static_cast<int>(sizeof(bf16_t));
  dim3 grid(kRouterN / kRouterWarps, kRouterSplits);
  router_fp32_splitk_kernel<BM><<<grid, kRouterWarps * 32, smem, stream>>>(partial, x, w, M);
  int total = M * kRouterN;
  constexpr int threads = 256;
  int blocks = (total + threads - 1) / threads;
  router_fp32_reduce_kernel<<<blocks, threads, 0, stream>>>(out, partial, M, kRouterSplits);
}

void router_fp32_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor partial, torch::Tensor out) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && partial.is_cuda() && out.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat);
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(N == kRouterN && K == kRouterK && w.size(1) == K);
  TORCH_CHECK(M >= 1 && M <= 8);  // l-m8-gate: scalar FP32 is the wrong machine past m=8
  TORCH_CHECK(out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(partial.numel() >= static_cast<int64_t>(kRouterSplits) * M * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* pp = reinterpret_cast<float*>(partial.data_ptr());
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  auto* wp = reinterpret_cast<const float*>(w.data_ptr());
  auto* op = reinterpret_cast<float*>(out.data_ptr());
  if (M % 8 == 0) launch_router_fp32<8>(pp, xp, wp, op, M, stream);
  else launch_router_fp32<1>(pp, xp, wp, op, M, stream);
}

// Unpooler m=1 N=1024 is 128 CTAs. Split K by 4 → 512 CTAs; fp32 partials, one bf16 reduce.
template <uint32_t N, uint32_t K, uint32_t SPLITS, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps>
__global__ void __launch_bounds__(kNumWarps * 32)
    gemv_splitk_kernel(float* __restrict__ partial, const bf16_t* __restrict__ x, const bf16_t* __restrict__ w) {
  constexpr uint32_t KCHUNK = K / SPLITS;
  __shared__ bf16_t sx[KCHUNK];
  const uint32_t split = blockIdx.y;
  const uint32_t k0 = split * KCHUNK;
  const uint32_t tid = threadIdx.x;
  for (uint32_t i = tid * kGemvVecSize; i < KCHUNK; i += kNumWarps * 32 * kGemvVecSize) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + k0 + i);
  }
  __syncthreads();

  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t r0 = (blockIdx.x * kNumWarps + warp) * kRows;
  if (r0 >= N) return;

  float acc[kRows];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) acc[r] = 0.f;

  constexpr uint32_t kStep = 32 * kGemvVecSize * kUnroll;
  for (uint32_t k = lane * kGemvVecSize * kUnroll; k < KCHUNK; k += kStep) {
    float4 xv[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) xv[u] = *reinterpret_cast<const float4*>(sx + k + u * kGemvVecSize);
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k0 + k;
      float4 wv[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u)
        wv[u] = __ldcs(reinterpret_cast<const float4*>(wr + u * kGemvVecSize));
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) acc[r] += dot8_f32(wv[u], xv[u]);
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) partial[split * N + r0 + r] = acc[r];
  }
}

__global__ void gemv_splitk_reduce(bf16_t* __restrict__ out, const float* __restrict__ partial, int n, int splits) {
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) return;
  float s = 0.f;
  for (int sp = 0; sp < splits; ++sp) s += partial[static_cast<size_t>(sp) * n + row];
  out[row] = cast<bf16_t>(s);
}

void hopper_bf16_gemv_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor partial) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda() && partial.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.size(0) == 1);
  constexpr int N = 1024, K = 4096, SPLITS = 4;
  TORCH_CHECK(w.size(0) == N && w.size(1) == K && out.numel() == N);
  TORCH_CHECK(partial.numel() >= SPLITS * N);
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  auto* wp = reinterpret_cast<const bf16_t*>(w.data_ptr());
  auto* op = reinterpret_cast<bf16_t*>(out.data_ptr());
  auto* pp = reinterpret_cast<float*>(partial.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int warps = 8;
  constexpr int blocks_n = N / warps;  // rows=1
  gemv_splitk_kernel<N, K, SPLITS, 1, 4, warps><<<dim3(blocks_n, SPLITS), warps * 32, 0, stream>>>(pp, xp, wp);
  gemv_splitk_reduce<<<(N + 255) / 256, 256, 0, stream>>>(op, pp, N, SPLITS);
}
"""

_CPP_DECL = r"""
void hopper_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl);
void hopper_bf16_gemv_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor partial);
void router_fp32_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor partial, torch::Tensor out);
void z_pack(torch::Tensor w, torch::Tensor payload, torch::Tensor exc_k, torch::Tensor exc_w, torch::Tensor exc_count,
            int64_t base);
void z_gemv(torch::Tensor x, torch::Tensor payload, torch::Tensor exc_k, torch::Tensor exc_w, torch::Tensor exc_count,
            torch::Tensor out, torch::Tensor partial, int64_t base, int64_t M, int64_t N, int64_t K);
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
    build_dir = Path.home() / ".cache" / "muse_bf16_impl_m" / arch
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"muse_bf16_impl_m_{arch}",
        cpp_sources=[_CPP_DECL],
        cuda_sources=[_CUDA_SRC],
        functions=[
            "hopper_bf16_gemv",
            "hopper_bf16_gemv_splitk",
            "router_fp32_splitk",
            "z_pack",
            "z_gemv",
            "bf16_mm_pick_nosplit",
            "bf16_mm_run_nosplit",
        ],
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


_Z_MAX = 32


def _z_base(w: torch.Tensor) -> int:
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e7 = ((bits >> 8) & 0x7F).reshape(-1)
    hist = torch.bincount(e7, minlength=128)
    c = torch.cumsum(hist, 0)
    tail = c[7:]
    prev = torch.zeros_like(tail)
    prev[1:] = c[:120]
    return int(torch.argmax(tail - prev))


def _pack_z(ext, w: torch.Tensor) -> dict[str, Any] | None:
    n, k = w.shape
    if k % 256 != 0 or n % 16 != 0:
        return None
    base = _z_base(w)
    device = w.device
    payload = torch.empty(n * (k // 256) * 384, dtype=torch.uint8, device=device)
    exc_k = torch.empty(n * _Z_MAX, dtype=torch.int16, device=device)
    exc_w = torch.empty(n * _Z_MAX, dtype=torch.float32, device=device)
    exc_count = torch.empty(n, dtype=torch.int32, device=device)
    ext.z_pack(w, payload, exc_k, exc_w, exc_count, base)
    if int(exc_count.max()) > _Z_MAX:
        return None
    return {
        "payload": payload,
        "exc_k": exc_k,
        "exc_w": exc_w,
        "exc_count": exc_count,
        "base": base,
        "n": n,
        "k": k,
    }


def _use_z(m: int, n: int, k: int) -> bool:
    return m == 1 and (n, k) in {
        (2304, 8192),
        (8192, 1024),
        (4096, 8192),
        (1024, 4096),
        (8192, 4096),
    }


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
    state["router_partial"] = None
    state["gemv_partial"] = None
    state["z"] = {}
    state["z_partial"] = None
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
            state["router_partial"] = torch.empty(8 * 1024 * 256, dtype=torch.float32, device=device)
            state["gemv_partial"] = torch.empty(4 * 1024, dtype=torch.float32, device=device)
            state["z_partial"] = torch.empty(4 * 1024, dtype=torch.float32, device=device)
            for name in ("qkv_gate", "o", "pooler", "unpooler"):
                packed = _pack_z(ext, state[name])
                if packed is not None:
                    state["z"][name] = packed
            for layer, m, n, k in (
                ("qkv_gate", 1, 2304, 8192),
                ("o", 1, 8192, 1024),
                ("pooler", 1, 4096, 8192),
                ("unpooler", 1, 1024, 4096),
            ):
                packed = state["z"].get(layer)
                if packed is None:
                    continue
                xw = torch.zeros(m, k, dtype=torch.bfloat16, device=device)
                yw = torch.empty(m, n, dtype=torch.bfloat16, device=device)
                ext.z_gemv(
                    xw,
                    packed["payload"],
                    packed["exc_k"],
                    packed["exc_w"],
                    packed["exc_count"],
                    yw,
                    state["z_partial"],
                    packed["base"],
                    m,
                    n,
                    k,
                )
            torch.cuda.synchronize(device)
    return state


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x, layer = data["x"], data["config"]["layer"]
    if layer == "router":
        m = x.numel() // x.shape[-1]
        ext = state["ext"]
        if ext is not None and state["router_partial"] is not None and 1 <= m <= ROUTER_FP32_MAX_TOKENS:
            x2 = _align16(x.view(m, -1))
            out = torch.empty((m, state["router"].shape[0]), dtype=torch.float32, device=x.device)
            ext.router_fp32_splitk(x2, state["router"], state["router_partial"], out)
            return out
        if m <= ROUTER_FP32_MAX_TOKENS:
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    n = data["config"]["n"]
    W = state[layer][:n]
    m = x.numel() // x.shape[-1]
    ext = state["ext"]
    packed = state["z"].get(layer) if ext is not None else None
    if (
        packed is not None
        and _use_z(m, W.shape[0], W.shape[1])
        and packed["k"] == W.shape[1]
        and W.shape[0] <= packed["n"]
    ):
        x2 = _align16(x.view(m, -1))
        out = torch.empty((m, W.shape[0]), dtype=x.dtype, device=x.device)
        ext.z_gemv(
            x2,
            packed["payload"],
            packed["exc_k"],
            packed["exc_w"],
            packed["exc_count"],
            out,
            state["z_partial"],
            packed["base"],
            m,
            W.shape[0],
            W.shape[1],
        )
        return out.view(*x.shape[:-1], -1)
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
        if W.shape[0] == 1024 and W.shape[1] == 4096 and state["gemv_partial"] is not None:
            ext.hopper_bf16_gemv_splitk(x1, W, out, state["gemv_partial"])
        else:
            ext.hopper_bf16_gemv(x1, W, out, bool(state["use_pdl"]))
        return out.view(*x.shape[:-1], -1)

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
