# okc: problem=muse_bf16_proj
"""Impl K: FP32 router split across the SMs, SM90 WGMMA for m=256, no-split nvjet on thin-m.

Router is a direct FP32 GEMM. 256 CTAs (4 K-splits x 64 N-blocks), x in smem, W via ld.cs,
one FP32 reduce. m=256 qkv/pooler/unpooler use wgmma.m64n128k16; qkv's 72 tiles take a
2-way K split (144 CTAs). Thin qkv/pooler m in {16,32} keep SPLITK_NUM<=1 nvjet.
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
_WGMMA_M256 = frozenset({"qkv_gate", "pooler", "unpooler"})
_H200_SMS = 132


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

// {{FP32 router, K-split across the SMs}}{{256 CTAs stream W once; register acc; one small FP32 reduce.}}
__device__ __forceinline__ float dot8_bf16_f32(float4 w0, float4 w1, float4 xv) {
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
  float acc = 0.f;
  float2 a = __bfloat1622float2(x2[0]);
  acc = fmaf(w0.x, a.x, acc);
  acc = fmaf(w0.y, a.y, acc);
  a = __bfloat1622float2(x2[1]);
  acc = fmaf(w0.z, a.x, acc);
  acc = fmaf(w0.w, a.y, acc);
  a = __bfloat1622float2(x2[2]);
  acc = fmaf(w1.x, a.x, acc);
  acc = fmaf(w1.y, a.y, acc);
  a = __bfloat1622float2(x2[3]);
  acc = fmaf(w1.z, a.x, acc);
  acc = fmaf(w1.w, a.y, acc);
  return acc;
}

// N=256, K=8192. Grid (4, 64): K-chunk 2048, 4 rows/block. 256 CTAs on 132 SMs.
__global__ void __launch_bounds__(128)
router_fp32_splitk(float* __restrict__ partial, const bf16_t* __restrict__ x,
                   const float* __restrict__ w, int M) {
  constexpr int K = 8192;
  constexpr int N = 256;
  constexpr int Chunk = 2048;
  constexpr int Unroll = 4;
  constexpr int Vec = 8;
  __shared__ __align__(16) bf16_t sx[Chunk];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int split = static_cast<int>(blockIdx.x);
  const int n0 = static_cast<int>(blockIdx.y) * 4 + warp;
  const int k0 = split * Chunk;
  const float* wrow = w + (static_cast<size_t>(n0) * K + k0);

  for (int mi = 0; mi < M; ++mi) {
    const bf16_t* xrow = x + (static_cast<size_t>(mi) * K + k0);
    for (int i = tid * Vec; i < Chunk; i += 128 * Vec) {
      *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(xrow + i);
    }
    __syncthreads();
    if (n0 < N) {
      float acc = 0.f;
      constexpr int step = 32 * Vec * Unroll;
#pragma unroll
      for (int k = lane * Vec * Unroll; k < Chunk; k += step) {
        float4 xv[Unroll], a0[Unroll], a1[Unroll];
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
          const int kk = k + u * Vec;
          xv[u] = *reinterpret_cast<const float4*>(sx + kk);
          a0[u] = __ldcs(reinterpret_cast<const float4*>(wrow + kk));
          a1[u] = __ldcs(reinterpret_cast<const float4*>(wrow + kk + 4));
        }
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
          acc += dot8_bf16_f32(a0[u], a1[u], xv[u]);
        }
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
      }
      if (lane == 0) {
        partial[(static_cast<size_t>(split) * M + mi) * N + n0] = acc;
      }
    }
    __syncthreads();
  }
}

__global__ void router_fp32_reduce(float* __restrict__ out, const float* __restrict__ partial, int M) {
  constexpr int N = 256;
  constexpr int Splits = 4;
  const int total = M * N;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < total;
       i += static_cast<int>(blockDim.x * gridDim.x)) {
    const int m = i / N;
    const int n = i - m * N;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < Splits; ++s) {
      a += partial[(static_cast<size_t>(s) * M + m) * N + n];
    }
    out[i] = a;
  }
}

void router_fp32(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor partial) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda() && partial.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kFloat);
  TORCH_CHECK(out.scalar_type() == at::kFloat && partial.scalar_type() == at::kFloat);
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(K == 8192 && N == 256 && w.size(1) == K);
  TORCH_CHECK(out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(partial.numel() >= static_cast<int64_t>(4) * M * N);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous() && partial.is_contiguous());
  auto stream = at::cuda::getCurrentCUDAStream();
  router_fp32_splitk<<<dim3(4, N / 4), 128, 0, stream>>>(
      partial.data_ptr<float>(), reinterpret_cast<const bf16_t*>(x.data_ptr()), w.data_ptr<float>(), M);
  const int threads = 256;
  const int total = M * N;
  int blocks = (total + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;
  router_fp32_reduce<<<blocks, threads, 0, stream>>>(out.data_ptr<float>(), partial.data_ptr<float>(), M);
}

#define WGMMA_FMA(acc, da, db, scale_d) \
  asm volatile( \
    "{\n" \
    ".reg .pred p;\n" \
    "setp.ne.b32 p, %66, 0;\n" \
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " \
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n" \
    "}\n" \
    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63]) \
    : "l"(da), "l"(db), "r"(scale_d));

// {{SM90 WGMMA m64n128k16}}{{m=256 BF16. cp.async into K-major smem (no TMA: tensor-map addr is capture-baked). qkv uses 2-way K split.}}
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem));
}

// K-major INTERLEAVE: groups of 8 K, each group is rows*8 packed (CuTe Layout_K_INTER_Atom).
__device__ __forceinline__ void cp_tile(bf16_t* smem, const bf16_t* row0, int rows, int ld, int k0, int tid) {
  constexpr int BK = 64;
  constexpr int G = 8;
  const int chunks = BK / G;
  const int total = rows * chunks;
  for (int i = tid; i < total; i += 128) {
    const int r = i / chunks;
    const int g = i - r * chunks;
    cp_async_16(smem + (g * rows + r) * G, row0 + static_cast<size_t>(r) * ld + k0 + g * G);
  }
}

// Field values are CuTe make_gmma_desc<Major::K> on that layout: leading=rows, stride=8, type=0.
__device__ __forceinline__ uint64_t desc_kmajor(const bf16_t* smem, int rows) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint64_t d = 0;
  d |= static_cast<uint64_t>(addr >> 4) & 0x3fff;
  d |= (static_cast<uint64_t>(rows) & 0x3fff) << 16;
  d |= static_cast<uint64_t>(8) << 32;
  return d;
}

__global__ void wgmma_split_reduce(bf16_t* __restrict__ c, const float* __restrict__ partial, int M, int N) {
  const int total = M * N;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < total;
       i += static_cast<int>(blockDim.x * gridDim.x)) {
    c[i] = __float2bfloat16(partial[i] + partial[total + i]);
  }
}

template <int kSplits>
__global__ void __launch_bounds__(128)
bf16_wgmma_64x128(bf16_t* __restrict__ c, const bf16_t* __restrict__ a, const bf16_t* __restrict__ b,
                  float* __restrict__ partial, int M, int N, int K) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int BM = 64;
  constexpr int BN = 128;
  constexpr int BK = 64;
  __shared__ __align__(128) bf16_t As[BM * BK];
  __shared__ __align__(128) bf16_t Bs[BN * BK];

  const int tid = threadIdx.x;
  const int split = static_cast<int>(blockIdx.z);
  const int m0 = static_cast<int>(blockIdx.y) * BM;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const int k_per = K / kSplits;
  const int k_begin = split * k_per;
  const int k_end = k_begin + k_per;

  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.f;

  int scale = 0;
  for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    cp_tile(As, a + static_cast<size_t>(m0) * K, BM, K, k0, tid);
    cp_tile(Bs, b + static_cast<size_t>(n0) * K, BN, K, k0, tid);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      const uint64_t da = desc_kmajor(As + kk * BM, BM);
      const uint64_t db = desc_kmajor(Bs + kk * BN, BN);
      WGMMA_FMA(acc, da, db, scale);
      scale = 1;
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      asm volatile("" : "+f"(acc[i])::"memory");
    }
    __syncthreads();
  }

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int group = lane >> 2;
  const int inner = lane & 3;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    const int quad = i >> 2;
    const int sub = i & 3;
    const int r = group + ((sub >> 1) << 3) + (warp << 4);
    const int col = (quad << 3) + (sub & 1) + (inner << 1);
    const float v = acc[i];
    if constexpr (kSplits == 1) {
      c[(static_cast<size_t>(m0 + r) * N) + n0 + col] = __float2bfloat16(v);
    } else {
      partial[(static_cast<size_t>(split) * M + (m0 + r)) * N + n0 + col] = v;
    }
  }
#else
  (void)c;
  (void)a;
  (void)b;
  (void)partial;
  (void)M;
  (void)N;
  (void)K;
#endif
}

void bf16_wgmma_m256(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor partial, int64_t splits) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda());
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 && c.scalar_type() == at::kBFloat16);
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous());
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(0));
  TORCH_CHECK(b.size(1) == K && c.size(0) == M && c.size(1) == N);
  TORCH_CHECK(M % 64 == 0 && N % 128 == 0 && K % 64 == 0);
  TORCH_CHECK(splits == 1 || splits == 2);
  TORCH_CHECK(K % (static_cast<int>(splits) * 64) == 0);
  float* pp = nullptr;
  if (splits == 2) {
    TORCH_CHECK(partial.is_cuda() && partial.scalar_type() == at::kFloat && partial.is_contiguous());
    TORCH_CHECK(partial.numel() >= static_cast<int64_t>(2) * M * N);
    pp = partial.data_ptr<float>();
  }
  auto* ap = reinterpret_cast<const bf16_t*>(a.data_ptr());
  auto* bp = reinterpret_cast<const bf16_t*>(b.data_ptr());
  auto* cp = reinterpret_cast<bf16_t*>(c.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(N / 128, M / 64, static_cast<unsigned>(splits));
  if (splits == 2) {
    bf16_wgmma_64x128<2><<<grid, 128, 0, stream>>>(cp, ap, bp, pp, M, N, K);
    const int threads = 256;
    const int total = M * N;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 2048) blocks = 2048;
    wgmma_split_reduce<<<blocks, threads, 0, stream>>>(cp, pp, M, N);
  } else {
    bf16_wgmma_64x128<1><<<grid, 128, 0, stream>>>(cp, ap, bp, nullptr, M, N, K);
  }
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
void router_fp32(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor partial);
void bf16_wgmma_m256(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor partial, int64_t splits);
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
    if major >= 9:
        arch = "sm_90a"
        gencode = "-gencode=arch=compute_90a,code=sm_90a"
    else:
        arch = f"sm_{major}{minor}"
        gencode = f"-gencode=arch=compute_{major}{minor},code={arch}"
    build_dir = Path.home() / ".cache" / "muse_bf16_impl_k" / arch
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"muse_bf16_impl_k_{arch}",
        cpp_sources=[_CPP_DECL],
        cuda_sources=[_CUDA_SRC],
        functions=[
            "hopper_bf16_gemv",
            "bf16_mm_pick_nosplit",
            "bf16_mm_run_nosplit",
            "router_fp32",
            "bf16_wgmma_m256",
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
    state["sm90"] = False
    state["nosplit"] = {}
    state["nosplit_ws"] = None
    state["wgmma_partial"] = None
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        state["use_pdl"] = major >= 9
        state["sm90"] = major >= 9
        state["ext"] = _build_ext(device)
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
            xw = torch.zeros(8, 8192, dtype=torch.bfloat16, device=device)
            out = torch.empty(8, 256, dtype=torch.float32, device=device)
            partial = torch.empty(4 * 8 * 256, dtype=torch.float32, device=device)
            ext.router_fp32(xw, state["router"], out, partial)
            if state["sm90"]:
                state["wgmma_partial"] = torch.empty(2 * 256 * 2304, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device)
    return state


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x, layer = data["x"], data["config"]["layer"]
    ext = state["ext"]
    if layer == "router":
        m = x.shape[0]
        k = x.shape[-1]
        w = state["router"]
        if ext is not None and tuple(w.shape) == (256, 8192) and k == 8192:
            x2 = _align16(x.reshape(m, k).contiguous())
            out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
            partial = torch.empty((4, m, 256), dtype=torch.float32, device=x.device)
            ext.router_fp32(x2, w, out, partial.view(-1))
            return out.view(*x.shape[:-1], 256)
        if m <= ROUTER_FP32_MAX_TOKENS:
            return F.linear(x.float(), w)
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    n = data["config"]["n"]
    W = state[layer][:n]
    m = x.numel() // x.shape[-1]
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

    k = x.shape[-1]
    if (
        ext is not None
        and state["sm90"]
        and m == 256
        and layer in _WGMMA_M256
        and n % 128 == 0
        and k % 64 == 0
        and W.is_contiguous()
    ):
        tiles = (m // 64) * (n // 128)
        splits = 1 if tiles >= _H200_SMS else 2
        if k % (splits * 64) == 0:
            x2 = _align16(x.view(m, k).contiguous())
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            if splits == 2:
                need = 2 * m * n
                buf = state["wgmma_partial"]
                if buf is None or buf.numel() < need:
                    buf = torch.empty(need, dtype=torch.float32, device=x.device)
                ext.bf16_wgmma_m256(x2, W, out, buf, splits)
            else:
                ext.bf16_wgmma_m256(x2, W, out, out, splits)
            return out.view(*x.shape[:-1], n)

    if state["nosplit"].get((layer, m)) and ext is not None:
        x2 = x.view(-1, x.shape[-1])
        out = torch.empty(x2.shape[0], n, dtype=x.dtype, device=x.device)
        state["ext"].bf16_mm_run_nosplit(x2, state[f"{layer}_t"][:, :n], out, state["nosplit_ws"])
        return out.view(*x.shape[:-1], n)

    return F.linear(x, W, None)
