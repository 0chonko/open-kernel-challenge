# okc: problem=muse_bf16_proj

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

ROUTER_FP32_MAX_TOKENS = 8

_MM_BF16 = frozenset({
    ("pooler", 8), ("pooler", 16), ("pooler", 32), ("pooler", 128), ("pooler", 256),
    ("unpooler", 16), ("unpooler", 32), ("unpooler", 128), ("unpooler", 256),
    ("qkv_gate", 16), ("qkv_gate", 32), ("qkv_gate", 128), ("qkv_gate", 256),
    ("o", 128), ("o", 256),
})

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

// .cg: L2 resident, L1 bypass. __ldcs is evict-first and drops W before the next graph replay.
__device__ __forceinline__ float4 ldg16_cg(const void* p) {
  float4 r;
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ float4 lds16(const void* p) {
  float4 r;
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
               : "r"(addr));
  return r;
}

__device__ __forceinline__ void sts16(void* p, float4 v) {
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
}

__device__ __forceinline__ float2 cvt_bf16x2(unsigned v) {
  float a, b;
  asm volatile("{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %2;\n\t"
               "cvt.rn.f32.bf16 %0, lo;\n\tcvt.rn.f32.bf16 %1, hi;\n\t}\n"
               : "=f"(a), "=f"(b)
               : "r"(v));
  return float2{a, b};
}

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float d;
  asm volatile("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

__device__ __forceinline__ float dot8_f32(float4 wv, float4 xv) {
  unsigned w[4] = {__float_as_uint(wv.x), __float_as_uint(wv.y), __float_as_uint(wv.z), __float_as_uint(wv.w)};
  unsigned x[4] = {__float_as_uint(xv.x), __float_as_uint(xv.y), __float_as_uint(xv.z), __float_as_uint(xv.w)};
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 wf = cvt_bf16x2(w[i]);
    float2 xf = cvt_bf16x2(x[i]);
    acc = fma_rn(wf.x, xf.x, acc);
    acc = fma_rn(wf.y, xf.y, acc);
  }
  return acc;
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps, bool kUsePDL>
__global__ void __launch_bounds__(kNumWarps * 32)
    hopper_bf16_gemv_kernel(bf16_t* __restrict__ out, const bf16_t* __restrict__ x, const bf16_t* __restrict__ w) {
  __shared__ __align__(16) bf16_t sx[K];

  PDLTriggerSecondary<kUsePDL>();
  const uint32_t tid = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();
  for (uint32_t i = tid * kGemvVecSize; i < K; i += kNumWarps * 32 * kGemvVecSize) {
    sts16(sx + i, ldg16_cg(x + i));
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
      for (uint32_t u = 0; u < kUnroll; ++u) xv[u] = lds16(sx + k + u * kGemvVecSize);
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          acc[r] += dot8_f32(ldg16_cg(wr + u * kGemvVecSize), xv[u]);
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
            acc[r] += dot8_f32(ldg16_cg(wr + u * kGemvVecSize), lds16(sx + k + u * kGemvVecSize));
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

// Prefer SPLITK_NUM<=1 for skinny M under CUDA graphs: SplitK partials thrash a cold L2
// more than they help SM fill when torch.mm already tiles well. No GROUPED/swizzle APIs used.
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

constexpr int kRouterN = 256;
constexpr int kRouterK = 8192;
constexpr int kRouterWarps = 4;

template <int SPLITS, int BM>
__global__ void __launch_bounds__(kRouterWarps * 32)
    router_fp32_splitk_kernel(float* __restrict__ partial, const bf16_t* __restrict__ x,
                              const float* __restrict__ w, int M) {
  constexpr int kRouterKChunk = kRouterK / SPLITS;
  constexpr int kVecIters = (kRouterKChunk / 4) / 32;
  static_assert(kRouterK % SPLITS == 0, "K must divide evenly");
  static_assert(kRouterKChunk % 8 == 0 && kVecIters >= 1, "chunk must be float4-friendly");
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
  for (int s = 0; s < kVecIters; ++s) wreg[s] = ldg16_cg(wrow4 + lane + s * 32);
#pragma unroll
  for (int s = 0; s < kVecIters; ++s) srow4[lane + s * 32] = wreg[s];
  __syncthreads();

  for (int m0 = 0; m0 < M; m0 += BM) {
#pragma unroll
    for (int t = 0; t < BM; ++t) {
      const bf16_t* src = x + (static_cast<size_t>(m0 + t) * kRouterK + k0);
      bf16_t* dst = sx + t * kRouterKChunk;
      for (int i = tid * 8; i < kRouterKChunk; i += kRouterWarps * 32 * 8) {
        *reinterpret_cast<float4*>(dst + i) = *reinterpret_cast<const float4*>(src + i);
      }
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
        const unsigned* xp = reinterpret_cast<const unsigned*>(sx + t * kRouterKChunk + kk);
        float2 a = cvt_bf16x2(xp[0]);
        float2 b = cvt_bf16x2(xp[1]);
        acc[t] = fma_rn(wv.x, a.x, acc[t]);
        acc[t] = fma_rn(wv.y, a.y, acc[t]);
        acc[t] = fma_rn(wv.z, b.x, acc[t]);
        acc[t] = fma_rn(wv.w, b.y, acc[t]);
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

template <int SPLITS, int BM>
void launch_router_fp32(float* partial, const bf16_t* x, const float* w, float* out, int M, cudaStream_t stream) {
  constexpr int kRouterKChunk = kRouterK / SPLITS;
  int smem = kRouterWarps * kRouterKChunk * static_cast<int>(sizeof(float)) +
             BM * kRouterKChunk * static_cast<int>(sizeof(bf16_t));
  dim3 grid(kRouterN / kRouterWarps, SPLITS);
  router_fp32_splitk_kernel<SPLITS, BM><<<grid, kRouterWarps * 32, smem, stream>>>(partial, x, w, M);
  int total = M * kRouterN;
  constexpr int threads = 256;
  int blocks = (total + threads - 1) / threads;
  router_fp32_reduce_kernel<<<blocks, threads, 0, stream>>>(out, partial, M, SPLITS);
}

void router_fp32_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor partial, torch::Tensor out, int64_t splits) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && partial.is_cuda() && out.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat);
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(N == kRouterN && K == kRouterK && w.size(1) == K);
  TORCH_CHECK((M >= 1 && M <= 8) || M == 16);
  TORCH_CHECK(out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(splits == 4 || splits == 8 || splits == 16);
  TORCH_CHECK(partial.numel() >= splits * M * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* pp = reinterpret_cast<float*>(partial.data_ptr());
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  auto* wp = reinterpret_cast<const float*>(w.data_ptr());
  auto* op = reinterpret_cast<float*>(out.data_ptr());
#define ROUTER_DISPATCH(SP)                                                       \
  do {                                                                            \
    if (M % 8 == 0) launch_router_fp32<SP, 8>(pp, xp, wp, op, M, stream);         \
    else launch_router_fp32<SP, 1>(pp, xp, wp, op, M, stream);                    \
  } while (0)
  if (splits == 4) ROUTER_DISPATCH(4);
  else if (splits == 8) ROUTER_DISPATCH(8);
  else ROUTER_DISPATCH(16);
#undef ROUTER_DISPATCH
}

// Unpooler m=1 N=1024 → 128 CTAs (8 warps × 1 row). H200 ≈132 SMs: base grid alone nearly fills;
// SplitK multiplies CTAs for latency hiding, but too many splits add reduce + partial traffic.
template <uint32_t N, uint32_t K, uint32_t SPLITS, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps>
__global__ void __launch_bounds__(kNumWarps * 32)
    gemv_splitk_kernel(float* __restrict__ partial, const bf16_t* __restrict__ x, const bf16_t* __restrict__ w) {
  constexpr uint32_t KCHUNK = K / SPLITS;
  __shared__ __align__(16) bf16_t sx[KCHUNK];
  const uint32_t split = blockIdx.y;
  const uint32_t k0 = split * KCHUNK;
  const uint32_t tid = threadIdx.x;
  for (uint32_t i = tid * kGemvVecSize; i < KCHUNK; i += kNumWarps * 32 * kGemvVecSize) {
    sts16(sx + i, ldg16_cg(x + k0 + i));
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
    for (uint32_t u = 0; u < kUnroll; ++u) xv[u] = lds16(sx + k + u * kGemvVecSize);
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k0 + k;
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) acc[r] += dot8_f32(ldg16_cg(wr + u * kGemvVecSize), xv[u]);
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

void hopper_bf16_gemv_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor partial,
                             int64_t splits) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda() && partial.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.size(0) == 1);
  constexpr int N = 1024, K = 4096;
  TORCH_CHECK(w.size(0) == N && w.size(1) == K && out.numel() == N);
  TORCH_CHECK(splits == 2 || splits == 4 || splits == 8);
  TORCH_CHECK(partial.numel() >= splits * N);
  auto* xp = reinterpret_cast<const bf16_t*>(x.data_ptr());
  auto* wp = reinterpret_cast<const bf16_t*>(w.data_ptr());
  auto* op = reinterpret_cast<bf16_t*>(out.data_ptr());
  auto* pp = reinterpret_cast<float*>(partial.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int warps = 8;
  constexpr int blocks_n = N / warps;  // rows=1
#define GEMV_SPLITK(SP)                                                                                      \
  do {                                                                                                       \
    gemv_splitk_kernel<N, K, SP, 1, 4, warps><<<dim3(blocks_n, SP), warps * 32, 0, stream>>>(pp, xp, wp);  \
    gemv_splitk_reduce<<<(N + 255) / 256, 256, 0, stream>>>(op, pp, N, SP);                                   \
  } while (0)
  if (splits == 2) GEMV_SPLITK(2);
  else if (splits == 4) GEMV_SPLITK(4);
  else GEMV_SPLITK(8);
#undef GEMV_SPLITK
}
"""

_CPP_DECL = r"""
void hopper_bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl);
void hopper_bf16_gemv_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor partial,
                             int64_t splits);
void router_fp32_splitk(torch::Tensor x, torch::Tensor w, torch::Tensor partial, torch::Tensor out,
                        int64_t splits);
int64_t bf16_mm_pick_nosplit(int64_t M, int64_t N, int64_t K);
void bf16_mm_run_nosplit(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor workspace);
"""


def _use_hopper_bf16_gemv(m: int, n: int, k: int) -> bool:
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
    gencode = f"-gencode=arch=compute_{major}{minor},code={arch}"
    build_dir = Path.home() / ".cache" / "muse_bf16_impl_splitk" / arch
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=f"muse_bf16_impl_splitk_{arch}",
        cpp_sources=[_CPP_DECL],
        cuda_sources=[_CUDA_SRC],
        functions=[
            "hopper_bf16_gemv",
            "hopper_bf16_gemv_splitk",
            "router_fp32_splitk",
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


def _graph_median_ms(fn) -> float | None:
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        samples = []
        for _ in range(7):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        samples.sort()
        return samples[len(samples) // 2]
    except Exception:
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
    state["router_partial"] = None
    state["gemv_partial"] = None
    state["gemv_splits"] = 4  # fallback: 128 CTAs × 4 → 512; fill H200 without heavy reduce
    state["router_splits"] = 8
    state["nosplit_qkv"] = set()
    state["nosplit_ws"] = None
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
            # Max probed splits: gemv 8 × N=1024; router 16 × M=16 × N=256
            state["router_partial"] = torch.empty(16 * 16 * 256, dtype=torch.float32, device=device)
            state["gemv_partial"] = torch.empty(8 * 1024, dtype=torch.float32, device=device)

            # Graph-time unpooler m=1 SplitK ∈ {2,4,8}; K=4096 divides all.
            Wu = state["unpooler"][:1024]
            xu = torch.randn(1, 4096, dtype=torch.bfloat16, device=device)
            ou = torch.empty(1, 1024, dtype=torch.bfloat16, device=device)
            gp = state["gemv_partial"]
            best_g, best_gt = 4, None
            for sp in (2, 4, 8):
                t = _graph_median_ms(
                    lambda s=sp: ext.hopper_bf16_gemv_splitk(xu, Wu, ou, gp, s)
                )
                if t is not None and (best_gt is None or t < best_gt):
                    best_g, best_gt = sp, t
            state["gemv_splits"] = best_g
            print(f"[muse_bf16] gemv SplitK wins: {best_g} (median_ms={best_gt})", flush=True)

            # Router SplitK ∈ {4,8,16}; K=8192 divides; keep FP32 path.
            Wr = state["router"]
            xr = torch.randn(8, 8192, dtype=torch.bfloat16, device=device)
            or_ = torch.empty(8, 256, dtype=torch.float32, device=device)
            rp = state["router_partial"]
            best_r, best_rt = 8, None
            for sp in (4, 8, 16):
                t = _graph_median_ms(
                    lambda s=sp: ext.router_fp32_splitk(xr, Wr, rp, or_, s)
                )
                if t is not None and (best_rt is None or t < best_rt):
                    best_r, best_rt = sp, t
            state["router_splits"] = best_r
            print(f"[muse_bf16] router SplitK wins: {best_r} (median_ms={best_rt})", flush=True)

            Wt = state["qkv_gate_t"]
            k, n = Wt.shape
            picks = [(m, int(ext.bf16_mm_pick_nosplit(m, n, k))) for m in (16, 32)]
            picks = [(m, ws) for m, ws in picks if ws >= 0]
            if picks:
                buf = torch.empty(max(ws for _, ws in picks), dtype=torch.uint8, device=device)
                kept = set()
                for m, _ws in picks:
                    xw = torch.randn(m, k, dtype=torch.bfloat16, device=device)
                    ref = torch.empty(m, n, dtype=torch.bfloat16, device=device)
                    alt = torch.empty(m, n, dtype=torch.bfloat16, device=device)
                    ref_us = _graph_median_ms(lambda: torch.mm(xw, Wt, out=ref))
                    alt_us = _graph_median_ms(lambda: ext.bf16_mm_run_nosplit(xw, Wt, alt, buf))
                    if ref_us is not None and alt_us is not None and alt_us < ref_us:
                        kept.add(m)
                if kept:
                    state["nosplit_qkv"] = kept
                    state["nosplit_ws"] = buf
    return state


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x, layer = data["x"], data["config"]["layer"]
    if layer == "router":
        m = x.numel() // x.shape[-1]
        ext = state["ext"]
        if ext is not None and state["router_partial"] is not None and (
            1 <= m <= ROUTER_FP32_MAX_TOKENS or m == 16
        ):
            x2 = _align16(x.view(m, -1))
            out = torch.empty((m, state["router"].shape[0]), dtype=torch.float32, device=x.device)
            ext.router_fp32_splitk(
                x2, state["router"], state["router_partial"], out, int(state["router_splits"])
            )
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
            ext.hopper_bf16_gemv_splitk(
                x1, W, out, state["gemv_partial"], int(state["gemv_splits"])
            )
        else:
            ext.hopper_bf16_gemv(x1, W, out, bool(state["use_pdl"]))
        return out.view(*x.shape[:-1], -1)

    if layer == "qkv_gate" and m in state["nosplit_qkv"]:
        x2 = x.view(-1, x.shape[-1])
        out = torch.empty(x2.shape[0], n, dtype=x.dtype, device=x.device)
        state["ext"].bf16_mm_run_nosplit(x2, state["qkv_gate_t"][:, :n], out, state["nosplit_ws"])
        return out.view(*x.shape[:-1], n)

    if state["mm_bf16"] is not None and (layer, m) in _MM_BF16:
        return state["mm_bf16"](
            x.view(-1, x.shape[-1]), state[f"{layer}_t"][:, :n], backend=state["mm_backend"]
        ).view(*x.shape[:-1], n)

    return F.linear(x, W, None)
