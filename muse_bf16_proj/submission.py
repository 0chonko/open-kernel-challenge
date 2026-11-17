# okc: problem=muse_bf16_proj
"""Piece submission extracted from proven mega paste; unclaimed shapes fall back to SGLang."""

import hashlib
import zlib
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]

class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str

class Inputs(TypedDict):
    x: torch.Tensor
    config: Config

# ---- wide2/wide2_kernels.py ----

def mma(n):
    regs=','.join(f'%{i}' for i in range(n//2))
    outs=','.join(f'"+f"(a[{i}])' for i in range(n//2))
    return f'''__device__ void mma{n}(float* a,uint64_t x,uint64_t y) {{
      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(y));
    }}'''

CUDA=r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ uint32_t addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ void waitbar(uint32_t b,int p) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");
}
__device__ void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
__device__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
__device__ void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");
}
MMA_FUNCTIONS
template<int BN,bool PLAIN>
__global__ __launch_bounds__(256,1) void ss_gemm(const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    __nv_bfloat16* out,const uint8_t* w,int m,int n,int k,int split,int stages,int hint) {
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int AB=64*128,SB=(64+BN)*128;
  uint32_t base=addr(sm),full=base+stages*SB,empty=full+stages*8;
  int tid=threadIdx.x,rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;
  int kb0=(k/64)*rank/split,nkb=(k/64)*(rank+1)/split-kb0;
  if(tid==0) {
    for(int s=0;s<stages;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],%2; mbarrier.init.shared::cta.b64 [%1],1;" :: "r"(full+s*8),"r"(empty+s*8),"r"(PLAIN?129:1) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if(tid>=128) {
    if constexpr(PLAIN) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    else asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    if(tid==128 || PLAIN) {
      uint64_t pw,px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
      if(!hint) { asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw)); px=pw; }
      for(int i=0;i<nkb;i++) {
        int s=i%stages,ph=(i/stages)&1;
        waitbar(empty+s*8,ph^1);
        if(tid==128) {
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(PLAIN?BN*128:SB) : "memory");
          if constexpr(!PLAIN) tma(base+s*SB,&tw,(kb0+i)*64,nt*64,full+s*8,pw);
          tma(base+s*SB+AB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);
        }
        if constexpr(PLAIN) {
          uint4 v[4];
          const uint8_t* src=w+(long)(nt*(k/64)+kb0+i)*8192;
          #pragma unroll
          for(int j=0;j<4;j++) {
            int q=tid-128+j*128;
            asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(v[j].x),"=r"(v[j].y),"=r"(v[j].z),"=r"(v[j].w) : "l"(src+q*16) : "memory");
          }
          #pragma unroll
          for(int j=0;j<4;j++) {
            int q=tid-128+j*128,r=q/8,c=q%8;
            uint32_t dst=base+s*SB+r*128+((c^(r&7))*16);
            asm volatile("st.shared.v4.u32 [%0],{%1,%2,%3,%4};" :: "r"(dst),"r"(v[j].x),"r"(v[j].y),"r"(v[j].z),"r"(v[j].w) : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          arrive(full+s*8);
        }
      }
    }
  } else {
    if constexpr(PLAIN) asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
    else asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) a[j]=0;
    for(int i=0;i<nkb;i++) {
      int s=i%stages,ph=(i/stages)&1;
      waitbar(full+s*8,ph);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<4;kk++) {
        if constexpr(BN==64) mma64(a,desc(base+s*SB+kk*32),desc(base+s*SB+AB+kk*32));
        if constexpr(BN==128) mma128(a,desc(base+s*SB+kk*32),desc(base+s*SB+AB+kk*32));
        if constexpr(BN==256) mma256(a,desc(base+s*SB+kk*32),desc(base+s*SB+AB+kk*32));
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      if(i>0 && tid==0) arrive(empty+((i-1)%stages)*8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    int row=(tid/32)*16+(tid%32)/4,col=2*(tid%4);
    // All producer copies are complete when the final MMA finishes.
    asm volatile("bar.sync 1,128;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      #pragma unroll
      for(int h=0;h<2;h++) {
        ((float*)sm)[(col+8*j)*68+row+8*h]=a[4*j+2*h];
        ((float*)sm)[(col+8*j+1)*68+row+8*h]=a[4*j+2*h+1];
      }
    }
  }
  __syncthreads();
  if(split>1) cluster_sync();
  for(int q=tid+rank*256;q<BN*16;q+=256*split) {
    int token=q/16,row=(q%16)*4;
    if(mt*BN+token>=m) continue;
    float4 sum=make_float4(0,0,0,0);
    for(int s=0;s<split;s++) {
      uint32_t remote=base+(token*68+row)*4;
      if(split>1) asm volatile("mapa.shared::cluster.u32 %0,%1,%2;" : "=r"(remote) : "r"(remote),"r"(s));
      float4 v;
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "r"(remote) : "memory");
      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
    }
    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
    uint2 pk=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);
    *(uint2*)(out+(long)(mt*BN+token)*n+nt*64+row)=pk;
  }
  __syncthreads();
  if(split>1) cluster_sync();
}
using EncodeFn=decltype(&cuTensorMapEncodeTiled);
int map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {
  static EncodeFn encode=nullptr;
  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}
  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,bool PLAIN> int launch_t(const void* x,const void* w,void* out,int m,int n,int k,int split,int stages,int hint,cudaStream_t st) {
  CUtensorMap tx,tw;int r=map(&tx,x,m,k,BN);if(r)return r;r=map(&tw,w,n,k,64);if(r)return r;
  int smem=max(stages*((64+BN)*128+16),BN*68*4);
  cudaFuncSetAttribute(ss_gemm<BN,PLAIN>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  cudaFuncSetAttribute(ss_gemm<BN,PLAIN>,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/64)*split,(m+BN-1)/BN);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=smem;cfg.stream=st;
  cudaLaunchAttribute a={};a.id=cudaLaunchAttributeClusterDimension;a.val.clusterDim={unsigned(split),1,1};
  if(split>1) {cfg.attrs=&a;cfg.numAttrs=1;}
  return int(cudaLaunchKernelEx(&cfg,ss_gemm<BN,PLAIN>,tw,tx,(__nv_bfloat16*)out,(const uint8_t*)w,m,n,k,split,stages,hint));
}
int launch_ss(const void* x,const void* w,void* out,int m,int n,int k,int bn,int split,int stages,int hint,int plain,cudaStream_t st) {
  if(bn==64)return launch_t<64,false>(x,w,out,m,n,k,split,stages,hint,st);
  if(bn==128)return launch_t<128,false>(x,w,out,m,n,k,split,stages,hint,st);
  if(bn==256)return launch_t<256,false>(x,w,out,m,n,k,split,stages,hint,st);
  return -1;
}
'''
CPP=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int launch_ss(const void*,const void*,void*,int,int,int,int,int,int,int,int,cudaStream_t);
int launch(torch::Tensor x,torch::Tensor w,torch::Tensor y,int bn,int split,int stages,int hint,int plain) {
  return launch_ss(x.data_ptr(),w.data_ptr(),y.data_ptr(),x.size(0),y.size(1),x.size(1),bn,split,stages,hint,plain,at::cuda::getCurrentCUDAStream());
}
'''

def wide2_build():
    source=CUDA.replace('MMA_FUNCTIONS','\n'.join(mma(n) for n in [64,128,256]))
    return load_inline('wide2_final_v1',CPP,cuda_sources=source,functions=['launch'],
        extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'])

CANDIDATE_CONFIGS = {
    (128, "qkv_gate"): (128, 3, 8, 1),
    (128, "o"): (128, 1, 8, 1),
    (128, "pooler"): (128, 2, 8, 1),
    (128, "unpooler"): (128, 1, 9, 1),
    (256, "qkv_gate"): (256, 3, 5, 1),
    (256, "o"): (256, 1, 5, 1),
    (256, "pooler"): (256, 2, 5, 1),
    (256, "unpooler"): (256, 1, 5, 1),
}

def prepare_wide2(weights) -> dict:
    device = weights["qkv_gate"].device
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0):
        return {"configs": {}}
    h200_keys = [(128, "qkv_gate"), (128, "pooler"), (256, "pooler")]
    keys = h200_keys
    configs = {key: CANDIDATE_CONFIGS[key] for key in keys}
    return {
        "ext": wide2_build(),
        "weights": {name: weight.contiguous() for name, weight in weights.items() if name != "router"},
        "configs": configs,
    }

def run_wide2(x, layer, n, st):
    m = x.shape[0]
    cfg = st["configs"].get((m, layer))
    if cfg is None or m > 256 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    weight = st["weights"].get(layer)
    if weight is None or weight.shape != (n, x.shape[1]) or weight.dtype != torch.bfloat16:
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = st["ext"].launch(x, weight, out, *cfg, 0)
    if rc:
        raise RuntimeError(f"wide2 launch failed: {rc}")
    return out

# ---- olayer/olayer_kernels.py ----

OLAYER_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b,int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");
}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
@@MMA@@

template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void ol_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        @@CALL@@
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);
      uint32_t d=base+2*(cb*72+rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  for(int q=tid;q<T*(R/8);q+=TH) {
    int token=q/(R/8),row=(q%(R/8))*8;
    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);
  }
}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  return int(cudaGetLastError());
}
int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {@@CASES@@ default:return -99;}
}
'''

OLAYER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
'''

def olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join(f'%{i}' for i in range(n // 2))
        outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
        functions.append(f'''OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{
          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));
        }}''')
    cases = '\n'.join(f'case {i}: return ol_launch_t<{",".join(map(str,tuple(c)+(1,) if len(c)==6 else c))}>(x,w,out,m,stream,query);' for i,c in enumerate(configs))
    calls = '\n'.join(f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns)
    return OLAYER_CUDA.replace('@@MMA@@','\n'.join(functions)).replace('@@CASES@@',cases).replace('@@CALL@@',calls)

def build_olayer(configs):
    for cfg in configs:
        bn,bk,st,pipe,pack,hint,*groups = cfg
        tw = groups[0] if groups else 1
        if (bn<16 or bn>256 or bn%8 or bk not in (64,128,256) or st<1
                or st*((64+bn*tw)*bk*2+16)>227000 or pipe not in (0,1,2,3)
                or (pipe&1 and st<2) or pack not in (0,1,2) or hint not in (0,1)
                or tw not in (1,2) or bn*tw>256 or (tw>1 and not pipe&2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = olayer_source(configs)
    name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name,OLAYER_CPP,cuda_sources=source,functions=['ol_launch'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True,with_pytorch_error_handling=False)

def pack_olayer(weight) -> torch.Tensor:
    columns = torch.arange(64,device=weight.device)[None,:] ^ ((torch.arange(64,device=weight.device)[:,None]&7)*8)
    tiled = weight.reshape(128,64,16,64).permute(0,2,1,3)
    return tiled.gather(-1,columns.expand(128,16,64,64)).contiguous()

# Token tile, K tile, stages, pipeline flags, packed weight, L2 hints.
OLAYER_PRIMARY_CONFIGS = [(16,256,3,3,1,1),(32,128,8,1,1,1),(64,128,4,3,1,1),
                         (128,128,4,3,1,1),(128,64,5,3,1,1,2)]
OLAYER_NEARBY_CONFIGS = {
    bn: (bn,128,4,3,1,1) if bn<=128 else (bn,64,min(8,227000//((64+bn)*128+16)),3,1,1)
    for bn in range(72,249,8)
}
OLAYER_CONFIGS = list(dict.fromkeys(OLAYER_PRIMARY_CONFIGS+list(OLAYER_NEARBY_CONFIGS.values())))

def prepare_olayer_candidates(weights,configs) -> dict:
    weight = weights['o'].contiguous()
    return {'enabled':True,'ext':build_olayer(configs),'configs':configs,'weight':weight,
            'packed':pack_olayer(weight) if any(c[4]==1 for c in configs) else None,
            'tiled32':weight.reshape(256,32,16,64).permute(0,2,1,3).contiguous() if any(c[4]==2 for c in configs) else None}

def run_olayer_candidate(x,st,index) -> torch.Tensor:
    pack = st['configs'][index][4]
    weight = st['tiled32'] if pack==2 else st['packed'] if pack==1 else st['weight']
    out = torch.empty((x.shape[0],8192),device=x.device,dtype=torch.bfloat16)
    rc = st['ext'].ol_launch(index,x.data_ptr(),weight.data_ptr(),out.data_ptr(),x.shape[0],torch.cuda.current_stream().cuda_stream,0)
    if rc:
        raise RuntimeError(f'olayer launch failed: {rc}, config={st["configs"][index]}')
    return out

def prepare_olayer(weights) -> dict:
    weight = weights.get('o')
    if weight is None or weight.shape != (8192,1024) or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(weight.device) != (9,0):
        return {'enabled':False}
    st = prepare_olayer_candidates(weights,OLAYER_CONFIGS)
    st['token_configs'] = {bn:OLAYER_CONFIGS.index(cfg) for bn,cfg in OLAYER_NEARBY_CONFIGS.items()}
    for index,cfg in enumerate(OLAYER_CONFIGS):
        m = cfg[0]*(cfg[6] if len(cfg)>6 else 1)
        run_olayer_candidate(torch.zeros((m,1024),dtype=torch.bfloat16,device=weight.device),st,index)
    torch.cuda.synchronize(weight.device)
    return st

def run_olayer(x,layer,n,st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'o' or n != 8192:
        return None
    if x.ndim != 2 or x.shape[1] != 1024 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr()%16 or not 1 <= x.shape[0] <= 256:
        return None
    m = x.shape[0]
    if 65<=m<=224:
        index = st['token_configs'][((m+7)//8)*8]
        return run_olayer_candidate(x,st,index)
    return None

# ---- cpstage/cpstage_kernels.py ----

CPS32_CUDA_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define M32D __device__ __forceinline__

M32D int4 cps32_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint4 cps32_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint32_t cps32_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

M32D uint32_t cps32_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

M32D void cps32_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
M32D void cps32_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = cps32_prmt(T0, T1, cm);
  const uint32_t hhi = cps32_prmt(T0, T1, cm >> 16);
  a[0] = cps32_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = cps32_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = cps32_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = cps32_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void cps32_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
M32D void cps32_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) cps32_never(p);
}

M32D uint32_t cps32_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

M32D uint32_t cps32_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

M32D uint32_t cps32_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

M32D void cps32_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

M32D void cps32_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

M32D void cps32_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
M32D void cps32_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
M32D void cps32_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

M32D void cps32_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

M32D bool cps32_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

M32D uint64_t cps32_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
M32D void cps32_wait_bar(uint32_t bar) {
  if (cps32_mbar_try(bar)) return;
  const uint64_t t0 = cps32_gtime();
  while (!cps32_mbar_try(bar))
    if (cps32_gtime() - t0 > 20000000ull) __trap();
}

M32D void cps32_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

M32D float4 cps32_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

M32D void cps32_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
M32D void cps32_store(float* p, float v) { *p = v; }

template <int FMT> struct M32Fmt;
template <> struct M32Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct M32Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct M32Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct M32Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
M32D void cps32_decode(const int4 (&w)[M32Fmt<FMT>::NL], uint32_t (&A)[M32Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < M32Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < M32Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      cps32_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      cps32_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      cps32_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      cps32_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
M32D void cps32_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
M32D void cps32_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) cps32_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>
__global__ void __launch_bounds__(NTMAX) cps32_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = M32Fmt<FMT>::NL, NA = M32Fmt<FMT>::NA, MTT = MT * TG;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3 || RED == 4) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
  const int rawbytes = CP * WR * WK * RT * NL * 512;
  const int wbytes = rawbytes + (XM == 4 ? CWR * CWK * RT * NA * 2048 : 0);
  int4* sw = reinterpret_cast<int4*>(smem + xbytes);
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;
    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MTT, (m + 1) >> 1);
    const uint32_t bar = cps32_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = cps32_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        cps32_mbar_init(bar, 1);
        cps32_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);
        cps32_fence_init();
      }
      cps32_cl_arrive();
    }

    if constexpr (XM == 2 || XM == 3 || XM == 5) {
      const int ks = K / S;
      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {
        const int t = v / (ks / 8), c = v % (ks / 8);
        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;
        void* dst = sx + t * (ks + 8) + c * 8;
        if constexpr (XM == 3 || XM == 5) {
          const uint32_t sa = cps32_smem_u32(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst) = cps32_ldx(src);
      }
      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");
      else __syncthreads();
    }
    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = cps32_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);

    int4* rawring = reinterpret_cast<int4*>(smem + xbytes);
    auto stage_weights = [&](int slot, int si) {
      if constexpr (CP > 0) {
        if (si < KU) {
#pragma unroll
          for (int j=0; j<RT; ++j)
#pragma unroll
            for (int l=0; l<NL; ++l) {
              const uint32_t dst=cps32_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);
              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
#pragma unroll
    for (int q=0; q<CP; ++q) stage_weights(q,q);
    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) if (XM != 4 || tg == 0) w[slot][j][l] = cps32_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = cps32_ldx(xp);
          xv[slot][n][1] = cps32_ldx(xp + (XM >= 9 ? 32 : 8));
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    cps32_order_point(m, outp);
    if constexpr (XM == 3 || XM == 5) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          cps32_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
          if constexpr (XM == 7 || XM == 8 || XM == 11) {
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;
              const uint4 z0=cps32_ldx(xp), z1=cps32_ldx(xp+(XM >= 9 ? 32 : 8));
              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;
              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;
            }
            cps32_order_point(m, outp);
          }
          if constexpr (CP > 0) {
            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");
#pragma unroll
            for (int j=0; j<RT; ++j)
#pragma unroll
              for (int l=0; l<NL; ++l) {
                const uint32_t src=cps32_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);
                int4& v=w[s][j][l];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");
              }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) if (XM != 4 || tg == 0) cps32_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if constexpr (CP > 0) stage_weights(si%CP,si+CP);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          cps32_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) cps32_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
          if constexpr (XM == 4) {
            if (tg == 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q)
                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =
                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);
            }
            __syncthreads();
            if (tg != 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q) {
                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];
                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;
                }
            }
          }
          if constexpr (XM == 6 || XM == 8 || XM == 10) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int n = 0; n < MT; ++n)
#pragma unroll
                for (int j = 0; j < RT; ++j)
                  cps32_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
          } else if constexpr (XM == 5) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
              for (int np = 0; np < (MT + 1) / 2; ++np) {
                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);
                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;
                const uint32_t addr = cps32_smem_u32(sx + tok * (K / S + 8) + col);
                uint32_t b0,b1,b2,b3;
                if constexpr (MT == 1)
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
                else
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                  cps32_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);
                  if constexpr (MT > 1) cps32_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);
                }
              }
            }
          } else {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              uint32_t bx[8];
              if constexpr (XM >= 1 && XM <= 4 && !NOX) {
                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);
                const __nv_bfloat16* xp;
                if constexpr (XM == 2 || XM == 3 || XM == 5)
                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;
                else xp = x0 + (size_t)tok * K + si * 64;
                uint4 z0, z1;
                if constexpr (XM == 2 || XM == 3 || XM == 5) {
                  z0 = *reinterpret_cast<const uint4*>(xp);
                  z1 = *reinterpret_cast<const uint4*>(xp + 8);
                } else {
                  z0 = cps32_ldx(xp); z1 = cps32_ldx(xp + (XM >= 9 ? 32 : 8));
                }
                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;
                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;
              }
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);
                } else {
                  cps32_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                  if constexpr (HILO) cps32_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                }
              }
            }
          }
          if constexpr (XM == 4) __syncthreads();
        }
      }
    }

    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          cps32_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) cps32_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                cps32_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) cps32_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) cps32_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          const int NI = WR * RT * MTT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            cps32_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MTT * 32;
            cps32_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) cps32_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2 || RED == 4) {
            cps32_cl_wait();
            if (RED == 4 || t < TV) {
              const uint32_t rbar = cps32_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = (tg * MT + n) * 4 + t;
                  if (RED == 4 || tn < TV) {
                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else cps32_st_async4(cps32_mapa(cps32_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            cps32_wait_bar(bar);
            if constexpr (RED == 4) cps32_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);
            else cps32_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (RED == 4 || t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if ((tg * MT + n) * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            cps32_cl_wait();
            const uint32_t rbar = cps32_mapa(bar, 0);
            cps32_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else cps32_st_async4(cps32_mapa(cps32_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            cps32_wait_bar(bar);
            cps32_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int cps32_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>
static int cps32_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = cps32_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG, CP>;
  const int policy = query / 100;
  query %= 100;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * TG * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT * TG, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;
    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;
  if (XM == 4) sm += (size_t)WR * WK * RT * M32Fmt<FMT>::NA * 2048;
  sm += (size_t)CP * NW * RT * M32Fmt<FMT>::NL * 512;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = cps32_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[2];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);
    lc.numAttrs = 2;
  }
  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }
  if (query == 4) return (int)sm;
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int cps32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
'''

CPS32_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int cps32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

CPS32_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")

CPS32_NL = {0: 4, 1: 3, 2: 8, 3: 6}

CPS32_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}

CPS32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

CPS32_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}

CPS32_ESC_CAP = 32

def cps32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def cps32_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def cps32_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def cps32_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

def cps32_bf16_bytes(v):
    r, c = v.shape[:2]
    words = cps32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def cps32_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPS32_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = cps32_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def cps32_c12_unpack(data, base, esc):
    """Inverse of cps32_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPS32_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def cps32_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def cps32_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [cps32_units(hi), cps32_units(lo)]
    else:
        mats = [cps32_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([cps32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = cps32_c12_bytes(v)
        if not torch.equal(cps32_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(cps32_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def cps32_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def cps32_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = cps32_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, CPS32_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > CPS32_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = cps32_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.cps32_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

CPS32_TABLE_M = {}

CPS32_MAX_M = 32

CPS32_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def cps32_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def cps32_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def cps32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if cps32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def cps32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def cps32_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def cps32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = cps32_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = cps32_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = cps32_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = cps32_verify_onehot(ext, inst_id[inst], sp, cps32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            cps32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

CPSSM_CUDA_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define CPSSD __device__ __forceinline__

CPSSD int4 cpssm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint4 cpssm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint32_t cpssm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

CPSSD uint32_t cpssm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

CPSSD void cpssm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
CPSSD void cpssm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = cpssm_prmt(T0, T1, cm);
  const uint32_t hhi = cpssm_prmt(T0, T1, cm >> 16);
  a[0] = cpssm_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = cpssm_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = cpssm_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = cpssm_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void cpssm_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
CPSSD void cpssm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) cpssm_never(p);
}

CPSSD uint32_t cpssm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

CPSSD uint32_t cpssm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

CPSSD uint32_t cpssm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

CPSSD void cpssm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

CPSSD void cpssm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

CPSSD void cpssm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
CPSSD void cpssm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
CPSSD void cpssm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

CPSSD void cpssm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

CPSSD bool cpssm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

CPSSD uint64_t cpssm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
CPSSD void cpssm_wait_bar(uint32_t bar) {
  if (cpssm_mbar_try(bar)) return;
  const uint64_t t0 = cpssm_gtime();
  while (!cpssm_mbar_try(bar))
    if (cpssm_gtime() - t0 > 20000000ull) __trap();
}

CPSSD void cpssm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

CPSSD float4 cpssm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

CPSSD void cpssm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
CPSSD void cpssm_store(float* p, float v) { *p = v; }

template <int FMT> struct CpsSmFmt;
template <> struct CpsSmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct CpsSmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct CpsSmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct CpsSmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
CPSSD void cpssm_decode(const int4 (&w)[CpsSmFmt<FMT>::NL], uint32_t (&A)[CpsSmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < CpsSmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < CpsSmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      cpssm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      cpssm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      cpssm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      cpssm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
CPSSD void cpssm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
CPSSD void cpssm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) cpssm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
__global__ void __launch_bounds__(NTMAX) cpssm_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = CpsSmFmt<FMT>::NL, NA = CpsSmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int rawbytes = CP * WR * WK * RT * NL * 512;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = cpssm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = cpssm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        cpssm_mbar_init(bar, 1);
        cpssm_mbar_expect(bar, parts * RT * TV * 128u);
        cpssm_fence_init();
      }
      cpssm_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = cpssm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4* rawring = reinterpret_cast<int4*>(smem);
    auto stage_weights = [&](int slot, int si) {
      if constexpr (CP > 0) {
        if (si < KU) {
#pragma unroll
          for (int j=0; j<RT; ++j)
#pragma unroll
            for (int l=0; l<NL; ++l) {
              const uint32_t dst=cpssm_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);
              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
#pragma unroll
    for (int q=0; q<CP; ++q) stage_weights(q,q);
    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) w[slot][j][l] = cpssm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = cpssm_ldx(xp);
          xv[slot][n][1] = cpssm_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    cpssm_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          cpssm_order_point(m, outp);
        } else {
          if constexpr (CP > 0) {
            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");
#pragma unroll
            for (int j=0; j<RT; ++j)
#pragma unroll
              for (int l=0; l<NL; ++l) {
                const uint32_t src=cpssm_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);
                int4& v=w[s][j][l];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");
              }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) cpssm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if constexpr (CP > 0) stage_weights(si%CP,si+CP);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          cpssm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) cpssm_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  cpssm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) cpssm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          cpssm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) cpssm_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                cpssm_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + rawbytes);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            cpssm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            cpssm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) cpssm_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + rawbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            cpssm_cl_wait();
            if (t < TV) {
              const uint32_t rbar = cpssm_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else cpssm_st_async4(cpssm_mapa(cpssm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            cpssm_wait_bar(bar);
            cpssm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            cpssm_cl_wait();
            const uint32_t rbar = cpssm_mapa(bar, 0);
            cpssm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else cpssm_st_async4(cpssm_mapa(cpssm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            cpssm_wait_bar(bar);
            cpssm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int cpssm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
static int cpssm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = cpssm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, CP>;
  const int policy = query / 100;
  query %= 100;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  sm += (size_t)CP * NW * RT * CpsSmFmt<FMT>::NL * 512;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = cpssm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[2];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);
    lc.numAttrs = 2;
  }
  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }
  if (query == 4) return (int)sm;
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
'''

CPSSM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

CPSSM_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")

CPSSM_NL = {0: 4, 1: 3, 2: 8, 3: 6}

CPSSM_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}

CPSSM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

CPSSM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}

CPSSM_ESC_CAP = 32

def cpssm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPSSM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def cpssm_c12_unpack(data, base, esc):
    """Inverse of cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPSSM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def cpssm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [cpssm_units(hi), cpssm_units(lo)]
    else:
        mats = [cpssm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = cpssm_c12_bytes(v)
        if not torch.equal(cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(cpssm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > CPSSM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = cpstage___to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.cpssm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

CPSSM_TABLE_M = {
    ("router", 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8)),
}

CPSSM_TABLE_MT = {
    ("pooler", 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)),
    ("qkv_gate", 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)),
    ("o", 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("o", 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("unpooler", 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)),
    ("router", 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)),
}

CPSSM_MAX_M = 32

CPSSM_TABLE_M.update({
    ("router", 8): CPSSM_TABLE_MT[("router", 1)],
    ("qkv_gate", 8): CPSSM_TABLE_MT[("qkv_gate", 1)],
    ("pooler", 8): CPSSM_TABLE_MT[("pooler", 1)],
    ("o", 8): CPSSM_TABLE_MT[("o", 1)],
    ("unpooler", 1): CPSSM_TABLE_MT[("unpooler", 1)],
    ("unpooler", 8): CPSSM_TABLE_MT[("unpooler", 1)],
})

CPSSM_TABLE_MT[('pooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))

CPSSM_TABLE_MT[('pooler', 4)] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))

CPSSM_TABLE_MT[('qkv_gate', 2)] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))

CPSSM_TABLE_MT[('unpooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def cpssm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def cpssm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = cpssm_verify_onehot(ext, inst_id[inst], sp, cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def cpstage___to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def cpstage___to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def cps_build(family, instances, depth):
    is32 = family == 'm32'
    prefix = 'cps32_' if is32 else 'cpssm_'
    template = CPS32_CUDA_TEMPLATE if is32 else CPSSM_CUDA_TEMPLATE
    cpp = CPS32_CPP if is32 else CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == (16 if is32 else 14) else inst + (0, 1) if is32 and len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name, cpp, cuda_sources=source, functions=[prefix+'launch'],
                       extra_cuda_cflags=['-O3','-lineinfo','-Xptxas=-v','-gencode=arch=compute_90a,code=sm_90a'],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def prepare_cps(weights, depth=None, layers=CPS_LAYERS, buckets=(2,4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple(layer for layer in layers if layer in ('pooler','unpooler'))
        for bucket,selected_depth in ((2,2),(4,1)):
            if bucket in buckets and selected_layers:
                part = prepare_cps(weights,selected_depth,selected_layers,(bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0,1,2,3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm','m32'):
        table = CPSSM_TABLE_MT if family == 'sm' else CPS32_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and layer in CPS_LAYERS and bucket == 1 and inst[4] in (2,3)
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and (layer,bucket) in CPS32_TABLE_MT:
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0,rows,CPSSM_LAYER_K[layer],*cfg)
            entries.append(((layer,'mt',bucket),(inst,cfg)))
        if include_policy and family == 'sm':
            for (layer,m),(inst,cfg) in CPSSM_TABLE_M.items():
                if layer not in CPS_LAYERS or layer not in weights or inst[4] not in (2,3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m,rows,CPSSM_LAYER_K[layer],*cfg)
                entries.append(((layer,'m',m),(inst,cfg)))
        if not entries:
            continue
        twin = cpssm_bf16_twin if family == 'sm' else cps32_bf16_twin
        instances = list(dict.fromkeys(inst for _,(inst,_) in entries for inst in (inst,twin(inst)) if inst is not None))
        ext = cps_build(family,instances,depth)
        ids = {inst:i for i,inst in enumerate(instances)}
        plan_fn = cpssm_plan if family == 'sm' else cps32_plan
        packs, streams = {}, {}
        for key,(inst,cfg) in entries:
            sp = plan_fn(ext,ids,packs,streams,weights[key[0]],key[0],key,inst,cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext,ids,packs,streams,weights[key[0]],key[0],key,chosen,cfg)
            if sp is not None:
                st['plans'][key] = dict(sp,ext=ext,family=family,inst=chosen,depth=depth)
    return st

def cps_launch(sp, x, out, n, policy=0, query=0):
    fn = cps32_launch_raw if sp['family'] == 'm32' else cpssm_launch_raw
    return fn(sp['ext'],sp['inst_id'],sp,x,out,n,sp['cfg'],query+policy*100)

def run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer,'m',m)) or st['plans'].get((layer,'mt',cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or x.shape[1] != CPS32_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = torch.empty((m,n),dtype=torch.float32 if layer == 'router' else torch.bfloat16,device=x.device)
    rc = cps_launch(sp,x,out,n,st.get('policy',0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out

def cps_resources(st, layer, m, policy=0):
    sp=st['plans'].get((layer,'m',m)) or st['plans'][layer,'mt',cps32_mt(m)]
    x=torch.empty((m,CPS32_LAYER_K[layer]),dtype=torch.bfloat16,device=sp['wp'].device)
    out=torch.empty((m,sp['rows']),dtype=torch.float32 if layer=='router' else torch.bfloat16,device=x.device)
    attrs=cps_launch(sp,x,out,sp['rows'],policy,2)
    return {'registers':attrs//1000,'local_bytes':attrs%1000,
            'active_clusters_or_blocks':cps_launch(sp,x,out,sp['rows'],policy,1),
            'blocks_per_sm':cps_launch(sp,x,out,sp['rows'],policy,3),
            'dynamic_smem':cps_launch(sp,x,out,sp['rows'],policy,4),
            'fmt':sp['fmt'],'inst':sp['inst'],'cfg':sp['cfg']}

"""BF16 cuBLASLt projections with offline descriptors and a cold-L2 tuning probe."""
import statistics
import time
import torch

def _fallback(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor:
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= 8:
        return F.linear(x.float(), state["router_weight"])
    out = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    out += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return out

def _fallback(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor:
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= 8:
        return F.linear(x.float(), state["router_weight"])
    out = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    out += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return out

def _base_state(weights: Weights) -> dict:
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = weights["router"].to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = (weights["router"] - hi.float()).to(torch.bfloat16).t().contiguous()
    state["router_weight"] = weights["router"]
    return state

def prepare(weights: Weights) -> dict:
    state = _base_state(weights)
    state["wide2"] = prepare_wide2(weights)
    state["olayer"] = prepare_olayer(weights)
    state["cps"] = prepare_cps(weights)
    torch.cuda.synchronize(weights["qkv_gate"].device)
    return state

def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data["x"], data["config"]["layer"], data["config"]["n"]
    m = x.shape[0]
    out = None
    if layer == "o" and 65 <= m <= 224:
        out = run_olayer(x, layer, n, state["olayer"])
    elif layer in ("pooler", "unpooler") and 9 <= m <= 16:
        out = run_cps(x, layer, n, state["cps"])
    if out is None and 64 < m <= 256:
        out = run_wide2(x, layer, n, state["wide2"])
    return _fallback(x, layer, n, state) if out is None else out

SELECTED_ROUTES = [
    {"m": 128, "layer": "qkv_gate", "name": "wide2", "module": "wide2"},
    {"m": 128, "layer": "pooler", "name": "wide2", "module": "wide2"},
    {"m": 256, "layer": "pooler", "name": "wide2", "module": "wide2"},
    {"m": 32, "layer": "o", "name": "olayer", "module": "olayer"},
]
