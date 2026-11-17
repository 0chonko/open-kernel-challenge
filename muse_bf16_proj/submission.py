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

# ---- qkv/qkv_kernels.py ----

def qkv_mma(n):
    regs = ",".join(f"%{i}" for i in range(n // 2))
    outs = ",".join(f'"+f"(a[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{
      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));
    }}'''

QKV_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define QD __device__ __forceinline__
QD uint32_t q_addr(const void* p) {return (uint32_t)__cvta_generic_to_shared(p);}
QD void q_init(uint32_t p,int n) {asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(p),"r"(n):"memory");}
QD void q_expect(uint32_t p,int n) {asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(p),"r"(n):"memory");}
QD void q_wait(uint32_t p,int phase) {
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra L;}"::"r"(p),"r"(phase):"memory");
}
QD void q_sync() {asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");}
QD uint32_t q_map(uint32_t p,int rank) {uint32_t d;asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(d):"r"(p),"r"(rank));return d;}
QD void q_push(uint32_t p,float4 v,uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"::"r"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
QD uint64_t q_desc(uint32_t p) {return uint64_t((p&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}
QD uint64_t q_time() {uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
QD void q_tma(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k),"r"(r),"r"(b),"l"(policy):"memory");
}
@@MMA@@
template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>
__global__ __launch_bounds__((PIPE&2)?256:128) void q_kernel(const __grid_constant__ CUtensorMap tw,
    const __grid_constant__ CUtensorMap tx,__nv_bfloat16* out,int m,unsigned long long* stamps) {
  constexpr int KT=(PIPE&32)?2:1, LR=(2304+G-1)/G, WB=NR*128, XB=TM*128, SB=KT*(WB+XB);
  constexpr int NV=NR*TM/4, OWN=(PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128;
  constexpr int RB=ST*SB+ST*16, RECV=RED==3?0:RB+16;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid=threadIdx.x, lane=tid&31, rank=blockIdx.x%S, group=blockIdx.x/S;
  const int tok=blockIdx.y*TM, row=group*2304/G, rows=(group+1)*2304/G-row;
  const int kb=(128/KT)*rank/S, nkb=(128/KT)*(rank+1)/S-kb, own=min(OWN,NV-rank*OWN);
  const uint32_t base=q_addr(sm), bars=base+ST*SB, empty=bars+ST*8, rb=base+RB;
  uint64_t begin=0, stream_end=0;
  if constexpr(PROFILE) begin=q_time();
  if constexpr(LR<NR) {
    for(int i=tid;i<ST*(NR-LR)*32;i+=blockDim.x) {
      int stage=i/((NR-LR)*32), j=i%((NR-LR)*32);
      ((uint32_t*)(sm+stage*SB+LR*128))[j]=0;
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
  }
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(px));
  auto load=[&](int i) {
    const int stage=i%ST;
    q_expect(bars+stage*8,KT*(LR*128+XB));
    if constexpr(KT==2)q_tma(base+stage*SB,&tw,0,(group*128+(kb+i)*KT)*NR,bars+stage*8,pw);
    else q_tma(base+stage*SB,&tw,(kb+i)*64,group*LR,bars+stage*8,pw);
#pragma unroll
    for(int q=0;q<KT;q++)q_tma(base+stage*SB+KT*WB+q*XB,&tx,((kb+i)*KT+q)*64,tok,bars+stage*8,px);
  };
  if(tid==((PIPE&2)?128:0)) {
#pragma unroll
    for(int i=0;i<ST;i++) {q_init(bars+i*8,1);if constexpr(PIPE&2)q_init(empty+i*8,4);}
    if constexpr(S>1) q_init(rb,1);
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    if constexpr(S>1) {q_expect(rb,(S-1)*own*16);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
#pragma unroll
    for(int i=0;i<ST;i++) if(i<nkb)load(i);
  }
  __syncthreads();
  if constexpr(S>1 && RED!=3) asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");
  if constexpr(PIPE&2) {
    if(tid>=128) {
      if(tid==128) for(int i=ST;i<nkb;i++) {q_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
      if constexpr(S>1) {
        if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
        if constexpr(RED!=2) q_sync();
      }
      return;
    }
  }
  float a[(TM/64)*(NR/2)];
#pragma unroll
  for(int h=0;h<TM/64;h++)
#pragma unroll
    for(int j=0;j<NR/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(a[h*(NR/2)+j]));
#pragma unroll 1
  for(int i=0;i<nkb;i++) {
    const int stage=i%ST;
    q_wait(bars+stage*8,(i/ST)&1);
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/2;j++) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
#pragma unroll
    for(int k=0;k<4*KT;k++) {
      if constexpr(PIPE&8) {
        uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);
        @@SWAP@@
      } else {
#pragma unroll
        for(int h=0;h<TM/64;h++) {
          uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+h*8192+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);
          @@CALL@@
        }
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
    else asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/2;j++) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");
    int done=i-((PIPE&1)?1:0);
    if(done>=0 && done+ST<nkb) {
      if constexpr(PIPE&2) {if(lane==0)asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(empty+(done%ST)*8):"memory");}
      else if(tid==0)load(done+ST);
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  if constexpr(PROFILE) stream_end=q_time();
  auto emit=[&](int v,float4 z) {
    int t=v%128, j=(v/128)%(NR/8), h=(v/128)/(NR/8);
    int rr=h*64+(t/32)*16+(t%32)/4, cc=j*8+(t%4)*2;
    float vv[4]={z.x,z.y,z.z,z.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      int tr,wc;
      if constexpr(PIPE&8) {tr=tok+(v/128)*8+(t%4)*2+(c&1);wc=(t/32)*16+(t%32)/4+(c/2)*8;}
      else {tr=tok+rr+(c/2)*8;wc=cc+(c&1);}
      if(tr<m && wc<rows)out[tr*2304+row+wc]=__float2bfloat16_rn(vv[c]);
    }
  };
  if constexpr(S==1) {
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/8;j++)emit((h*(NR/8)+j)*128+tid,make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]));
  } else {
    if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/8;j++) {
        int v=(h*(NR/8)+j)*128+tid,dst=v/OWN,local=v%OWN;
        float4 z=make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]);
        uint32_t p=base+RECV+(rank*OWN+local)*16;
        if(dst==rank)*(float4*)(sm+RECV+(rank*OWN+local)*16)=z;
        else q_push(q_map(p,dst),z,q_map(rb,dst));
      }
    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;":::"memory");else __syncthreads();
    q_wait(rb,0);
    for(int t=tid;t<own;t+=128) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) {
        float4 z=*(float4*)(sm+RECV+(r*OWN+t)*16);
        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;
      }
      emit(rank*OWN+t,sum);
    }
    if constexpr(RED!=2) q_sync();
  }
  if constexpr(PROFILE) if(lane==0) {
    unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
    int idx=((blockIdx.y*gridDim.x+blockIdx.x)*4+tid/32)*4;
    stamps[idx]=smid;stamps[idx+1]=begin;stamps[idx+2]=stream_end;stamps[idx+3]=q_time();
  }
}
using QEncode=decltype(&cuTensorMapEncodeTiled);
int q_tensor_map(CUtensorMap* tm,int64_t p,int inner,int rows,int box) {
  static QEncode encode=nullptr;
  if(!encode){cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);if(!encode)return -100;}
  uint64_t dims[2]={(uint64_t)inner,(uint64_t)rows},stride[1]={(uint64_t)inner*2};uint32_t b[2]={64,(uint32_t)box},es[2]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>
int q_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {
  constexpr int LR=(2304+G-1)/G,NV=NR*TM/4,OWN=(PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128;
  constexpr int KT=(PIPE&32)?2:1;
  constexpr int bytes=ST*KT*(NR+TM)*128+ST*16+16+(S>1 && RED!=3?S*OWN*16:0);
  auto fn=q_kernel<NR,G,TM,S,ST,PIPE,RED,PROFILE>;
  static bool ready=false;
  if(!ready){cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return -(int)e;
    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);if(e!=cudaSuccess)return -(int)e;ready=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(G*S,(m+TM-1)/TM);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={S,1,1};if(S>1){cfg.attrs=&attr;cfg.numAttrs=1;}
  if(query==1){int v=0;cudaError_t e=S>1?cudaOccupancyMaxActiveClusters(&v,fn,&cfg):cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,fn,cfg.blockDim.x,bytes);return e==cudaSuccess?v:-(int)e;}
  CUtensorMap tw,tx;int e=q_tensor_map(&tw,w,KT==2?64:8192,KT==2?G*128*NR:G*LR,KT==2?KT*NR:LR);if(e)return e;e=q_tensor_map(&tx,x,8192,m,TM);if(e)return e;
  return int(cudaLaunchKernelEx(&cfg,fn,tw,tx,(__nv_bfloat16*)out,m,(unsigned long long*)stats));
}
int q_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {
  switch(id){@@CASES@@ default:return -99;}
}
'''

def build_qkv(configs, profile=False):
    ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
    source = QKV_CUDA.replace("@@MMA@@", "\n".join(qkv_mma(n) for n in ns))
    source = source.replace("@@CALL@@", " ".join(f"if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);" for n in ns))
    source = source.replace("@@SWAP@@", " ".join(f"if constexpr(TM=={n})q_mma_{n}(a,dw,dx);" for n in ns))
    source = source.replace("@@CASES@@", "\n".join(
        f'case {i}: return q_launch_t<{",".join(map(str,c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);'
        for i,c in enumerate(configs)))
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint q_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
    return load_inline("qkv_"+hashlib.sha256(source.encode()).hexdigest()[:16],cpp,cuda_sources=source,
        functions=["q_launch"],no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=["-O3","-std=c++20","-gencode=arch=compute_90a,code=sm_90a","--ptxas-options=-v"])

def qkv_pack(w, groups, kt=1):
    rows = (2304 + groups - 1) // groups
    if kt==2:
        return w.view(groups, rows, 128, 64).permute(0,2,1,3).contiguous().reshape(-1,64)
    if 2304 % groups == 0:
        return w.contiguous()
    out = torch.zeros((groups, rows, 8192), dtype=torch.bfloat16, device=w.device)
    for g in range(groups):
        lo, hi = g * 2304 // groups, (g + 1) * 2304 // groups
        out[g, :hi-lo].copy_(w[lo:hi])
    return out.reshape(-1,8192)

def prepare_qkv_candidates(weights, configs, profile=False):
    w = weights["qkv_gate"]
    return {"ext": build_qkv(configs, profile), "configs": list(configs),
        "packed": {(g,kt):qkv_pack(w,g,kt) for g,kt in {(c[1],2 if c[5]&32 else 1) for c in configs}}, "device":w.device}

def run_qkv_candidate(x, st, index, stats=None, query=0):
    out = torch.empty((x.shape[0],2304), dtype=torch.bfloat16, device=x.device)
    cfg = st["configs"][index]
    rc = st["ext"].q_launch(index, st["packed"][cfg[1],2 if cfg[5]&32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(),
        x.shape[0],torch.cuda.current_stream().cuda_stream,query,stats.data_ptr() if stats is not None else 0)
    if query:
        return rc
    if rc:
        raise RuntimeError(f"qkv launch {cfg} failed: {rc}")
    return out

# ---- qkv_sm/qkv_sm_kernels.py ----

QKV_SM_CUDA_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define QKV_SMD __device__ __forceinline__

QKV_SMD int4 qkv_sm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

QKV_SMD uint4 qkv_sm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

QKV_SMD uint32_t qkv_sm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

QKV_SMD uint32_t qkv_sm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

QKV_SMD void qkv_sm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
QKV_SMD void qkv_sm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = qkv_sm_prmt(T0, T1, cm);
  const uint32_t hhi = qkv_sm_prmt(T0, T1, cm >> 16);
  a[0] = qkv_sm_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = qkv_sm_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = qkv_sm_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = qkv_sm_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void qkv_sm_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
QKV_SMD void qkv_sm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) qkv_sm_never(p);
}

QKV_SMD uint32_t qkv_sm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

QKV_SMD uint32_t qkv_sm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

QKV_SMD uint32_t qkv_sm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

QKV_SMD void qkv_sm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

QKV_SMD void qkv_sm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

QKV_SMD void qkv_sm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
QKV_SMD void qkv_sm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
QKV_SMD void qkv_sm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

QKV_SMD void qkv_sm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

QKV_SMD bool qkv_sm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

QKV_SMD uint64_t qkv_sm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
QKV_SMD void qkv_sm_wait_bar(uint32_t bar) {
  if (qkv_sm_mbar_try(bar)) return;
  const uint64_t t0 = qkv_sm_gtime();
  while (!qkv_sm_mbar_try(bar))
    if (qkv_sm_gtime() - t0 > 20000000ull) __trap();
}

QKV_SMD void qkv_sm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

QKV_SMD float4 qkv_sm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

QKV_SMD void qkv_sm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
QKV_SMD void qkv_sm_store(float* p, float v) { *p = v; }

template <int FMT> struct QKV_SMFmt;
template <> struct QKV_SMFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct QKV_SMFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct QKV_SMFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct QKV_SMFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
QKV_SMD void qkv_sm_decode(const int4 (&w)[QKV_SMFmt<FMT>::NL], uint32_t (&A)[QKV_SMFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < QKV_SMFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < QKV_SMFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      qkv_sm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      qkv_sm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      qkv_sm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      qkv_sm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
QKV_SMD void qkv_sm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
QKV_SMD void qkv_sm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) qkv_sm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
__global__ void __launch_bounds__(NTMAX) qkv_sm_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = QKV_SMFmt<FMT>::NL, NA = QKV_SMFmt<FMT>::NA, MTT = MT * TG;
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
  const int wbytes = XM == 4 ? CWR * CWK * RT * NA * 2048 : 0;
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
    const uint32_t bar = qkv_sm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = qkv_sm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        qkv_sm_mbar_init(bar, 1);
        qkv_sm_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);
        qkv_sm_fence_init();
      }
      qkv_sm_cl_arrive();
    }

    if constexpr (XM == 2 || XM == 3 || XM == 5) {
      const int ks = K / S;
      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {
        const int t = v / (ks / 8), c = v % (ks / 8);
        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;
        void* dst = sx + t * (ks + 8) + c * 8;
        if constexpr (XM == 3 || XM == 5) {
          const uint32_t sa = qkv_sm_smem_u32(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst) = qkv_sm_ldx(src);
      }
      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");
      else __syncthreads();
    }
    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = qkv_sm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if (XM != 4 || tg == 0) w[slot][j][l] = qkv_sm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = qkv_sm_ldx(xp);
          xv[slot][n][1] = qkv_sm_ldx(xp + (XM >= 9 ? 32 : 8));
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    qkv_sm_order_point(m, outp);
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
          qkv_sm_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
          if constexpr (XM == 7 || XM == 8 || XM == 11) {
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;
              const uint4 z0=qkv_sm_ldx(xp), z1=qkv_sm_ldx(xp+(XM >= 9 ? 32 : 8));
              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;
              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;
            }
            qkv_sm_order_point(m, outp);
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) if (XM != 4 || tg == 0) qkv_sm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
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
          qkv_sm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) qkv_sm_patch<NA * 16>(A[j], nxt);
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
                  qkv_sm_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
          } else if constexpr (XM == 5) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
              for (int np = 0; np < (MT + 1) / 2; ++np) {
                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);
                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;
                const uint32_t addr = qkv_sm_smem_u32(sx + tok * (K / S + 8) + col);
                uint32_t b0,b1,b2,b3;
                if constexpr (MT == 1)
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
                else
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                  qkv_sm_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);
                  if constexpr (MT > 1) qkv_sm_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);
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
                  z0 = qkv_sm_ldx(xp); z1 = qkv_sm_ldx(xp + (XM >= 9 ? 32 : 8));
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
                  qkv_sm_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                  if constexpr (HILO) qkv_sm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                }
              }
            }
          }
          if constexpr (XM == 4) __syncthreads();
        }
      }
    }

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
          qkv_sm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) qkv_sm_store(out, 1.f);
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
                qkv_sm_store(out + (size_t)tok * N + row, r[j][n][c]);
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
            if (tok < m) qkv_sm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) qkv_sm_store(out + (size_t)tok * N + row, vv[c]);
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
            qkv_sm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MTT * 32;
            qkv_sm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) qkv_sm_add4(v, *cl.map_shared_rank(csum + idx, q));
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
            qkv_sm_cl_wait();
            if (RED == 4 || t < TV) {
              const uint32_t rbar = qkv_sm_mapa(bar, 0);
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
                    else qkv_sm_st_async4(qkv_sm_mapa(qkv_sm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            qkv_sm_wait_bar(bar);
            if constexpr (RED == 4) qkv_sm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);
            else qkv_sm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
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
            qkv_sm_cl_wait();
            const uint32_t rbar = qkv_sm_mapa(bar, 0);
            qkv_sm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else qkv_sm_st_async4(qkv_sm_mapa(qkv_sm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            qkv_sm_wait_bar(bar);
            qkv_sm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int qkv_sm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
static int qkv_sm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = qkv_sm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG>;
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
  if (XM == 4) sm += (size_t)WR * WK * RT * QKV_SMFmt<FMT>::NA * 2048;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = qkv_sm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
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
    lc.numAttrs = 1;
  }
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

int qkv_sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
'''

QKV_SM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int qkv_sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

QKV_SM_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")
QKV_SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
QKV_SM_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}
QKV_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
QKV_SM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
QKV_SM_ESC_CAP = 32

def qkv_sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def qkv_sm_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,)))
        cases.append(f"    case {i}: return qkv_sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return QKV_SM_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))

def qkv_sm_build(instances, name="qkv_sm_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, QKV_SM_CPP + extra_cpp, cuda_sources=qkv_sm_cuda_source(instances) + extra_cuda,
                       functions=["qkv_sm_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def qkv_sm_units(w, layout=0):
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

def qkv_sm_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def qkv_sm_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

def qkv_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = qkv_sm_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def qkv_sm_c12_bytes(v):
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
    pos = torch.tensor(QKV_SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = qkv_sm_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def qkv_sm_c12_unpack(data, base, esc):
    """Inverse of qkv_sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(QKV_SM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def qkv_sm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def qkv_sm_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [qkv_sm_units(hi), qkv_sm_units(lo)]
    else:
        mats = [qkv_sm_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([qkv_sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = qkv_sm_c12_bytes(v)
        if not torch.equal(qkv_sm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(qkv_sm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def qkv_sm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def qkv_sm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = qkv_sm_geometry(R, C, rt, wr, wk, ku, s)
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
    esc = torch.full((G, QKV_SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
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
        if int(counts.max()) > QKV_SM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = qkv_sm_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.qkv_sm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

# H100-selected candidates; H200 confirmation is recorded separately.
QKV_SM_TABLE_M = {}
QKV_SM_MAX_M = 32
QKV_SM_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def qkv_sm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def qkv_sm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def qkv_sm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def qkv_sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if qkv_sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def qkv_sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
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
        if qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def qkv_sm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def qkv_sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = qkv_sm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = qkv_sm_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = qkv_sm_stream(packs[pkey], inst[2], *cfg)
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
        streams[vkey] = qkv_sm_verify_onehot(ext, inst_id[inst], sp, qkv_sm_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            qkv_sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def prepare_qkv_sm(weights, table_m=None, table_mt=None, name="qkv_sm_gemv"):
    table_m = dict(QKV_SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(QKV_SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = qkv_sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, QKV_SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if qkv_sm_bf16_twin(inst):
            instances.append(qkv_sm_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = qkv_sm_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and qkv_sm_bf16_twin(inst):
            plan = qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, qkv_sm_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    return st

def run_qkv_sm(x, layer, n, st):
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if not 9 <= m <= QKV_SM_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", qkv_sm_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != QKV_SM_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if qkv_sm_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
    return out

# H100 finalists. H200 routing stays disabled until the dated judge gate is used.
QKV_LARGE_CONFIGS = [(64,36,128,3,8,27,2), (64,36,128,3,8,25,2), (64,36,128,3,4,59,2)]
QKV_SMALL_INST = (1,2,1,1,4,0,256,0,2304,8192,1,8,8,2,7,1)
QKV_SMALL_GEOMETRY = (1,8,8,2)

def prepare_qkv(weights) -> dict:
    w = weights.get("qkv_gate")
    if w is None or w.shape != (2304,8192) or w.dtype != torch.bfloat16 or not w.is_cuda:
        return {"is_enabled":False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {"is_enabled":False}
    small = prepare_qkv_sm({"qkv_gate":w}, table_m={},
        table_mt={("qkv_gate",2):(QKV_SMALL_INST,QKV_SMALL_GEOMETRY)}, name="qkv_small_final")
    large = prepare_qkv_candidates(weights,QKV_LARGE_CONFIGS)
    warm = torch.zeros((128,8192),dtype=torch.bfloat16,device=w.device)
    for index in range(len(QKV_LARGE_CONFIGS)):
        run_qkv_candidate(warm,large,index)
    torch.cuda.synchronize(w.device)
    return {"is_enabled":"H100" in torch.cuda.get_device_name(w.device), "device":w.device,
        "small":small, "large":large, "large_index":0}

def run_qkv(x, layer, n, st) -> torch.Tensor | None:
    if not st.get("is_enabled") or layer != "qkv_gate" or n != 2304:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda:
        return None
    if x.device != st["device"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if 9 <= m <= 16:
        return run_qkv_sm(x,layer,n,st["small"])
    if m == 128:
        return run_qkv_candidate(x,st["large"],st["large_index"])
    return None



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
    state["qkv"] = prepare_qkv(weights)
    torch.cuda.synchronize(weights["qkv_gate"].device)
    return state

def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data["x"], data["config"]["layer"], data["config"]["n"]
    out = run_qkv(x, layer, n, state["qkv"])
    return _fallback(x, layer, n, state) if out is None else out

SELECTED_ROUTES = [{"m": 128, "layer": "qkv_gate", "name": "qkv", "module": "qkv"}]
