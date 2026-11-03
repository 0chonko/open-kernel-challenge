# okc: problem=muse_attention_decode
"""Flash-Decoding gated attention decode: baseline Muse pipeline with FA3 KV splits.

Hong, Dao et al. 2023 (Flash-Decoding): for L=1 decode over long KV, split the KV
sequence across SMs, compute partial online-softmax, then reduce — applied here via
flash_attn_with_kvcache(num_splits=...).
"""

from typing import TypedDict  # typed dicts for Config / Inputs field shapes

import torch  # tensor ops and dtypes used throughout the kernel
from sgl_kernel.flash_attn import flash_attn_with_kvcache  # FA3 paged attention (+ Flash-Decoding splits)

from sglang.kernels.ops.kvcache.kvcache import store_cache  # write FP8 K/V into the paged cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm  # fused Q/K RMSNorm (+ RoPE)
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate  # output RMSNorm * sigmoid(gate)


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: q (Hq*64), keys (Hkv*64),
                               # values (Hkv*64), the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5  # RMSNorm epsilon shared with SGLang Muse Spark norms
FP8 = torch.float8_e4m3fn  # E4M3 dtype for paged KV cache and FA3 FP8 query path
_NORMS = {}  # cache of FusedQKNorm modules keyed by (hq, hkv, d)
_ROPE = {}  # cache of precomputed cos|sin tables keyed by rope_theta
ROPE_POSITIONS = 1 << 21  # RoPE table length matching SGLang RotaryEmbedding (2^21 positions)


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)  # identity for this head layout
    if key not in _NORMS:  # build once per layout; reuse across CUDA-graph captures
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]  # return the cached fused Q/K norm module


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ROPE.get(theta)  # look up an existing cos|sin table for this theta
    if cache is None:  # first use of this theta: build the full position table once
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # store [pos, d] cos|sin
    return cache  # reuse the cached table (shape-stable; graph-safe)


def _flash_decoding_splits(cfg):
    """Pick FA3 num_splits for Flash-Decoding occupancy on H200 (132 SMs).

    num_splits>1 splits KV along sequence so many SMs work on L=1 decode; 0 = FA3 heuristic.
    Config/shape only — no tensor .item() — so CUDA-graph capture stays valid.
    """
    if cfg["q_len"] != 1:  # verify blocks (L=8) already occupy better; keep FA3 heuristic
        return 0
    if cfg["window"]:  # sliding window ~2048 tokens: mild split still helps small batches
        return 8
    return 32  # full-context decode (~80K KV): aggressive split across SMs


def kernel_fn(data):
    cfg = data["config"]  # shape / layer knobs for this timed call
    batch, length = cfg["batch"], cfg["q_len"]  # B and L (1=decode, 8=verify)
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]  # head layout
    tokens = batch * length  # flatten [B, L] tokens for fused ops
    qkv = data["qkv"].view(tokens, -1)  # fused projection: [tokens, (2*Hq+2*Hkv)*d]
    device = qkv.device  # GPU device for RoPE table construction

    if cfg["rope_theta"]:  # sliding layer: fused QK-norm with interleaved RoPE
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:  # full layer: fused QK-norm only (no RoPE)
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]  # raw values slice from fused qkv (pre-cache)

    k_cache, v_cache = data["k_cache"], data["v_cache"]  # paged FP8 caches (written in place)
    scaled = cfg["kv_scale"] != 1.0  # whether cache uses non-unit FP8 scales
    k = k.reshape(tokens, hkv * d)  # flatten K for store_cache's byte view API
    if scaled:  # divide by scales so E4M3 store reconstitutes with *scale later
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)

    window = cfg["window"]  # 2048 sliding, or 0 for full attention
    descale = {}  # optional FA3 descales when kv_scale != 1
    if scaled:  # FA3 needs per-batch/head descales matching the store path
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))

    num_splits = _flash_decoding_splits(cfg)  # Flash-Decoding: how many KV chunks for this shape
    # Prebuilt scheduler_metadata assumes num_splits=0; drop it when we override splits.
    sched = None if num_splits else data["scheduler_metadata"]

    out = flash_attn_with_kvcache(  # FA3 paged MHA; num_splits enables Flash-Decoding path
        q=q.view(tokens, hq, d).to(FP8),  # FA3 FP8 query path (as prepare_paged_mha_query)
        k_cache=k_cache,  # paged FP8 keys (just updated for this block)
        v_cache=v_cache,  # paged FP8 values (just updated for this block)
        page_table=data["page_table"],  # logical→physical page map
        cache_seqlens=data["cache_seqlens"],  # attended KV length per request (seq_lens+L)
        cu_seqlens_q=data["cu_seqlens_q"],  # packed query offsets 0, L, 2L, ...
        max_seqlen_q=length,  # max query length in the packed batch
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,  # causal within the verify block and vs past KV
        window_size=(window - 1, 0) if window else (-1, -1),  # sliding left window or full
        scheduler_metadata=sched,  # FA3 tile schedule only when using heuristic splits
        num_splits=num_splits,  # >1: split KV across SMs (Flash-Decoding); 0: heuristic
        **descale,  # empty or k_descale/v_descale for scaled caches
    )
    if isinstance(out, tuple):  # some builds return (out, ...); keep the attention tensor
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]  # gate logits from fused projection
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)  # per-head RMSNorm(attn) * sigmoid(gate)
    return o.view(batch, length, hq * d)  # bfloat16 [B, L, Hq*d] as required
