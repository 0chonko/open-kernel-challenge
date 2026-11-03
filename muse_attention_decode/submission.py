# okc: problem=muse_attention_decode
"""FlashAttention-3 gated decode with fused prep + fused RMSNorm×sigmoid epilogue.

Paper: FlashAttention-3 (Shah, Dao, et al., 2024) — async WGMMA/TMA attention on
Hopper with FP8 KV. Strategy vs SGLang baseline: keep FA3 (`flash_attn_with_kvcache`)
as the spine (scheduler_metadata, window, softmax_scale = sm_scale * qk_scale_factor),
but (1) fuse QK-RMSNorm[+RoPE]+FP8 cache store[+FP8 Q cast] into one Triton launch
when kv_scale==1, and (2) fuse FA output cast + per-head RMSNorm × sigmoid(gate)
into one Triton epilogue — fewer launches / less HBM traffic around FA3.
"""

from typing import TypedDict  # shape/dtype contracts for Inputs / Config

import torch  # tensors, dtypes, device ops (CUDA-graph safe: no .item/.cpu/sync in kernel_fn)
import triton  # JIT GPU compiler used for prep + epilogue fusion
import triton.language as tl  # Triton DSL: loads, stores, math inside device kernels

# FA3 paged decode entry (Hopper FP8 path used by SGLang Muse)
from sgl_kernel.flash_attn import flash_attn_with_kvcache
# separate store path when cache scales are not 1 (fused prep assumes unity scales)
from sglang.kernels.ops.kvcache.kvcache import store_cache
# baseline QK-norm[+RoPE] when we cannot use the unity-scale fused prep
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm


class Config(TypedDict):
    batch: int            # B — number of requests in the step
    q_len: int            # L — 1 decode token, or 8 speculative verify tokens
    num_q_heads: int      # Hq — 16 under TP8 on the scored shapes
    num_kv_heads: int     # Hkv — 2; GQA groups share each KV head
    head_dim: int         # D — 64 for Muse Spark
    page_size: int        # 1 scored; 64 in some correctness tests
    window: int           # 2048 sliding, or 0 = full-context attention
    rope_theta: float     # 500000 on sliding; 0 disables RoPE (full layers)
    qk_scale_factor: float  # 5.473 query gain; folded into FA softmax_scale
    sm_scale: float       # 1/8 attention temperature
    kv_scale: float       # cache dequant scale; 1 scored, 1/64 in some tests


class Inputs(TypedDict):
    qkv: torch.Tensor          # bf16 [B,L,(2*Hq+2*Hkv)*D]: Q | K | V | gate logits
    positions: torch.Tensor    # int64 [B*L] absolute positions for RoPE / causality
    cache_loc: torch.Tensor    # int64 [B*L] flat FP8 cache rows to overwrite this step
    cache_seqlens: torch.Tensor  # int32 [B] = seq_lens + L — FA3 KV length per request
    cu_seqlens_q: torch.Tensor   # int32 [B+1] prefix sums of L — FA3 query packing
    scheduler_metadata: torch.Tensor | None  # FA3 tile schedule on full-attn decode; else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, D], updated in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, D], updated in place
    k_scale: torch.Tensor     # float32 [] — key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [] — value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages] logical→physical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B] cached tokens before this block
    config: Config            # static shape knobs for this load (graph-capture safe to branch on)


RMS_EPS = 1e-5  # Muse / SGLang RMSNorm epsilon (matches baseline + reference)
FP8 = torch.float8_e4m3fn  # E4M3 cache + FA3 query dtype on the scored FP8 path
FP8_E4M3_MAX = 448.0  # saturate before FP8 cast (same bound as FusedQKNorm)
_NORMS = {}  # cache FusedQKNorm modules keyed by (hq, hkv, d) — config-stable, not per-input
_ROPE = {}  # cache float32 cos|sin tables keyed by rope_theta — built once per theta
ROPE_POSITIONS = 1 << 21  # SGLang RotaryEmbedding table length (covers long contexts)


def _qk_norm(hq, hkv, d):
    """Return a cached FusedQKNorm for this head layout (fallback / scaled-KV path)."""
    key = (hq, hkv, d)  # module identity depends only on static head geometry
    if key not in _NORMS:  # first use for this geometry: construct once
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]  # reuse across CUDA-graph captures of the same shape family


def _cos_sin(theta, d, device):
    """Build or reuse [2^21, D] float32 cos|sin (interleaved-half layout) for RoPE."""
    cache = _ROPE.get(theta)  # theta is a config constant, safe to cache globally
    if cache is None:  # cold theta: materialize the full position table on device
        # inv_freq[i] = theta^(-2i/D) for pair index i — classic RoPE base frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        # freqs[p, i] = p * inv_freq[i] for every absolute position p in [0, 2^21)
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        # SGLang layout: concat(cos, sin) along last dim → width D (D/2 + D/2)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache  # subsequent calls hit the module-level dict


@triton.jit
def _fused_qk_rope_store_kernel(
    QKV,  # [T, qkv_width] fused projection: Q heads | K heads | V heads | gate
    Q_OUT,  # [T, Hq*D] contiguous queries for FA3 (FP8 when FP8_KV_STORE)
    POS,  # [T] positions; read only if HAS_ROPE
    COS_SIN,  # [max_pos, D] fp32 cos then sin halves; read only if HAS_ROPE
    K_CACHE,  # [N, Hkv*D] flat FP8/BF16 K pool rows
    V_CACHE,  # [N, Hkv*D] flat FP8/BF16 V pool rows
    CACHE_LOC,  # [T] destination row per token (slot 0 is LIVE here — do not skip)
    eps,  # RMSNorm epsilon
    stride_qkv,  # elements between successive tokens in QKV
    n_rows,  # T * (Hq + 2*Hkv): one program row per (token, Q/K/V head)
    NUM_Q_HEADS: tl.constexpr,  # Hq
    NUM_KV_HEADS: tl.constexpr,  # Hkv
    D: tl.constexpr,  # head_dim
    BD: tl.constexpr,  # next_power_of_2(D) load width
    BLOCK_M: tl.constexpr,  # rows per program
    HAS_ROPE: tl.constexpr,  # 1 on sliding layers
    FP8_KV_STORE: tl.constexpr,  # 1 → saturate + store E4M3, emit FP8 Q
    FP8_MAX: tl.constexpr,  # 448.0
):
    """One launch: per-head RMSNorm Q/K, optional RoPE, FP8 KV store, FP8 Q out."""
    pid = tl.program_id(0)  # which BLOCK_M-chunk of (token, head) rows we own
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # absolute row ids in this block
    row_mask = rows < n_rows  # mask padded lanes past the last live row
    cols = tl.arange(0, BD)  # column vector covering head_dim (possibly padded)
    col_mask = cols < D  # ignore power-of-two padding columns
    mask = row_mask[:, None] & col_mask[None, :]  # 2D validity for head loads/stores

    HEADS: tl.constexpr = NUM_Q_HEADS + 2 * NUM_KV_HEADS  # Q then K then V packing
    token = rows // HEADS  # which token this row belongs to
    head = rows % HEADS  # which head index within Q|K|V
    is_q = head < NUM_Q_HEADS  # query-head lanes
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)  # key-head lanes
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS  # value-head lanes (raw V, no RMSNorm)

    src = token * stride_qkv + head * D  # base offset of this head's D elements in QKV
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)  # bf16/fp16 load
    x = raw.to(tl.float32)  # RMS + RoPE math in fp32
    var = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D  # mean square over head_dim
    rstd = tl.rsqrt(var + eps)  # reciprocal RMS
    x = x * rstd[:, None]  # normalized Q/K (V overwritten later from raw)

    if HAS_ROPE:  # sliding-window layers only
        # match separate-RoPE numerics: rotate the bf16-rounded normalized row
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
        partner_cols = cols ^ 1  # interleaved pair partner (…0↔1, 2↔3, …)
        xp = tl.load(QKV + src[:, None] + partner_cols[None, :], mask=mask, other=0.0).to(tl.float32)
        xp = (xp * rstd[:, None]).to(QKV.dtype.element_ty).to(tl.float32)  # partner, same rounding
        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)  # absolute position
        freq = cols // 2  # pair index into cos/sin half-tables
        cs_mask = row_mask[:, None] & (cols < D)[None, :]
        cos = tl.load(COS_SIN + pos[:, None] * D + freq[None, :], mask=cs_mask, other=1.0)
        sin = tl.load(COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=cs_mask, other=0.0)
        is_even = (cols % 2 == 0)[None, :]  # even idx = cos branch of the pair
        x = tl.where(is_even, x * cos - xp * sin, xp * sin + x * cos)  # GPT-J interleaved RoPE

    if FP8_KV_STORE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)  # round-trip before FP8 saturate
    # V stores the raw projection (no norm/RoPE); Q/K keep normalized[+rotated] x
    x = tl.where(is_v[:, None], raw.to(tl.float32), x)
    if FP8_KV_STORE:
        x = tl.minimum(tl.maximum(x, -FP8_MAX), FP8_MAX)  # clamp to E4M3 dynamic range

    loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)  # dest row
    # IMPORTANT vs SGLang FusedQKNorm: challenge marks slot 0 as a live cache row
    # (baseline store_cache uses reserved_skip_index=-1), so we never skip loc==0.
    kv_mask = mask  # every valid lane may write its cache slot
    kv_head = tl.where(is_v, head - NUM_Q_HEADS - NUM_KV_HEADS, head - NUM_Q_HEADS)
    kv_dst = loc * (NUM_KV_HEADS * D) + kv_head * D  # flat row-major KV address
    tl.store(K_CACHE + kv_dst[:, None] + cols[None, :], x, mask=kv_mask & is_k[:, None])
    tl.store(V_CACHE + kv_dst[:, None] + cols[None, :], x, mask=kv_mask & is_v[:, None])

    q_dst = tl.where(is_q, (token * NUM_Q_HEADS + head) * D, 0)  # pack Q for FA3
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])


@triton.jit
def _rms_norm_gate_epilogue_kernel(
    X,  # [tokens*Hq, D] FA3 attention output (any float dtype FA returned)
    GATE,  # [tokens*Hq, D] gate logits from the fused QKV projection
    OUT,  # [tokens*Hq, D] bf16 final gated normalized output
    eps,  # RMSNorm epsilon
    n_rows,  # tokens * Hq
    stride_x,  # row stride of X in elements
    stride_g,  # row stride of GATE
    stride_o,  # row stride of OUT
    D: tl.constexpr,  # head_dim
    BD: tl.constexpr,  # next_power_of_2(D)
):
    """Fuse FA→bf16 cast + per-head RMSNorm + sigmoid(gate) into one write."""
    pid = tl.program_id(0)  # one CUDA program = one (token, head) row
    if pid >= n_rows:  # guard grid overshoot
        return
    cols = tl.arange(0, BD)  # columns covering head_dim
    mask = cols < D  # drop padding lanes
    # load FA output and gate; promote to fp32 for stable RMS + sigmoid
    x = tl.load(X + pid * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(GATE + pid * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D  # mean square over the head
    rstd = tl.rsqrt(var + eps)  # 1 / RMS
    sig = 1.0 / (1.0 + tl.exp(-g))  # sigmoid(gate) elementwise
    y = (x * rstd) * sig  # rmsnorm(a) * sigmoid(gate) — Muse gated attention epilogue
    tl.store(OUT + pid * stride_o + cols, y.to(tl.bfloat16), mask=mask)  # final bf16 store


def _fused_qk_rope_store(qkv, positions, cos_sin, k_cache, v_cache, cache_loc, hq, hkv, d, has_rope):
    """Host wrapper: launch fused prep; returns FP8 Q ready for FA3 (unity kv_scale)."""
    tokens = qkv.shape[0]  # B*L flattened token count
    heads = hq + 2 * hkv  # Q+K+V heads processed as rows
    n_rows = tokens * heads  # total (token, head) rows
    block_m = 4 if n_rows < 1024 else (8 if n_rows < 8192 else (16 if n_rows < 65536 else 32))
    q_out = torch.empty(tokens, hq * d, dtype=FP8, device=qkv.device)  # FA3 wants FP8 queries
    # flatten paged caches to [rows, Hkv*D] so the kernel can address by cache_loc
    k_flat = k_cache.view(-1, hkv * d)
    v_flat = v_cache.view(-1, hkv * d)
    _fused_qk_rope_store_kernel[(triton.cdiv(n_rows, block_m),)](
        qkv,  # fused projection
        q_out,  # FP8 Q destination
        positions if has_rope else qkv,  # dummy ptr when no RoPE (not read)
        cos_sin if has_rope else qkv,  # dummy ptr when no RoPE
        k_flat,  # K cache rows
        v_flat,  # V cache rows
        cache_loc,  # per-token destinations
        RMS_EPS,  # norm eps
        qkv.stride(0),  # token stride
        n_rows,  # grid bound
        hq,  # NUM_Q_HEADS
        hkv,  # NUM_KV_HEADS
        d,  # D
        triton.next_power_of_2(d),  # BD
        block_m,  # BLOCK_M
        HAS_ROPE=has_rope,  # compile-time RoPE switch
        FP8_KV_STORE=True,  # always FP8 cache on this problem
        FP8_MAX=FP8_E4M3_MAX,  # saturate bound
        num_warps=4,  # matches FusedQKNorm sweep default
    )
    return q_out  # [T, Hq*D] float8_e4m3fn


def _fused_rms_norm_gate(attn_out, gate, tokens, hq, d):
    """Host wrapper: FA output + gate → bf16 [T, Hq*D] via one Triton epilogue."""
    # FA3 may return [T, Hq, D] or [T, Hq*D]; force a contiguous row-per-head view
    x = attn_out.reshape(tokens * hq, d)
    if not x.is_contiguous():  # rare non-contiguous FA view — one contig copy
        x = x.contiguous()
    g = gate.reshape(tokens * hq, d)  # gate already lives inside qkv; view only
    if not g.is_contiguous():
        g = g.contiguous()
    out = torch.empty(tokens, hq * d, dtype=torch.bfloat16, device=x.device)  # final buffer
    out_rows = out.view(tokens * hq, d)  # kernel writes one head-row at a time
    n_rows = tokens * hq
    _rms_norm_gate_epilogue_kernel[(n_rows,)](
        x,  # attention values
        g,  # gate logits
        out_rows,  # bf16 destination
        RMS_EPS,  # eps
        n_rows,  # row count
        x.stride(0),  # X row stride
        g.stride(0),  # gate row stride
        out_rows.stride(0),  # out row stride
        d,  # D
        triton.next_power_of_2(d),  # BD
        num_warps=4,  # small head_dim=64: 4 warps is enough
    )
    return out  # [T, Hq*D] bfloat16


def kernel_fn(data: Inputs) -> torch.Tensor:
    """Timed entry: fused prep (or scaled fallback) → FA3 → fused gated RMS epilogue."""
    cfg = data["config"]  # static knobs for this captured graph
    batch, length = cfg["batch"], cfg["q_len"]  # B, L
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]  # head geometry
    tokens = batch * length  # flatten decode/verify tokens for FA3 varlen API
    qkv = data["qkv"].view(tokens, -1)  # [T, (2*Hq+2*Hkv)*D]
    device = qkv.device  # stay on the capture device
    scaled = cfg["kv_scale"] != 1.0  # non-unity scales need descales + divide-before-store
    has_rope = bool(cfg["rope_theta"])  # sliding layers only

    k_cache, v_cache = data["k_cache"], data["v_cache"]  # in-place FP8 pools

    if not scaled:
        # Unity scales: one Triton launch does QK-norm[+RoPE], FP8 KV write, FP8 Q.
        cos_sin = _cos_sin(cfg["rope_theta"], d, device) if has_rope else None
        q = _fused_qk_rope_store(
            qkv, data["positions"], cos_sin, k_cache, v_cache, data["cache_loc"],
            hq, hkv, d, has_rope,
        )
        q = q.view(tokens, hq, d)  # FA3 expects [T, Hq, D]
    else:
        # Non-unity scales: keep SGLang FusedQKNorm + store_cache + FA descales.
        if has_rope:
            q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
        else:
            q, k = _qk_norm(hq, hkv, d)(qkv)
        v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]  # raw V slice (no norm)
        k = k.reshape(tokens, hkv * d)  # flat for store_cache
        k, v = k / data["k_scale"], v / data["v_scale"]  # pre-divide so E4M3 stores dequant correctly
        # reserved_skip_index=-1: slot 0 is a live row in this challenge harness
        store_cache(
            k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
            k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
            data["cache_loc"], reserved_skip_index=-1,
        )
        q = q.view(tokens, hq, d).to(FP8)  # FA3 FP8 query cast (separate launch on this path)

    window = cfg["window"]  # 2048 or 0
    descale = {}  # FA3 optional KV descale tensors
    if scaled:
        # broadcast scalar scales to [B, Hkv] as flash_attn_with_kvcache expects
        descale = dict(
            k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
            v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv),
        )

    # FlashAttention-3 paged decode: causal, windowed or full, with step schedule when present
    out = flash_attn_with_kvcache(
        q=q,  # FP8 queries [T, Hq, D]
        k_cache=k_cache,  # FP8 paged K
        v_cache=v_cache,  # FP8 paged V
        page_table=data["page_table"],  # logical→physical pages
        cache_seqlens=data["cache_seqlens"],  # KV lengths including this block
        cu_seqlens_q=data["cu_seqlens_q"],  # query prefix sums
        max_seqlen_q=length,  # L
        # fold Muse's 5.473 query gain into softmax scale (same as SGLang)
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,  # decode/verify causality inside the block
        # sliding: last `window` tokens → FA window (window-1, 0); full → (-1, -1)
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],  # FA3 tile schedule on full decode
        **descale,  # empty on scored unity-scale loads
    )
    if isinstance(out, tuple):  # some FA builds return (out, softmax_lse, ...)
        out = out[0]

    gate = qkv[:, (hq + 2 * hkv) * d:]  # gate logits share the fused projection buffer
    # fused epilogue: avoid separate .to(bf16) materialization + rms_norm_gate launch
    o = _fused_rms_norm_gate(out, gate, tokens, hq, d)
    return o.view(batch, length, hq * d)  # judge expects bf16 [B, L, Hq*D]
