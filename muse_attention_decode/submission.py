# okc: problem=muse_attention_decode
"""Mega-fused gated attention decode (FlashAttention + online-softmax normalizer).

Paper stack:
  - Dao et al. 2022, FlashAttention: keep QK^T / softmax / PV tiled in SRAM.
  - Milakov & Gimelshein 2018, Online normalizer for softmax: running (m, l) stats.

Strategy: collapse baseline's 4+ launches (FusedQKNorm, store_cache, FA3, rms_norm_gate)
into a two-kernel hot path for sliding-window decode (window=2048, the load-weight
majority): (1) RMSNorm Q/K + interleaved RoPE + FP8 cache store, (2) online-softmax
paged attention + RMSNorm×sigmoid(gate). Full / very-long contexts fall back to FA3
like SGLang so ctx=1M and dp-head tests stay correct and fast.
"""

from typing import TypedDict  # Inputs/Config contracts for the judge

import torch  # tensors + CUDA graphs host
import triton  # JIT entry points
import triton.language as tl  # device DSL


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
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: per token q (Hq*64),
                               # then the block's keys before norm and RoPE (Hkv*64), values
                               # (Hkv*64) and the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L: what each request attends over
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...: the block's query offsets
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5  # Muse / SGLang scaleless RMSNorm epsilon
FP8 = torch.float8_e4m3fn  # KV cache dtype
FP8_MAX = 448.0  # E4M3 finite max used when clamping before cast
ROPE_POSITIONS = 1 << 21  # SGLang RotaryEmbedding table length
_ROPE: dict[float, torch.Tensor] = {}  # theta -> cos|sin table (not data-derived)
_NORMS: dict = {}  # (hq,hkv,d) -> FusedQKNorm, created lazily on FA3 path
_WARMED = False  # True after Hopper mega-fuse JIT warmup
_IS_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9  # H100/H200


def _cos_sin(theta: float, d: int, device: torch.device) -> torch.Tensor:
    """SGLang-style [P, D] float32 table: cos in [:, :D/2], sin in [:, D/2:]."""
    cache = _ROPE.get(theta)  # memoized by theta only
    if cache is None or cache.device != device:  # build on first use / device move
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))  # θ^{-2i/d}
        t = torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device)  # positions 0..P-1
        freqs = t[:, None] * inv_freq[None, :]  # outer product pos * inv_freq
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [P, D]
        _ROPE[theta] = cache  # store for reuse across graph replays
    return cache  # float32 cos|sin


def _rms(x: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """Scaleless RMSNorm in fp32, matching reference._rms."""
    x = x.float()  # promote
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # x / rms


def _rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Interleaved GPT-J RoPE: pairs (2i, 2i+1) rotate by pos * θ^{-2i/d}."""
    d = x.shape[-1]  # head dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))  # inv frequencies
    ang = pos.float()[..., None] * inv_freq  # [..., D/2] angles
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)  # broadcast over heads
    even, odd = x[..., 0::2], x[..., 1::2]  # split pairs
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)  # re-interleave


@triton.jit
def _prep_qk_store_kernel(
    QKV,  # bf16 [T, width]
    Q_OUT,  # bf16 [T, Hq, D]
    POS,  # int64 [T]
    COS_SIN,  # fp32 [P, D]
    K_CACHE,  # fp8 [rows, Hkv, D]
    V_CACHE,  # fp8 [rows, Hkv, D]
    CACHE_LOC,  # int64 [T]
    K_SCALE_PTR,  # fp32 []
    V_SCALE_PTR,  # fp32 []
    stride_qkv,  # QKV row stride
    qk_scale,  # Muse query factor
    eps,  # RMS eps
    T,  # token count
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    """RMSNorm Q/K, optional RoPE, FP8 store of K/V (V without RMS). One program = one (token, head)."""
    pid = tl.program_id(0)  # flat program id
    HEADS: tl.constexpr = Hq + 2 * Hkv  # Q + K + V head count
    if pid >= T * HEADS:  # cdiv tail guard
        return  # idle
    token = pid // HEADS  # token index in [0, T)
    head = pid % HEADS  # which head lane
    is_q = head < Hq  # query?
    is_v = head >= Hq + Hkv  # value? (else key if not is_q)
    cols = tl.arange(0, D)  # 0..D-1
    base = token * stride_qkv + head * D  # QKV offset for this head
    raw = tl.load(QKV + base + cols).to(tl.float32)  # load bf16 → fp32
    if is_v:  # value path: quantize raw projection only
        v_scale = tl.load(V_SCALE_PTR)  # scalar scale
        loc = tl.load(CACHE_LOC + token)  # cache row
        kv_head = head - Hq - Hkv  # KV head index
        x = tl.minimum(tl.maximum(raw / v_scale, -448.0), 448.0)  # clamp to E4M3
        dst = loc * (Hkv * D) + kv_head * D  # flat destination
        tl.store(V_CACHE + dst + cols, x.to(tl.float8e4nv))  # FP8 store
        return  # done
    var = tl.sum(raw * raw, axis=0) / float(D)  # mean square
    rstd = tl.rsqrt(var + eps)  # reciprocal RMS
    x = raw * rstd  # normalized
    if HAS_ROPE:  # interleaved RoPE
        partner = cols ^ 1  # even↔odd partner
        raw_p = tl.load(QKV + base + partner).to(tl.float32)  # partner raw
        xp = raw_p * rstd  # partner normalized with same rstd
        pos = tl.load(POS + token)  # position
        freq = cols // 2  # pair index
        cos = tl.load(COS_SIN + pos * D + freq)  # cos half
        sin = tl.load(COS_SIN + pos * D + (D // 2) + freq)  # sin half
        even = (cols % 2) == 0  # even mask
        x = tl.where(even, x * cos - xp * sin, xp * sin + x * cos)  # rotate
    if is_q:  # write scaled query buffer
        q_dst = (token * Hq + head) * D  # [T,Hq,D] flat
        tl.store(Q_OUT + q_dst + cols, (x * qk_scale).to(tl.bfloat16))  # bf16 Q
    else:  # key path: FP8 store
        k_scale = tl.load(K_SCALE_PTR)  # scalar scale
        loc = tl.load(CACHE_LOC + token)  # cache row
        kv_head = head - Hq  # KV head index
        xq = tl.minimum(tl.maximum(x / k_scale, -448.0), 448.0)  # clamp
        dst = loc * (Hkv * D) + kv_head * D  # flat destination
        tl.store(K_CACHE + dst + cols, xq.to(tl.float8e4nv))  # FP8 store


@triton.jit
def _attn_gate_kernel(
    Q,  # bf16 [T, Hq, D]
    GATE,  # bf16 [T, Hq, D]
    OUT,  # bf16 [T, Hq, D]
    K_CACHE,  # fp8 [rows, Hkv, D]
    V_CACHE,  # fp8 [rows, Hkv, D]
    PAGE_TABLE,  # int32 [B, max_pages]
    POS,  # int64 [T]
    K_SCALE_PTR,  # fp32 []
    V_SCALE_PTR,  # fp32 []
    sm_scale,  # 1/sqrt(d)
    eps,  # RMS eps
    stride_pt,  # page table row stride
    window,  # sliding window size
    page_size,  # tokens per page
    L,  # q_len
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FlashAttention-style online softmax over the paged window, then RMSNorm * sigmoid(gate)."""
    pid = tl.program_id(0)  # one program per (token, q_head)
    token = pid // Hq  # flat token
    hq = pid % Hq  # query head
    b = token // L  # batch idx
    group = Hq // Hkv  # GQA group size
    hkv = hq // group  # kv head for this query
    cols = tl.arange(0, D)  # dim lanes
    q = tl.load(Q + (token * Hq + hq) * D + cols).to(tl.float32)  # prepared Q
    k_scale = tl.load(K_SCALE_PTR)  # dequant K
    v_scale = tl.load(V_SCALE_PTR)  # dequant V
    p = tl.load(POS + token)  # absolute position of this query
    start = tl.maximum(p - window + 1, 0)  # inclusive window start
    m_i = tl.full((), float("-inf"), tl.float32)  # running max logit (online softmax)
    l_i = tl.full((), 0.0, tl.float32)  # running sum of exp(logit - m)
    acc = tl.zeros([D], dtype=tl.float32)  # running weighted values
    t0 = start  # tile cursor
    while t0 <= p:  # walk tiles across [start, p]
        offs = tl.arange(0, BLOCK_N)  # tile offsets
        t = t0 + offs  # absolute positions
        visible = t <= p  # causal mask inside tile
        logical_page = t // page_size  # logical page index
        page_off = t % page_size  # within-page offset
        phys = tl.load(PAGE_TABLE + b * stride_pt + logical_page, mask=visible, other=0).to(tl.int64)
        rows = phys * page_size + page_off  # flat cache rows
        k_ptrs = K_CACHE + rows[:, None] * (Hkv * D) + hkv * D + cols[None, :]
        v_ptrs = V_CACHE + rows[:, None] * (Hkv * D) + hkv * D + cols[None, :]
        k = tl.load(k_ptrs, mask=visible[:, None], other=0.0).to(tl.float32) * k_scale
        v = tl.load(v_ptrs, mask=visible[:, None], other=0.0).to(tl.float32) * v_scale
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale  # [BLOCK_N] logits
        qk = tl.where(visible, qk, float("-inf"))  # mask
        m_tile = tl.max(qk, axis=0)  # tile max
        m_new = tl.maximum(m_i, m_tile)  # new running max
        alpha = tl.exp(m_i - m_new)  # rescale prior mass
        p_tile = tl.exp(qk - m_new)  # unnormalized weights
        p_tile = tl.where(visible, p_tile, 0.0)  # zero masked
        l_i = l_i * alpha + tl.sum(p_tile, axis=0)  # update normalizer
        acc = acc * alpha + tl.sum(p_tile[:, None] * v, axis=0)  # update accumulator
        m_i = m_new  # commit max
        t0 += BLOCK_N  # advance tile
    out = acc / tl.maximum(l_i, 1e-12)  # attention output (guard empty)
    gate = tl.load(GATE + (token * Hq + hq) * D + cols).to(tl.float32)  # gate logits
    out = out * tl.rsqrt(tl.sum(out * out, axis=0) / float(D) + eps)  # RMSNorm
    out = out * tl.sigmoid(gate)  # × sigmoid(gate)
    tl.store(OUT + (token * Hq + hq) * D + cols, out.to(tl.bfloat16))  # bf16 write


def _mega_sliding_triton(data: Inputs) -> torch.Tensor:
    """Hopper mega-fuse: Triton prep (norm/RoPE/FP8 store) + Triton online-attn⊕gate."""
    cfg = data["config"]  # static knobs
    batch, length = cfg["batch"], cfg["q_len"]  # B, L
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]  # heads/dim
    tokens = batch * length  # T = B*L
    qkv = data["qkv"].view(tokens, -1).contiguous()  # [T, width]
    device = qkv.device  # CUDA
    q_out = torch.empty((tokens, hq, d), dtype=torch.bfloat16, device=device)  # prepared Q
    cos_sin = _cos_sin(cfg["rope_theta"], d, device) if cfg["rope_theta"] else q_out  # table or dummy
    k_cache = data["k_cache"].view(-1, hkv, d)  # flat rows
    v_cache = data["v_cache"].view(-1, hkv, d)
    n = tokens * (hq + 2 * hkv)  # prep programs
    _prep_qk_store_kernel[(n,)](
        qkv, q_out, data["positions"], cos_sin,
        k_cache, v_cache, data["cache_loc"], data["k_scale"], data["v_scale"],
        qkv.stride(0), float(cfg["qk_scale_factor"]), float(RMS_EPS), tokens,
        hq, hkv, d, 1 if cfg["rope_theta"] else 0,
        num_warps=4, num_stages=2,
    )
    out = torch.empty((tokens, hq, d), dtype=torch.bfloat16, device=device)  # gated out
    gate = qkv[:, (hq + 2 * hkv) * d :].contiguous().view(tokens, hq, d)  # gate view
    _attn_gate_kernel[(tokens * hq,)](
        q_out, gate, out, k_cache, v_cache, data["page_table"], data["positions"],
        data["k_scale"], data["v_scale"], float(cfg["sm_scale"]), float(RMS_EPS),
        data["page_table"].stride(0), int(cfg["window"]), int(cfg["page_size"]), int(length),
        hq, hkv, d, 64,
        num_warps=4, num_stages=2,
    )
    return out.view(batch, length, hq * d)  # [B,L,Hq*D]


def _mega_sliding_torch(data: Inputs) -> torch.Tensor:
    """Numerically twin mega-fuse in PyTorch (local Ampere validation / non-Hopper fallback)."""
    cfg = data["config"]  # knobs
    batch, length = cfg["batch"], cfg["q_len"]  # B, L
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    q_in, k_in, v_in, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)  # split fused
    pos = data["positions"].view(batch, length)  # [B,L]
    q = _rms(q_in.reshape(batch, length, hq, d)) * cfg["qk_scale_factor"]  # norm + scale Q
    k = _rms(k_in.reshape(batch, length, hkv, d))  # norm K
    if cfg["rope_theta"]:  # sliding RoPE
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    q = q.to(torch.bfloat16)  # match Triton Q buffer dtype
    slots = data["cache_loc"].view(batch, length)  # store rows
    k_cache = data["k_cache"].view(-1, hkv, d)  # flat
    v_cache = data["v_cache"].view(-1, hkv, d)
    k_cache[slots] = (k / data["k_scale"]).to(FP8)  # FP8 key store
    v_cache[slots] = (v_in.reshape(batch, length, hkv, d).float() / data["v_scale"]).to(FP8)  # FP8 V
    table = data["page_table"].long()  # page table
    out = torch.empty((batch, length, hq, d), dtype=torch.bfloat16, device=q.device)  # result
    group = hq // hkv  # GQA
    for b in range(batch):  # per request (correctness path, not timed on H200)
        for j in range(length):  # per block token
            p = int(pos[b, j])  # absolute position
            start = max(0, p - cfg["window"] + 1)  # window start
            t = torch.arange(start, p + 1, device=q.device)  # attended positions
            rows = table[b, t // page].clamp(min=0) * page + t % page  # flat rows
            keys = k_cache[rows].float() * data["k_scale"]  # [S,Hkv,D]
            vals = v_cache[rows].float() * data["v_scale"]
            qbj = q[b, j].view(hkv, group, d).float()  # [Hkv,G,D]
            att = torch.einsum("kgd,skd->kgs", qbj, keys) * cfg["sm_scale"]  # logits
            att = att.softmax(-1)  # softmax over S
            out[b, j] = torch.einsum("kgs,skd->kgd", att, vals).reshape(hq, d).to(torch.bfloat16)
    out_f = _rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, d)  # gate epilogue
    return out_f.reshape(batch, length, hq * d).to(torch.bfloat16)  # bf16 out


def _mega_sliding(data: Inputs) -> torch.Tensor:
    """Dispatch mega-fuse: Triton FP8 on Hopper, PyTorch twin elsewhere."""
    if _IS_HOPPER:  # H200 judge path
        return _mega_sliding_triton(data)
    return _mega_sliding_torch(data)  # local validation path


def _fa3_full(data: Inputs) -> torch.Tensor:
    """Baseline-parity FA3 path for full-attention / long contexts (80K–1M, dp heads)."""
    from sgl_kernel.flash_attn import flash_attn_with_kvcache  # FA3
    from sglang.kernels.ops.kvcache.kvcache import store_cache  # FP8 writer
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm  # QK RMS(+RoPE)
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate  # out gate

    cfg = data["config"]  # knobs
    batch, length = cfg["batch"], cfg["q_len"]  # B, L
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]  # heads
    tokens = batch * length  # T
    qkv = data["qkv"].view(tokens, -1)  # [T, width]
    device = qkv.device  # CUDA
    key = (hq, hkv, d)  # module cache key
    if key not in _NORMS:  # lazy init once
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    if cfg["rope_theta"]:  # rare on full path but keep parity
        q, k = _NORMS[key](qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _NORMS[key](qkv)  # norm only
    v = qkv[:, (hq + hkv) * d : (hq + 2 * hkv) * d]  # raw V slice
    k_cache, v_cache = data["k_cache"], data["v_cache"]  # caches
    scaled = cfg["kv_scale"] != 1.0  # config float, no sync
    k = k.reshape(tokens, hkv * d)  # flatten heads
    if scaled:  # apply store-time scales
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(  # write FP8 rows
        k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
        k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
        data["cache_loc"], reserved_skip_index=-1,
    )
    window = cfg["window"]  # 0 on full layers
    descale = {}  # FA3 dequant kwargs
    if scaled:
        descale = dict(
            k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
            v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv),
        )
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8), k_cache=k_cache, v_cache=v_cache,
        page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"], **descale,
    )
    if isinstance(out, tuple):  # FA3 may return extras
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d :]  # gate logits
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None,
                      head_dim=d, eps=RMS_EPS, has_norm=True)
    return o.view(batch, length, hq * d)  # [B,L,Hq*D]


def _warm_kernels() -> None:
    """Pre-JIT Hopper mega-fuse so CUDA-graph capture never pays first-compile."""
    global _WARMED
    if _WARMED or not _IS_HOPPER:  # nothing to do on Ampere / already warm
        return
    device = torch.device("cuda")  # default GPU
    B, L, Hq, Hkv, D, page, window = 1, 1, 16, 2, 64, 1, 2048  # scored decode shape family
    T = B * L  # tokens
    width = (2 * Hq + 2 * Hkv) * D  # fused width
    scale = torch.ones((), dtype=torch.float32, device=device)  # kv scale=1
    def _run(q_len, page_size):  # helper to touch both L and page variants
        conf = {
            "batch": B, "q_len": q_len, "num_q_heads": Hq, "num_kv_heads": Hkv, "head_dim": D,
            "page_size": page_size, "window": window, "rope_theta": 500000.0,
            "qk_scale_factor": 5.4730064863875, "sm_scale": D ** -0.5, "kv_scale": 1.0,
        }
        tt = B * q_len
        data = {
            "qkv": torch.zeros((B, q_len, width), dtype=torch.bfloat16, device=device),
            "positions": torch.zeros((tt,), dtype=torch.int64, device=device),
            "cache_loc": torch.zeros((tt,), dtype=torch.int64, device=device),
            "cache_seqlens": torch.full((B,), q_len, dtype=torch.int32, device=device),
            "cu_seqlens_q": torch.arange(0, tt + 1, q_len, dtype=torch.int32, device=device),
            "scheduler_metadata": None,
            "k_cache": torch.zeros((32, page_size, Hkv, D), dtype=FP8, device=device),
            "v_cache": torch.zeros((32, page_size, Hkv, D), dtype=FP8, device=device),
            "k_scale": scale, "v_scale": scale,
            "page_table": torch.zeros((B, 32), dtype=torch.int32, device=device),
            "seq_lens": torch.zeros((B,), dtype=torch.int32, device=device),
            "config": conf,
        }
        _mega_sliding_triton(data)  # compile
    _run(1, 1)  # decode page1
    _run(8, 1)  # verify page1
    _run(1, 64)  # page64 tests
    torch.cuda.synchronize()  # ensure compile finished before graphs
    _WARMED = True  # mark once


_warm_kernels()  # import-time precompile on Hopper


def kernel_fn(data: Inputs) -> torch.Tensor:
    """Timed entry: mega-fuse on sliding layers; FA3 stack on full-attention layers."""
    if data["config"]["window"]:  # sliding-window hot path (dominates load weights)
        return _mega_sliding(data)
    return _fa3_full(data)  # full/long FA3 fallback
