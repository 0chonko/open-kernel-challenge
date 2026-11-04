# okc: problem=muse_bf16_proj
# Large CUDA/C++ source strings are stored zlib-compressed to fit the 1 MB submission limit;
# _zsrc() restores each one byte-for-byte at import. Nothing else changes.
import base64 as _zsrc_b64, zlib as _zsrc_zlib
def _zsrc(payload: str) -> str:
    return _zsrc_zlib.decompress(_zsrc_b64.b64decode(payload)).decode()
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

# ---- m32/m32_kernels.py ----

M32_CUDA_TEMPLATE = _zsrc("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")

M32_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int m32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

M32_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")
M32_NL = {0: 4, 1: 3, 2: 8, 3: 6}
M32_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}
M32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
M32_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
M32_ESC_CAP = 32

def m32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def m32_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,)))
        cases.append(f"    case {i}: return m32_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return M32_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))

def m32_build(instances, name="m32_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, M32_CPP + extra_cpp, cuda_sources=m32_cuda_source(instances) + extra_cuda,
                       functions=["m32_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def m32_units(w, layout=0):
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

def m32_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def m32_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

def m32_bf16_bytes(v):
    r, c = v.shape[:2]
    words = m32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def m32_c12_bytes(v):
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
    pos = torch.tensor(M32_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = m32_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def m32_c12_unpack(data, base, esc):
    """Inverse of m32_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(M32_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def m32_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def m32_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [m32_units(hi), m32_units(lo)]
    else:
        mats = [m32_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([m32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = m32_c12_bytes(v)
        if not torch.equal(m32_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(m32_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def m32_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def m32_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = m32_geometry(R, C, rt, wr, wk, ku, s)
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
    esc = torch.full((G, M32_ESC_CAP), -1, dtype=torch.int32, device=dev)
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
        if int(counts.max()) > M32_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = m32_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def m32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

# H100-selected candidates; H200 confirmation is recorded separately.
M32_TABLE_M = {}
M32_MAX_M = 32
M32_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def m32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def m32_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def m32_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def m32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if m32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def m32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
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
        if m32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def m32_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def m32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = m32_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = m32_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = m32_stream(packs[pkey], inst[2], *cfg)
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
        streams[vkey] = m32_verify_onehot(ext, inst_id[inst], sp, m32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            m32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def prepare_m32(weights, table_m=None, table_mt=None, name="m32_gemv"):
    table_m = dict(M32_TABLE_M if table_m is None else table_m)
    table_mt = dict(M32_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = m32_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, M32_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if m32_bf16_twin(inst):
            instances.append(m32_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = m32_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and m32_bf16_twin(inst):
            plan = m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, m32_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    st["outputs"] = {layer: torch.empty((M32_MAX_M, w.shape[0]), dtype=w.dtype, device=w.device)
                     for layer, w in weights.items() if any(key[0] == layer for key in st["plans"])}
    return st

def run_m32(x, layer, n, st):
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if not 9 <= m <= M32_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", m32_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != M32_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=st["outputs"][layer].dtype, device=x.device)
    if m32_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
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
    state["m32"] = prepare_m32(weights)
    torch.cuda.synchronize(weights["qkv_gate"].device)
    return state

def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data["x"], data["config"]["layer"], data["config"]["n"]
    out = run_m32(x, layer, n, state["m32"])
    return _fallback(x, layer, n, state) if out is None else out

SELECTED_ROUTES = [{"m": m, "layer": layer, "name": "m32", "module": "m32"} for (layer, m) in sorted(M32_TABLE_M)]
