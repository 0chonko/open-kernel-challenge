# okc: problem=muse_bf16_proj
"""Dense proj: hi/lo router_cat + Triton K-split for skinny router/unpooler.

Ideas (H200 v3 profile, ~4100 score_us class):
- Router FP32 → BF16 hi + residual lo, concat once in prepare; one mm + add.
- Skinny M (router M≤8, unpooler M=1): Triton Split-K partials + reduce.
- Other M=1: SGLang hopper_bf16_gemv; larger M: torch mm / linear.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def prepare(weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    state: dict[str, Any] = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_cat"] = torch.cat([state["router_hi_t"], state["router_lo_t"]], dim=1)
    for name, w in weights.items():
        if name != "router":
            state[name + "_t"] = w.T.contiguous()
    if router.is_cuda:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        state["native_gemv"] = hopper_bf16_gemv
    return state


def _torch(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    w = state[layer][:n]
    if layer == "router":
        if not x.is_cuda:
            return x.float() @ w.T
        both = torch.mm(x, state["router_cat"], out_dtype=torch.float32)
        width = state["router"].shape[0]
        return both[:, :n] + both[:, width : width + n]
    if layer == "qkv_gate" and x.shape[0] >= 192:
        return torch.nn.functional.linear(x, w)
    return torch.mm(x, state[layer + "_t"][:, :n])


if triton is not None:

    @triton.jit
    def _partial(
        X,
        W,
        T,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        SX: tl.constexpr,
        SK: tl.constexpr,
        WN: tl.constexpr,
        WK: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        SPLITS: tl.constexpr,
        CHUNK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        token, split = tl.program_id(1), tl.program_id(2)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BN, BK), tl.float32)
        for step in range(tl.cdiv(CHUNK, BK)):
            k = split * CHUNK + step * BK + kk
            mask = (k < K) & (k < (split + 1) * CHUNK)
            x = tl.load(X + token * SX + k * SK, mask, other=0).to(tl.float32)
            w = tl.load(
                W + rows[:, None] * WN + k[None, :] * WK,
                (rows[:, None] < N) & mask[None, :],
                other=0,
            ).to(tl.float32)
            acc += w * x[None, :]
        tl.store(T + (split * M + token) * N + rows, tl.sum(acc, axis=1), rows < N)

    @triton.jit
    def _reduce(T, Y, SIZE: tl.constexpr, SPLITS: tl.constexpr, B: tl.constexpr):
        idx = tl.program_id(0) * B + tl.arange(0, B)
        acc = tl.full((B,), 0, tl.float32)
        for split in range(SPLITS):
            acc += tl.load(T + split * SIZE + idx, idx < SIZE, other=0)
        tl.store(Y + idx, acc, idx < SIZE)


def kernel_fn(data, state):
    x = data["x"]
    if not x.is_cuda:
        return _torch(data, state)
    if triton is None:
        raise RuntimeError("Triton is required for GPU candidates")

    cfg = data["config"]
    layer = cfg["layer"]
    m = x.shape[0]
    unpool_m1 = layer == "unpooler" and m == 1

    if m == 1 and layer != "router" and not unpool_m1:
        return state["native_gemv"](x, state[layer][: cfg["n"]])
    if m > 8 or (layer != "router" and not unpool_m1):
        return _torch(data, state)

    n, k = cfg["n"], x.shape[1]
    splits = 4
    partial = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    y = torch.empty(
        (m, n),
        device=x.device,
        dtype=torch.float32 if layer == "router" else torch.bfloat16,
    )
    w = state[layer]
    _partial[(triton.cdiv(n, 4), m, splits)](
        x,
        w,
        partial,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        4,
        256,
        splits,
        triton.cdiv(k, splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _reduce[(triton.cdiv(m * n, 256),)](partial, y, m * n, splits, 256)
    return y
