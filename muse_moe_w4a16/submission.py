# okc: problem=muse_moe_w4a16
"""Starter: the MoE experts in plain PyTorch, correct but slow. Every local expert runs on
every token, kept by the token's routing weight (zero where it wasn't routed there), and
the NVFP4 weights are expanded to BF16 on every call.

Implement kernel_fn, and prepare to repack the weights (called once, untimed)."""

from typing import TypedDict

import torch

from nvfp4 import dequantize


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


def prepare(weights: Weights) -> Weights:
    """Called once, untimed, before any case. Repack the weights as you like."""
    return weights


def post_expert_norm(y, w, eps=1e-8):
    """Scale by (1 + w), round to BF16, then RMSNorm."""
    y = (y.float() * (1.0 + w.float())).to(torch.bfloat16).float()
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)


def kernel_fn(data: Inputs, weights: Weights) -> torch.Tensor:
    """This GPU's share of the MoE output, bfloat16 [M, H]: for each token, the sum over its
    experts that are local of weight * post_expert_norm(expert(x)). Rows whose experts are
    all elsewhere are zero."""
    cfg = data["config"]
    x = data["hidden_states"]
    d_expert = cfg["d_expert"]
    ids = data["topk_ids"].long()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(cfg["local_start"], cfg["local_start"] + cfg["local_count"]):
        # Each token's routing weight for expert e: zero unless e is among its top 8.
        g = (data["topk_weights"] * (ids == e)).sum(-1)
        w13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).to(torch.bfloat16)
        h = x @ w13.T
        inter = torch.nn.functional.silu(h[:, :d_expert]) * h[:, d_expert:]
        w2 = dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).to(torch.bfloat16)
        y = post_expert_norm(inter @ w2.T, weights["post_norm_w"])
        out += g[:, None] * y
    return out.to(torch.bfloat16)


# === UNDERLYING STACK INDEX (muse_moe_w4a16) ===
# Baseline call chain (SGLang a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f):
#   prepare:
#     prepare_moe_nvfp4_layer_for_marlin  marlin_utils_fp4.py
#       -> gptq_marlin_repack             gptq_marlin_repack.py/.cuh
#       -> marlin_permute_scales          marlin_utils.py
#     marlin_make_workspace               marlin_utils.py
#   kernel_fn:
#     moe_align_block_size / single_token  (CUDA AOT/JIT default; Triton only for tiny-batch)
#     moe_wna16_marlin_gemm (w13)         moe_wna16_marlin.py/.cuh + marlin_template.h
#     silu_and_mul                        activation.py/.cuh
#     moe_wna16_marlin_gemm (w2)
#     fused_post_expert_norm_weighted_sum Triton
# Full huge CUDA: ./underlying/
# Blocks below are reference-only assignments (not called by kernel_fn).



# --- underlying ref: marlin_utils_fp4.py ---
_REF_MARLIN_UTILS_FP4 = r'''
FILE: marlin_utils_fp4.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/quantization/marlin_utils_fp4.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
from __future__ import annotations

import logging
from typing import Callable

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()
logger = logging.getLogger(__name__)


def nvfp4_marlin_process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
    if not (marlin_scales >= 0).all():
        # NVFP4 ModelOpt scales are expected to be non-negative. Keep this as
        # a warning so unusual checkpoints can still load for diagnosis.
        logger.warning_once(
            "NVFP4 Marlin assumes non-negative scales, but negative scales "
            "were found. Accuracy may be degraded."
        )

    marlin_scales = marlin_scales.to(torch.half)
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )
    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    return marlin_scales[:, 1::2].contiguous()


def nvfp4_marlin_process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    global_scale_shape = global_scale.shape
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    global_scale = global_scale * (2.0 ** (exponent_bias - 7))
    if global_scale_shape == torch.Size([]):
        global_scale = global_scale.reshape(1)
    return global_scale


def fake_apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    del weight, weight_scale, weight_global_scale, workspace, size_k, bias
    out_shape = input.shape[:-1] + (size_n,)
    return input.new_empty(out_shape)


@register_custom_op(fake_impl=fake_apply_fp4_marlin_linear)
def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("NVFP4 Marlin requires FP16 or BF16 activations.")

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)
    # Recover the physical Marlin tile dimensions from the repacked weight.
    # They can exceed the logical TP shard dimensions when preparation padded a
    # misaligned shard (e.g. N=928 -> 960 for TP=4).
    padded_size_k = weight.size(0) * 16
    padded_size_n = weight.size(1) * 8 // 16
    if padded_size_k != size_k:
        reshaped_x = torch.nn.functional.pad(reshaped_x, (0, padded_size_k - size_k))

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=padded_size_n,
        k=padded_size_k,
        device=input.device,
        dtype=input.dtype,
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=weight_global_scale,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=padded_size_n,
        size_k=padded_size_k,
        is_k_full=True,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    # A narrowed N dimension has the padded row stride, so materialize it
    # before reshaping. This is only needed for a TP shard that was padded.
    if padded_size_n != size_n:
        output = output[:, :size_n].contiguous()
    return output.reshape(out_shape)


def prepare_nvfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    if getattr(layer, "quant_config", None) is not None:
        group_size = layer.quant_config.group_size
        if group_size != 16:
            raise ValueError(f"NVFP4 Marlin requires group_size=16, got {group_size}.")

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = getattr(layer, "params_dtype", getattr(layer, "orig_dtype", None))
    if param_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("NVFP4 Marlin requires FP16 or BF16 activation dtype.")

    assert layer.weight.shape == (part_size_n, part_size_k // 2)

    # Marlin accepts either N%64/K%128 or N%128/K%64. Select the smaller
    # padded shape, matching vLLM's marlin_padded_nk helper.
    padded_size_n, padded_size_k = min(
        (
            ((part_size_n + 63) // 64 * 64, (part_size_k + 127) // 128 * 128),
            ((part_size_n + 127) // 128 * 128, (part_size_k + 63) // 64 * 64),
        ),
        key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]),
    )

    if (padded_size_n, padded_size_k) != (part_size_n, part_size_k):
        pad_rows = padded_size_n - part_size_n
        pad_cols = (padded_size_k - part_size_k) // 2
        scale_pad_cols = (padded_size_k - part_size_k) // 16
        layer.weight = torch.nn.Parameter(
            torch.nn.functional.pad(layer.weight, (0, pad_cols, 0, pad_rows)),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.nn.functional.pad(
                layer.weight_scale, (0, scale_pad_cols, 0, pad_rows)
            ),
            requires_grad=False,
        )

    device = layer.weight.device
    layer.workspace = marlin_make_workspace(device)

    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = layer.weight.view(torch.int32).T.contiguous()
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=padded_size_k,
        size_n=padded_size_n,
        num_bits=4,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    weight_scale = layer.weight_scale.T.contiguous().to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=padded_size_k,
        size_n=padded_size_n,
        group_size=16,
    )
    weight_scale = nvfp4_marlin_process_scales(weight_scale)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

    weight_global_scale = layer.weight_global_scale.to(param_dtype)
    weight_global_scale = nvfp4_marlin_process_global_scale(weight_global_scale)
    layer.weight_global_scale = torch.nn.Parameter(
        weight_global_scale, requires_grad=False
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = torch.nn.functional.pad(layer.bias, (0, padded_size_n - part_size_n))
        bias = marlin_permute_bias(bias)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input_dtype is None or input_dtype.itemsize == 2:
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
            marlin_scales.size(0), -1
        )
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        assert marlin_scales.max() <= 249
        # exponent_bias (fp4->fp8) = 2 ** 3 - 2 ** 1 = 6
        marlin_scales = marlin_scales + 6
        marlin_scales = marlin_scales.view(torch.float8_e8m0fnu)
    return marlin_scales


def _normalize_scale_tensor(
    scales: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    # The kernel consumes E8M0 exponents. Regardless of the placeholder dtype
    # the loader used, we want the *numerical* value 2**e in ``target_dtype``.
    # float32/bfloat16/float16 containers hold the numerical 2**e directly
    # (they were filled via a dtype-promoting copy from uint8/e8m0).
    # uint8/int8 containers hold the raw E8M0 byte and must be reinterpreted.
    if scales.dtype == torch.float8_e8m0fnu:
        return scales.to(target_dtype)
    if scales.dtype == torch.uint8:
        return scales.view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype == torch.int8:
        return scales.view(torch.uint8).view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return scales.to(target_dtype)
    raise TypeError(f"Unsupported MXFP4 scale dtype for Marlin: {scales.dtype}")


def _get_optional_param(layer: torch.nn.Module, *names: str) -> torch.Tensor | None:
    for name in names:
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def deinterleave_moe_mxfp4_w13_for_marlin(layer: torch.nn.Module) -> None:
    """Convert GPT-OSS interleaved w13 rows to Marlin's contiguous halves.

    GPT-OSS stores gate/up rows as [gate0, up0, gate1, up1, ...]. The Marlin
    fused activation consumes [all_gate_rows, all_up_rows].
    """

    w13 = layer.w13_weight.data
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")

    if w13.shape[1] % 2 != 0:
        raise ValueError(f"Expected even w13 row dimension, got {w13.shape}.")

    e, n, k = w13.shape
    layer.w13_weight.data = (
        w13.view(e, n // 2, 2, k).permute(0, 2, 1, 3).contiguous().view(e, n, k)
    )

    if w13_scale is not None:
        scale = w13_scale.data
        if scale.shape[1] != n:
            raise ValueError(
                f"Expected w13 scale row dimension {n}, got {scale.shape}."
            )
        w13_scale.data = (
            scale.view(e, n // 2, 2, scale.shape[-1])
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(e, n, scale.shape[-1])
        )

    if w13_bias is not None:
        bias = w13_bias.data
        if bias.shape[1] != n:
            raise ValueError(f"Expected w13 bias row dimension {n}, got {bias.shape}.")
        w13_bias.data = bias.view(e, n // 2, 2).permute(0, 2, 1).contiguous().view(e, n)


def _repack_moe_fp4_weight_for_marlin(
    weight: torch.Tensor,
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    perm: torch.Tensor,
) -> torch.Tensor:
    assert weight.shape == (num_experts, size_n, size_k // 2)

    tensor_list = []
    for i in range(num_experts):
        qweight = weight[i].view(torch.int32).T.contiguous()
        marlin_qweight = gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
        )
        tensor_list.append(marlin_qweight)
    return torch.stack(tensor_list)


def _permute_moe_fp4_scales_for_marlin(
    scales: torch.Tensor,
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    group_size: int,
    process_scales: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    tensor_list = []
    for i in range(num_experts):
        scale = scales[i].T.contiguous()
        marlin_scales = marlin_permute_scales(
            s=scale,
            size_k=size_k,
            size_n=size_n,
            group_size=group_size,
        )
        tensor_list.append(process_scales(marlin_scales))
    return torch.stack(tensor_list)


def prepare_moe_mxfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    group_size = 32
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = _get_optional_param(layer, "w13_weight_scale", "w13_weight_scale_inv")
    w2_scale = _get_optional_param(layer, "w2_weight_scale", "w2_weight_scale_inv")
    w13_bias = _get_optional_param(layer, "w13_weight_bias", "w13_bias")
    w2_bias = _get_optional_param(layer, "w2_weight_bias", "w2_bias")

    if w13_scale is None or w2_scale is None:
        raise ValueError("MXFP4 Marlin requires w13/w2 weight scales.")

    w13_scale_data = w13_scale.data if hasattr(w13_scale, "data") else w13_scale
    w2_scale_data = w2_scale.data if hasattr(w2_scale, "data") else w2_scale
    w13_bias_data = w13_bias.data if hasattr(w13_bias, "data") else w13_bias
    w2_bias_data = w2_bias.data if hasattr(w2_bias, "data") else w2_bias

    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2
    hidden_size = w13.shape[2] * 2
    if hidden_size % 128 == 0:
        padded_intermediate_size = ((intermediate_size + 63) // 64) * 64
    else:
        if hidden_size % 64 != 0:
            raise ValueError(
                f"MXFP4 Marlin requires hidden_size to be divisible by 64, "
                f"got {hidden_size}."
            )
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128
    param_dtype = getattr(
        layer,
        "orig_dtype",
        w13_bias_data.dtype if w13_bias_data is not None else torch.bfloat16,
    )

    device = w13.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _pad_w13(x: torch.Tensor) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        x = x.view(num_experts, 2, intermediate_size, x.shape[-1])
        x = torch.nn.functional.pad(
            x, (0, 0, 0, padded_intermediate_size - intermediate_size)
        )
        return x.reshape(num_experts, 2 * padded_intermediate_size, -1)

    def _pad_w2(x: torch.Tensor, packing: int) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return x
        return torch.nn.functional.pad(
            x, (0, (padded_intermediate_size - intermediate_size) // packing)
        )

    w13 = _pad_w13(w13)
    w2 = _pad_w2(w2, packing=2)
    w13_scale_data = _pad_w13(_normalize_scale_tensor(w13_scale_data, param_dtype))
    w2_scale_data = _pad_w2(
        _normalize_scale_tensor(w2_scale_data, param_dtype),
        packing=group_size,
    )
    if w13_bias_data is not None:
        w13_bias_data = _pad_w13(w13_bias_data.unsqueeze(-1)).squeeze(-1)

    w13_size_n, w13_size_k = padded_intermediate_size * 2, hidden_size
    w2_size_n, w2_size_k = hidden_size, padded_intermediate_size

    def _process_scales(marlin_scales: torch.Tensor) -> torch.Tensor:
        return mxfp4_marlin_process_scales(marlin_scales, input_dtype=param_dtype)

    def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        tensor_list = []
        for i in range(num_experts):
            tensor_list.append(marlin_permute_bias(bias[i].to(param_dtype)))
        return torch.stack(tensor_list)

    w13_marlin = _repack_moe_fp4_weight_for_marlin(
        w13, num_experts=num_experts, size_n=w13_size_n, size_k=w13_size_k, perm=perm
    )
    w2_marlin = _repack_moe_fp4_weight_for_marlin(
        w2, num_experts=num_experts, size_n=w2_size_n, size_k=w2_size_k, perm=perm
    )
    w13_scale_marlin = _permute_moe_fp4_scales_for_marlin(
        w13_scale_data,
        num_experts=num_experts,
        size_n=w13_size_n,
        size_k=w13_size_k,
        group_size=group_size,
        process_scales=_process_scales,
    )
    w2_scale_marlin = _permute_moe_fp4_scales_for_marlin(
        w2_scale_data,
        num_experts=num_experts,
        size_n=w2_size_n,
        size_k=w2_size_k,
        group_size=group_size,
        process_scales=_process_scales,
    )

    layer.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    if w13_bias_data is not None:
        layer.w13_weight_bias = torch.nn.Parameter(
            _permute_bias(w13_bias_data), requires_grad=False
        )
    if w2_bias_data is not None:
        layer.w2_weight_bias = torch.nn.Parameter(
            _permute_bias(w2_bias_data), requires_grad=False
        )

    # Marlin uses the repacked scales; release the loader-format parameters.
    for stale in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        if hasattr(layer, stale):
            delattr(layer, stale)


def prepare_moe_nvfp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    if layer.quant_config.group_size != 16:
        raise ValueError(
            f"NVFP4 Marlin MoE requires group_size=16, got {layer.quant_config.group_size}."
        )

    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = layer.w13_weight_scale.data
    w2_scale = layer.w2_weight_scale.data
    w13_global_scale = layer.w13_weight_scale_2.data
    w2_global_scale = layer.w2_weight_scale_2.data
    w13_bias = getattr(layer, "w13_bias", None)
    w2_bias = getattr(layer, "w2_bias", None)

    num_experts = w13.shape[0]
    num_shards = 2 if layer.moe_runner_config.is_gated else 1
    intermediate_size = layer.intermediate_size_per_partition
    hidden_size = w13.shape[2] * 2
    param_dtype = layer.params_dtype
    if param_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("NVFP4 Marlin MoE requires FP16 or BF16 activations.")

    device = w13.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    if not layer.moe_runner_config.is_gated:
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128
        intermediate_size_pad = padded_intermediate_size - intermediate_size
        if intermediate_size_pad:
            w13 = torch.nn.functional.pad(w13, (0, 0, 0, intermediate_size_pad))
            w13_scale = torch.nn.functional.pad(
                w13_scale, (0, 0, 0, intermediate_size_pad)
            )
            w2 = torch.nn.functional.pad(w2, (0, intermediate_size_pad // 2, 0, 0))
            w2_scale = torch.nn.functional.pad(
                w2_scale, (0, intermediate_size_pad // 16)
            )
            if w13_bias is not None:
                w13_bias = torch.nn.functional.pad(w13_bias, (0, intermediate_size_pad))
            intermediate_size = padded_intermediate_size

    w13_size_n, w13_size_k = intermediate_size * num_shards, hidden_size
    w2_size_n, w2_size_k = hidden_size, intermediate_size

    def _process_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
        return nvfp4_marlin_process_global_scale(global_scale.to(param_dtype))

    def _permute_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        tensor_list = []
        for i in range(num_experts):
            tensor_list.append(marlin_permute_bias(bias[i].to(param_dtype)))
        return torch.stack(tensor_list)

    layer.w13_weight = torch.nn.Parameter(
        _repack_moe_fp4_weight_for_marlin(
            w13,
            num_experts=num_experts,
            size_n=w13_size_n,
            size_k=w13_size_k,
            perm=perm,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        _repack_moe_fp4_weight_for_marlin(
            w2, num_experts=num_experts, size_n=w2_size_n, size_k=w2_size_k, perm=perm
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        _permute_moe_fp4_scales_for_marlin(
            w13_scale.to(param_dtype),
            num_experts=num_experts,
            size_n=w13_size_n,
            size_k=w13_size_k,
            group_size=16,
            process_scales=nvfp4_marlin_process_scales,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        _permute_moe_fp4_scales_for_marlin(
            w2_scale.to(param_dtype),
            num_experts=num_experts,
            size_n=w2_size_n,
            size_k=w2_size_k,
            group_size=16,
            process_scales=nvfp4_marlin_process_scales,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        _process_global_scale(w13_global_scale), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        _process_global_scale(w2_global_scale), requires_grad=False
    )

    if w13_bias is not None:
        layer.w13_bias = torch.nn.Parameter(
            _permute_bias(w13_bias), requires_grad=False
        )
    if w2_bias is not None:
        layer.w2_bias = torch.nn.Parameter(_permute_bias(w2_bias), requires_grad=False)

'''


# --- underlying ref: marlin_utils.py ---
_REF_MARLIN_UTILS = r'''
FILE: marlin_utils.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/quantization/marlin_utils.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy
import torch

from sglang.srt.layers.quantization.utils import (
    get_scalar_types,
    pack_cols,
    unpack_cols,
)
from sglang.srt.utils import get_device_capability, is_cuda
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm

logger = logging.getLogger(__name__)

ScalarType, scalar_types = get_scalar_types()

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
# NVFP SUPPORT 16, while MXFP4 supports 32 and 16.
FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16, 32]

# In case there is a performance issue with Marlin, the variable below can be
# changed to False, which allows Marlin to perform global reductions in fp16
# precision (instead of fp32), and therefore, save on some memory movements.
USE_FP32_REDUCE_DEFAULT = True


@dataclass
class MarlinLinearLayerConfig:
    full_weight_shape: tuple[int, int]  # [in, out]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool


# For binary size and compile time, we don't support the same types for with and
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ.
#  TODO: we may want to move this into the C++ so its closer to the actual impl
def query_marlin_supported_quant_types(
    has_zp: Optional[bool] = None,
    include_fp_type: bool = True,
    device_capability: Optional[int] = None,
):
    if device_capability is None:
        major, minor = get_device_capability()
        capability = major * 10 + minor if major is not None else None
        device_capability = -1 if capability is None else capability

    if device_capability < 80:
        return []

    # - has_zp is True: return quant_types that has zero points
    # - has_zp is False: return quant_types that has not zero points
    # - has_zp is None: both
    if has_zp is None:
        types0 = query_marlin_supported_quant_types(
            False, include_fp_type, device_capability
        )
        types1 = query_marlin_supported_quant_types(
            True, include_fp_type, device_capability
        )
        return types0 + types1

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4]
    else:
        # GPTQ style, unsigned + symmetric bias
        res = [scalar_types.uint4b8, scalar_types.uint8b128]
        if include_fp_type:
            res += [scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f]
        return res


def _check_marlin_supported(
    quant_type: ScalarType,
    group_size: Optional[int],
    has_zp: bool,
    device_capability: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    if device_capability is None:
        major, minor = get_device_capability()
        capability = major * 10 + minor if major is not None else None
        device_capability = -1 if capability is None else capability

    supported_types = query_marlin_supported_quant_types(
        has_zp, True, device_capability
    )

    if quant_type not in supported_types:
        return (
            False,
            f"Marlin does not support weight_bits = {quant_type}. "
            f"Only types = {supported_types} "
            f"are supported (for group_size = {group_size}, "
            f"device_capability = {device_capability}, zp = {has_zp}).",
        )
    if quant_type == scalar_types.float4_e2m1f:
        allowed_group_sizes = FP4_MARLIN_SUPPORTED_GROUP_SIZES
    else:
        allowed_group_sizes = MARLIN_SUPPORTED_GROUP_SIZES
    if group_size is None or group_size not in allowed_group_sizes:
        return (
            False,
            f"Marlin does not support group_size = {group_size} for "
            f"quant_type = {quant_type}. Only group_sizes = {allowed_group_sizes} "
            "are supported.",
        )

    return True, None


def check_marlin_supported(
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool = False,
    device_capability: Optional[int] = None,
) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp, device_capability)
    return cond


def verify_marlin_supported(
    quant_type: ScalarType, group_size: int, has_zp: bool = False
) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"Weight output_size_per_partition = "
            f"{output_size_per_partition} is not divisible by "
            f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible "
            f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )


def check_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> tuple[bool, Optional[str]]:
    try:
        verify_marlin_supports_shape(
            output_size_per_partition, input_size_per_partition, input_size, group_size
        )
    except ValueError as e:
        return False, e.__str__()
    return True, None


def check_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    output_size_per_partition = (
        getattr(layer, "output_size_per_partition", None) or layer.output_size
    )
    input_size_per_partition = (
        getattr(layer, "input_size_per_partition", None) or layer.input_size
    )

    return check_marlin_supports_shape(
        output_size_per_partition=output_size_per_partition,
        input_size_per_partition=input_size_per_partition,
        input_size=layer.input_size,
        group_size=group_size,
    )[0]


def check_moe_marlin_supports_layer(
    layer: FusedMoE, group_size: int, allow_tile_padding: bool = False
) -> bool:
    hidden_size = layer.hidden_size
    intermediate_size_per_partition = layer.intermediate_size_per_partition
    # apply_router_weight_on_input is not supported for moe marlin
    supports_router_weight = not layer.moe_runner_config.apply_router_weight_on_input
    if layer.moe_runner_config.is_gated:
        supports_activation = layer.moe_runner_config.activation in {"silu", "situ"}
    else:
        supports_activation = layer.moe_runner_config.activation in {
            "silu",
            "relu2",
        }

    if allow_tile_padding:
        # Thread-tile misalignment can be fixed by zero-padding the expert
        # intermediate dimension before Marlin repack. The original K still
        # needs to fit a Marlin tile family, and quant groups must stay whole.
        supports_shape = (
            hidden_size % 64 == 0 and intermediate_size_per_partition % group_size == 0
        )
    else:
        # gate-up: (n, k) = (intermediate_size_per_partition * 2, hidden_size)
        # down: (n, k) = (hidden_size, intermediate_size_per_partition)
        # moe marlin requires n % 128 == 0 and k % 64 == 0
        supports_shape = (
            hidden_size % 128 == 0
            and intermediate_size_per_partition % max(64, group_size) == 0
        )
    supports_group_size = group_size in [-1, 32, 64, 128]
    return (
        supports_shape
        and supports_group_size
        and supports_router_weight
        and supports_activation
    )


def marlin_make_workspace(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(
    act_order: bool, group_size: int, is_row_parallel: bool
) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def marlin_moe_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    num_experts = s.shape[0]
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # AWQ zero-points are quantized and packed on the column dim.
    # In addition, the values are permuted based on dequantizer.
    # Here we undo both of these, and then apply marlin permutation
    # and pack it back.
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo interleaving (use argsort(..) to get inverse perm)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)
    return marlin_zp


def moe_awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
):
    num_experts = q_zp_packed.shape[0]
    output = torch.empty(
        (num_experts, q_zp_packed.shape[1], q_zp_packed.shape[2]),
        device=q_zp_packed.device,
        dtype=q_zp_packed.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_to_marlin_zero_points(q_zp_packed[e], size_k, size_n, num_bits)
    return output


def maybe_warn_marlin_atomic_add(device, dtype):
    if torch.compiler.is_dynamo_compiling():
        return
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        logger.info_once(
            "You are running Marlin kernel with bf16 on GPUs before SM90. "
            "You can consider change to fp16 to achieve better performance "
            "if possible."
        )


def maybe_warn_marlin_atomic_add_env():
    if torch.compiler.is_dynamo_compiling():
        return
    # TODO(yiyun): Need to add sglang's MARLIN_USE_ATOMIC_ADD: bool = False
    if True:
        return
    # if envs.VLLM_MARLIN_USE_ATOMIC_ADD:
    #     return
    logger.info_once(
        "Marlin kernel can achieve better performance for small size_n "
        "with experimental use_atomic_add feature. "
        "You can consider set environment variable "
        "VLLM_MARLIN_USE_ATOMIC_ADD to 1 if possible."
    )


def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # disable atomicAdd reduce by default,
    # one can enable it with VLLM_MARLIN_USE_ATOMIC_ADD=1
    # TODO: Need to add sglang's MARLIN_USE_ATOMIC_ADD: bool = False
    if not True:
        maybe_warn_marlin_atomic_add_env()
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        maybe_warn_marlin_atomic_add(device, dtype)
        return False

    return True


def apply_gptq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    forward_context = get_tc_piecewise_forward_context()
    if forward_context is None:
        output = gptq_marlin_gemm(
            reshaped_x,
            None,
            weight,
            weight_scale,
            None,
            weight_zp,
            g_idx,
            g_idx_sort_indices,
            workspace,
            wtype,
            size_m=reshaped_x.shape[0],
            size_n=output_size_per_partition,
            size_k=input_size_per_partition,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )
    else:
        output = unified_apply_gptq_marlin_gemm_with_wtype(
            input=reshaped_x,
            weight=weight,
            weight_scale=weight_scale,
            weight_zp=weight_zp,
            g_idx=g_idx,
            g_idx_sort_indices=g_idx_sort_indices,
            workspace=workspace,
            wtype_id=wtype.id,
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def apply_awq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    forward_context = get_tc_piecewise_forward_context()
    if forward_context is None:
        output = gptq_marlin_gemm(
            reshaped_x,
            None,
            weight,
            weight_scale,
            None,
            weight_zp,
            g_idx,
            g_idx_sort_indices,
            workspace,
            quant_type,
            size_m=reshaped_x.shape[0],
            size_n=output_size_per_partition,
            size_k=input_size_per_partition,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )
    else:
        output = unified_apply_gptq_marlin_gemm(
            input=reshaped_x,
            weight=weight,
            weight_scale=weight_scale,
            weight_zp=weight_zp,
            g_idx=g_idx,
            g_idx_sort_indices=g_idx_sort_indices,
            workspace=workspace,
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def fake_unified_apply_gptq_marlin_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    output_size_per_partition: int,
    input_size_per_partition: int,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    return input.new_empty(
        (input.shape[0], output_size_per_partition), dtype=input.dtype
    )


@register_custom_op(fake_impl=fake_unified_apply_gptq_marlin_gemm)
def unified_apply_gptq_marlin_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    output_size_per_partition: int,
    input_size_per_partition: int,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    quant_config = get_tc_piecewise_forward_context().quant_config
    quant_type = quant_config.quant_type
    return gptq_marlin_gemm(
        input,
        None,
        weight,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        quant_type,
        size_m=input.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=is_zp_float,
    )


def fake_unified_apply_gptq_marlin_gemm_with_wtype(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype_id: int,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    return input.new_empty(
        (input.shape[0], output_size_per_partition), dtype=input.dtype
    )


@register_custom_op(fake_impl=fake_unified_apply_gptq_marlin_gemm_with_wtype)
def unified_apply_gptq_marlin_gemm_with_wtype(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype_id: int,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    # Reconstruct ScalarType from id
    wtype = None
    for attr_name in dir(scalar_types):
        if not attr_name.startswith("_"):
            st = getattr(scalar_types, attr_name)
            if hasattr(st, "id") and st.id == wtype_id:
                wtype = st
                break
    return gptq_marlin_gemm(
        input,
        None,
        weight,
        weight_scale,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=input.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=is_zp_float,
    )

'''


# --- underlying ref: moe_align_block_size.py ---
_REF_MOE_ALIGN_BLOCK = r'''
FILE: moe_align_block_size.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
from __future__ import annotations

from typing import Tuple

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_xpu

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()

if _is_cuda or _is_hip or _is_xpu or _is_musa:
    from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size

if _is_cuda:
    from sglang.kernels.ops.moe.moe_align_small_numel import (
        SMALL_NUMEL_LIMIT,
        moe_align_small_numel,
    )

# Where the CUDA kernel's own small-batch single-block path stops: its
# per-thread histogram costs 4 * (buckets + 1) ** 2 bytes of shared memory.
_CUDA_SMALL_BATCH_MAX_BUCKETS = 64


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    # ===== TO BE REFACTORED ====
    if _SGLANG_EXPERIMENTAL_LORA_OPTI:
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        if lora_envs.SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN.get() and num_experts <= 8191:
            from sglang.kernels.ops.lora.moe.virtual_experts import (
                _align_block_size_jit,
            )

            return _align_block_size_jit(topk_ids, block_size, num_experts)
    # ===== END TO BE REFACTORED ====

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    # The align kernel pads sorted_ids with int4 stores.
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    # In EP, expert_ids for filtered experts are -1. We have num_experts + 1 ids in total.
    cumsum_buffer = torch.empty(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )

    # Tiny-batch fast path (bs=1 decode): one single-CTA triton launch replaces
    # the generic align + count_and_sort pair, covering the corner the CUDA
    # small-batch kernel cannot reach. Below that bucket limit the CUDA kernel
    # is already a single launch and does O(numel) work where this one does
    # O(numel ** 2) pairwise, so leave that side to it. ignore_invalid_expert is
    # a different contract from the "+1 offset" convention this kernel implements.
    if (
        _is_cuda
        and topk_ids.numel() <= SMALL_NUMEL_LIMIT
        and num_experts + 1 > _CUDA_SMALL_BATCH_MAX_BUCKETS
        and not ignore_invalid_expert
    ):
        moe_align_small_numel(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

    # ===== TO BE REFACTORED ====
    use_jit_align = False
    if _SGLANG_EXPERIMENTAL_LORA_OPTI:
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        use_jit_align = lora_envs.SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN.get()
    if use_jit_align:
        from sglang.kernels.ops.moe.moe_align import (
            moe_align_block_size as jit_moe_align_block_size,
        )

        jit_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
        )
    # ===== END TO BE REFACTORED ====
    else:
        sgl_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
            ignore_invalid_expert,
        )
    return sorted_ids, expert_ids, num_tokens_post_pad

'''


# --- underlying ref: moe_align.py ---
_REF_MOE_ALIGN_PY = r'''
FILE: moe_align.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/moe/moe_align.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_align_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_align_block_size",
        *args,
        cuda_files=["moe/moe_align_kernel.cu"],
        cuda_wrappers=[
            ("moe_align_block_size", f"MoeAlignBlockSizeKernel<{args}>::run"),
        ],
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
) -> None:
    module = _jit_moe_align_module(topk_ids.dtype)
    module.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )

'''


# --- underlying ref: moe_align_single_token.py ---
_REF_MOE_ALIGN_SINGLE = r'''
FILE: moe_align_single_token.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/moe/moe_align_single_token.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
"""CUDA JIT single-warp moe_align_block_size for M == 1 decode batches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_align_single_token_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_align_single_token",
        *args,
        cuda_files=["moe/align_single_token.cuh"],
        cuda_wrappers=[("run", f"AlignSingleTokenKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def moe_align_single_token(
    topk_ids: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """moe_align_block_size for a single token (distinct expert ids).

    Returns (sorted_token_ids, expert_ids, num_tokens_post_padded) with the
    same layout as moe_align_block_size: experts ascending, one block per
    expert, padding value = numel.
    """
    topk = topk_ids.shape[1]
    device = topk_ids.device
    sorted_ids = torch.empty((topk * block_size,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((topk,), dtype=torch.int32, device=device)
    num_post = torch.empty((1,), dtype=torch.int32, device=device)
    _jit_align_single_token_module().run(
        topk_ids, sorted_ids, expert_ids, num_post, block_size
    )
    return sorted_ids, expert_ids, num_post

'''


# --- underlying ref: moe_align_small_numel.py ---
_REF_MOE_ALIGN_SMALL = r'''
FILE: moe_align_small_numel.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/moe/moe_align_small_numel.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
"""Single-launch moe_align for tiny batches with many experts.

The CUDA small-batch align kernel is gated to ``num_experts <= 64`` (its shared
memory grows as O(threads x experts)), so bs=1 decode on a MoE with a wider
expert dimension always paid the generic two-kernel (align + count_and_sort)
path. This kernel covers that corner in a single launch, at any expert count,
for ``numel <= SMALL_NUMEL_LIMIT``.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

# Largest numel routed to this kernel. Its [NP, NP] pairwise tensors fit in
# registers at NP=64 (~4 us, on par with the two CUDA launches it replaces) but
# spill to local memory at NP=256 (~230 us measured).
SMALL_NUMEL_LIMIT = 64


@triton.jit
def _moe_align_small_numel_kernel(
    topk_ids_ptr,  # [numel] int, flattened (token, slot) expert ids, -1 = filtered
    sorted_token_ids_ptr,  # [max_num_tokens_padded] int32
    expert_ids_ptr,  # [max_num_m_blocks] int32
    num_tokens_post_pad_ptr,  # [1] int32
    num_experts,  # E + 1 (the "+1 offset" convention's bucket count)
    block_size,
    numel,
    NP: tl.constexpr,  # power-of-2 >= numel
    NB: tl.constexpr,  # power-of-2 >= max blocks used
    USE_GDC: tl.constexpr = False,
):
    """Single-CTA moe_align for tiny batches with MANY experts.

    Everything works on the PAIR axis ([NP, NP] pairwise comparisons plus a
    rank-0 representative per bucket) -- an expert-axis formulation (histogram
    / cumsum over ~1k buckets) is ~3x more single-SM work and measured slower
    than the two-kernel path it replaces.

    Reference semantics reproduced:
    - "+1 offset" convention: expert -1 (EP-filtered) maps to bucket 0 and its
      blocks get expert_ids = -1 (skipped by fused_moe's filter_expert);
    - every bucket is padded to a block_size multiple, offsets in bucket order;
    - pad slots inside [0, num_tokens_post_pad) hold `numel`.

    Intended deviations, both invisible to fused_moe:
    - intra-bucket order is stable in pair index (the reference's atomicAdd
      order is scheduling-dependent; every pair writes its own output row);
    - sorted_token_ids beyond num_tokens_post_pad is left unwritten (the
      reference pre-fills the whole buffer; consumers only read below the
      published total).
    """
    if USE_GDC:
        # Consumer side of the router top-k that produced topk_ids.
        tl.extra.cuda.gdc_wait()

    offs_p = tl.arange(0, NP)
    mask_p = offs_p < numel
    ids = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=-2)
    # Padded lanes get an out-of-range bucket and are masked out everywhere.
    bucket = tl.where(mask_p, (ids + 1).to(tl.int32), num_experts)

    # Pairwise stats: stable rank within the bucket and bucket population.
    same = (bucket[None, :] == bucket[:, None]) & mask_p[None, :] & mask_p[:, None]
    earlier = offs_p[None, :] < offs_p[:, None]
    rank = tl.sum((same & earlier).to(tl.int32), axis=1)  # [NP]
    cnt = tl.sum(same.to(tl.int32), axis=1)  # [NP], own-bucket population
    padded_cnt = ((cnt + block_size - 1) // block_size) * block_size
    is_rep = (rank == 0) & mask_p  # one representative pair per bucket

    # Bucket-ordered exclusive offsets: sum the padded counts of every
    # representative with a strictly smaller bucket id.
    smaller_rep = (bucket[None, :] < bucket[:, None]) & is_rep[None, :]
    excl = tl.sum(smaller_rep.to(tl.int32) * padded_cnt[None, :], axis=1)  # [NP]

    total = tl.sum(tl.where(is_rep, padded_cnt, 0), axis=0)
    tl.store(num_tokens_post_pad_ptr, total.to(tl.int32))

    # expert_ids per used block: representative r owns blocks
    # [excl[r], excl[r] + padded_cnt[r]); the written id is bucket - 1
    # (bucket 0 = filtered -> -1).
    offs_b = tl.arange(0, NB)
    block_start = offs_b * block_size
    in_range = (
        (block_start[:, None] >= excl[None, :])
        & (block_start[:, None] < (excl + padded_cnt)[None, :])
        & is_rep[None, :]
    )
    eid = tl.sum(in_range.to(tl.int32) * (bucket[None, :] - 1), axis=1)
    tl.store(expert_ids_ptr + offs_b, eid.to(tl.int32), mask=block_start < total)

    # Fill the used region's pad slots with `numel`, then scatter the real
    # pair indices over them. The barrier is required: fill and scatter run on
    # different warps of this CTA, and a scatter store must not be overtaken
    # by a later-warp fill store to the same address.
    n_fill = (total + NP - 1) // NP
    for it in range(n_fill):
        f_offs = it * NP + offs_p
        tl.store(
            sorted_token_ids_ptr + f_offs,
            tl.full([NP], 0, tl.int32) + numel,
            mask=f_offs < total,
        )
    tl.debug_barrier()
    pos = excl + rank
    tl.store(sorted_token_ids_ptr + pos, offs_p.to(tl.int32), mask=mask_p)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def moe_align_small_numel(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Align and sort expert token ids into block-padded buffers, in one launch.

    Buffer contract matches ``sglang.kernels.ops.moe.moe_align_block_size``
    (minus its ``cumsum_buffer``, which a single CTA does not need):
    ``num_experts`` is the bucket count ``E + 1`` under the "+1 offset"
    convention, and the three output buffers are written in place.

    Callers gate on ``topk_ids.numel() <= SMALL_NUMEL_LIMIT``; the kernel stays
    correct above it, but its pairwise tensors spill and it stops being faster
    than the two-kernel path.
    """
    numel = topk_ids.numel()
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    _moe_align_small_numel_kernel[(1,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        num_experts,
        block_size,
        numel,
        NP=triton.next_power_of_2(max(numel, 2)),
        NB=triton.next_power_of_2(max(expert_ids.numel(), 2)),
        num_warps=4,
        **pdl_kwargs,
    )

'''


# --- underlying ref: moe_wna16_marlin.py ---
_REF_MOE_WNA16_PY = r'''
FILE: moe_wna16_marlin.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/moe/moe_wna16_marlin.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

# Constants matching device::marlin_moe:: in marlin.cuh
_MAX_THREAD_N = 256


@cache_once
def _jit_moe_wna16_marlin_module(
    dtype: torch.dtype, is_ep: bool, has_bias: bool
) -> Module:
    args = make_cpp_args(dtype, is_ep, has_bias)
    return load_jit(
        "moe_wna16_marlin",
        *args,
        cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],
        cuda_wrappers=[
            (
                "moe_wna16_marlin_gemm",
                f"moe_wna16_marlin_gemm<{args}>",
            )
        ],
    )


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


@debug_kernel_api
def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    device = a.device

    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.empty((size_m * top_k, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # Determine activation ordering
    has_act_order = (
        g_idx_or_none is not None
        and perm_or_none is not None
        and g_idx_or_none.numel() > 0
        and perm_or_none.numel() > 0
        and g_idx_or_none.size(-1) > 0
        and perm_or_none.size(-1) > 0
    )

    # Determine has_zp
    has_zp = b_zeros_or_none is not None and b_zeros_or_none.numel() > 0

    # Determine has_bias
    has_bias = b_bias_or_none is not None

    # Derive num_groups and group_size from b_scales
    num_groups = b_scales.size(1)
    if has_act_order:
        if is_k_full:
            group_size = size_k // num_groups
        else:
            group_size = 0
    else:
        if num_groups > 1:
            group_size = size_k // num_groups
        else:
            group_size = -1

    # Allocate a_tmp for act_order column permutation
    if has_act_order:
        a_tmp = torch.empty((size_m * top_k, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce and not use_atomic_add:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        # max num of threadblocks is sms * 4
        max_c_tmp_size = min(
            size_n * sorted_token_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = torch.empty(max_c_tmp_size, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Convert Optional tensors to empty tensors
    g_idx_t = _or_empty(g_idx_or_none, device, torch.int32)
    perm_t = _or_empty(perm_or_none, device, torch.int32)
    b_zeros_t = _or_empty(b_zeros_or_none, device, a.dtype)
    b_bias_t = _or_empty(b_bias_or_none, device, a.dtype)
    global_scale_t = _or_empty(global_scale_or_none, device, a.dtype)

    module = _jit_moe_wna16_marlin_module(a.dtype, is_ep, has_bias)
    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias_t,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        perm_t,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        a_tmp,
        c_tmp,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        has_act_order,
        has_bias,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    return c

'''


# --- underlying ref: activation.py ---
_REF_ACTIVATION_PY = r'''
FILE: activation.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/activation/activation.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_activation_cuda_cflags,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def activation_module(dtype: torch.dtype, *, fast_math: bool = True) -> Module:
    fast_math_flags = get_activation_cuda_cflags()
    if not fast_math and not fast_math_flags:
        return activation_module(dtype)
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "activation" if fast_math else "rounded_activation",
        *args,
        cuda_files=["elementwise/activation.cuh"],
        extra_cuda_cflags=fast_math_flags if fast_math else [],
        cuda_wrappers=[
            ("run_activation", f"ActivationKernel<{args}>::run_activation"),
            (
                "run_activation_with_rounding",
                f"ActivationKernel<{args}>::run_activation_with_rounding",
            ),
            (
                "run_activation_with_rounding_input_inplace",
                f"ActivationKernel<{args}>::run_activation_with_rounding_input_inplace",
            ),
            (
                "run_activation_filtered",
                f"ActivationKernel<{args}>::run_activation_filtered",
            ),
            (
                "run_unary_activation",
                f"ActivationKernel<{args}>::run_unary_activation",
            ),
        ],
    )


SUPPORTED_ACTIVATIONS = {"silu", "gelu", "gelu_tanh"}
SUPPORTED_UNARY_ACTIVATIONS = {"relu2"}


@register_custom_op(mutates_args=["out"])
def _run_activation_inplace(
    op_name: str, input: torch.Tensor, out: torch.Tensor
) -> None:
    hidden_size = input.shape[-1] // 2
    module = activation_module(input.dtype)
    input_2d = input.view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)
    module.run_activation(input_2d, out_2d, op_name)


@register_custom_op(mutates_args=["out"])
def _run_activation_with_rounding_inplace(
    op_name: str, input: torch.Tensor, out: torch.Tensor, clamp_limit: float = 0.0
) -> None:
    hidden_size = input.shape[-1] // 2
    # Fast-math changes FP16 SiLU at eager rounding boundaries on SM90.
    module = activation_module(input.dtype, fast_math=False)
    input_2d = input.view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)
    module.run_activation_with_rounding(input_2d, out_2d, op_name, clamp_limit)


@register_custom_op(mutates_args=["input"])
def _run_silu_and_mul_with_rounding_inplace(input: torch.Tensor) -> None:
    hidden_size = input.shape[-1] // 2
    module = activation_module(input.dtype, fast_math=False)
    input_2d = input.view(-1, hidden_size * 2)
    module.run_activation_with_rounding_input_inplace(
        input_2d, input_2d[:, :hidden_size], "silu"
    )


@register_custom_op(mutates_args=["out"])
def _run_activation_filtered_inplace(
    op_name: str,
    input: torch.Tensor,
    out: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_step: int,
) -> None:
    hidden_size = input.shape[-1] // 2
    module = activation_module(input.dtype)
    input_2d = input.view(-1, hidden_size * 2)
    out_2d = out.view(-1, hidden_size)
    module.run_activation_filtered(input_2d, out_2d, expert_ids, expert_step, op_name)


def run_activation(
    op_name: str,
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    """Apply ``op_name`` activation followed by element-wise multiplication.

    When ``expert_ids`` is provided, output rows are skipped for tokens whose
    routed expert id is ``-1``. ``expert_step`` is 1 for per-token routing and
    ``BLOCK_SIZE_M`` for sorted/TMA routing — i.e. ``expert_ids[token_id //
    expert_step]`` is consulted before computing each row.
    """
    assert op_name in SUPPORTED_ACTIVATIONS, f"Unsupported activation: {op_name}"
    hidden_size = input.shape[-1] // 2
    if out is None:
        out = input.new_empty(*input.shape[:-1], hidden_size)
    if expert_ids is None:
        _run_activation_inplace(op_name, input, out)
    else:
        # The JIT kernel indexes expert ids as int32. Routing ids may arrive as
        # int64 (e.g. from torch.topk) and torch.compile realizes them at their
        # true dtype, so normalize here instead of asserting downstream.
        if expert_ids.dtype != torch.int32:
            expert_ids = expert_ids.to(torch.int32)
        _run_activation_filtered_inplace(op_name, input, out, expert_ids, expert_step)
    return out


@register_custom_op(mutates_args=["out"])
def _run_unary_activation_inplace(
    op_name: str, input: torch.Tensor, out: torch.Tensor, fast_math: bool = True
) -> None:
    last = input.shape[-1]
    module = activation_module(input.dtype, fast_math=fast_math)
    module.run_unary_activation(input.view(-1, last), out.view(-1, last), op_name)


def run_unary_activation(
    op_name: str,
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    fast_math: bool = True,
) -> torch.Tensor:
    """Apply a standalone (non-gated) element-wise activation: ``out = act(input)``.

    Unlike :func:`run_activation`, there is no gate/up split — ``input`` and
    ``out`` share the same shape.
    """
    assert op_name in SUPPORTED_UNARY_ACTIVATIONS, (
        f"Unsupported unary activation: {op_name}"
    )
    if out is None:
        out = torch.empty_like(input)
    _run_unary_activation_inplace(op_name, input, out, fast_math)
    return out


def relu2(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    fast_math: bool = True,
) -> torch.Tensor:
    """Squared ReLU; disable fast math to preserve BF16 subnormal results."""
    return run_unary_activation("relu2", input, out, fast_math=fast_math)


def silu_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    return run_activation("silu", input, out, expert_ids, expert_step)


def silu_and_mul_with_activation_rounding(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    clamp_limit: float = 0.0,
) -> torch.Tensor:
    hidden_size = input.shape[-1] // 2
    if out is None:
        out = input.new_empty(*input.shape[:-1], hidden_size)
    _run_activation_with_rounding_inplace("silu", input, out, clamp_limit)
    return out


def silu_and_mul_with_activation_rounding_(input: torch.Tensor) -> torch.Tensor:
    hidden_size = input.shape[-1] // 2
    _run_silu_and_mul_with_rounding_inplace(input)
    return input[..., :hidden_size]


def gelu_and_mul_with_activation_rounding(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    hidden_size = input.shape[-1] // 2
    if out is None:
        out = input.new_empty(*input.shape[:-1], hidden_size)
    _run_activation_with_rounding_inplace("gelu", input, out)
    return out


def gelu_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    return run_activation("gelu", input, out, expert_ids, expert_step)


def gelu_tanh_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    return run_activation("gelu_tanh", input, out, expert_ids, expert_step)

'''


# --- underlying ref: fused_post_expert_norm_weighted_sum_triton.py ---
_REF_POST_EXPERT_TRITON = r'''
FILE: fused_post_expert_norm_weighted_sum_triton.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
from typing import Optional

import torch
import triton
import triton.language as tl

_MIN_TOKENS_FOR_FUSED = 32


@triton.jit
def _post_expert_norm_weighted_sum_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + tok * TOPK + k) >= 0
        if live:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            acc += y * s

    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_post_expert_norm_weighted_sum(
    expert_output: torch.Tensor,
    topk_scores: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    weight_first: bool = True,
    apply_norm: bool = True,
    slot_expert_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = expert_output.contiguous()
    tokens, topk, hidden = x.shape
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    if tokens < _MIN_TOKENS_FOR_FUSED:
        y = x
        if slot_expert_ids is not None:
            y = torch.where(
                (slot_expert_ids.reshape(tokens, topk) >= 0).unsqueeze(-1),
                y,
                torch.zeros((), dtype=y.dtype, device=y.device),
            )
        if apply_norm:
            f = y.float()
            if norm_weight is not None and weight_first:
                f = f * norm_weight.float()
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            elif norm_weight is not None:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
                f = f * norm_weight.float()
            else:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            y = f
        else:
            y = y.float()
        y = y * topk_scores.float().unsqueeze(-1)
        return y.sum(dim=1).to(x.dtype)

    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x,
        norm_weight if norm_weight is not None else x,
        topk_scores.contiguous(),
        out,
        slot_expert_ids if slot_expert_ids is not None else x,
        TOPK=topk,
        H=hidden,
        eps=eps,
        APPLY_NORM=apply_norm,
        HAS_WEIGHT=norm_weight is not None,
        WEIGHT_FIRST=weight_first,
        HAS_IDS=slot_expert_ids is not None,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


__all__ = ["fused_post_expert_norm_weighted_sum"]

'''


# --- underlying ref: gptq_marlin_repack.py ---
_REF_GPTQ_REPACK_PY = r'''
FILE: gptq_marlin_repack.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/quantization/gptq_marlin_repack.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Constants matching device::marlin:: in marlin.cuh
_TILE_SIZE = 16


@cache_once
def _jit_gptq_marlin_repack_module() -> Module:
    return load_jit(
        "gptq_marlin_repack",
        cuda_files=["gemm/marlin/gptq_marlin_repack.cuh"],
        cuda_wrappers=[("gptq_marlin_repack", "gptq_marlin_repack")],
    )


@debug_kernel_api
def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits

    # Allocate output tensor
    out = torch.empty(
        (size_k // _TILE_SIZE, size_n * _TILE_SIZE // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )

    module = _jit_gptq_marlin_repack_module()
    module.gptq_marlin_repack(b_q_weight, perm, out, size_k, size_n, num_bits)
    return out

'''


# --- underlying ref: scalar_type.py ---
_REF_SCALAR_TYPE_PY = r'''
FILE: scalar_type.py
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/aot/python/sgl_kernel/scalar_type.py
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

_SCALAR_TYPES_ID_MAP = {}


# Mirrors enum in `core/scalar_type.hpp`
class NanRepr(Enum):
    NONE = 0  # nans are not supported
    IEEE_754 = 1  # nans are: Exp all 1s, mantissa not all 0s
    EXTD_RANGE_MAX_MIN = 2  # nans are: Exp all 1s, mantissa all 1s


# This ScalarType class is a parallel implementation of the C++ ScalarType
# class found in csrc/core/scalar_type.hpp.  These two classes should be kept
# in sync until the inductor fully supports custom C++ classes.
@dataclass(frozen=True)
class ScalarType:
    """
    ScalarType can represent a wide range of floating point and integer
    types, in particular it can be used to represent sub-byte data types
    (something that torch.dtype currently does not support). It is also
    capable of  representing types with a bias, i.e.:
      `stored_value = value + bias`,
    this is useful for quantized types (e.g. standard GPTQ 4bit uses a bias
    of 8). The implementation for this class can be found in
    csrc/core/scalar_type.hpp, these type signatures should be kept in sync
    with that file.
    """

    exponent: int
    """
    Number of bits in the exponent if this is a floating point type
    (zero if this an integer type)
    """

    mantissa: int
    """
    Number of bits in the mantissa if this is a floating point type,
    or the number bits representing an integer excluding the sign bit if
    this an integer type.
    """

    signed: bool
    "If the type is signed (i.e. has a sign bit)"

    bias: int
    """
    bias used to encode the values in this scalar type
    (value = stored_value - bias, default 0) for example if we store the
    type as an unsigned integer with a bias of 128 then the value 0 will be
    stored as 128 and -1 will be stored as 127 and 1 will be stored as 129.
    """

    _finite_values_only: bool = False
    """
    Private: if infs are supported, used `has_infs()` instead.
    """

    nan_repr: NanRepr = NanRepr.IEEE_754
    """
    How NaNs are represent in this scalar type, returns NanRepr value.
    (not applicable for integer types)
    """

    def _floating_point_max_int(self) -> int:
        assert self.mantissa <= 52 and self.exponent <= 11, (
            f"Cannot represent max/min as a double for type {self.__str__()}"
        )

        max_mantissa = (1 << self.mantissa) - 1
        if self.nan_repr == NanRepr.EXTD_RANGE_MAX_MIN:
            max_mantissa = max_mantissa - 1

        max_exponent = (1 << self.exponent) - 2
        if self.nan_repr == NanRepr.EXTD_RANGE_MAX_MIN or self.nan_repr == NanRepr.NONE:
            assert self.exponent < 11, (
                f"Cannot represent max/min as a double for type {self.__str__()}"
            )
            max_exponent = max_exponent + 1

        # adjust the exponent to match that of a double
        # for now we assume the exponent bias is the standard 2^(e-1) -1, (where
        # e is the exponent bits), there is some precedent for non-standard
        # biases, example `float8_e4m3b11fnuz` here:
        # https://github.com/jax-ml/ml_dtypes but to avoid premature over
        # complication we are just assuming the standard exponent bias until
        # there is a need to support non-standard biases
        exponent_bias = (1 << (self.exponent - 1)) - 1
        exponent_bias_double = (1 << 10) - 1  # double e = 11

        max_exponent_double = max_exponent - exponent_bias + exponent_bias_double

        # shift the mantissa and exponent into the proper positions for an
        # IEEE double and bitwise-or them together.
        return (max_mantissa << (52 - self.mantissa)) | (max_exponent_double << 52)

    def _floating_point_max(self) -> float:
        double_raw = self._floating_point_max_int()
        return struct.unpack("!d", struct.pack("!Q", double_raw))[0]

    def _raw_max(self) -> Union[int, float]:
        if self.is_floating_point():
            return self._floating_point_max()
        else:
            assert self.size_bits < 64 or self.size_bits == 64 and self.is_signed(), (
                "Cannot represent max as an int"
            )
            return (1 << self.mantissa) - 1

    def _raw_min(self) -> Union[int, float]:
        if self.is_floating_point():
            assert self.is_signed(), (
                "We currently assume all floating point types are signed"
            )
            sign_bit_double = 1 << 63

            max_raw = self._floating_point_max_int()
            min_raw = max_raw | sign_bit_double
            return struct.unpack("!d", struct.pack("!Q", min_raw))[0]
        else:
            assert not self.is_signed() or self.size_bits <= 64, (
                "Cannot represent min as a int64_t"
            )

            if self.is_signed():
                return -(1 << (self.size_bits - 1))
            else:
                return 0

    @functools.cached_property
    def id(self) -> int:
        """
        Convert the ScalarType to an int which can be passed to pytorch custom
        ops. This layout of the int must be kept in sync with the C++
        ScalarType's from_id method.
        """
        val = 0
        offset = 0

        def or_and_advance(member, bit_width):
            nonlocal val
            nonlocal offset
            bit_mask = (1 << bit_width) - 1
            val = val | (int(member) & bit_mask) << offset
            offset = offset + bit_width

        or_and_advance(self.exponent, 8)
        or_and_advance(self.mantissa, 8)
        or_and_advance(self.signed, 1)
        or_and_advance(self.bias, 32)
        or_and_advance(self._finite_values_only, 1)
        or_and_advance(self.nan_repr.value, 8)

        assert offset <= 64, f"ScalarType fields too big {offset} to fit into an int64"

        _SCALAR_TYPES_ID_MAP[val] = self

        return val

    @property
    def size_bits(self) -> int:
        return self.exponent + self.mantissa + int(self.signed)

    def min(self) -> Union[int, float]:
        """
        Min representable value for this scalar type.
        (accounting for bias if there is one)
        """
        return self._raw_min() - self.bias

    def max(self) -> Union[int, float]:
        """
        Max representable value for this scalar type.
        (accounting for bias if there is one)
        """
        return self._raw_max() - self.bias

    def is_signed(self) -> bool:
        """
        If the type is signed (i.e. has a sign bit), same as `signed`
        added for consistency with:
        https://pytorch.org/docs/stable/generated/torch.Tensor.is_signed.html
        """
        return self.signed

    def is_floating_point(self) -> bool:
        "If the type is a floating point type"
        return self.exponent != 0

    def is_integer(self) -> bool:
        "If the type is an integer type"
        return self.exponent == 0

    def has_bias(self) -> bool:
        "If the type has a non-zero bias"
        return self.bias != 0

    def has_infs(self) -> bool:
        "If the type is floating point and supports infinity"
        return not self._finite_values_only

    def has_nans(self) -> bool:
        return self.nan_repr != NanRepr.NONE

    def is_ieee_754(self) -> bool:
        """
        If the type is a floating point type that follows IEEE 754
        conventions
        """
        return self.nan_repr == NanRepr.IEEE_754 and not self._finite_values_only

    def __str__(self) -> str:
        """
        naming generally follows: https://github.com/jax-ml/ml_dtypes
        for floating point types (leading f) the scheme is:
        `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
        flags:
          - no-flags: means it follows IEEE 754 conventions
          - f: means finite values only (no infinities)
          - n: means nans are supported (non-standard encoding)
        for integer types the scheme is:
          `[u]int<size_bits>[b<bias>]`
          - if bias is not present it means its zero
        """
        if self.is_floating_point():
            ret = (
                "float"
                + str(self.size_bits)
                + "_e"
                + str(self.exponent)
                + "m"
                + str(self.mantissa)
            )

            if not self.is_ieee_754():
                if self._finite_values_only:
                    ret = ret + "f"
                if self.nan_repr != NanRepr.NONE:
                    ret = ret + "n"

            return ret
        else:
            ret = ("int" if self.is_signed() else "uint") + str(self.size_bits)
            if self.has_bias():
                ret = ret + "b" + str(self.bias)
            return ret

    def __repr__(self) -> str:
        return "ScalarType." + self.__str__()

    # __len__ needs to be defined (and has to throw TypeError) for pytorch's
    # opcheck to work.
    def __len__(self) -> int:
        raise TypeError

    #
    # Convenience Constructors
    #

    @classmethod
    def int_(cls, size_bits: int, bias: Optional[int]) -> "ScalarType":
        "Create a signed integer scalar type (size_bits includes sign-bit)."
        ret = cls(0, size_bits - 1, True, bias if bias else 0)
        ret.id  # noqa B018: make sure the id is cached
        return ret

    @classmethod
    def uint(cls, size_bits: int, bias: Optional[int]) -> "ScalarType":
        """Create a unsigned integer scalar type."""
        ret = cls(0, size_bits, False, bias if bias else 0)
        ret.id  # noqa B018: make sure the id is cached
        return ret

    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> "ScalarType":
        """
        Create a standard floating point type
        (i.e. follows IEEE 754 conventions).
        """
        assert mantissa > 0 and exponent > 0
        ret = cls(exponent, mantissa, True, 0)
        ret.id  # noqa B018: make sure the id is cached
        return ret

    @classmethod
    def float_(
        cls, exponent: int, mantissa: int, finite_values_only: bool, nan_repr: NanRepr
    ) -> "ScalarType":
        """
        Create a non-standard floating point type
        (i.e. does not follow IEEE 754 conventions).
        """
        assert mantissa > 0 and exponent > 0
        assert nan_repr != NanRepr.IEEE_754, (
            "use `float_IEEE754` constructor for floating point types that "
            "follow IEEE 754 conventions"
        )
        ret = cls(exponent, mantissa, True, 0, finite_values_only, nan_repr)
        ret.id  # noqa B018: make sure the id is cached
        return ret

    @classmethod
    def from_id(cls, scalar_type_id: int):
        if scalar_type_id not in _SCALAR_TYPES_ID_MAP:
            raise ValueError(f"scalar_type_id {scalar_type_id} doesn't exists.")
        return _SCALAR_TYPES_ID_MAP[scalar_type_id]


# naming generally follows: https://github.com/jax-ml/ml_dtypes
# for floating point types (leading f) the scheme is:
#  `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
#  flags:
#  - no-flags: means it follows IEEE 754 conventions
#  - f: means finite values only (no infinities)
#  - n: means nans are supported (non-standard encoding)
# for integer types the scheme is:
#  `[u]int<size_bits>[b<bias>]`
#  - if bias is not present it means its zero


class scalar_types:
    int4 = ScalarType.int_(4, None)
    uint4 = ScalarType.uint(4, None)
    int8 = ScalarType.int_(8, None)
    uint8 = ScalarType.uint(8, None)
    float8_e4m3fn = ScalarType.float_(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN)
    float8_e5m2 = ScalarType.float_IEEE754(5, 2)
    float16_e8m7 = ScalarType.float_IEEE754(8, 7)
    float16_e5m10 = ScalarType.float_IEEE754(5, 10)

    # fp6, https://github.com/usyd-fsalab/fp6_llm/tree/main
    float6_e3m2f = ScalarType.float_(3, 2, True, NanRepr.NONE)

    # fp4, https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)

    # "gptq" types
    uint2b2 = ScalarType.uint(2, 2)
    uint3b4 = ScalarType.uint(3, 4)
    uint4b8 = ScalarType.uint(4, 8)
    uint8b128 = ScalarType.uint(8, 128)

    # colloquial names
    bfloat16 = float16_e8m7
    float16 = float16_e5m10

'''


# --- underlying ref: align_single_token.cuh ---
_REF_ALIGN_SINGLE_CUH = r'''
FILE: align_single_token.cuh
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/moe/align_single_token.cuh
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
// Tiny moe_align_block_size for M == 1 decode: one warp replaces the
// moe_align_block_size + count_and_sort_expert_tokens kernel pair (~4.4us)
// with a single ~1.5us launch.
//
// For a single token the top-k expert ids are distinct, so the aligned
// layout is exactly: experts sorted ascending, one block per expert,
// slot i of block b = flat topk index for b's expert, remaining block
// slots padded with numel (= topk). num_tokens_post_padded = topk * block.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct AlignSingleTokenParams {
  const int32_t* __restrict__ topk_ids;  // [1, topk]
  int32_t* __restrict__ sorted_ids;      // [topk * block_size]
  int32_t* __restrict__ expert_ids;      // [topk]
  int32_t* __restrict__ num_post;        // [1]
  uint32_t topk;
  uint32_t block_size;
};

template <bool kUsePDL>
__global__ void align_single_token_kernel(const AlignSingleTokenParams __grid_constant__ params) {
  using namespace device;
  const uint32_t lane = threadIdx.x;  // one warp
  const uint32_t topk = params.topk;
  const uint32_t bs = params.block_size;

  PDLWaitPrimary<kUsePDL>();

  int32_t my_id = (lane < topk) ? params.topk_ids[lane] : INT32_MAX;

  // Rank of my expert id among the topk (ids are distinct for one token;
  // tie-break on lane keeps this robust anyway).
  uint32_t rank = 0;
  for (uint32_t j = 0; j < topk; ++j) {
    int32_t other = __shfl_sync(0xffffffff, my_id, j);
    if (other < my_id || (other == my_id && j < lane)) {
      rank++;
    }
  }

  if (lane < topk) {
    params.expert_ids[rank] = my_id;
    // block `rank`: first slot is my flat index (token 0, slot `lane`),
    // rest padded with numel (= topk).
    params.sorted_ids[rank * bs] = static_cast<int32_t>(lane);
  }
  // Fill padding cooperatively: positions not equal to a block start.
  for (uint32_t p = lane; p < topk * bs; p += 32) {
    if (p % bs != 0) {
      params.sorted_ids[p] = static_cast<int32_t>(topk);
    }
  }
  if (lane == 0) {
    params.num_post[0] = static_cast<int32_t>(topk * bs);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct AlignSingleTokenKernel {
  static constexpr auto kernel = align_single_token_kernel<kUsePDL>;

  static void
  run(const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView sorted_ids,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView num_post,
      int64_t block_size) {
    using namespace host;

    auto One_ = SymbolicSize{"one"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({One_, K_}).with_dtype<int32_t>().with_device(device).verify(topk_ids);

    const auto topk = static_cast<uint32_t>(K_.unwrap());
    RuntimeCheck(One_.unwrap() == 1, "moe_align_single_token requires M == 1");
    RuntimeCheck(topk <= 32, "moe_align_single_token requires topk <= 32");

    const auto params = AlignSingleTokenParams{
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .sorted_ids = static_cast<int32_t*>(sorted_ids.data_ptr()),
        .expert_ids = static_cast<int32_t*>(expert_ids.data_ptr()),
        .num_post = static_cast<int32_t*>(num_post.data_ptr()),
        .topk = topk,
        .block_size = static_cast<uint32_t>(block_size),
    };

    LaunchKernel(dim3(1), 32, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang

'''


# --- underlying ref: moe_align_kernel.cu ---
_REF_MOE_ALIGN_CU = r'''
FILE: moe_align_kernel.cu
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/moe/moe_align_kernel.cu
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
# NOTE: truncated; full file at underlying/moe_align_kernel.cu (19129 bytes)
/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <sgl_kernel/bits.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define VEC_SIZE 4
namespace sglang {

using Vec = int4;

namespace moe {

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size,
    int32_t max_num_tokens_padded) {
  // Use a separate thread block to populate sorted_token_ids
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      Vec fill_vec;
      fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
      // Whole vectors only; a partial last vector would write past the buffer.
      int32_t total_vecs = max_num_tokens_padded / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        out_ptr[i] = fill_vec;
      }
      for (int32_t i = total_vecs * VEC_SIZE + threadIdx.x; i < max_num_tokens_padded; i += blockDim.x) {
        sorted_token_ids[i] = numel;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                  // [num_experts]
  int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i] + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }

  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

#ifndef __CUDA_ARCH__  // HIP

  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }

  __syncthreads();

  // Blelloch scan
  int offset = 1;
#pragma unroll
  for (int d = scan_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      scan_buf[bi] += scan_buf[ai];
    }
    offset <<= 1;
    __syncthreads();
  }

  // down-sweep
  if (tid == 0) {
    prefix[num_experts] = scan_buf[scan_size - 1];
    scan_buf[scan_size - 1] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int d = 1; d < scan_size; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      if (bi < scan_size) {
        int temp = scan_buf[ai];
        scan_buf[ai] = scan_buf[bi];
        scan_buf[bi] += temp;
      }
    }
    __syncthreads();
  }

  if (tid < num_experts) {
    prefix[tid] = scan_buf[tid];
  }

  if (tid == 0) {
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

#else  // CUDA

  // Intra warp prefix sum
  int32_t* warp_sums = scan_buf + scan_size;  // [<= 32]
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;
  const int warp_sum = device::warp::inclusive_sum<32>(padded_count);
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;
  __syncthreads();

  // warp0 accumulate all the block's prefix sum
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = device::warp::inclusive_sum<32>(val);
    warp_sums[tid] = incl;
  }
  __syncthreads();

  // Every thread obtains the whole block's sum
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  // Fill 0 to scan_buf extended area (tid >= num_expert)
  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();

  // Perform 2 level exclusive-prefix-sum to scan_buf
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = device::warp::inclusive_sum<32>(v) - v;
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan

# ... [7249 chars omitted; full file in underlying/moe_align_kernel.cu] ...

read_prefix + running;
    }
    running += local_padded[i];
  }

  // Last thread writes total
  if (tid == blockDim.x - 1) {
    cumsum[num_experts] = thread_prefix + thread_sum;
    *total_tokens_post_pad = thread_prefix + thread_sum;
  }

  __syncthreads();

  // Phase 3: Fill expert_ids (eid - 1 to match sgl-kernel convention)
  for (int i = 0; i < EXPERTS_PER_THREAD; ++i) {
    int eid = my_start + i;
    if (eid < num_experts) {
      for (int j = cumsum[eid]; j < cumsum[eid + 1]; j += block_size) {
        expert_ids[j / block_size] = eid - 1;
      }
    }
  }
}

}  // namespace moe

template <typename scalar_t>
struct MoeAlignBlockSizeKernel {
  static void
  run(tvm::ffi::TensorView topk_ids,
      int64_t num_experts,
      int64_t block_size,
      tvm::ffi::TensorView sorted_token_ids,
      tvm::ffi::TensorView expert_ids,
      tvm::ffi::TensorView num_tokens_post_pad,
      tvm::ffi::TensorView cumsum_buffer,
      bool pad_sorted_token_ids) {
    using namespace host;

    auto device = topk_ids.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);

    int threads = 1024;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    int64_t max_num_tokens_padded = sorted_token_ids.size(0);

    // num_experts from Python is actual_num_experts + 1 (for EP offset convention).
    // The v2 kernel (>1024 experts) uses 1024 threads with EXPERTS_PER_THREAD up
    // to 8, covering at most 8192 expert indices. This supports up to 8191 real
    // experts, sufficient for LoRA virtual experts (num_moe_experts * max_loras).
    RuntimeCheck(num_experts <= 8192, "moe_align_block_size: num_experts must be <= 8192, got ", num_experts);

    const scalar_t* topk_ids_ptr = static_cast<const scalar_t*>(topk_ids.data_ptr());
    int32_t* sorted_token_ids_ptr = static_cast<int32_t*>(sorted_token_ids.data_ptr());
    int32_t* expert_ids_ptr = static_cast<int32_t*>(expert_ids.data_ptr());
    int32_t* num_tokens_post_pad_ptr = static_cast<int32_t*>(num_tokens_post_pad.data_ptr());
    int32_t* cumsum_buffer_ptr = static_cast<int32_t*>(cumsum_buffer.data_ptr());
    size_t numel = topk_ids.numel();

    bool small_batch_expert_mode = (numel < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode) {
      const int32_t num_thread = std::max((int32_t)num_experts, (int32_t)WARP_SIZE);
      constexpr int32_t fill_threads = 256;
      const int32_t shared_mem_size = ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

      auto kernel = moe::moe_align_block_size_small_batch_expert_kernel<scalar_t, fill_threads>;
      LaunchKernel(dim3(1), dim3(fill_threads + num_thread), stream, shared_mem_size)(
          kernel,
          topk_ids_ptr,
          sorted_token_ids_ptr,
          expert_ids_ptr,
          num_tokens_post_pad_ptr,
          (int32_t)num_experts,
          (int32_t)block_size,
          numel,
          pad_sorted_token_ids,
          (int32_t)max_num_tokens_padded);
    } else if (num_experts <= 1024) {
      const size_t scan_size = host::round_up_pow2(static_cast<uint32_t>(num_experts));
      const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size + WARP_SIZE) * sizeof(int32_t);

      auto align_kernel = moe::moe_align_block_size_kernel<scalar_t>;
      LaunchKernel(dim3(2), dim3(threads), stream, shared_mem_size)(
          align_kernel,
          topk_ids_ptr,
          sorted_token_ids_ptr,
          expert_ids_ptr,
          num_tokens_post_pad_ptr,
          (int32_t)num_experts,
          (int32_t)block_size,
          numel,
          cumsum_buffer_ptr,
          pad_sorted_token_ids,
          (int32_t)scan_size,
          (int32_t)max_num_tokens_padded);

      const int block_threads = std::min(256, threads);
      const int num_blocks = (numel + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      auto sort_kernel = moe::count_and_sort_expert_tokens_kernel<scalar_t>;
      LaunchKernel(dim3(actual_blocks), dim3(block_threads), stream)(
          sort_kernel, topk_ids_ptr, sorted_token_ids_ptr, cumsum_buffer_ptr, numel);
    } else {
      // v2 path for >1024 experts: two-level warp scan with EXPERTS_PER_THREAD
      int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
      size_t shared_mem_size = (padded_num_experts + WARP_SIZE) * sizeof(int32_t);

      auto launch_v2 = [&](auto ept_tag) {
        constexpr int EPT = decltype(ept_tag)::value;
        auto v2_kernel = moe::moe_align_block_size_kernel_v2<scalar_t, EPT>;
        LaunchKernel(dim3(2), dim3(threads), stream, shared_mem_size)(
            v2_kernel,
            topk_ids_ptr,
            sorted_token_ids_ptr,
            expert_ids_ptr,
            num_tokens_post_pad_ptr,
            (int32_t)num_experts,
            (int32_t)padded_num_experts,
            (int32_t)block_size,
            numel,
            cumsum_buffer_ptr,
            pad_sorted_token_ids,
            (int32_t)max_num_tokens_padded);
      };

      if (padded_num_experts <= 2048) {
        launch_v2(std::integral_constant<int, 2>{});
      } else if (padded_num_experts <= 4096) {
        launch_v2(std::integral_constant<int, 4>{});
      } else {
        launch_v2(std::integral_constant<int, 8>{});
      }

      const int block_threads = std::min(256, threads);
      const int num_blocks = (numel + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      auto sort_kernel = moe::count_and_sort_expert_tokens_kernel<scalar_t>;
      LaunchKernel(dim3(actual_blocks), dim3(block_threads), stream)(
          sort_kernel, topk_ids_ptr, sorted_token_ids_ptr, cumsum_buffer_ptr, numel);
    }
  }
};

}  // namespace sglang

'''


# --- underlying ref: activation.cuh ---
_REF_ACTIVATION_CUH = r'''
FILE: activation.cuh
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/elementwise/activation.cuh
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
# NOTE: truncated; full file at underlying/activation.cuh (12269 bytes)
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace sglang {

enum class ActivationKind : uint32_t {
  kSiLU,
  kGELU,
  kGELUTanh,
  kReLU2,
};

template <ActivationKind kAct>
SGL_DEVICE float apply_activation_f32(float x_f32) {
  if constexpr (kAct == ActivationKind::kSiLU) {
    return x_f32 / (1.0f + expf(-x_f32));
  } else if constexpr (kAct == ActivationKind::kGELU) {
    constexpr auto kSqrt1Over2 = 0.7071067811865475f;
    return x_f32 * (0.5f * (1.0f + erff(x_f32 * kSqrt1Over2)));
  } else if constexpr (kAct == ActivationKind::kGELUTanh) {
    constexpr auto kGeluTanhAlpha = 0.044715f;
    constexpr auto kGeluTanhBeta = 0.7978845608028654f;
    const float cdf = 0.5f * (1.0f + tanhf(kGeluTanhBeta * (x_f32 + kGeluTanhAlpha * x_f32 * x_f32 * x_f32)));
    return x_f32 * cdf;
  } else if constexpr (kAct == ActivationKind::kReLU2) {
    const float relu = x_f32 > 0.0f ? x_f32 : 0.0f;
    return relu * relu;
  } else {
    static_assert(host::dependent_false_v<decltype(kAct)>, "unsupported activation kind");
    return 0.0f;
  }
}

struct ActivationParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  uint32_t hidden_dim;
  uint32_t num_tokens;
  // Optional MoE expert filtering: when expert_ids != nullptr, a token is
  // skipped if expert_ids[token_id / expert_step] == -1. expert_step is 1
  // for per-token routing and BLOCK_SIZE_M for sorted/TMA routing.
  const int32_t* __restrict__ expert_ids;
  uint32_t expert_step;
  float clamp_limit;
};

template <
    typename T,
    ActivationKind kAct,
    bool kUsePDL,
    bool kFilterExpert,
    bool kRoundActivation = false,
    bool kReuseInput = false,
    bool kClamp = false>
__global__ void act_and_mul_kernel(const __grid_constant__ ActivationParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;
  const auto num_vecs = params.hidden_dim / kVecSize;  // per token
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto token_id = tid / num_vecs;

  if (token_id >= params.num_tokens) return;
  if constexpr (kFilterExpert) {
    if (params.expert_ids[token_id / params.expert_step] == -1) return;
  }
  const auto offset = tid % num_vecs;
  const auto input_offset = token_id * (num_vecs * 2) + offset;
  const auto output_offset = kReuseInput ? input_offset : tid;
  PDLWaitPrimary<kUsePDL>();
  const auto gate = device::load_as<vec_t>(params.input, input_offset);
  const auto up = device::load_as<vec_t>(params.input, input_offset + num_vecs);
  const float limit = device::cast<fp32_t>(device::cast<T>(params.clamp_limit));
  vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    float gate_f32 = device::cast<fp32_t>(gate[i]);
    float up_f32 = device::cast<fp32_t>(up[i]);
    if constexpr (kClamp) {
      static_assert(kAct == ActivationKind::kSiLU);
      gate_f32 = gate_f32 > limit ? limit : gate_f32;
      up_f32 = up_f32 > limit ? limit : (up_f32 < -limit ? -limit : up_f32);
    }
    if constexpr (kRoundActivation) {
      const T activated = device::cast<T>(apply_activation_f32<kAct>(gate_f32));
      out[i] = device::cast<T>(device::cast<fp32_t>(activated) * up_f32);
    } else {
      out[i] = device::cast<T>(apply_activation_f32<kAct>(gate_f32) * up_f32);
    }
  }
  if constexpr (kReuseInput) {
    device::store_as<vec_t>(const_cast<void*>(params.input), out, output_offset);
  } else {
    device::store_as<vec_t>(params.out, out, output_offset);
  }
  PDLTriggerSecondary<kUsePDL>();
}

struct UnaryActivationParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  uint32_t num_vecs;
};

template <typename T, ActivationKind kAct, bool kUsePDL>
__global__ void act_kernel(const __grid_constant__ UnaryActivationParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;
  const auto vec_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_id >= params.num_vecs) return;
  PDLWaitPrimary<kUsePDL>();
  const auto in = device::load_as<vec_t>(params.input, vec_id);
  vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    out[i] = device::cast<T>(apply_activation_f32<kAct>(device::cast<fp32_t>(in[i])));
  }
  device::store_as<vec_t>(params.out, out, vec_id);
  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct ActivationKernel {
  static constexpr auto kVecSize = device::kMaxVecBytes / sizeof(T);
  static constexpr auto kBlockSize = 256u;

  using kernel_fn_t = decltype(&act_and_mul_kernel<T, ActivationKind::kSiLU, kUsePDL, false>);
  using unary_kernel_fn_t = decltype(&act_kernel<T, ActivationKind::kReLU2, kUsePDL>);

  template <ActivationKind kAct, bool kFilterExpert, bool kRoundActivation = false, bool kReuseInput = false>
  static constexpr kernel_fn_t activation_kernel =
      act_and_mul_kernel<T, kAct, kUsePDL, kFilterExpert, kRoundActivation, kReuseInput>;

  static_assert(device::kMaxVecBytes % sizeof(T) == 0, "unsupported data type");

  template <bool kFilterExpert, bool kRoundActivation = false, bool kReuseInput = false>
  static kernel_fn_t select_kernel(const std::string& type) {
    using namespace host;
    if (type == "silu") {
      return activation_kernel<ActivationKind::kSiLU, kFilterExpert, kRoundActivation, kReuseInput>;
    } else if (type == "gelu") {
      return activation_kernel<ActivationKind::kGELU, kFilterExpert, kRoundActivation, kReuseInput>;
    } else if (type == "gelu_tanh") {
      return activation_kernel<ActivationKind::kGELUTanh, kFilterExpert, kRoundActivation, kReuseInput>;
    } else {
      Panic("unsupported activation type: ", type);
    }
    return nullptr;
  }

  template <bool kRoundActivation = false, bool kReuseInput = false, bool kClamp = false>
  static void launch(
      const tvm::ffi::TensorView& input,
      const tvm::ffi::TensorView& out,
      const std::string& type,
      const int32_t* expert_ids,
      uint32_t expert_step,
      float clamp_limit = 0.0f) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D_in = SymbolicSize{"input_width"};
    auto D_out = SymbolicSize{"output_width"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    if constexpr (kReuseInput) {
      TensorMatcher({N, D_out}).with_strides({D_in, 1}).with_dtype<T>().with_device(device_).verify(out);
    } else {
      TensorMatcher({N, D_out}).with_dtype<T>().with_device(device_).verify(out);
    }
    TensorMatcher({N, D_in})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(input);

    const auto hidden_size = static_cast<uint32_t>(D_out.unwrap());
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto device = device_.unwrap();
    if (num_tokens == 0) return;
    RuntimeCheck(hidden_size * 2 == D_in.unwrap(), "invalid activation dimension");
    RuntimeCheck(hidden_size % kVecSize == 0, "hidden size must be divisible by vector size");
    if constexpr (kReuseInput) {
      RuntimeCheck(input.data_ptr() == out.data_ptr(), "in-place activation output must alias input");
    }
    // only get once to avoid overhead
    const auto num_total_items = num_tokens * (hidden_size / kVecSize);
    RuntimeCheck(num_total_items <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");
    const auto num_blocks = div_ceil(static_cast<uint32_t>(num_total_items), kBlockSize);
    const auto params = ActivationParams{
        .input = input.data_ptr(),
        .out = kReuseInput ? nullptr : out.data_ptr(),
        .hidden_dim = hidden_size,
        .num_tokens = num_tokens,
        .expert_ids = expert_ids,
        .expert_step = expert_step,
        .clamp_limit = clamp_limit,
    };
    if constexpr (kClamp) {
      RuntimeCheck(type == "silu" && expert_ids == nullptr, "clamping requires unfiltered SiLU");
      const auto kernel =
          act_and_mul_kernel<T, ActivationKind::kSiLU, kUsePDL, false, kRoundActivation, kReuseInput, true>;
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
    } else if (expert_ids != nullptr) {
      RuntimeCheck(expert_step > 0, "expert_step must be positive");
      const auto kernel = select_kernel<true, kRoundActivation, kReuseInput>(type);
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
    } else {
      const auto kernel = select_kernel<false, kRoundActivation, kReuseInput>(type);
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
    }
  }

  static void run_activation(const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type) {
    launch(input, out, type, /*expert_ids=*/nullptr, /*expert_step=*/1);
  }

  static void run_activation_with_rounding(
      const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type, double clamp_limit) {
    if (clamp_limit > 0) {
      launch<true, false, true>(input, out, type, /*expert_ids=*/nullptr, /*expert_step=*/1, clamp_limit);
    } else {
      launch<true>(input, out, type, /*expert_ids=*/nullptr, /*expert_step=*/1);
    }
  }

  static void run_activation_with_rounding_input_inplace(
      const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type) {
    launch<true, true>(input, out, type, /*expert_ids=*/nullptr, /*expert_step=*/1);
  }

  static void run_activation_filtered(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView expert_ids,
      int64_t expert_step,
      std::string type) {
    using namespace host;
    RuntimeCheck(is_type<int32_t>(expert_ids.dtype()), "expert_ids must have dtype int32");
    RuntimeCheck(expert_step >= 1, "expert_step must be positive");
    launch(input, out, type, static_cast<const int32_t*>(expert_ids.data_ptr()), static_cast<uint32_t>(expert_step));
  }

  template <ActivationKind kAct>
  static constexpr auto unary_kernel = act_kernel<T, kAct, kUsePDL>;

  // Use the explicit non-const function-pointer type (mirrors select_kernel's
  // kernel_fn_t) rather than a trailing `decltype(unary_kernel<...>)` return,
  // which deduces a const-qualified pointer that clang-HIP (gfx942) refuses to
  // initialize from an lvalue / nullptr. nvcc accepts both; this form works for
  // CUDA and ROCm alike.
  static unary_kernel_fn_t select_unary_kernel(const std::string& type) {
    using namespace host;
    if (type == "relu2") {
      return ActivationKernel::template unary_kernel<ActivationKind::kReLU2>;
    } else {
      Panic("unsupported unary activation type: ", type);
    }
    return nullptr;
  }

  static void run_unary_activation(const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(out)
        .verify(input);

    const auto num_elems = static_cast<int64_t>(N.unwrap()) * D.unwrap();
    const auto device = device_.unwrap();
    if (num_elems == 0) return;
    RuntimeCheck(num_elems % kVecSize == 0, "num elements must be divisible by vector size");
    const auto num_vecs = num_elems / kVecSize;
    RuntimeCheck(num_vecs <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");
    const auto num_blocks = div_ceil(static_cast<uint32_t>(num_vecs), kBlockSize);
    const auto params = UnaryActivationParams{
        .input = input.data_ptr(),
        .out = out.data_ptr(),
        .num_vecs = static_cast<uint32_t>(num_vecs),
    };
    const auto kernel = select_unary_kernel(type);
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang

'''


# --- underlying ref: moe_wna16_marlin.cuh ---
_REF_MOE_WNA16_CUH = r'''
FILE: moe_wna16_marlin.cuh
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
# NOTE: truncated; full file at underlying/moe_wna16_marlin.cuh (39363 bytes)
/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/scalar_type.hpp>

#include "kernel.h"
#include "marlin_template.h"

namespace sglang {

namespace device::marlin_moe {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int size_m,
    int size_k,
    int top_k) {};

#else

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int size_m,
    int size_k,
    int top_k) {
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);
  int32_t block_sorted_ids[moe_block_size];
  int block_num_valid_tokens = 0;
  int64_t old_expert_id = 0;
  int64_t expert_id = 0;
  int row_stride = size_k * sizeof(half) / 16;

  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
    for (int i = 0; i < moe_block_size / 4; i++) {
      tmp_block_sorted_ids[i] = ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
    }
    for (int i = 0; i < moe_block_size; i++) {
      if (block_sorted_ids[i] >= size_m * top_k) {
        block_num_valid_tokens = i;
        break;
      };
    }
  };

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int in_offset = (row / top_k) * row_stride;
    int out_offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + in_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {
    old_expert_id = expert_id;
    int tmp_expert_id = expert_ids_ptr[index];
    if (tmp_expert_id == -1) continue;
    expert_id = tmp_expert_id;
    perm_int_ptr += (expert_id - old_expert_id) * size_k;
    read_moe_block_data(index);

    for (int i = 0; i < block_num_valid_tokens; i++)
      permute_row(block_sorted_ids[i]);
  }
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

constexpr int kSharedMemoryValidityMargin = 512;
constexpr int kSharedMemoryLaunchReserve = 1024;

int get_scales_cache_size(
    thread_config_t const& th_config,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups = tb_groups * pipe_stages * 2;  // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);             // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

int get_kernel_cache_size(
    thread_config_t const& th_config,
    bool m_block_size_8,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    int is_zp_float) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights
  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)
  int sh_block_meta_size = tb_m * 4;
  int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;
  int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full);
  int sh_g_idx_size = has_act_order && !is_k_full ? pipe_stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(
    thread_config_t const& th_config,
    bool m_block_size_8,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    int is_zp_float,
    int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config,
      m_block_size_8,
      thread_m_blocks,
      prob_m,
     

# ... [23483 chars omitted; full file in underlying/moe_wna16_marlin.cuh] ...

t.size(1),
      "Shape mismatch: b_q_weight.size(1) = ",
      b_q_weight.size(1),
      ", size_k = ",
      size_k,
      ", tile_size = ",
      device::marlin::tile_size);
  RuntimeCheck(
      b_q_weight.size(2) % device::marlin::tile_size == 0,
      "b_q_weight.size(2) = ",
      b_q_weight.size(2),
      " is not divisible by tile_size = ",
      device::marlin::tile_size);
  int64_t actual_size_n = (b_q_weight.size(2) / device::marlin::tile_size) * pack_factor;
  RuntimeCheck(size_n == actual_size_n, "size_n = ", size_n, ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);

  device.verify(b_q_weight.device());
  RuntimeCheck(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  device.verify(b_scales.device());
  RuntimeCheck(b_scales.is_contiguous(), "b_scales is not contiguous");

  // thread_k, thread_n, sms
  int thread_k = -1;
  int thread_n = -1;
  int sms = -1;
  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

  // Verify c (allocation done in Python)
  device.verify(c.device());
  RuntimeCheck(c.is_contiguous(), "c is not contiguous");
  RuntimeCheck(
      c.size(0) == size_m * top_k, "Shape mismatch: c.size(0) = ", c.size(0), ", size_m * topk = ", size_m * top_k);
  RuntimeCheck(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1), ", size_n = ", size_n);

  // Alloc c_tmp: SKIP, done in Python

  // Detect groupsize: b_scales rank and dims
  RuntimeCheck(b_scales.dim() == 3, "b_scales rank = ", b_scales.dim(), " is not 3");
  RuntimeCheck(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2), " is not size_n = ", size_n);
  RuntimeCheck(
      b_scales.size(1) == num_groups, "b_scales dim 1 = ", b_scales.size(1), " is not num_groups = ", num_groups);

  // Validate g_idx, perm (Optional unwrap done in Python; empty tensors when absent)
  if (g_idx.size(g_idx.dim() - 1) > 0 && perm.size(perm.dim() - 1) > 0) {
    device.verify(g_idx.device());
    RuntimeCheck(g_idx.is_contiguous(), "g_idx is not contiguous");
    device.verify(perm.device());
    RuntimeCheck(perm.is_contiguous(), "perm is not contiguous");

    int64_t g_idx_last = g_idx.size(g_idx.dim() - 1);
    int64_t perm_last = perm.size(perm.dim() - 1);
    RuntimeCheck(
        (g_idx_last == 0 && perm_last == 0) || (g_idx_last == size_k && perm_last == size_k),
        "Unexpected g_idx.size(-1) = ",
        g_idx_last,
        " and perm.size(-1) = ",
        perm_last,
        ", where size_k = ",
        size_k);
  }
  // has_act_order derivation: SKIP (passed as param)

  // Verify group_size consistency
  if (has_act_order) {
    // SKIP: a_tmp allocation done in Python
    if (is_k_full) {
      RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");
      RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
    }
  } else {
    if (num_groups > 1) {
      RuntimeCheck(
          size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by b_scales.size(1) = ", num_groups);
    }
  }

  // Verify global_scale (Optional unwrap done in Python)
  int64_t global_scale_size = global_scale.size(0);
  if (global_scale_size > 0) {
    RuntimeCheck(b_q_type == kFE2M1f && group_size == 16, "global_scale can only be used for nvfp4 format.");
  } else {
    RuntimeCheck(
        !(b_q_type == kFE2M1f && group_size == 16), "the global_scale parameter must be passed for nvfp4 format.");
  }

  // Verify b_bias (Optional unwrap done in Python)
  if (has_bias) {
    device.verify(b_bias.device());
    RuntimeCheck(b_bias.is_contiguous(), "b_bias is not contiguous");
    RuntimeCheck(b_bias.size(1) == size_n, "b_bias.size(0) != size_n");
    RuntimeCheck(b_bias.stride(1) == 1, "b_bias.stride(1) != 1");
  }

  // b_zeros Optional unwrap + has_zp derivation: SKIP (done in Python)

  // Verify b_q_type vs has_zp
  if (has_zp) {
    device.verify(b_zeros.device());
    RuntimeCheck(b_zeros.is_contiguous(), "b_zeros is not contiguous");
    RuntimeCheck(
        b_q_type == kU4 || b_q_type == kU8, "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());
  } else {
    RuntimeCheck(
        b_q_type == kU4B8 || b_q_type == kU8B128 || b_q_type == kFE4M3fn || b_q_type == kFE2M1f,
        "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or "
        "float4_e2m1f when "
        "has_zp = False. Got = ",
        b_q_type.str());
  }

  if (has_zp && is_zp_float) {
    RuntimeCheck(
        std::is_same<scalar_t, fp16_t>::value,
        "Computation type must be float16 (half) when using float zero "
        "points.");
  }

  if (b_q_type == kFE2M1f) {
    RuntimeCheck(
        group_size == 16 || group_size == 32,
        "float4_e2m1f only supports group_size == 16 (NVFP4) or group_size == 32 (MXFP4). Got group_size = ",
        group_size);
    RuntimeCheck(
        group_size != 32 || std::is_same<scalar_t, nv_bfloat16>::value,
        "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 activations.");
  }

  // Verify b_zeros
  if (has_zp) {
    RuntimeCheck(b_zeros.dim() == 3, "b_zeros rank = ", b_zeros.dim(), " is not 3");
    if (is_zp_float) {
      RuntimeCheck(b_zeros.size(2) == size_n, "b_zeros dim 2 = ", b_zeros.size(2), " is not size_n = ", size_n);
      RuntimeCheck(
          num_groups == b_zeros.size(1), "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);
      RuntimeCheck(num_groups != -1, "num_groups must be != -1");
    } else {
      RuntimeCheck(
          b_zeros.size(1) == num_groups, "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);
      RuntimeCheck(
          b_zeros.size(2) == size_n / pack_factor,
          "b_zeros dim 2 = ",
          b_zeros.size(2),
          " is not size_n / pack_factor = ",
          size_n / pack_factor);
    }
  }

  // Verify workspace size
  RuntimeCheck(
      size_n % device::marlin::min_thread_n == 0,
      "size_n = ",
      size_n,
      ", is not divisible by min_thread_n = ",
      device::marlin::min_thread_n);

  int64_t max_n_tiles = size_n / device::marlin::min_thread_n;
  int64_t min_workspace_size =
      std::min(max_n_tiles * (sorted_token_ids.size(0) / moe_block_size), static_cast<int64_t>(sms) * 4);
  RuntimeCheck(
      workspace.size(0) >= min_workspace_size,
      "workspace.numel = ",
      workspace.size(0),
      " is below min_workspace_size = ",
      min_workspace_size);

  // Early return for zero-size M (moved after all validation)
  if (size_m == 0) return;

  device::marlin_moe::marlin_mm<scalar_t, kIsEP, kHasBias>(
      a.data_ptr(),
      b_q_weight.data_ptr(),
      c.data_ptr(),
      c_tmp.data_ptr(),
      b_bias.data_ptr(),
      b_scales.data_ptr(),
      global_scale.data_ptr(),
      b_zeros.data_ptr(),
      g_idx.data_ptr(),
      perm.data_ptr(),
      a_tmp.data_ptr(),
      sorted_token_ids.data_ptr(),
      expert_ids.data_ptr(),
      num_tokens_post_padded.data_ptr(),
      topk_weights.data_ptr(),
      static_cast<int>(moe_block_size),
      static_cast<int>(top_k),
      mul_topk_weights,
      is_ep,
      static_cast<int>(size_m),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      workspace.data_ptr(),
      b_q_type,
      has_bias,
      has_act_order,
      is_k_full,
      has_zp,
      static_cast<int>(num_groups),
      static_cast<int>(group_size),
      dev,
      stream,
      thread_k,
      thread_n,
      sms,
      use_atomic_add,
      use_fp32_reduce,
      is_zp_float);
}

}  // namespace sglang

'''


# --- underlying ref: marlin_template.h ---
_REF_MARLIN_TEMPLATE = r'''
FILE: marlin_template.h
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
# NOTE: truncated; full file at underlying/marlin_template.h (78037 bytes)
/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#include <sgl_kernel/scalar_type.hpp>

#include "../marlin/dequant.h"
#include "../marlin/marlin.cuh"
#include "../marlin/marlin_dtypes.cuh"
#include <type_traits>

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)                                        \
  static_assert(                                                                         \
      std::is_same<scalar_t, half>::value || std::is_same<scalar_t, nv_bfloat16>::value, \
      "only float16 and bfloat16 is supported");

namespace sglang {

namespace device::marlin_moe {
using namespace device::marlin;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <
    typename scalar_t,                   // compute dtype, half or nv_float16
    const host::ScalarTypeId w_type_id,  // weight ScalarType id
    const int threads,                   // number of threads in a threadblock
    const int thread_m_blocks,           // number of 16x16 blocks in the m
                                         // dimension (batchsize) of the
                                         // threadblock
    const int thread_n_blocks,           // same for n dimension (output)
    const int thread_k_blocks,           // same for k dimension (reduction)
    const bool m_block_size_8,           // whether m_block_size == 8
                                         // only works when thread_m_blocks == 1
    const int stages,                    // number of stages for the async global->shared
                                         // fetch pipeline
    const int group_blocks,              // number of consecutive 16x16 blocks
                                         // with a separate quantization scale
    const bool is_zp_float,              // is zero point of float16 type?
    const bool kIsEP,                    // expert parallelism
    const bool kHasBias                  // has per-expert bias
    >
__global__ void Marlin(
    const int4* __restrict__ A,                              // fp16 input matrix of shape mxk
    const int4* __restrict__ B,                              // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,                                    // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,                                // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,                     // fp16 quantization scales of shape
                                                             // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,                         // 4bit packed zero-points of shape
                                                             // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,                           // int32 group indices of shape k
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,                                               // num of experts per token
    bool mul_topk_weights,                                   // mul topk weights or not
    bool is_ep,                                              // expert parallelism
    int num_groups,                                          // number of scale groups per output channel
    int prob_m,                                              // batch dimension m
    int prob_n,                                              // output dimension n
    int prob_k,                                              // reduction dimension k
    int* locks,                                              // extra global storage for barrier synchronization
    bool use_atomic_add,                                     // whether to use atomic add to reduce
    bool use_fp32_reduce,                                    // whether to use fp32 global reduce
    int max_shared_mem) {}

}  // namespace device::marlin_moe

#else

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
template <typename scalar_t>
__device__ inline void
mma(const typename ScalarType<scalar_t>::FragA& a_frag,
    const typename ScalarType<scalar_t>::FragB& frag_b,
    typename ScalarType<scalar_t>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  if constexpr (std::is_same<scalar_t, half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])

# ... [66157 chars omitted; full file in underlying/marlin_template.h] ...

h_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();

      if constexpr (kHasBias) {
        if (last) {
          __syncthreads();
          cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd], threadIdx.x < 16 * thread_n_blocks / 8);
          cp_async_fence();
        }
      }

      if constexpr (!has_act_order && group_blocks == -1 && (has_zp && dequant_skip_flop || !has_zp)) {
        if (w_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              scalar_t2* frag_s_half2 = reinterpret_cast<scalar_t2*>(frag_s);
#pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] = Dtype::num2num2(reinterpret_cast<scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (
          !has_act_order && group_blocks == -1 && w_type.size_bits() == 8 && (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][0][0]), frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]), frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][1][0]), frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][1][2]), frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      if (slice_count > 1 && !use_atomic_add) {
        // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
      }

      if constexpr (kHasBias) {
        if (last) {
          cp_async_wait<0>();
          __syncthreads();
          reinterpret_cast<int4*>(&frag_bias)[0] = sh_bias[bias_sh_rd];
          reinterpret_cast<int4*>(&frag_bias)[1] = sh_bias[bias_sh_rd + 4];
          __syncthreads();
        }
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0) wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
        // only the last block in a slice actually writes the result
        write_result(last);
      int old_slice_row = slice_row;
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      is_first_matmul_in_slice = true;
      init_slice();

      // Should we load A matrix in next slice?
      // `slice_col == 0`: when move to a new moe block
      // `old_slice_row > 0`:
      //    when the last slice is not starting from k_index == 0
      //    (only happen when it is the first slice of a threadblock)
      // `prob_k > thread_k_blocks * 16 * stages * max_num_stage_groups`:
      //    when the required shared memory size is larger than
      //    the remaining shared memory
      if (slice_col == 0 || old_slice_row || prob_k > thread_k_blocks * 16 * stages * max_num_stage_groups) {
        should_load_a = true;
      } else {
        should_load_a = false;
      }

      if (slice_iters) {
        a_gl_rd_col = (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
#pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr[i] -= b_gl_stride;
        }

        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        // Update slice k/n for scales loading
        if constexpr (has_act_order) {
          slice_k_start = tb_k * slice_row;
          slice_k_finish = slice_k_start + tb_k * slice_iters;
          slice_k_start_shared_fetch = slice_k_start;
          slice_n_offset = act_s_col_tb_stride * slice_col;
        } else {
          if constexpr (group_blocks == -1) {
            s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
            zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
          } else if constexpr (group_blocks >= thread_k_blocks) {
            s_gl_rd =
                s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;
            zp_gl_rd =
                zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + zp_sh_stride * slice_col + threadIdx.x;
          } else {
            s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks + threadIdx.x / s_sh_stride) +
                      s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
            zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks + threadIdx.x / zp_sh_stride) +
                       zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
          }
        }
        start_pipes();
      }
    }
  }
}

}  // namespace device::marlin_moe

#endif

}  // namespace sglang

'''


# --- underlying ref: gptq_marlin_repack.cuh ---
_REF_GPTQ_REPACK_CUH = r'''
FILE: gptq_marlin_repack.cuh
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/gemm/marlin/gptq_marlin_repack.cuh
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
# NOTE: truncated; full file at underlying/gptq_marlin_repack.cuh (12114 bytes)
/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include "marlin.cuh"

namespace sglang {

namespace device::marlin {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k,
    int size_n) {
  return;
}
#else
template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k,
    int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  int k_tiles = size_k / tile_k_size;
  int n_tiles = size_n / tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int perm_size = tile_k_size / 4;

  int4* sh_perm_ptr = sh;
  int4* sh_pipe_ptr = sh_perm_ptr;
  if constexpr (has_perm) {
    sh_pipe_ptr += perm_size;
  }

  constexpr int tile_ints = tile_k_size / pack_factor;

  constexpr int stage_n_threads = tile_n_size / 4;
  constexpr int stage_k_threads = has_perm ? tile_k_size : tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto load_perm_to_shared = [&](int k_tile_id) {
    int first_k_int4 = (k_tile_id * tile_k_size) / 4;

    int4 const* perm_int4_ptr = reinterpret_cast<int4 const*>(perm_ptr);

    if (threadIdx.x < perm_size) {
      sh_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
    }
    __syncthreads();
  };

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * tile_n_size;

    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if constexpr (has_perm) {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;

        uint32_t const* sh_perm_int_ptr = reinterpret_cast<uint32_t const*>(sh_perm_ptr);

        int src_k = sh_perm_int_ptr[k_id];
        int src_k_packed = src_k / pack_factor;

        cp_async4(
            &sh_ptr[k_id * stage_n_threads + n_id],
            reinterpret_cast<int4 const*>(&(b_q_weight_ptr[src_k_packed * size_n + first_n + (n_id * 4)])));
      }

    } else {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;

        int first_k = k_tile_id * tile_k_size;
        int first_k_packed = first_k / pack_factor;

        cp_async4(
            &sh_ptr[k_id * stage_n_threads + n_id],
            reinterpret_cast<int4 const*>(&(b_q_weight_ptr[(first_k_packed + k_id) * size_n + first_n + (n_id * 4)])));
      }
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;

    constexpr int sh_stride = 64;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    uint32_t* sh_perm_int_ptr = reinterpret_cast<uint32_t*>(sh_perm_ptr);

    uint32_t vals[8];

    if con

# ... [2234 chars omitted; full file in underlying/gptq_marlin_repack.cuh] ...

int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    if constexpr (has_perm) {
      load_perm_to_shared(k_tile_id);
    }

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id, n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}
#endif

}  // namespace device::marlin

#define CALL_IF_REPACK(NUM_BITS, HAS_PERM)                                                                        \
  else if (num_bits == NUM_BITS && has_perm == HAS_PERM) {                                                        \
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(                                                                \
        device::marlin::gptq_marlin_repack_kernel<device::marlin::repack_threads, NUM_BITS, HAS_PERM>,            \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                                                              \
        max_shared_mem));                                                                                         \
    host::LaunchKernel(blocks, device::marlin::repack_threads, stream, static_cast<std::size_t>(max_shared_mem))( \
        device::marlin::gptq_marlin_repack_kernel<device::marlin::repack_threads, NUM_BITS, HAS_PERM>,            \
        b_q_weight_ptr,                                                                                           \
        perm_ptr,                                                                                                 \
        out_ptr,                                                                                                  \
        size_k,                                                                                                   \
        size_n);                                                                                                  \
  }

void gptq_marlin_repack(
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView perm,
    tvm::ffi::TensorView out,
    int64_t size_k,
    int64_t size_n,
    int64_t num_bits) {
  using namespace host;

  // Validate num_bits
  RuntimeCheck(num_bits == 4 || num_bits == 8, "num_bits must be 4 or 8. Got = ", num_bits);
  int const pack_factor = 32 / static_cast<int>(num_bits);

  // Validate size alignment
  RuntimeCheck(
      size_k % device::marlin::tile_k_size == 0,
      "size_k = ",
      size_k,
      " is not divisible by tile_k_size = ",
      device::marlin::tile_k_size);
  RuntimeCheck(
      size_n % device::marlin::tile_n_size == 0,
      "size_n = ",
      size_n,
      " is not divisible by tile_n_size = ",
      device::marlin::tile_n_size);

  // Validate b_q_weight
  auto bqw_dim0 = SymbolicSize{"bqw_dim0"};
  auto bqw_dim1 = SymbolicSize{"bqw_dim1"};
  bqw_dim0.set_value(size_k / pack_factor);
  bqw_dim1.set_value(size_n);
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({bqw_dim0, bqw_dim1}).with_dtype<int32_t>().with_device(device_).verify(b_q_weight);

  // Validate out
  auto out_dim0 = SymbolicSize{"out_dim0"};
  auto out_dim1 = SymbolicSize{"out_dim1"};
  out_dim0.set_value(size_k / device::marlin::tile_size);
  out_dim1.set_value(size_n * device::marlin::tile_size / pack_factor);
  TensorMatcher({out_dim0, out_dim1}).with_dtype<int32_t>().with_device(device_).verify(out);

  // Detect if there is act_order
  bool has_perm = perm.size(0) != 0;

  // Get ptrs
  uint32_t const* b_q_weight_ptr = reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr());
  uint32_t const* perm_ptr = reinterpret_cast<uint32_t const*>(perm.data_ptr());
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr());

  // Get dev info
  DLDevice dl_device = device_.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  int blocks;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev));

  int max_shared_mem = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  RuntimeCheck(max_shared_mem > 0, "max_shared_mem must be > 0");

  if (false) {
  }
  CALL_IF_REPACK(4, false)
  CALL_IF_REPACK(4, true)
  CALL_IF_REPACK(8, false)
  CALL_IF_REPACK(8, true)
  else {
    Panic("Unsupported repack config: num_bits = ", num_bits, ", has_perm = ", has_perm);
  }
}

}  // namespace sglang

#undef CALL_IF_REPACK

'''


# --- underlying ref: dequant.h ---
_REF_DEQUANT_H = r'''
FILE: dequant.h
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/gemm/marlin/dequant.h
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
# NOTE: truncated; full file at underlying/dequant.h (19313 bytes)
/*
Fast Dequantization (Converting INT4/INT8/FP4/FP8 to FP16/BF16)

The process of fast dequantization can be summarized as a combination
of bitwise operations and floating-point computations:

weight =>(bit_op / bitwise operations)=>
f16_value =>(flop / floating-point computation)=>
dequantized_weight

Since the dequantized weights typically require subtracting the zero point and
applying a scale factor, the floating-point computation step can be fused with
the zero-point subtraction and scaling operations.

The following are the parts that need to be modified for the fused operation
of zero-point subtraction and scaling.

## INT4 => FP16/BF16 or INT8 => FP16

The floating-point computation is `__hsub2`

If has zero points:

    flop(bit_op(weight)) - flop(bit_op(zp))
  = sub(bit_op(weight), bias) - sub(bit_op(zp), bias)
  = bit_op(weight) - bit_op(zp)

so we don't need additional modification.

If has float zero points:

    flop(bit_op(weight)) - fzp
  = sub(bit_op(weight), bias) - fzp
  = bit_op(weight) - (fzp + bias)

where the `fzp + bias` can be computed at weight loading. But this
may have accuracy issue, so we should not use this in most cases.

If has not zero points:

    scale(flop(bit_op(weight)))
  = scale(sub(bit_op(weight), bias))
  = scale(bit_op(weight)) - scale(bias)
  = fma(bit_op(weight), scale_factor, scale(bias))

where the `scale(bias)` can be cached. But this may have accuracy issue,
so we should not use this in most cases.


## INT8 => BF16

INT8 => BF16 is a special case, it use byte_perm instead of flop.
We cannot fused byte_perm with scaling.


## FP4/FP8 => FP16/BF16

    scale(flop(bit_op(weight)))
  = scale(mul(bit_op(weight), multiplier))
  = mul(bit_op(weight), scale_factor * multiplier)

where `scale_factor * multiplier` can be computed at weight loading.

*/

#include "marlin_dtypes.cuh"

namespace sglang {

namespace device::marlin {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(res) : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t2, host::ScalarTypeId w_type_id, bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);

//
// Efficiently dequantize 4bit values packed in an int32 value into a full
// B-fragment of 4 fp16 values. We mostly follow the strategy in the link below,
// with some small changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
//
template <>
__device__ inline void dequant<half2, host::kU4B8.id(), true>(int q, half2* frag_b) {
  const int MASK = 0x000f000f;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, host::kU4B8.id(), false>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // clang-format on
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(
      *reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<half2, host::kU4.id(), true>(int q, half2* frag_b) {
  dequant<half2, host::kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, host::kU4.id(), false>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off

# ... [9433 chars omitted; full file in underlying/dequant.h] ...

equant<half2, host::kFE2M1f.id(), true>(q, frag_b);

  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, FP16_EXPONENT = 5;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

  // Convert to half2 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, host::kFE2M1f.id(), true>(int q, nv_bfloat162* frag_b) {
  // Constants for FP4 (E2M1) and FP16 formats
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  // Extract and shift FP4 values to FP16 format
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, host::kFE2M1f.id(), false>(int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, host::kFE2M1f.id(), true>(q, frag_b);

  // Constants for FP4 (E2M1) and BF16 formats
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;

  // Construct and apply exponent bias
  constexpr int BIAS_OFFSET = (1 << (BF16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  // Add 127 (float exponent bias) to BIAS_OFFSET and shift to float exponent
  // position
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  // Convert to half2 and apply bias
  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <typename scalar_t2>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2>(int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  ;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162>(int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to BF16 format
  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
};

// New version with s_type_id parameter for marlin_moe_wna16_v2
template <typename scalar_t2, host::ScalarTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2, host::kFE4M3fn.id()>(int q, half2* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  ;
  q <<= 8;
  int Out2 = (q & 0xFF00FF00) >> 1;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
  frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, host::kFE4M3fn.id()>(int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  // Extract and shift FP8 values to BF16 format
  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, host::kFE8M0fnu.id()>(int q, nv_bfloat162* frag_b) {
  // In this conversion, 2 ** -127 in FP8E8M0 would become 0 in BF16,
  // but we assume that such a extreme value would not occur in real models.
  int Out1 = (q & 0xFF00FF00) >> 1;
  q <<= 7;
  int Out2 = q & 0x7F807F80;

  // Note: reverse indexing is intentional because weights are permuted
  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

#endif

}  // namespace device::marlin

}  // namespace sglang

'''


# --- underlying ref: kernel.h ---
_REF_MARLIN_KERNEL_H = r'''
FILE: kernel.h
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/gemm/marlin_moe/kernel.h
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================

#include <sgl_kernel/scalar_type.hpp>

#include "../marlin/marlin.cuh"
#include "../marlin/marlin_dtypes.cuh"

#define MARLIN_KERNEL_PARAMS                                                                                         \
  const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp,            \
      const int4 *__restrict__ b_bias_ptr, const int4 *__restrict__ scales_ptr,                                      \
      const uint16_t *__restrict__ scale2_ptr, const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx,       \
      const int32_t *__restrict__ sorted_token_ids_ptr, const int32_t *__restrict__ expert_ids_ptr,                  \
      const int32_t *__restrict__ num_tokens_past_padded_ptr, const float *__restrict__ topk_weights_ptr, int top_k, \
      bool mul_topk_weights, bool is_ep, int num_groups, int prob_m, int prob_n, int prob_k, int *locks,             \
      bool has_bias, bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem

namespace sglang {

namespace device::marlin_moe {
template <
    typename scalar_t,                   // compute dtype, half or nv_float16
    const host::ScalarTypeId w_type_id,  // weight ScalarType id
    const host::ScalarTypeId s_type_id,  // weight scale ScalarType id
    const int threads,                   // number of threads in a threadblock
    const int thread_m_blocks,           // number of 16x16 blocks in the m
                                         // dimension (batchsize) of the
                                         // threadblock
    const int thread_n_blocks,           // same for n dimension (output)
    const int thread_k_blocks,           // same for k dimension (reduction)
    const bool m_block_size_8,           // whether m_block_size == 8
                                         // only works when thread_m_blocks == 1
    const int stages,                    // number of stages for the async global->shared
                                         // fetch pipeline
    const int group_blocks,              // number of consecutive 16x16 blocks
                                         // with a separate quantization scale
    const bool is_zp_float,              // is zero point of float16 type?
    const bool kIsEP,                    // expert parallelism
    const bool kHasBias                  // has per-expert bias
    >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}  // namespace device::marlin_moe

}  // namespace sglang

'''


# --- underlying ref: marlin.cuh ---
_REF_MARLIN_CUH = r'''
FILE: marlin.cuh
URL:  https://raw.githubusercontent.com/sgl-project/sglang/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/gemm/marlin/marlin.cuh
SGLang commit: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
========================================================================
#pragma once

#include <sgl_kernel/utils.cuh>

#include <iostream>

namespace sglang {

namespace device::marlin {
// Marlin params

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages = 4;  // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// Helpers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// No support for async
#else

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem),
      "l"(glob_ptr),
      "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr),
      "n"(BYTES));
}

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

#endif

}  // namespace device::marlin

}  // namespace sglang

'''
