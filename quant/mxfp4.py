"""OCP-style MXFP4 reference utilities.

Storage format:
- one FP4 E2M1 code per element (packed two nibbles into one uint8)
- one E8M0 shared scale per block
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import torch


_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_FP4_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_POS_THRESHOLDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
_INT4_QMIN = -8
_INT4_QMAX = 7


def pack_u4(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit unsigned codes [0, 15] into uint8 tensor."""
    if codes.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"codes must be integer tensor, got {codes.dtype}")
    flat = codes.to(torch.int16).reshape(-1)
    if flat.numel() > 0:
        if flat.min().item() < 0 or flat.max().item() > 15:
            raise ValueError("4-bit codes out of range [0, 15]")
    u4 = flat.to(torch.uint8)
    if u4.numel() % 2 == 1:
        u4 = torch.cat([u4, u4.new_zeros(1)], dim=0)
    lo = u4[0::2]
    hi = u4[1::2] << 4
    return (lo | hi).contiguous()


def unpack_u4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack uint8-packed nibbles into uint8 codes [0, 15]."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    p = packed.reshape(-1)
    out = torch.empty(p.numel() * 2, dtype=torch.uint8, device=p.device)
    out[0::2] = p & 0x0F
    out[1::2] = (p >> 4) & 0x0F
    return out[:numel]


def fp4_e2m1_decode(codes: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode FP4 E2M1 codes into floating-point values."""
    idx = codes.to(torch.long)
    if idx.numel() > 0:
        if idx.min().item() < 0 or idx.max().item() > 15:
            raise ValueError("FP4 codes out of range [0, 15]")
    lut = _FP4_E2M1_LUT.to(device=codes.device)
    return lut[idx].to(dtype=out_dtype)


def fp4_e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Encode normalized float32 values to nearest FP4 E2M1 code."""
    x32 = torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=6.0, neginf=-6.0)
    x32 = x32.clamp(min=-6.0, max=6.0)
    ax = x32.abs()
    thresholds = _FP4_POS_THRESHOLDS.to(device=x.device)
    pos_idx = torch.bucketize(ax, thresholds, right=False).to(torch.uint8)
    zero_mask = ax < 0.25
    pos_idx = torch.where(zero_mask, torch.zeros_like(pos_idx), pos_idx)
    sign_mask = (x32 < 0) & (~zero_mask)
    return torch.where(sign_mask, pos_idx | 0x08, pos_idx)


def e8m0_decode(scale_code: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode E8M0 scale code: code 0..254 => 2 ** (code - 127), 255 => NaN."""
    code = scale_code.to(torch.int32)
    out = torch.pow(
        torch.tensor(2.0, device=scale_code.device, dtype=torch.float32),
        (code - 127).to(torch.float32),
    )
    out = torch.where(code == 255, torch.full_like(out, float("nan")), out)
    return out.to(dtype=out_dtype)


def e8m0_encode_from_exponent(exp: torch.Tensor) -> torch.Tensor:
    """Encode integer exponent to E8M0 code, clamped to [0, 254]."""
    code = exp.to(torch.int32) + 127
    return code.clamp_(0, 254).to(torch.uint8)


def _group_pad_in_features(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    out_features, in_features = weight.shape
    pad = (group_size - (in_features % group_size)) % group_size
    if pad == 0:
        return weight, in_features
    padded = torch.zeros((out_features, in_features + pad), dtype=weight.dtype, device=weight.device)
    padded[:, :in_features] = weight
    return padded, in_features + pad


def _quantize_linear_weight_sym_int4(
    weight: torch.Tensor,
    group_size: int = 64,
    clip_ratio: float = 1.0,
    scheme: str = "sym_int4",
) -> Dict[str, Any]:
    w = weight.detach().to(torch.float32).contiguous()
    w_pad, padded_in = _group_pad_in_features(w, group_size)
    out_features, _ = w_pad.shape
    num_groups = padded_in // group_size
    w_groups = w_pad.view(out_features, num_groups, group_size)
    absmax = w_groups.abs().amax(dim=-1) * float(clip_ratio)
    scale = (absmax / float(_INT4_QMAX)).clamp(min=1e-8)
    q = torch.round(w_groups / scale.unsqueeze(-1)).clamp(_INT4_QMIN, _INT4_QMAX).to(torch.int16)
    q_codes = (q & 0x0F).to(torch.uint8).view(out_features, padded_in)
    return {
        "qweight_packed": pack_u4(q_codes).cpu(),
        "scales": scale.to(torch.float32).cpu(),
        "meta": {
            "scheme": scheme,
            "group_size": int(group_size),
            "scale_dtype": "float32",
            "element_dtype": "int4_sym",
            "clip_ratio": float(clip_ratio),
            "out_features": int(weight.shape[0]),
            "in_features": int(weight.shape[1]),
            "padded_in_features": int(padded_in),
            "num_groups": int(num_groups),
        },
    }


def _dequantize_linear_weight_sym_int4(
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    group_size: int,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = qweight_packed.device
    scales = scales.to(device=device, dtype=torch.float32)
    out_features, num_groups = scales.shape
    padded_in = num_groups * group_size
    q_codes = unpack_u4(qweight_packed.to(device=device, dtype=torch.uint8), out_features * padded_in)
    q = q_codes.to(torch.int16)
    q = torch.where(q >= 8, q - 16, q).to(torch.float32)
    q = q.view(out_features, num_groups, group_size)
    w = q * scales.unsqueeze(-1)
    w = w.reshape(out_features, padded_in)[:, :in_features]
    return w.to(dtype=out_dtype)


def quantize_linear_weight_mxfp4(
    weight: torch.Tensor,
    group_size: int = 32,
    clip_ratio: float = 1.0,
    scale_method: str = "ocp_floor",
) -> Dict[str, Any]:
    """Quantize [out_features, in_features] weight into MXFP4 packed FP4 + E8M0 scales."""
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
    if group_size != 32:
        raise ValueError(f"MXFP4 reference expects group_size=32, got {group_size}")
    if clip_ratio <= 0:
        raise ValueError(f"clip_ratio must be > 0, got {clip_ratio}")
    if scale_method not in ("ocp_floor", "safe_ceil"):
        raise ValueError(f"Unsupported scale_method: {scale_method}")

    w = weight.detach().to(torch.float32).contiguous()
    w_pad, padded_in = _group_pad_in_features(w, group_size)
    out_features, _ = w_pad.shape
    num_groups = padded_in // group_size
    w_groups = w_pad.view(out_features, num_groups, group_size)

    amax = w_groups.abs().amax(dim=-1) * float(clip_ratio)
    zero_mask = amax == 0
    safe_amax = torch.where(zero_mask, torch.ones_like(amax), amax)

    if scale_method == "ocp_floor":
        floor_pow2_amax_exp = torch.floor(torch.log2(safe_amax))
        block_scale_exp = floor_pow2_amax_exp - 2.0
    else:
        block_scale_exp = torch.ceil(torch.log2(safe_amax / 6.0))

    block_scale_exp = torch.where(zero_mask, torch.zeros_like(block_scale_exp), block_scale_exp)
    scale_code = e8m0_encode_from_exponent(block_scale_exp.to(torch.int32))
    scale_code = torch.where(zero_mask, torch.full_like(scale_code, 127), scale_code)
    block_scale = e8m0_decode(scale_code, out_dtype=torch.float32)

    y = w_groups / block_scale.unsqueeze(-1)
    fp4_codes = fp4_e2m1_encode(y).view(out_features, padded_in)
    fp4_codes = torch.where(
        zero_mask.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_features, padded_in),
        torch.zeros_like(fp4_codes),
        fp4_codes,
    )

    return {
        "qweight_packed": pack_u4(fp4_codes).cpu(),
        "scales": scale_code.to(torch.uint8).cpu(),
        "meta": {
            "scheme": "mxfp4",
            "group_size": int(group_size),
            "scale_dtype": "e8m0",
            "element_dtype": "fp4_e2m1",
            "clip_ratio": float(clip_ratio),
            "scale_method": scale_method,
            "out_features": int(weight.shape[0]),
            "in_features": int(weight.shape[1]),
            "padded_in_features": int(padded_in),
            "num_groups": int(num_groups),
        },
    }


def _compute_mxfp4_group_scales(
    w_groups: torch.Tensor,
    clip_ratio: float,
    scale_method: str = "ocp_floor",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return E8M0 scale codes and decoded scales for [out, groups, 32] weights."""
    if scale_method not in ("ocp_floor", "safe_ceil"):
        raise ValueError(f"Unsupported scale_method: {scale_method}")
    amax = w_groups.abs().amax(dim=-1) * float(clip_ratio)
    zero_mask = amax == 0
    safe_amax = torch.where(zero_mask, torch.ones_like(amax), amax)

    if scale_method == "ocp_floor":
        floor_pow2_amax_exp = torch.floor(torch.log2(safe_amax))
        block_scale_exp = floor_pow2_amax_exp - 2.0
    else:
        block_scale_exp = torch.ceil(torch.log2(safe_amax / 6.0))

    block_scale_exp = torch.where(zero_mask, torch.zeros_like(block_scale_exp), block_scale_exp)
    scale_code = e8m0_encode_from_exponent(block_scale_exp.to(torch.int32))
    scale_code = torch.where(zero_mask, torch.full_like(scale_code, 127), scale_code)
    block_scale = e8m0_decode(scale_code, out_dtype=torch.float32)
    return scale_code, block_scale


def _encode_mxfp4_column_with_scale(w_col: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode/decode one [out] column with per-output-row MXFP4 group scale."""
    codes = fp4_e2m1_encode(w_col.to(torch.float32) / scale.to(torch.float32))
    q_col = fp4_e2m1_decode(codes, out_dtype=torch.float32) * scale.to(torch.float32)
    return codes, q_col


def quantize_linear_weight_groupwise_gptq(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    group_size: int = 32,
    clip_ratio: float = 1.0,
    damp_percent: float = 0.01,
    block_size: int = 128,
    scheme: str = "mxfp4",
    dynamic_group_scale: bool = False,
) -> Dict[str, Any]:
    """GPTQ-MXFP4 quantization for SVDQuant main residual Linear weights.

    The returned qweight/scales/meta layout is intentionally identical to
    `quantize_linear_weight_groupwise(..., scheme="mxfp4")`, so existing
    `QuantLinear` checkpoints and dequantization remain compatible. GPTQ uses
    the provided main-branch Hessian to compensate column errors while directly
    emitting final FP4 E2M1 packed codes; it does not quantize a float GPTQ
    result a second time.
    """
    if scheme != "mxfp4":
        raise ValueError(f"GPTQ-MXFP4 currently only supports scheme='mxfp4', got {scheme!r}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
    if hessian.dim() != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"hessian must be square 2D, got shape {tuple(hessian.shape)}")
    if group_size != 32:
        raise ValueError(f"MXFP4 GPTQ expects group_size=32, got {group_size}")
    if clip_ratio <= 0:
        raise ValueError(f"clip_ratio must be > 0, got {clip_ratio}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if block_size % group_size != 0:
        warnings.warn(
            f"GPTQ block_size={block_size} is not a multiple of MXFP4 group_size={group_size}; "
            "column processing remains correct, but group boundaries may cross GPTQ blocks.",
            RuntimeWarning,
            stacklevel=2,
        )

    out_features, in_features = weight.shape
    if hessian.shape != (in_features, in_features):
        raise ValueError(
            f"hessian shape mismatch: expected ({in_features}, {in_features}), got {tuple(hessian.shape)}"
        )

    device = weight.device
    w = weight.detach().to(device=device, dtype=torch.float32).clone().contiguous()
    h = hessian.detach().to(device=device, dtype=torch.float32).clone().contiguous()
    h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    h = 0.5 * (h + h.T)

    diag = torch.diag(h)
    dead = diag == 0
    if bool(dead.any().item()):
        h[dead, dead] = 1.0
        w[:, dead] = 0.0
        diag = torch.diag(h)

    damp = float(damp_percent) * float(diag.mean().item())
    if not torch.isfinite(torch.tensor(damp, device=device)) or damp <= 0.0:
        damp = 1e-6
    h.diagonal().add_(damp)

    eye = torch.eye(in_features, device=device, dtype=torch.float32)
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            l = torch.linalg.cholesky(h)
            hinv = torch.cholesky_inverse(l)
            hinv = torch.linalg.cholesky(hinv, upper=True)
            break
        except RuntimeError as exc:
            last_error = exc
            h = h + eye * (damp * (10.0 ** attempt))
    else:
        raise RuntimeError("Failed to compute GPTQ Hessian inverse Cholesky") from last_error

    w_pad, padded_in = _group_pad_in_features(w, group_size)
    num_groups = padded_in // group_size
    w_groups = w_pad.view(out_features, num_groups, group_size)
    scale_code, block_scale = _compute_mxfp4_group_scales(
        w_groups,
        clip_ratio=clip_ratio,
        scale_method="ocp_floor",
    )
    q_codes = torch.zeros((out_features, padded_in), dtype=torch.uint8, device=device)

    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        block_cols = block_end - block_start
        w_block = w[:, block_start:block_end].clone()
        hinv_block = hinv[block_start:block_end, block_start:block_end]
        err_block = torch.zeros((out_features, block_cols), dtype=torch.float32, device=device)

        for i in range(block_cols):
            col = block_start + i
            group_id = col // group_size
            if dynamic_group_scale and col % group_size == 0:
                group_start = group_id * group_size
                group_end = min(group_start + group_size, in_features)
                group_cur = torch.zeros((out_features, group_size), dtype=torch.float32, device=device)
                group_cur[:, : group_end - group_start] = w[:, group_start:group_end]
                cur_code, cur_scale = _compute_mxfp4_group_scales(
                    group_cur.view(out_features, 1, group_size),
                    clip_ratio=clip_ratio,
                    scale_method="ocp_floor",
                )
                scale_code[:, group_id] = cur_code[:, 0]
                block_scale[:, group_id] = cur_scale[:, 0]

            w_col = w_block[:, i]
            codes, q_col = _encode_mxfp4_column_with_scale(w_col, block_scale[:, group_id])
            q_codes[:, col] = codes

            denom = hinv_block[i, i].clamp_min(1e-12)
            err = (w_col - q_col) / denom
            w_block[:, i:] -= err[:, None] * hinv_block[i, i:][None, :]
            err_block[:, i] = err

        if block_end < in_features:
            w[:, block_end:] -= err_block @ hinv[block_start:block_end, block_end:]

    return {
        "qweight_packed": pack_u4(q_codes).cpu(),
        "scales": scale_code.to(torch.uint8).cpu(),
        "meta": {
            "scheme": "mxfp4",
            "group_size": int(group_size),
            "scale_dtype": "e8m0",
            "element_dtype": "fp4_e2m1",
            "clip_ratio": float(clip_ratio),
            "scale_method": "ocp_floor",
            "quant_method": "gptq_mxfp4",
            "damp_percent": float(damp_percent),
            "block_size": int(block_size),
            "dynamic_group_scale": bool(dynamic_group_scale),
            "out_features": int(out_features),
            "in_features": int(in_features),
            "padded_in_features": int(padded_in),
            "num_groups": int(num_groups),
        },
    }


def dequantize_linear_weight_mxfp4(
    qweight_packed: torch.Tensor,
    scale_e8m0: torch.Tensor,
    in_features: int,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Dequantize MXFP4 packed weight back to [out_features, in_features]."""
    if group_size != 32:
        raise ValueError(f"MXFP4 reference expects group_size=32, got {group_size}")
    if device is None:
        device = qweight_packed.device

    scale_e8m0 = scale_e8m0.to(device=device, dtype=torch.uint8)
    out_features, num_groups = scale_e8m0.shape
    padded_in = num_groups * group_size

    codes = unpack_u4(qweight_packed.to(device=device, dtype=torch.uint8), out_features * padded_in)
    fp4 = fp4_e2m1_decode(codes, out_dtype=torch.float32).view(out_features, num_groups, group_size)
    scales = e8m0_decode(scale_e8m0, out_dtype=torch.float32)
    w = fp4 * scales.unsqueeze(-1)
    w = w.reshape(out_features, padded_in)[:, :in_features]
    return w.to(dtype=out_dtype)


def quantize_linear_weight_groupwise(
    weight: torch.Tensor,
    group_size: int = 32,
    clip_ratio: float = 1.0,
    scheme: str = "mxfp4",
) -> Dict[str, Any]:
    if scheme == "mxfp4":
        return quantize_linear_weight_mxfp4(
            weight,
            group_size=group_size,
            clip_ratio=clip_ratio,
            scale_method="ocp_floor",
        )
    if scheme == "sym_int4":
        return _quantize_linear_weight_sym_int4(weight, group_size=group_size, clip_ratio=clip_ratio, scheme=scheme)
    raise ValueError(f"Unsupported quantization scheme: {scheme}")


def dequantize_linear_weight_groupwise(
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    group_size: int,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    scheme: str = "mxfp4",
) -> torch.Tensor:
    if scheme == "mxfp4":
        return dequantize_linear_weight_mxfp4(
            qweight_packed=qweight_packed,
            scale_e8m0=scales,
            in_features=in_features,
            group_size=group_size,
            out_dtype=out_dtype,
            device=device,
        )
    if scheme == "sym_int4":
        return _dequantize_linear_weight_sym_int4(
            qweight_packed=qweight_packed,
            scales=scales,
            in_features=in_features,
            group_size=group_size,
            out_dtype=out_dtype,
            device=device,
        )
    raise ValueError(f"Unsupported quantization scheme: {scheme}")


def export_quant_weight_state(
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "qweight_packed": qweight_packed.detach().cpu(),
        "scales": scales.detach().cpu(),
        "meta": dict(meta),
    }


def load_quant_weight_state(state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if "qweight_packed" not in state or "scales" not in state or "meta" not in state:
        raise KeyError("quant state must contain qweight_packed/scales/meta")
    return state["qweight_packed"], state["scales"], dict(state["meta"])
