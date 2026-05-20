"""QuantLinear for OCP-style MXFP4 reference quantization."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mxfp4 import (
    dequantize_linear_weight_groupwise,
    e8m0_decode,
    fp4_e2m1_decode,
    fp4_e2m1_encode,
    pack_u4,
    quantize_linear_weight_groupwise,
    quantize_linear_weight_groupwise_gptq,
    unpack_u4,
)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _DTYPE_MAP[key]


def _qdq_int4_activation_groupwise(
    x: torch.Tensor,
    group_size: int,
    clip_ratio: float,
    channels_dim: int = -1,
) -> torch.Tensor:
    """Reference groupwise int4 activation quantize-dequantize."""
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if clip_ratio <= 0:
        raise ValueError(f"clip_ratio must be > 0, got {clip_ratio}")
    if x.ndim == 0:
        return x

    cdim = channels_dim if channels_dim >= 0 else x.ndim + channels_dim
    if cdim < 0 or cdim >= x.ndim:
        raise ValueError(f"Invalid channels_dim={channels_dim} for input shape={tuple(x.shape)}")

    x32 = x.to(torch.float32)
    moved = cdim != x.ndim - 1
    if moved:
        x32 = x32.movedim(cdim, -1)

    orig_shape = x32.shape
    c = int(orig_shape[-1])
    x2 = x32.reshape(-1, c)
    pad = (group_size - (c % group_size)) % group_size
    if pad > 0:
        x2 = F.pad(x2, (0, pad))
    c_pad = x2.shape[-1]
    num_groups = c_pad // group_size
    xg = x2.view(-1, num_groups, group_size)

    absmax = xg.abs().amax(dim=-1) * float(clip_ratio)
    scale = (absmax / 7.0).clamp_min(1e-8)
    q = torch.round(xg / scale.unsqueeze(-1)).clamp(-8, 7)
    xg_dq = q * scale.unsqueeze(-1)
    x_dq = xg_dq.view(-1, c_pad)[..., :c].view(orig_shape)
    if moved:
        x_dq = x_dq.movedim(-1, cdim)
    return x_dq.to(dtype=x.dtype)


def _qdq_mxfp4_activation_groupwise(
    x: torch.Tensor,
    group_size: int = 32,
    clip_ratio: float = 1.0,
    channels_dim: int = -1,
    exp_offset: int = 0,
    scale_method: str = "safe_ceil",
    fp4_max: float = 6.0,
) -> torch.Tensor:
    """Reference MXFP4 activation quantize-dequantize."""
    if group_size != 32:
        raise ValueError(f"MXFP4 activation reference expects group_size=32, got {group_size}")
    if clip_ratio <= 0:
        raise ValueError(f"clip_ratio must be > 0, got {clip_ratio}")
    if fp4_max <= 0:
        raise ValueError(f"fp4_max must be > 0, got {fp4_max}")
    if x.ndim == 0:
        return x

    cdim = channels_dim if channels_dim >= 0 else x.ndim + channels_dim
    if cdim < 0 or cdim >= x.ndim:
        raise ValueError(f"Invalid channels_dim={channels_dim} for input shape={tuple(x.shape)}")

    x32 = x.to(torch.float32)
    moved = cdim != x.ndim - 1
    if moved:
        x32 = x32.movedim(cdim, -1)

    orig_shape = x32.shape
    c = int(orig_shape[-1])
    x2 = x32.reshape(-1, c)
    pad = (group_size - (c % group_size)) % group_size
    if pad > 0:
        x2 = F.pad(x2, (0, pad))
    c_pad = x2.shape[-1]
    num_groups = c_pad // group_size
    xg = x2.view(-1, num_groups, group_size)

    amax = xg.abs().amax(dim=-1) * float(clip_ratio)
    zero_mask = amax == 0
    safe_amax = torch.where(zero_mask, torch.ones_like(amax), amax)
    if scale_method == "ocp_floor":
        block_scale_exp = torch.floor(torch.log2(safe_amax)) - 2.0
    elif scale_method == "safe_ceil":
        block_scale_exp = torch.ceil(torch.log2(safe_amax / float(fp4_max)))
    else:
        raise ValueError(f"Unsupported MXFP4 activation scale_method: {scale_method}")
    block_scale_exp = block_scale_exp + int(exp_offset)
    block_scale_exp = torch.where(zero_mask, torch.zeros_like(block_scale_exp), block_scale_exp)
    scale_code = (block_scale_exp.to(torch.int32) + 127).clamp_(0, 254).to(torch.uint8)
    scale_code = torch.where(zero_mask, torch.full_like(scale_code, 127), scale_code)
    block_scale = e8m0_decode(scale_code, out_dtype=torch.float32)

    x_scaled = xg / block_scale.unsqueeze(-1)
    x_scaled = x_scaled.clamp(-float(fp4_max), float(fp4_max))
    codes = fp4_e2m1_encode(x_scaled)
    packed = pack_u4(codes.reshape(-1))
    codes_rt = unpack_u4(packed.to(device=x.device), codes.numel()).view_as(codes)
    xg_dq = fp4_e2m1_decode(codes_rt, out_dtype=torch.float32) * block_scale.unsqueeze(-1)
    x_dq = xg_dq.view(-1, c_pad)[..., :c].view(orig_shape)
    if moved:
        x_dq = x_dq.movedim(-1, cdim)
    return x_dq.to(dtype=x.dtype)


class QuantLinear(nn.Module):
    """Linear layer with offline MXFP4: packed FP4 E2M1 + E8M0 block scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_group_size: int = 32,
        act_group_size: int = 32,
        weight_clip_ratio: float = 1.0,
        act_clip_ratio: float = 1.0,
        quantize_input: bool = False,
        skip_input_quant: bool = False,
        compute_dtype: str | torch.dtype = "bfloat16",
        scheme: str = "mxfp4",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_group_size = int(weight_group_size)
        self.act_group_size = int(act_group_size)
        self.weight_clip_ratio = float(weight_clip_ratio)
        self.act_clip_ratio = float(act_clip_ratio)
        self.quantize_input = bool(quantize_input)
        self.skip_input_quant = bool(skip_input_quant)
        self.compute_dtype = resolve_dtype(compute_dtype)
        self.scheme = scheme
        self.act_exp_offset = 0
        self.act_scale_method = "safe_ceil"
        self.act_policy: Optional[dict] = None
        self.act_policy_context: Dict[str, Optional[str]] = {
            "expert_id": None,
            "timestep_bin": None,
        }
        self.module_name = ""

        self.register_buffer("qweight_packed", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("w_scales", torch.empty(0, 0, dtype=torch.uint8), persistent=True)
        self.register_buffer("w_in_features", torch.tensor(0, dtype=torch.int32), persistent=True)
        self.register_buffer("w_group_size", torch.tensor(self.weight_group_size, dtype=torch.int32), persistent=True)
        self.register_buffer("smooth_scale", torch.empty(0, dtype=torch.float32), persistent=True)
        self.register_buffer("input_channels_dim", torch.tensor(-1, dtype=torch.int32), persistent=True)
        self.register_buffer("smooth_enabled", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.quant_method = "rtn_mxfp4"

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_group_size: int = 32,
        act_group_size: int = 32,
        weight_clip_ratio: float = 1.0,
        act_clip_ratio: float = 1.0,
        quantize_input: bool = False,
        skip_input_quant: bool = False,
        compute_dtype: str | torch.dtype = "bfloat16",
        scheme: str = "mxfp4",
    ) -> "QuantLinear":
        out = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            weight_group_size=weight_group_size,
            act_group_size=act_group_size,
            weight_clip_ratio=weight_clip_ratio,
            act_clip_ratio=act_clip_ratio,
            quantize_input=quantize_input,
            skip_input_quant=skip_input_quant,
            compute_dtype=compute_dtype,
            scheme=scheme,
        )

        q = quantize_linear_weight_groupwise(
            linear.weight.detach(),
            group_size=weight_group_size,
            clip_ratio=weight_clip_ratio,
            scheme=scheme,
        )
        out.qweight_packed = q["qweight_packed"].contiguous()
        out.w_scales = q["scales"].contiguous()
        out.w_in_features = torch.tensor(q["meta"]["in_features"], dtype=torch.int32)
        out.w_group_size = torch.tensor(q["meta"]["group_size"], dtype=torch.int32)
        out.quant_method = str(q["meta"].get("quant_method", "rtn_mxfp4"))

        if linear.bias is not None:
            out.bias.data.copy_(linear.bias.detach().to(torch.bfloat16))

        return out

    def set_smooth_state(
        self,
        smooth_scale: Optional[torch.Tensor],
        input_channels_dim: int = -1,
        smooth_enabled: bool = True,
    ) -> None:
        if smooth_scale is None:
            self.smooth_scale = torch.empty(0, dtype=torch.float32)
            self.input_channels_dim = torch.tensor(int(input_channels_dim), dtype=torch.int32)
            self.smooth_enabled = torch.tensor(False, dtype=torch.bool)
            return
        if smooth_scale.ndim != 1:
            raise ValueError(f"smooth_scale must be 1D, got {tuple(smooth_scale.shape)}")
        if int(smooth_scale.numel()) != int(self.in_features):
            raise ValueError(
                f"smooth_scale size mismatch: expected {self.in_features}, got {int(smooth_scale.numel())}"
            )
        self.smooth_scale = smooth_scale.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.input_channels_dim = torch.tensor(int(input_channels_dim), dtype=torch.int32)
        self.smooth_enabled = torch.tensor(bool(smooth_enabled), dtype=torch.bool)

    def set_module_name(self, name: str) -> None:
        self.module_name = str(name)

    def set_act_policy(self, policy: Optional[dict]) -> None:
        self.act_policy = policy

    def set_act_policy_context(self, expert_id: Optional[str], timestep_bin: Optional[str]) -> None:
        self.act_policy_context = {
            "expert_id": None if expert_id is None else str(expert_id),
            "timestep_bin": None if timestep_bin is None else str(timestep_bin),
        }

    def get_current_act_quant_params(self) -> tuple[float, int, str]:
        clip_ratio = float(self.act_clip_ratio)
        exp_offset = int(self.act_exp_offset)
        scale_method = str(self.act_scale_method)
        policy = self.act_policy
        if policy is None or not self.module_name:
            return clip_ratio, exp_offset, scale_method

        expert_id = self.act_policy_context.get("expert_id")
        timestep_bin = self.act_policy_context.get("timestep_bin")

        def merge_entry(entry: Any) -> bool:
            nonlocal clip_ratio, exp_offset, scale_method
            if not isinstance(entry, dict):
                return False
            if "clip_ratio" in entry:
                clip_ratio = float(entry["clip_ratio"])
            if "exp_offset" in entry:
                exp_offset = int(entry["exp_offset"])
            if "scale_method" in entry:
                scale_method = str(entry["scale_method"])
            return True

        def lookup_in_module(module_policy: Any) -> bool:
            if not isinstance(module_policy, dict):
                return False
            if expert_id is not None:
                expert_policy = module_policy.get(expert_id)
                if isinstance(expert_policy, dict):
                    if timestep_bin is not None and merge_entry(expert_policy.get(timestep_bin)):
                        return True
                    if merge_entry(expert_policy.get("default")):
                        return True
            return merge_entry(module_policy.get("default"))

        if lookup_in_module(policy.get(self.module_name)):
            return clip_ratio, exp_offset, scale_method

        default_policy = policy.get("__default__")
        if isinstance(default_policy, dict) and expert_id is not None:
            expert_policy = default_policy.get(expert_id)
            if isinstance(expert_policy, dict):
                if timestep_bin is not None and merge_entry(expert_policy.get(timestep_bin)):
                    return clip_ratio, exp_offset, scale_method
                if merge_entry(expert_policy.get("default")):
                    return clip_ratio, exp_offset, scale_method
        merge_entry(default_policy)
        return clip_ratio, exp_offset, scale_method

    def _dequant_weight(self, device: torch.device) -> torch.Tensor:
        return dequantize_linear_weight_groupwise(
            qweight_packed=self.qweight_packed.to(device=device),
            scales=self.w_scales.to(device=device, dtype=torch.uint8),
            in_features=int(self.w_in_features.item()),
            group_size=int(self.w_group_size.item()),
            out_dtype=self.compute_dtype,
            device=device,
            scheme=self.scheme,
        )

    def _apply_input_smooth(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.smooth_enabled.item()):
            return x
        if self.smooth_scale.numel() == 0:
            return x

        cdim = int(self.input_channels_dim.item())
        cdim = cdim if cdim >= 0 else x.ndim + cdim
        if cdim < 0 or cdim >= x.ndim:
            raise ValueError(
                f"Invalid input_channels_dim={int(self.input_channels_dim.item())} for input shape={tuple(x.shape)}"
            )
        if int(x.shape[cdim]) != int(self.smooth_scale.numel()):
            raise ValueError(
                "smooth_scale/input shape mismatch: "
                f"shape={tuple(x.shape)} channels_dim={cdim} smooth_scale={int(self.smooth_scale.numel())}"
            )

        scale = self.smooth_scale.to(device=x.device, dtype=x.dtype)
        view_shape = [1] * x.ndim
        view_shape[cdim] = int(scale.numel())
        return x / scale.view(*view_shape)

    def _apply_input_quant_qdq(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.quantize_input) or self.skip_input_quant:
            return x
        if self.scheme == "mxfp4":
            clip_ratio, exp_offset, scale_method = self.get_current_act_quant_params()
            return _qdq_mxfp4_activation_groupwise(
                x,
                group_size=self.act_group_size,
                clip_ratio=clip_ratio,
                channels_dim=int(self.input_channels_dim.item()),
                exp_offset=exp_offset,
                scale_method=scale_method,
            )
        if self.scheme == "sym_int4":
            return _qdq_int4_activation_groupwise(
                x,
                group_size=self.act_group_size,
                clip_ratio=self.act_clip_ratio,
                channels_dim=int(self.input_channels_dim.item()),
            )
        raise ValueError(f"Unsupported activation quantization scheme: {self.scheme}")

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        x_smooth = self._apply_input_smooth(x)
        return self._apply_input_quant_qdq(x_smooth)

    def smooth_input_only(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_input_smooth(x)

    def forward_preprocessed(self, x_in: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        w = self._dequant_weight(device=x_in.device)
        b = self.bias
        if b is not None:
            b = b.to(device=x_in.device, dtype=self.compute_dtype)
        out = F.linear(x_in.to(self.compute_dtype), w, b)
        target_dtype = x_in.dtype if out_dtype is None else out_dtype
        return out.to(target_dtype)

    def export_quant_state(self) -> Dict[str, Any]:
        return {
            "qweight_packed": self.qweight_packed.detach().cpu(),
            "w_scales": self.w_scales.detach().cpu(),
            "w_in_features": int(self.w_in_features.item()),
            "w_group_size": int(self.w_group_size.item()),
            "in_features": self.in_features,
            "out_features": self.out_features,
            "scheme": self.scheme,
            "weight_clip_ratio": self.weight_clip_ratio,
            "act_group_size": self.act_group_size,
            "act_clip_ratio": self.act_clip_ratio,
            "act_exp_offset": self.act_exp_offset,
            "act_scale_method": self.act_scale_method,
            "quantize_input": self.quantize_input,
            "skip_input_quant": self.skip_input_quant,
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "bias": None if self.bias is None else self.bias.detach().cpu(),
            "smooth_scale": self.smooth_scale.detach().cpu(),
            "input_channels_dim": int(self.input_channels_dim.item()),
            "smooth_enabled": bool(self.smooth_enabled.item()),
            "quant_method": self.quant_method,
        }

    def load_quant_state(self, state: Dict[str, Any]) -> None:
        self.qweight_packed = state["qweight_packed"].detach().clone().to(torch.uint8)
        self.w_scales = state["w_scales"].detach().clone().to(torch.uint8)
        self.w_in_features = torch.tensor(int(state["w_in_features"]), dtype=torch.int32)
        self.w_group_size = torch.tensor(int(state["w_group_size"]), dtype=torch.int32)
        self.quantize_input = bool(state.get("quantize_input", self.quantize_input))
        self.skip_input_quant = bool(state.get("skip_input_quant", self.skip_input_quant))
        self.quant_method = str(state.get("quant_method", self.quant_method))
        self.act_exp_offset = int(state.get("act_exp_offset", 0))
        # Old PTQ states were produced with the original OCP floor exponent rule.
        # New modules default to safe_ceil, but missing checkpoint metadata must
        # stay on ocp_floor for backward-compatible activation QDQ behavior.
        self.act_scale_method = str(state.get("act_scale_method", "ocp_floor"))

        if self.bias is not None and state.get("bias") is not None:
            self.bias.data.copy_(state["bias"].to(torch.bfloat16))
        smooth_scale = state.get("smooth_scale", None)
        if smooth_scale is not None:
            smooth_scale = smooth_scale.detach().clone().to(torch.float32)
            if smooth_scale.numel() == 0:
                smooth_scale = None
        self.set_smooth_state(
            smooth_scale=smooth_scale,
            input_channels_dim=int(state.get("input_channels_dim", -1)),
            smooth_enabled=bool(state.get("smooth_enabled", smooth_scale is not None and smooth_scale.numel() > 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.preprocess_input(x)
        return self.forward_preprocessed(x_in, out_dtype=x.dtype)


class QuantLinearWithBranch(nn.Module):
    """Quant linear wrapper with an optional low-rank branch.

    Notes:
    - The quantized backbone is implemented by `QuantLinear`.
    - The low-rank branch stays in BF16/FP16 while the main branch uses MXFP4.
    """

    def __init__(self, quant_linear: QuantLinear, branch: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.quant_linear = quant_linear
        self.branch = branch

    @property
    def in_features(self) -> int:
        return self.quant_linear.in_features

    @property
    def out_features(self) -> int:
        return self.quant_linear.out_features

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_group_size: int = 32,
        act_group_size: int = 32,
        weight_clip_ratio: float = 1.0,
        act_clip_ratio: float = 1.0,
        compute_dtype: str | torch.dtype = "bfloat16",
    ) -> "QuantLinearWithBranch":
        q = QuantLinear.from_linear(
            linear,
            weight_group_size=weight_group_size,
            act_group_size=act_group_size,
            weight_clip_ratio=weight_clip_ratio,
            act_clip_ratio=act_clip_ratio,
            quantize_input=False,
            skip_input_quant=True,
            compute_dtype=compute_dtype,
            scheme="mxfp4",
        )
        return cls(quant_linear=q, branch=None)

    @classmethod
    def from_linear_svdquant(
        cls,
        linear: nn.Linear,
        rank: int,
        weight_group_size: int = 32,
        act_group_size: int = 32,
        weight_clip_ratio: float = 1.0,
        act_clip_ratio: float = 1.0,
        compute_dtype: str | torch.dtype = "bfloat16",
        gptq_hessian: Optional[torch.Tensor] = None,
        gptq_damp_percent: float = 0.01,
        gptq_block_size: int = 128,
        gptq_dynamic_group_scale: bool = False,
    ) -> "QuantLinearWithBranch":
        """Single-layer SVDQuant (weight-first low-rank split).

        Target form:
            W ~= Q(W_main) + W_branch
            W_main = W - W_branch

        where W_branch is extracted from truncated SVD of original W.
        """
        with torch.no_grad():
            w_ref = linear.weight.detach().to(torch.float32)
            dtype = resolve_dtype(compute_dtype)

            if rank <= 0:
                q = _quant_linear_from_weight_and_bias(
                    in_features=linear.in_features,
                    out_features=linear.out_features,
                    weight=w_ref,
                    bias=linear.bias.detach() if linear.bias is not None else None,
                    smooth_scale=None,
                    input_channels_dim=-1,
                    smooth_enabled=False,
                    weight_group_size=weight_group_size,
                    act_group_size=act_group_size,
                    weight_clip_ratio=weight_clip_ratio,
                    act_clip_ratio=act_clip_ratio,
                    compute_dtype=compute_dtype,
                    scheme="mxfp4",
                    gptq_hessian=gptq_hessian,
                    gptq_damp_percent=gptq_damp_percent,
                    gptq_block_size=gptq_block_size,
                    gptq_dynamic_group_scale=gptq_dynamic_group_scale,
                )
                return cls(quant_linear=q, branch=None)

            branch = LowRankWeightBranch.from_weight(
                weight=w_ref,
                rank=rank,
                compute_dtype=dtype,
            )
            w_branch = branch.effective_weight().to(torch.float32)
            w_main = w_ref - w_branch

            q = _quant_linear_from_weight_and_bias(
                in_features=linear.in_features,
                out_features=linear.out_features,
                weight=w_main,
                bias=linear.bias.detach() if linear.bias is not None else None,
                smooth_scale=None,
                input_channels_dim=-1,
                smooth_enabled=False,
                weight_group_size=weight_group_size,
                act_group_size=act_group_size,
                weight_clip_ratio=weight_clip_ratio,
                act_clip_ratio=act_clip_ratio,
                compute_dtype=compute_dtype,
                scheme="mxfp4",
                gptq_hessian=gptq_hessian,
                gptq_damp_percent=gptq_damp_percent,
                gptq_block_size=gptq_block_size,
                gptq_dynamic_group_scale=gptq_dynamic_group_scale,
            )
            return cls(quant_linear=q, branch=branch)

    @classmethod
    def from_smoothed_weight_and_bias(
        cls,
        in_features: int,
        out_features: int,
        smoothed_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        smooth_scale: Optional[torch.Tensor],
        rank: int,
        weight_group_size: int = 32,
        act_group_size: int = 32,
        weight_clip_ratio: float = 1.0,
        act_clip_ratio: float = 1.0,
        compute_dtype: str | torch.dtype = "bfloat16",
        input_channels_dim: int = -1,
        smooth_enabled: bool = True,
        svd_device: Optional[torch.device | str] = None,
        gptq_hessian: Optional[torch.Tensor] = None,
        gptq_damp_percent: float = 0.01,
        gptq_block_size: int = 128,
        gptq_dynamic_group_scale: bool = False,
    ) -> "QuantLinearWithBranch":
        """Build QuantLinearWithBranch from an already-smoothed weight tensor.

        Target form:
            W_s ~= Q(W_main) + W_branch
            W_main = W_s - W_branch

        where SVD is applied to `smoothed_weight`, not to the original linear weight.
        """
        if smoothed_weight.dim() != 2:
            raise ValueError(f"smoothed_weight must be 2D, got {tuple(smoothed_weight.shape)}")
        if int(smoothed_weight.shape[0]) != int(out_features) or int(smoothed_weight.shape[1]) != int(in_features):
            raise ValueError(
                "smoothed_weight shape mismatch: "
                f"expected ({out_features}, {in_features}), got {tuple(smoothed_weight.shape)}"
            )

        with torch.no_grad():
            w_ref = smoothed_weight.detach().to(torch.float32)
            dtype = resolve_dtype(compute_dtype)

            if rank <= 0:
                q = _quant_linear_from_weight_and_bias(
                    in_features=in_features,
                    out_features=out_features,
                    weight=w_ref,
                    bias=bias.detach() if bias is not None else None,
                    smooth_scale=smooth_scale,
                    input_channels_dim=input_channels_dim,
                    smooth_enabled=smooth_enabled,
                    weight_group_size=weight_group_size,
                    act_group_size=act_group_size,
                    weight_clip_ratio=weight_clip_ratio,
                    act_clip_ratio=act_clip_ratio,
                    compute_dtype=compute_dtype,
                    scheme="mxfp4",
                    gptq_hessian=gptq_hessian,
                    gptq_damp_percent=gptq_damp_percent,
                    gptq_block_size=gptq_block_size,
                    gptq_dynamic_group_scale=gptq_dynamic_group_scale,
                )
                return cls(quant_linear=q, branch=None)

            branch = LowRankWeightBranch.from_weight(
                weight=w_ref,
                rank=rank,
                compute_dtype=dtype,
                svd_device=svd_device,
            )
            w_branch = branch.effective_weight().to(torch.float32)
            w_main = w_ref - w_branch

            q = _quant_linear_from_weight_and_bias(
                in_features=in_features,
                out_features=out_features,
                weight=w_main,
                bias=bias.detach() if bias is not None else None,
                smooth_scale=smooth_scale,
                input_channels_dim=input_channels_dim,
                smooth_enabled=smooth_enabled,
                weight_group_size=weight_group_size,
                act_group_size=act_group_size,
                weight_clip_ratio=weight_clip_ratio,
                act_clip_ratio=act_clip_ratio,
                compute_dtype=compute_dtype,
                scheme="mxfp4",
                gptq_hessian=gptq_hessian,
                gptq_damp_percent=gptq_damp_percent,
                gptq_block_size=gptq_block_size,
                gptq_dynamic_group_scale=gptq_dynamic_group_scale,
            )
            return cls(quant_linear=q, branch=branch)

    def export_state(self) -> Dict[str, Any]:
        branch_state = None
        if isinstance(self.branch, LowRankWeightBranch):
            branch_state = self.branch.export_state()
        return {
            "quant_linear": self.quant_linear.export_quant_state(),
            "has_branch": self.branch is not None,
            "branch": branch_state,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.quant_linear.load_quant_state(state["quant_linear"])
        bstate = state.get("branch", None)
        if bstate is None:
            self.branch = None
            return
        btype = bstate.get("type", "")
        # backward compatibility with previous prototype type.
        if btype in ("low_rank_weight", "low_rank_residual"):
            self.branch = LowRankWeightBranch.from_state(bstate)
            return
        raise ValueError(f"Unsupported branch type: {btype}")

    def register_branch(self, branch: nn.Module) -> None:
        self.branch = branch

    def set_module_name(self, name: str) -> None:
        self.quant_linear.set_module_name(name)

    def set_act_policy(self, policy: Optional[dict]) -> None:
        self.quant_linear.set_act_policy(policy)

    def set_act_policy_context(self, expert_id: Optional[str], timestep_bin: Optional[str]) -> None:
        self.quant_linear.set_act_policy_context(expert_id, timestep_bin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_smooth = self.quant_linear.smooth_input_only(x)
        x_main = self.quant_linear._apply_input_quant_qdq(x_smooth)
        out = self.quant_linear.forward_preprocessed(x_main, out_dtype=x.dtype)
        # out = self.quant_linear.forward_preprocessed(x_smooth, out_dtype=x.dtype)
        if self.branch is not None:
            out = out + self.branch(x_smooth)
        return out


class LowRankWeightBranch(nn.Module):
    """Low-rank branch extracted from a reference weight tensor.

    For truncated SVD W ~= U_r S_r V_r^T, use:
      A = S_r^{1/2} V_r^T   (down weight, shape [r, in])
      B = U_r S_r^{1/2}     (up weight,   shape [out, r])
      W_branch = B @ A
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.compute_dtype = compute_dtype
        self.a = nn.Linear(self.in_features, self.rank, bias=False, dtype=compute_dtype)
        self.b = nn.Linear(self.rank, self.out_features, bias=False, dtype=compute_dtype)

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        rank: int,
        compute_dtype: torch.dtype = torch.bfloat16,
        svd_device: Optional[torch.device | str] = None,
    ) -> "LowRankWeightBranch":
        if weight.dim() != 2:
            raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
        out_features, in_features = weight.shape
        r = max(1, min(int(rank), int(min(out_features, in_features))))
        m = cls(in_features=in_features, out_features=out_features, rank=r, compute_dtype=compute_dtype)
        with torch.no_grad():
            if svd_device is None:
                svd_device = weight.device
            svd_device = torch.device(svd_device)
            w_svd = weight.detach().to(device=svd_device, dtype=torch.float32)
            u, s, vh = torch.linalg.svd(w_svd, full_matrices=False)
            sr_sqrt = torch.sqrt(s[:r])
            # A = S^{1/2} V^T, B = U S^{1/2}
            a = sr_sqrt[:, None] * vh[:r, :]
            b = u[:, :r] * sr_sqrt[None, :]
            m.a.weight.copy_(a.to(device=m.a.weight.device, dtype=compute_dtype))
            m.b.weight.copy_(b.to(device=m.b.weight.device, dtype=compute_dtype))
        return m

    def effective_weight(self) -> torch.Tensor:
        return self.b.weight.to(torch.float32) @ self.a.weight.to(torch.float32)

    def export_state(self) -> Dict[str, Any]:
        return {
            "type": "low_rank_weight",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "a.weight": self.a.weight.detach().cpu(),
            "b.weight": self.b.weight.detach().cpu(),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "LowRankWeightBranch":
        dtype = resolve_dtype(state.get("compute_dtype", "bfloat16"))
        m = cls(
            in_features=int(state["in_features"]),
            out_features=int(state["out_features"]),
            rank=int(state["rank"]),
            compute_dtype=dtype,
        )
        with torch.no_grad():
            m.a.weight.copy_(state["a.weight"].to(dtype=dtype))
            m.b.weight.copy_(state["b.weight"].to(dtype=dtype))
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b(self.a(x.to(self.compute_dtype)))
        return y.to(x.dtype)


def _quant_linear_from_weight_and_bias(
    in_features: int,
    out_features: int,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    smooth_scale: Optional[torch.Tensor],
    input_channels_dim: int,
    smooth_enabled: bool,
    weight_group_size: int,
    act_group_size: int,
    weight_clip_ratio: float,
    act_clip_ratio: float,
    compute_dtype: str | torch.dtype,
    scheme: str = "mxfp4",
    gptq_hessian: Optional[torch.Tensor] = None,
    gptq_damp_percent: float = 0.01,
    gptq_block_size: int = 128,
    gptq_dynamic_group_scale: bool = False,
) -> QuantLinear:
    q = QuantLinear(
        in_features=in_features,
        out_features=out_features,
        bias=(bias is not None),
        weight_group_size=weight_group_size,
        act_group_size=act_group_size,
        weight_clip_ratio=weight_clip_ratio,
        act_clip_ratio=act_clip_ratio,
        quantize_input=True,
        skip_input_quant=False,
        compute_dtype=compute_dtype,
        scheme=scheme,
    )
    if gptq_hessian is not None:
        qstate = quantize_linear_weight_groupwise_gptq(
            weight.detach(),
            hessian=gptq_hessian,
            group_size=weight_group_size,
            clip_ratio=weight_clip_ratio,
            scheme=scheme,
            damp_percent=gptq_damp_percent,
            block_size=gptq_block_size,
            dynamic_group_scale=gptq_dynamic_group_scale,
        )
    else:
        qstate = quantize_linear_weight_groupwise(
            weight.detach(),
            group_size=weight_group_size,
            clip_ratio=weight_clip_ratio,
            scheme=scheme,
        )
    q.qweight_packed = qstate["qweight_packed"].contiguous()
    q.w_scales = qstate["scales"].contiguous().to(torch.uint8)
    q.w_in_features = torch.tensor(qstate["meta"]["in_features"], dtype=torch.int32)
    q.w_group_size = torch.tensor(qstate["meta"]["group_size"], dtype=torch.int32)
    q.quant_method = str(qstate["meta"].get("quant_method", "rtn_mxfp4"))
    q.set_smooth_state(
        smooth_scale=smooth_scale,
        input_channels_dim=input_channels_dim,
        smooth_enabled=smooth_enabled,
    )
    if bias is not None and q.bias is not None:
        q.bias.data.copy_(bias.to(torch.bfloat16))
    return q


# Backward-compatible type alias for older code paths/caches.
LowRankResidualBranch = LowRankWeightBranch
