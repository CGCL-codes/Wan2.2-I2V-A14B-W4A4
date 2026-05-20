# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Simple MXFP4 per-tensor runtime modules and loader for WanModel."""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MXFP4_CODEBOOK = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
      0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0],
    dtype=torch.float32,
)


@dataclass
class QuantTensor:
    packed: torch.Tensor
    scale: torch.Tensor
    shape: Tuple[int, ...]
    numel: int


def _unpack_nibbles(packed: torch.Tensor, numel: int, device: torch.device) -> torch.Tensor:
    p = packed.to(device=device, dtype=torch.uint8)
    out = torch.empty(p.numel() * 2, device=device, dtype=torch.uint8)
    out[0::2] = p & 0x0F
    out[1::2] = (p >> 4) & 0x0F
    return out[:numel]


def dequantize_mxfp4_per_tensor(q: QuantTensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = _unpack_nibbles(q.packed, q.numel, device)
    cb = MXFP4_CODEBOOK.to(device=device)
    w = cb[idx.long()].reshape(q.shape) * q.scale.to(device=device)
    return w.to(dtype=dtype)


class MXFP4Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("w_scale", torch.ones((), dtype=torch.float32), persistent=True)
        self.register_buffer("w_shape", torch.empty(0, dtype=torch.int64), persistent=True)
        self.register_buffer("w_numel", torch.tensor(0, dtype=torch.int64), persistent=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "MXFP4Linear":
        out = cls(mod.in_features, mod.out_features, mod.bias is not None)
        if mod.bias is not None:
            out.bias.data.copy_(mod.bias.detach().to(torch.float32))
        return out

    def _dequant_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        qt = QuantTensor(
            packed=self.qweight,
            scale=self.w_scale,
            shape=tuple(self.w_shape.tolist()),
            numel=int(self.w_numel.item()),
        )
        return dequantize_mxfp4_per_tensor(qt, device=device, dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:
        # Compatibility shim for legacy code paths that query `.weight.device`.
        return self.qweight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight(device=x.device, dtype=x.dtype)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class MXFP4Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("w_scale", torch.ones((), dtype=torch.float32), persistent=True)
        self.register_buffer("w_shape", torch.empty(0, dtype=torch.int64), persistent=True)
        self.register_buffer("w_numel", torch.tensor(0, dtype=torch.int64), persistent=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, mod: nn.Conv3d) -> "MXFP4Conv3d":
        out = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
        )
        if mod.bias is not None:
            out.bias.data.copy_(mod.bias.detach().to(torch.float32))
        return out

    def _dequant_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        qt = QuantTensor(
            packed=self.qweight,
            scale=self.w_scale,
            shape=tuple(self.w_shape.tolist()),
            numel=int(self.w_numel.item()),
        )
        return dequantize_mxfp4_per_tensor(qt, device=device, dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:
        # Compatibility shim for legacy code paths that query `.weight.device`.
        return self.qweight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight(device=x.device, dtype=x.dtype)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.conv3d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def replace_with_mxfp4_modules(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, MXFP4Linear.from_float(child))
        elif isinstance(child, nn.Conv3d):
            setattr(module, name, MXFP4Conv3d.from_float(child))
        else:
            replace_with_mxfp4_modules(child)


def load_quantized_wan_model(model: nn.Module, quant_ckpt_path: str, strict: bool = True) -> nn.Module:
    """Replace float layers with MXFP4 runtime layers and load quantized state dict."""
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")

    replace_with_mxfp4_modules(model)
    state = torch.load(quant_ckpt_path, map_location="cpu")

    # Preload quant buffers directly to avoid shape mismatch (placeholder buffers are shape [0]).
    quant_buf_names = ("qweight", "w_scale", "w_shape", "w_numel")
    preloaded_quant_keys = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, (MXFP4Linear, MXFP4Conv3d)):
            continue
        prefix = f"{module_name}." if module_name else ""
        for buf_name in quant_buf_names:
            key = f"{prefix}{buf_name}"
            if key in state:
                module._buffers[buf_name] = state[key].detach().clone()
                preloaded_quant_keys.add(key)

    remain_state = {k: v for k, v in state.items() if k not in preloaded_quant_keys}
    incompat = model.load_state_dict(remain_state, strict=False)

    if strict:
        missing = [k for k in incompat.missing_keys if k not in preloaded_quant_keys]
        unexpected = list(incompat.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Error(s) in loading state_dict for WanModel after quant buffer preload:\n"
                f"  missing keys: {missing}\n"
                f"  unexpected keys: {unexpected}"
            )

    model.to(dev)
    model.eval().requires_grad_(False)
    return model
