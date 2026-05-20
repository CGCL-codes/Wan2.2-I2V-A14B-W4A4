#!/usr/bin/env python3
"""
Quantize Wan2.2 I2V A14B DiT weights into a simple MXFP4-like per-tensor format.

Notes:
- This is a practical, simple implementation:
  - 4-bit codebook quantization (FP4-like value set)
  - one shared scale per weight tensor (per-tensor)
- It quantizes nn.Linear and nn.Conv3d weights in WanModel.
- Outputs custom checkpoint files (not directly loadable by WanModel.from_pretrained).
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Tuple
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from wan.modules.model import WanModel

# 16-level FP4-like codebook. We use a symmetric finite set and pair it with per-tensor scale.
# This is intentionally simple and easy to run.
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


def _pack_nibbles(q_idx: torch.Tensor) -> torch.Tensor:
    """Pack uint8 indices [0, 15] into bytes (2 values per byte)."""
    flat = q_idx.reshape(-1).to(torch.uint8)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, flat.new_zeros(1)], dim=0)
    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    return lo | hi


def _unpack_nibbles(packed: torch.Tensor, numel: int, device: torch.device) -> torch.Tensor:
    """Unpack bytes to uint8 indices [0, 15]."""
    p = packed.to(device=device, dtype=torch.uint8)
    out = torch.empty(p.numel() * 2, device=device, dtype=torch.uint8)
    out[0::2] = p & 0x0F
    out[1::2] = (p >> 4) & 0x0F
    return out[:numel]


def quantize_mxfp4_per_tensor(w: torch.Tensor) -> QuantTensor:
    """Quantize one tensor with shared scale + FP4-like codebook."""
    w32 = w.detach().to(torch.float32).contiguous()
    max_abs = w32.abs().max()
    scale = torch.clamp(max_abs / 6.0, min=1e-8)  # map largest magnitude roughly to codebook max

    normed = w32 / scale
    cb = MXFP4_CODEBOOK.to(device=normed.device)

    # Nearest codebook entry for each element.
    # Shape: [N, 16] -> argmin over last dim.
    flat = normed.reshape(-1, 1)
    dist = (flat - cb.view(1, -1)).abs()
    q_idx = torch.argmin(dist, dim=1).to(torch.uint8)

    packed = _pack_nibbles(q_idx).cpu()
    return QuantTensor(
        packed=packed,
        scale=scale.detach().to(torch.float32).cpu(),
        shape=tuple(w.shape),
        numel=w.numel(),
    )


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
        qt = quantize_mxfp4_per_tensor(mod.weight)
        out.qweight = qt.packed
        out.w_scale = qt.scale
        out.w_shape = torch.tensor(qt.shape, dtype=torch.int64)
        out.w_numel = torch.tensor(qt.numel, dtype=torch.int64)
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
        qt = quantize_mxfp4_per_tensor(mod.weight)
        out.qweight = qt.packed
        out.w_scale = qt.scale
        out.w_shape = torch.tensor(qt.shape, dtype=torch.int64)
        out.w_numel = torch.tensor(qt.numel, dtype=torch.int64)
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
    """In-place replacement of float modules with MXFP4 modules."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, MXFP4Linear.from_float(child))
        elif isinstance(child, nn.Conv3d):
            setattr(module, name, MXFP4Conv3d.from_float(child))
        else:
            replace_with_mxfp4_modules(child)


def _quantize_submodel(src_root: str, subfolder: str, dst_root: str) -> None:
    print(f"[Quantize] Loading {subfolder} from: {src_root}")
    model = WanModel.from_pretrained(src_root, subfolder=subfolder, torch_dtype=torch.float32)
    model.eval().requires_grad_(False).cpu()

    print(f"[Quantize] Replacing modules for {subfolder}...")
    replace_with_mxfp4_modules(model)

    dst_sub = os.path.join(dst_root, subfolder)
    os.makedirs(dst_sub, exist_ok=True)

    # Save quantized state dict.
    q_path = os.path.join(dst_sub, "mxfp4_per_tensor_model.pt")
    torch.save(model.state_dict(), q_path)

    # Copy config for reconstruction.
    src_cfg = os.path.join(src_root, subfolder, "config.json")
    dst_cfg = os.path.join(dst_sub, "config.json")
    if os.path.exists(src_cfg):
        shutil.copy2(src_cfg, dst_cfg)

    meta = {
        "format": "mxfp4_per_tensor_simple",
        "codebook": MXFP4_CODEBOOK.tolist(),
        "quantized_modules": ["Linear", "Conv3d"],
        "checkpoint": os.path.basename(q_path),
    }
    with open(os.path.join(dst_sub, "quant_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Quantize] Done {subfolder}: {q_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize Wan2.2 I2V A14B to simple MXFP4 per-tensor")
    parser.add_argument("--src_ckpt_dir", type=str, required=True, help="Original Wan2.2-I2V-A14B checkpoint directory")
    parser.add_argument("--dst_ckpt_dir", type=str, required=True, help="Output quantized checkpoint directory")
    args = parser.parse_args()

    os.makedirs(args.dst_ckpt_dir, exist_ok=True)

    # Keep shared files for convenience.
    for fname in ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"]:
        src = os.path.join(args.src_ckpt_dir, fname)
        dst = os.path.join(args.dst_ckpt_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    _quantize_submodel(args.src_ckpt_dir, "low_noise_model", args.dst_ckpt_dir)
    _quantize_submodel(args.src_ckpt_dir, "high_noise_model", args.dst_ckpt_dir)

    print("[Done] Quantization finished.")
    print("This output is a custom MXFP4-per-tensor checkpoint and requires custom loader code.")


if __name__ == "__main__":
    print(sys.path[0])
    print(PROJECT_ROOT)
    main()
