"""Activation quantization observers / fake-quantizers."""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicActQuantizer(nn.Module):
    """Dynamic per-batch group-wise fake quant for activations [..., C]."""

    def __init__(
        self,
        group_size: int = 64,
        clip_ratio: float = 1.0,
        quant_min: int = -8,
        quant_max: int = 7,
        eps: float = 1e-8,
        enabled: bool = True,
        observer_type: str = "absmax",
    ) -> None:
        super().__init__()
        self.group_size = int(group_size)
        self.clip_ratio = float(clip_ratio)
        self.quant_min = int(quant_min)
        self.quant_max = int(quant_max)
        self.eps = float(eps)
        self.enabled = bool(enabled)
        self.observer_type = observer_type

        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}")

        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long), persistent=False)

    def _compute_scale(self, xg: torch.Tensor) -> torch.Tensor:
        if self.observer_type != "absmax":
            raise NotImplementedError(
                f"observer_type={self.observer_type} is not implemented yet. Use 'absmax'."
            )
        absmax = xg.abs().amax(dim=-1)
        absmax = absmax * self.clip_ratio
        scale = (absmax / float(self.quant_max)).clamp(min=self.eps)
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.enabled) or x.numel() == 0:
            return x

        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        c = x32.shape[-1]
        g = self.group_size
        pad = (g - (c % g)) % g

        if pad > 0:
            pad_shape = list(x32.shape)
            pad_shape[-1] = pad
            x32 = torch.cat([x32, x32.new_zeros(pad_shape)], dim=-1)

        new_c = x32.shape[-1]
        num_groups = new_c // g

        xg = x32.reshape(-1, num_groups, g)
        scale = self._compute_scale(xg)

        q = torch.round(xg / scale.unsqueeze(-1)).clamp(self.quant_min, self.quant_max)
        dq = q * scale.unsqueeze(-1)
        dq = dq.reshape(*x32.shape)

        if pad > 0:
            dq = dq[..., :c]

        self.num_batches += 1
        return dq.to(orig_dtype)
