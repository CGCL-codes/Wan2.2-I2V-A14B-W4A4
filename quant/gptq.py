"""GPTQ calibration helpers for SVDQuant main residual branches."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class GPTQHessianCollector:
    """Online X^T X Hessian accumulator for GPTQ-MXFP4 Linear quantization.

    The collector stores only the running Hessian and sample count. Inputs are
    flattened to `[-1, in_features]`, converted to float32, optionally token
    sampled, and accumulated without retaining activation tensors.
    """

    def __init__(
        self,
        in_features: int,
        device: str | torch.device = "cpu",
        sample_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.in_features = int(in_features)
        self.device = torch.device(device)
        self.sample_tokens = None if sample_tokens is None else int(sample_tokens)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.H = torch.zeros(self.in_features, self.in_features, dtype=torch.float32, device=self.device)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulate `x.T @ x` from an activation batch shaped `[..., in_features]`."""
        if x.numel() == 0:
            return
        if int(x.shape[-1]) != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got shape={tuple(x.shape)}")

        x2 = x.detach().reshape(-1, self.in_features)
        if self.sample_tokens is not None and self.sample_tokens > 0 and x2.shape[0] > self.sample_tokens:
            idx = torch.randperm(x2.shape[0], device=x2.device)[: self.sample_tokens]
            x2 = x2.index_select(0, idx)
        elif self.max_tokens is not None and self.max_tokens > 0 and x2.shape[0] > self.max_tokens:
            x2 = x2[: self.max_tokens]

        if x2.numel() == 0:
            return
        x2 = x2.to(device=self.device, dtype=torch.float32)
        self.H.add_(x2.T @ x2)
        self.nsamples += int(x2.shape[0])

    def get_hessian(self, normalize: bool = True) -> torch.Tensor:
        """Return the accumulated Hessian, optionally normalized by sample count."""
        if normalize and self.nsamples > 0:
            return self.H / float(self.nsamples)
        return self.H.clone()

    def export(self, normalize: bool = True) -> Dict[str, Any]:
        """Export a serializable Hessian record."""
        return {
            "hessian": self.get_hessian(normalize=normalize).cpu(),
            "nsamples": int(self.nsamples),
            "in_features": int(self.in_features),
        }

