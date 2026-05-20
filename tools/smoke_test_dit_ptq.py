#!/usr/bin/env python3
"""Smoke test for DiT PTQ skeleton (save/load cache workflow)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from quant.dit_ptq import DiTActivationQuantConfig, DiTQuantConfig, DiTWeightQuantConfig, ptq


class TinyAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = TinyAttn(dim)
        self.cross_attn = TinyAttn(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))


class TinyWanDiT(nn.Module):
    def __init__(self, dim: int = 64, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = x + blk.self_attn.o(blk.self_attn.v(x))
            x = x + blk.cross_attn.o(blk.cross_attn.v(x))
            x = x + blk.ffn(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Smoke test for DiT PTQ skeleton")
    parser.add_argument("--out_dir", type=str, default="./tmp_dit_ptq_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DiTQuantConfig(
        wgts=DiTWeightQuantConfig(enabled=True, enabled_low_rank=True, low_rank_rank=8),
        ipts=DiTActivationQuantConfig(enabled=True),
        opts=DiTActivationQuantConfig(enabled=False),
    )

    model = TinyWanDiT()
    ptq(model, cfg, save_dirpath=str(out_dir), save_model=True)

    expected = ["branch.pt", "wgts.pt", "acts.pt", "scale.pt"]
    for name in expected:
        p = out_dir / name
        if not p.exists():
            raise RuntimeError(f"missing cache file: {p}")

    # Reload path smoke.
    model2 = TinyWanDiT()
    ptq(model2, cfg, load_dirpath=str(out_dir), save_dirpath="")

    x = torch.randn(1, 4, 64)
    _ = model2(x)
    print("Smoke test passed.")
    print(f"Cache dir: {out_dir}")


if __name__ == "__main__":
    main()
