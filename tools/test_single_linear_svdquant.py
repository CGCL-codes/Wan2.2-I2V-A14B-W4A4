#!/usr/bin/env python3
"""Unit test for single nn.Linear SVDQuant prototype."""

from __future__ import annotations

import argparse
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import torch.nn as nn

from quant.quant_linear import QuantLinearWithBranch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Test single-linear SVDQuant prototype")
    parser.add_argument("--in_features", type=int, default=512)
    parser.add_argument("--out_features", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--weight_group_size", type=int, default=64)
    parser.add_argument("--ranks", type=str, default="0,4,8,16,32,64")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _metrics(ref: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    diff = (pred - ref).to(torch.float32)
    ref32 = ref.to(torch.float32)
    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    denom = ref32.abs().mean().item() + 1e-8
    rel_mae = mae / denom
    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "rel_mae": rel_mae,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    linear = nn.Linear(args.in_features, args.out_features, bias=True).to(torch.bfloat16)
    x = torch.randn(args.batch, args.seq, args.in_features, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = linear(x)

    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]

    print("Single-layer SVDQuant (W ~= Q(W_main) + W_branch) error report")
    print(
        f"shape: x=({args.batch},{args.seq},{args.in_features}), "
        f"W=({args.out_features},{args.in_features}), group={args.weight_group_size}"
    )
    print("rank\tmain_mse\tmain_mae\tmain_rel_mae\tmain_max_abs\tfull_mse\tfull_mae\tfull_rel_mae\tfull_max_abs")

    for rank in ranks:
        mod = QuantLinearWithBranch.from_linear_svdquant(
            linear,
            rank=rank,
            weight_group_size=args.weight_group_size,
            act_group_size=args.weight_group_size,
            weight_clip_ratio=1.0,
            act_clip_ratio=1.0,
            compute_dtype="bfloat16",
        )
        with torch.no_grad():
            pred_main = mod.quant_linear(x)  # only quantized main branch
            pred_full = mod(x)               # quantized main + low-rank branch
        m_main = _metrics(ref, pred_main)
        m_full = _metrics(ref, pred_full)
        print(
            f"{rank}\t"
            f"{m_main['mse']:.6e}\t{m_main['mae']:.6e}\t{m_main['rel_mae']:.6e}\t{m_main['max_abs']:.6e}\t"
            f"{m_full['mse']:.6e}\t{m_full['mae']:.6e}\t{m_full['rel_mae']:.6e}\t{m_full['max_abs']:.6e}"
        )


if __name__ == "__main__":
    main()
