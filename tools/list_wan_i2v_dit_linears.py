#!/usr/bin/env python3
"""List all nn.Linear modules in Wan2.2 I2V DiT with role flags."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traverse Wan2.2 I2V DiT and print all nn.Linear modules."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Optional checkpoint directory. If not set, instantiate model from config.",
    )
    parser.add_argument(
        "--noise",
        type=str,
        choices=["low", "high"],
        default="low",
        help="Which I2V DiT to load when --ckpt_dir is provided.",
    )
    return parser.parse_args()


def _build_i2v_dit(ckpt_dir: str | None, noise: str) -> WanModel:
    cfg = WAN_CONFIGS["i2v-A14B"]
    if ckpt_dir:
        subfolder = cfg.low_noise_checkpoint if noise == "low" else cfg.high_noise_checkpoint
        return WanModel.from_pretrained(ckpt_dir, subfolder=subfolder)
    return WanModel(
        model_type="i2v",
        patch_size=cfg.patch_size,
        text_len=cfg.text_len,
        in_dim=16,
        dim=cfg.dim,
        ffn_dim=cfg.ffn_dim,
        freq_dim=cfg.freq_dim,
        text_dim=cfg.text_dim,
        out_dim=16,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        window_size=cfg.window_size,
        qk_norm=cfg.qk_norm,
        cross_attn_norm=cfg.cross_attn_norm,
        eps=cfg.eps,
    )


def _classify(name: str) -> Tuple[bool, bool, bool, bool]:
    in_self_attn = ".self_attn." in name
    in_cross_attn = ".cross_attn." in name
    in_ffn = ".ffn." in name
    # Modulation in Wan I2V is parameter-driven; time_projection is the modulation generator.
    in_modulation = (".modulation" in name) or name.startswith("time_projection.")
    return in_self_attn, in_cross_attn, in_ffn, in_modulation


def _iter_linear_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def _bool01(v: bool) -> str:
    return "1" if v else "0"


def main() -> None:
    args = _parse_args()
    model = _build_i2v_dit(args.ckpt_dir, args.noise).eval()

    header = (
        "name\tin_features\tout_features\tself_attn\tcross_attn\tffn\tmodulation"
    )
    print(header)
    for name, linear in _iter_linear_modules(model):
        s, c, f, m = _classify(name)
        print(
            f"{name}\t{linear.in_features}\t{linear.out_features}\t"
            f"{_bool01(s)}\t{_bool01(c)}\t{_bool01(f)}\t{_bool01(m)}"
        )


if __name__ == "__main__":
    main()
