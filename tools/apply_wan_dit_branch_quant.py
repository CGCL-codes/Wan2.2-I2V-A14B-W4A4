#!/usr/bin/env python3
"""Apply QuantLinearWithBranch replacement to Wan DiT checkpoint (single-submodel)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from quant.replace import load_quant_config, replace_wan_dit_linear_with_branch_quant
from wan.modules.model import WanModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Apply branch quant replacement to Wan DiT")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Wan2.2 checkpoint root")
    parser.add_argument(
        "--subfolder",
        type=str,
        default="low_noise_model",
        choices=["low_noise_model", "high_noise_model"],
        help="Which DiT sub-model to process",
    )
    parser.add_argument("--quant_config", type=str, required=True, help="Path to quant config yaml/json")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Only print selected modules")
    parser.add_argument("--cache_dir", type=str, default="", help="Output dir for branch/wgts cache")
    parser.add_argument("--log_dir", type=str, default="", help="Output dir for replacement summary logs")
    parser.add_argument("--save_state_path", type=str, default="", help="Optional output path for replaced model state_dict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_quant_config(args.quant_config)

    model = WanModel.from_pretrained(args.ckpt_dir, subfolder=args.subfolder)
    summary = replace_wan_dit_linear_with_branch_quant(
        model=model,
        config=cfg,
        dry_run=args.dry_run,
        cache_dir=args.cache_dir or None,
        log_dir=args.log_dir or None,
        model_tag=args.subfolder,
    )

    if (not args.dry_run) and args.save_state_path:
        out = Path(args.save_state_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
