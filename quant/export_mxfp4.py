"""Export Wan2.2 I2V DiT W4A4 (engineering MXFP4) checkpoints."""

from __future__ import annotations
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from quant.quant_linear import QuantLinear
from quant.replace import QuantReplaceConfig, load_quant_config, replace_wan_dit_linear_with_quant
from wan.modules.model import WanModel


def _collect_unquantized_linear_modules(model: nn.Module) -> list[str]:
    out = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out.append(n)
    return out


def _collect_quant_linear_modules(model: nn.Module) -> list[str]:
    out = []
    for n, m in model.named_modules():
        if isinstance(m, QuantLinear):
            out.append(n)
    return out


def _extract_quant_only_state(model: nn.Module) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for n, m in model.named_modules():
        if isinstance(m, QuantLinear):
            prefix = n + "."
            state[prefix + "qweight_packed"] = m.qweight_packed.detach().cpu()
            state[prefix + "w_scales"] = m.w_scales.detach().cpu()
            state[prefix + "w_in_features"] = m.w_in_features.detach().cpu()
            state[prefix + "w_group_size"] = m.w_group_size.detach().cpu()
            if m.bias is not None:
                state[prefix + "bias"] = m.bias.detach().cpu()
    return state


def prepare_quantized_model(
    model: nn.Module,
    cfg: QuantReplaceConfig,
    log_dir: str | None,
    model_tag: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    has_quant = any(isinstance(m, QuantLinear) for _, m in model.named_modules())
    if has_quant:
        summary = {
            "model_tag": model_tag,
            "already_quantized": True,
            "quantized_modules": _collect_quant_linear_modules(model),
            "skipped_modules": [],
            "keep_block_indices": [],
            "block_map": [],
            "num_blocks": None,
        }
        return model, summary

    summary = replace_wan_dit_linear_with_quant(
        model=model,
        config=cfg,
        log_dir=log_dir,
        model_tag=model_tag,
    )
    summary["already_quantized"] = False
    return model, summary


def export_submodel(
    model: nn.Module,
    out_dir: Path,
    cfg: QuantReplaceConfig,
    model_tag: str,
    replace_log_dir: str | None = None,
) -> Dict[str, Any]:
    model, summary = prepare_quantized_model(model, cfg, replace_log_dir, model_tag)

    out_dir.mkdir(parents=True, exist_ok=True)
    full_state_path = out_dir / "mxfp4_state.pt"
    quant_only_path = out_dir / "mxfp4_quant_only.pt"

    torch.save(model.state_dict(), full_state_path)
    torch.save(_extract_quant_only_state(model), quant_only_path)

    unquantized = _collect_unquantized_linear_modules(model)
    quantized = _collect_quant_linear_modules(model)

    manifest = {
        "model_tag": model_tag,
        "full_state": full_state_path.name,
        "quant_only_state": quant_only_path.name,
        "num_quantized_linear": len(quantized),
        "num_unquantized_linear": len(unquantized),
        "quantized_linear_modules": quantized,
        "unquantized_linear_whitelist": unquantized,
        "replace_summary": summary,
    }
    (out_dir / "mxfp4_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export Wan2.2 I2V MXFP4 W4A4 checkpoints")
    parser.add_argument("--src_ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--quant_config",
        type=str,
        default="configs/wan_i2v_w4a4_mxfp4.yaml",
        help="Quant config yaml/json",
    )
    parser.add_argument(
        "--subfolders",
        type=str,
        default="low_noise_model,high_noise_model",
        help="Comma separated subfolders to export",
    )
    parser.add_argument(
        "--from_replaced",
        action="store_true",
        default=False,
        help="If set, assume loaded model may already contain QuantLinear and export directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_quant_config(args.quant_config)
    (out_root / "quant_config_resolved.json").write_text(
        json.dumps(cfg.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    subfolders = [s.strip() for s in args.subfolders.split(",") if s.strip()]
    all_manifest: Dict[str, Any] = {"submodels": {}}

    for sf in subfolders:
        logging.info("Loading %s from %s", sf, args.src_ckpt_dir)
        model = WanModel.from_pretrained(args.src_ckpt_dir, subfolder=sf)
        model.eval().requires_grad_(False)

        if args.from_replaced:
            replaced_state = Path(args.src_ckpt_dir) / sf / "mxfp4_state.pt"
            if replaced_state.exists():
                # Build QuantLinear graph first, then load replaced state.
                _ = replace_wan_dit_linear_with_quant(
                    model=model,
                    config=cfg,
                    log_dir=str(out_root / "logs"),
                    model_tag=f"{sf}_from_replaced_runtime",
                )
                model.load_state_dict(torch.load(replaced_state, map_location="cpu"), strict=False)
                logging.info("Loaded existing replaced state: %s", replaced_state)
            else:
                logging.warning(
                    "--from_replaced is set but %s not found, fallback to direct BF16->quant export.",
                    replaced_state,
                )

        replace_log_dir = str(out_root / "logs")
        manifest = export_submodel(
            model=model,
            out_dir=out_root / sf,
            cfg=cfg,
            model_tag=sf,
            replace_log_dir=replace_log_dir,
        )
        all_manifest["submodels"][sf] = manifest

        logging.info(
            "%s exported: quantized=%d unquantized=%d",
            sf,
            manifest["num_quantized_linear"],
            manifest["num_unquantized_linear"],
        )

    (out_root / "mxfp4_export_all_manifest.json").write_text(
        json.dumps(all_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Export complete: %s", out_root)


if __name__ == "__main__":
    main()
