#!/usr/bin/env python3
"""Offline Wan DiT PTQ from existing act.pt and wgt.pt artifacts.

This script builds SVDQuant-style PTQ modules with:
- main branch: MXFP4 reference quantization
- low-rank branch: BF16/FP16/FP32 depending on compute_dtype
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

import wan
from quant.quant_linear import QuantLinearWithBranch
from wan.configs import WAN_CONFIGS


LOGGER = logging.getLogger("dit_ptq")


def _find_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    cur = root
    for p in full_name.split("."):
        cur = getattr(cur, p)
    return cur


def _find_parent_and_leaf(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    if not parts:
        raise ValueError("empty module path")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _set_module_by_name(root: nn.Module, full_name: str, module: nn.Module) -> None:
    parent, leaf = _find_parent_and_leaf(root, full_name)
    parent._modules[leaf] = module


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Offline Wan DiT PTQ from act.pt + wgt.pt")
    p.add_argument("--task", type=str, default="i2v-A14B", choices=["i2v-A14B"])
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--act_path", type=str, required=True)
    p.add_argument("--wgt_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--weight_group_size", type=int, default=32)
    p.add_argument("--act_group_size", type=int, default=32)
    p.add_argument("--weight_clip_ratio", type=float, default=1.0)
    p.add_argument("--act_clip_ratio", type=float, default=1.0)
    p.add_argument("--compute_dtype", type=str, default="bfloat16")
    p.add_argument("--save_model_state", action="store_true", default=False)
    p.add_argument("--main_quant_method", "--main-quant-method", type=str, default="rtn_mxfp4", choices=["rtn_mxfp4", "gptq_mxfp4"])
    p.add_argument("--gptq_hessian_path", "--gptq-hessian-path", type=str, default="")
    p.add_argument("--gptq_damp_percent", "--gptq-damp-percent", type=float, default=0.01)
    p.add_argument("--gptq_block_size", "--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq_dynamic_group_scale", "--gptq-dynamic-group-scale", action="store_true", default=False)
    p.add_argument("--collect_gptq_hessian", "--collect-gptq-hessian", action="store_true", default=False)
    p.add_argument("--save_gptq_hessian", "--save-gptq-hessian", type=str, default="")
    return p.parse_args()


def _load_artifact(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict artifact at {path}, got {type(obj).__name__}")
    return obj


def _validate_artifacts(act_stats: Dict[str, Any], wgt_payload: Dict[str, Any]) -> None:
    for model_name in ("high_noise_model", "low_noise_model"):
        if model_name not in act_stats:
            raise KeyError(f"act.pt missing {model_name}")
        if model_name not in wgt_payload:
            raise KeyError(f"wgt.pt missing {model_name}")
        model_wgts = wgt_payload[model_name]
        if not isinstance(model_wgts, dict):
            raise TypeError(f"wgt.pt[{model_name}] must be dict")
        model_acts = act_stats[model_name]
        if not isinstance(model_acts, dict):
            raise TypeError(f"act.pt[{model_name}] must be dict")
        missing = [name for name in model_wgts.keys() if name not in model_acts]
        if missing:
            raise KeyError(f"act.pt[{model_name}] missing {len(missing)} module stats, e.g. {missing[:3]}")


def _validate_ptq_args(args: argparse.Namespace) -> None:
    if int(args.weight_group_size) != 32:
        raise ValueError(f"MXFP4 reference weight_group_size must be 32, got {args.weight_group_size}")
    if int(args.act_group_size) != 32:
        raise ValueError(f"MXFP4 reference act_group_size must be 32, got {args.act_group_size}")
    if args.main_quant_method == "gptq_mxfp4":
        if args.collect_gptq_hessian:
            raise ValueError("dit_ptq.py cannot collect GPTQ Hessians because it has no calibration samples; use tools/ptq_wan_i2v_opens2v_svdquant.py or pass --gptq_hessian_path.")
        if not args.gptq_hessian_path:
            raise ValueError("--main_quant_method=gptq_mxfp4 requires --gptq_hessian_path in dit_ptq.py")


def _load_gptq_hessians(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict GPTQ Hessian artifact at {path}, got {type(obj).__name__}")
    return obj


def _build_wan_i2v_pipe(args: argparse.Namespace) -> wan.WanI2V:
    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=True,
        convert_model_dtype=False,
    )
    pipe.low_noise_model.eval().requires_grad_(False)
    pipe.high_noise_model.eval().requires_grad_(False)
    return pipe


def _resolve_svd_device(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.device_id}")
    return torch.device("cpu")


def _build_one_submodel(
    model: nn.Module,
    model_name: str,
    act_stats: Dict[str, Any],
    wgt_payload: Dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    gptq_hessians: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    svd_device = _resolve_svd_device(args)
    model_states: Dict[str, Any] = {
        "__meta__": {
            "version": 2,
            "artifact": "wan_dit_ptq_state",
            "model_name": model_name,
            "task": args.task,
            "quant_scheme": "mxfp4",
            "weight_format": "fp4_e2m1_packed+e8m0_scale",
            "activation_qdq_format": "mxfp4_reference",
            "weight_block_size": int(args.weight_group_size),
            "activation_block_size": int(args.act_group_size),
            "rank": int(args.rank),
            "weight_group_size": int(args.weight_group_size),
            "act_group_size": int(args.act_group_size),
            "weight_clip_ratio": float(args.weight_clip_ratio),
            "act_clip_ratio": float(args.act_clip_ratio),
            "compute_dtype": str(args.compute_dtype),
            "svd_device": str(svd_device),
            "source_act_path": str(args.act_path),
            "source_wgt_path": str(args.wgt_path),
            "main_quant_method": str(args.main_quant_method),
            "gptq_hessian_path": str(args.gptq_hessian_path),
            "gptq_damp_percent": float(args.gptq_damp_percent),
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_dynamic_group_scale": bool(args.gptq_dynamic_group_scale),
        }
    }

    replaced_modules = []
    total_weight_params = 0
    total_branch_params = 0

    for module_name, rec in wgt_payload[model_name].items():
        module = _find_module_by_name(model, module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {model_name}.{module_name}, got {type(module).__name__}")
        if not isinstance(rec, dict) or "weight" not in rec:
            raise ValueError(f"Invalid wgt record for {model_name}.{module_name}")

        smoothed_weight = rec["weight"]
        if not torch.is_tensor(smoothed_weight):
            raise TypeError(f"wgt weight must be tensor at {model_name}.{module_name}")
        smooth_scale = rec.get("scale", None)
        if smooth_scale is None or not torch.is_tensor(smooth_scale):
            raise ValueError(f"wgt scale tensor missing at {model_name}.{module_name}")
        input_channels_dim = int(rec.get("input_channels_dim", -1))
        gptq_hessian = None
        if args.main_quant_method == "gptq_mxfp4":
            if gptq_hessians is None or model_name not in gptq_hessians:
                raise KeyError(f"Missing GPTQ Hessian payload for {model_name}")
            hrec = gptq_hessians[model_name].get(module_name, None)
            if not isinstance(hrec, dict) or "hessian" not in hrec:
                raise KeyError(f"Missing GPTQ Hessian for {model_name}.{module_name}")
            gptq_hessian = hrec["hessian"]

        qmod = QuantLinearWithBranch.from_smoothed_weight_and_bias(
            in_features=module.in_features,
            out_features=module.out_features,
            smoothed_weight=smoothed_weight,
            bias=module.bias.detach() if module.bias is not None else None,
            smooth_scale=smooth_scale,
            rank=int(args.rank),
            weight_group_size=int(args.weight_group_size),
            act_group_size=int(args.act_group_size),
            weight_clip_ratio=float(args.weight_clip_ratio),
            act_clip_ratio=float(args.act_clip_ratio),
            compute_dtype=args.compute_dtype,
            input_channels_dim=input_channels_dim,
            smooth_enabled=True,
            svd_device=svd_device,
            gptq_hessian=gptq_hessian,
            gptq_damp_percent=float(args.gptq_damp_percent),
            gptq_block_size=int(args.gptq_block_size),
            gptq_dynamic_group_scale=bool(args.gptq_dynamic_group_scale),
        )
        _set_module_by_name(model, module_name, qmod)

        state = qmod.export_state()
        model_states[module_name] = state
        replaced_modules.append(module_name)
        total_weight_params += int(module.weight.numel())
        if qmod.branch is not None and hasattr(qmod.branch, "a") and hasattr(qmod.branch, "b"):
            total_branch_params += int(qmod.branch.a.weight.numel() + qmod.branch.b.weight.numel())

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "ptq_state.pt"
    torch.save(model_states, state_path)

    model_state_path = None
    if args.save_model_state:
        model_state_path = out_dir / "model_state_dict.pt"
        torch.save(model.state_dict(), model_state_path)

    summary = {
        "model_name": model_name,
        "quant_scheme": "mxfp4",
        "weight_format": "fp4_e2m1_packed+e8m0_scale",
        "activation_qdq_format": "mxfp4_reference",
        "weight_block_size": int(args.weight_group_size),
        "activation_block_size": int(args.act_group_size),
        "num_replaced_modules": len(replaced_modules),
        "replaced_modules": replaced_modules,
        "rank": int(args.rank),
        "weight_group_size": int(args.weight_group_size),
        "act_group_size": int(args.act_group_size),
        "weight_clip_ratio": float(args.weight_clip_ratio),
        "act_clip_ratio": float(args.act_clip_ratio),
        "compute_dtype": str(args.compute_dtype),
        "main_quant_method": str(args.main_quant_method),
        "gptq_hessian_path": str(args.gptq_hessian_path),
        "gptq_damp_percent": float(args.gptq_damp_percent),
        "gptq_block_size": int(args.gptq_block_size),
        "gptq_dynamic_group_scale": bool(args.gptq_dynamic_group_scale),
        "svd_device": str(svd_device),
        "total_weight_params": int(total_weight_params),
        "total_branch_params": int(total_branch_params),
        "ptq_state_path": str(state_path),
        "model_state_dict_path": None if model_state_path is None else str(model_state_path),
        "source_act_modules": len(act_stats[model_name]),
        "source_wgt_modules": len(wgt_payload[model_name]),
    }
    LOGGER.info(
        "%s | scheme=mxfp4 | weight_block=%d | act_block=%d | replaced_modules=%d | total_weight_params=%d | total_branch_params=%d | ptq_state=%s",
        model_name,
        int(args.weight_group_size),
        int(args.act_group_size),
        len(replaced_modules),
        total_weight_params,
        total_branch_params,
        state_path,
    )
    return summary


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    act_stats = _load_artifact(args.act_path)
    wgt_payload = _load_artifact(args.wgt_path)
    _validate_artifacts(act_stats, wgt_payload)
    _validate_ptq_args(args)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pipe = _build_wan_i2v_pipe(args)
    gptq_hessians = None
    if args.main_quant_method == "gptq_mxfp4":
        LOGGER.info("Loading GPTQ Hessians: %s", args.gptq_hessian_path)
        gptq_hessians = _load_gptq_hessians(args.gptq_hessian_path)
    model_map = {
        "high_noise_model": pipe.high_noise_model,
        "low_noise_model": pipe.low_noise_model,
    }

    summaries = []
    for model_name in ("high_noise_model", "low_noise_model"):
        summary = _build_one_submodel(
            model=model_map[model_name],
            model_name=model_name,
            act_stats=act_stats,
            wgt_payload=wgt_payload,
            args=args,
            out_dir=out_root / model_name,
            gptq_hessians=gptq_hessians,
        )
        summaries.append(summary)

    final_summary = {
        "artifact": "wan_dit_ptq",
        "task": args.task,
        "quant_scheme": "mxfp4",
        "weight_format": "fp4_e2m1_packed+e8m0_scale",
        "activation_qdq_format": "mxfp4_reference",
        "weight_block_size": int(args.weight_group_size),
        "activation_block_size": int(args.act_group_size),
        "device_id": int(args.device_id),
        "source_act_path": str(args.act_path),
        "source_wgt_path": str(args.wgt_path),
        "output_dir": str(out_root),
        "rank": int(args.rank),
        "weight_group_size": int(args.weight_group_size),
        "act_group_size": int(args.act_group_size),
        "weight_clip_ratio": float(args.weight_clip_ratio),
        "act_clip_ratio": float(args.act_clip_ratio),
        "compute_dtype": str(args.compute_dtype),
        "save_model_state": bool(args.save_model_state),
        "main_quant_method": str(args.main_quant_method),
        "gptq_hessian_path": str(args.gptq_hessian_path),
        "gptq_damp_percent": float(args.gptq_damp_percent),
        "gptq_block_size": int(args.gptq_block_size),
        "gptq_dynamic_group_scale": bool(args.gptq_dynamic_group_scale),
        "models": summaries,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved PTQ summary: %s", summary_path)


if __name__ == "__main__":
    main()
