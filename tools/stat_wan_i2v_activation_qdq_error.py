#!/usr/bin/env python3
"""Stat activation QDQ error for Wan I2V SVDQuant/GPTQ QuantLinear layers.

For each target QuantLinear / QuantLinearWithBranch, this script observes the
real input from the quantized model forward path and compares:

    x       = x_smooth, the actual input to activation quantization
    x_qdq   = activation_quant_dequant(x_smooth)
    error   = x_qdq - x_smooth

This is not weight error, isolated layer output error, or propagated layer
input/output error. Only scalar reductions are accumulated; full activations
are not saved to CPU, disk, or long-lived Python containers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from quant.quant_linear import QuantLinear, QuantLinearWithBranch
from quant.wan_smooth import _build_schedule, _register_linear_input_cast_hooks
from tools.infer_wan_i2v_svdquant import (
    _apply_ptq_state_to_model,
    _load_i2v_ptq_states,
    _parse_keep_fp_blocks,
)
from tools.stat_wan_i2v_isolated_layer_error import (
    ManifestSample,
    _build_filter_regex,
    _build_pipe,
    _expert_names,
    _load_manifest,
    _prep_sample,
    _size_to_hw,
)
from tools.stat_wan_i2v_propagated_layer_error import _remove_hooks, _run_model
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger("stat_wan_i2v_activation_qdq_error")


class ActivationQDQStats:
    """Streaming scalar QDQ error stats for one activation stream."""

    def __init__(self) -> None:
        self.calls = 0
        self.numel = 0
        self.sum_abs_err = 0.0
        self.sum_sq_err = 0.0
        self.max_abs_err = 0.0
        self.sum_x_abs = 0.0
        self.sum_x_qdq_abs = 0.0
        self.sum_x_sq = 0.0
        self.sum_x_qdq_sq = 0.0
        self.sum_dot = 0.0
        self.x_absmax = 0.0
        self.x_qdq_absmax = 0.0
        self.first_step: Optional[int] = None
        self.last_step: Optional[int] = None

    @torch.no_grad()
    def update(self, x: torch.Tensor, x_qdq: torch.Tensor, step_idx: Optional[int]) -> None:
        if tuple(x.shape) != tuple(x_qdq.shape):
            raise ValueError(f"Shape mismatch while comparing activation QDQ: x={tuple(x.shape)} x_qdq={tuple(x_qdq.shape)}")

        x32 = x.detach().to(torch.float32)
        x_qdq32 = x_qdq.detach().to(torch.float32)
        err = x_qdq32 - x32
        n = int(err.numel())
        if n == 0:
            return

        abs_err = err.abs()
        self.calls += 1
        self.numel += n
        self.sum_abs_err += float(abs_err.sum().item())
        self.sum_sq_err += float((err * err).sum().item())
        self.max_abs_err = max(self.max_abs_err, float(abs_err.amax().item()))
        self.sum_x_abs += float(x32.abs().sum().item())
        self.sum_x_qdq_abs += float(x_qdq32.abs().sum().item())
        self.sum_x_sq += float((x32 * x32).sum().item())
        self.sum_x_qdq_sq += float((x_qdq32 * x_qdq32).sum().item())
        self.sum_dot += float((x32 * x_qdq32).sum().item())
        self.x_absmax = max(self.x_absmax, float(x32.abs().amax().item()))
        self.x_qdq_absmax = max(self.x_qdq_absmax, float(x_qdq32.abs().amax().item()))
        if step_idx is not None:
            step_idx = int(step_idx)
            self.first_step = step_idx if self.first_step is None else min(self.first_step, step_idx)
            self.last_step = step_idx if self.last_step is None else max(self.last_step, step_idx)

    def to_row(self, expert: str, module_name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        denom = max(1, self.numel)
        mse = self.sum_sq_err / denom
        cosine = self.sum_dot / math.sqrt(max(self.sum_x_sq * self.sum_x_qdq_sq, 1e-30))
        row: Dict[str, Any] = {
            "expert": expert,
            "module": module_name,
            "calls": int(self.calls),
            "numel": int(self.numel),
            "mae": self.sum_abs_err / denom,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "rel_l2": math.sqrt(self.sum_sq_err / max(self.sum_x_sq, 1e-30)),
            "max_abs_error": self.max_abs_err,
            "mean_x_abs": self.sum_x_abs / denom,
            "mean_x_qdq_abs": self.sum_x_qdq_abs / denom,
            "x_absmax": self.x_absmax,
            "x_qdq_absmax": self.x_qdq_absmax,
            "cosine": cosine,
            "cosine_error": 1.0 - cosine,
            "first_step": "" if self.first_step is None else int(self.first_step),
            "last_step": "" if self.last_step is None else int(self.last_step),
        }
        row.update(meta)
        return row


class ActivationQDQCollector:
    def __init__(self, module_meta: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        self.current_step: Optional[int] = None
        self.current_expert: Optional[str] = None
        self.raw_timesteps: Dict[str, Dict[int, float]] = {"high": {}, "low": {}}
        self.module_meta = module_meta
        self.stats: Dict[str, Dict[str, ActivationQDQStats]] = {"high": {}, "low": {}}

    def begin_step(self, step_idx: int, raw_timestep: float, expert: str) -> None:
        self.current_step = int(step_idx)
        self.current_expert = expert
        self.raw_timesteps[expert][int(step_idx)] = float(raw_timestep)

    @torch.no_grad()
    def observe(self, expert: str, module_name: str, module: nn.Module, x_raw: torch.Tensor) -> None:
        if self.current_expert != expert:
            return
        qlinear = _inner_quant_linear(module)
        x_smooth = qlinear.smooth_input_only(x_raw.detach())
        x_qdq = qlinear._apply_input_quant_qdq(x_smooth)
        rec = self.stats[expert].setdefault(module_name, ActivationQDQStats())
        rec.update(x=x_smooth, x_qdq=x_qdq, step_idx=self.current_step)
        del x_smooth, x_qdq


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stat Wan I2V activation QDQ error")
    p.add_argument("--ckpt_dir", default="../Wan2.2-I2V-A14B-bf16", help="Wan I2V checkpoint directory")
    p.add_argument("--quant_ckpt_dir", default="", help="Checkpoint directory used to instantiate the quantized model; defaults to --ckpt_dir")
    p.add_argument("--ptq_dir", default="outputs/gptq_ptq", help="ptq_stats.pt path/dir or legacy split PTQ directory")
    p.add_argument("--manifest", default='/home/wjh/Wan2.2/opens2v_outputs/wan_bf16/generation_manifest.jsonl', help="jsonl manifest with image/used_image/img_paths and prompt")
    p.add_argument("--out_dir", default='outputs/activation_qdq_error')
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=5)
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--experts", type=str, default="both", choices=["high", "low", "both"])
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--target_regex", type=str, default="", help="Optional regex over module names; defaults to all quantized Linear modules")
    p.add_argument("--blocks", type=str, default="", help="Optional comma-separated block ids to filter modules")
    p.add_argument("--modules", type=str, default="", help="Optional comma-separated module suffixes, e.g. self_attn.q,ffn.2")
    p.add_argument("--list_linears", action="store_true", default=False)
    p.add_argument("--low_keep_fp_blocks", type=str, default="")
    p.add_argument("--high_keep_fp_blocks", type=str, default="")
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--offload_models", action="store_true", default=True)
    p.add_argument("--no_offload_models", action="store_false", dest="offload_models")
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _inner_quant_linear(module: nn.Module) -> QuantLinear:
    if isinstance(module, QuantLinearWithBranch):
        return module.quant_linear
    if isinstance(module, QuantLinear):
        return module
    raise TypeError(f"Expected QuantLinear or QuantLinearWithBranch, got {type(module).__name__}")


def _module_meta(module: nn.Module) -> Dict[str, Any]:
    qlinear = _inner_quant_linear(module)
    return {
        "act_group_size": int(qlinear.act_group_size),
        "act_clip_ratio": float(qlinear.act_clip_ratio),
        "scheme": str(qlinear.scheme),
        "quantize_input": bool(qlinear.quantize_input),
        "skip_input_quant": bool(qlinear.skip_input_quant),
        "smooth_enabled": bool(qlinear.smooth_enabled.item()),
        "input_channels_dim": int(qlinear.input_channels_dim.item()),
        "has_branch": bool(isinstance(module, QuantLinearWithBranch) and module.branch is not None),
        "quant_method": str(qlinear.quant_method),
    }


def _select_quant_targets(
    model: nn.Module,
    pattern: Optional[Any],
) -> Tuple[List[str], Dict[str, nn.Module], Dict[str, Dict[str, Any]]]:
    names: List[str] = []
    modules: Dict[str, nn.Module] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinearWithBranch):
            target = True
        elif isinstance(module, QuantLinear):
            target = not name.endswith(".quant_linear")
        else:
            target = False
        if not target:
            continue
        if pattern is not None and not pattern.search(name):
            continue
        names.append(name)
        modules[name] = module
        meta[name] = _module_meta(module)
    return names, modules, meta


def _register_activation_qdq_hooks(
    model: nn.Module,
    expert: str,
    module_names: Sequence[str],
    collector: ActivationQDQCollector,
) -> List[Any]:
    handles = []
    module_map = dict(model.named_modules())
    for name in module_names:
        module = module_map.get(name)
        if not isinstance(module, (QuantLinearWithBranch, QuantLinear)):
            LOGGER.warning("Quant target is not QuantLinear wrapper: %s (%s)", name, type(module).__name__)
            continue

        def _make_hook(module_name: str):
            def _hook(mod: nn.Module, inp: Tuple[Any, ...]) -> None:
                if not inp or not torch.is_tensor(inp[0]):
                    return
                collector.observe(expert=expert, module_name=module_name, module=mod, x_raw=inp[0])

            return _hook

        handles.append(module.register_forward_pre_hook(_make_hook(name), with_kwargs=False))
    return handles


def _write_outputs(out_dir: Path, collector: ActivationQDQCollector, metadata: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for expert in ("high", "low"):
        for module_name, rec in sorted(collector.stats[expert].items()):
            rows.append(rec.to_row(expert, module_name, collector.module_meta[expert][module_name]))

    fieldnames = [
        "expert",
        "module",
        "calls",
        "numel",
        "mae",
        "mse",
        "rmse",
        "rel_l2",
        "max_abs_error",
        "mean_x_abs",
        "mean_x_qdq_abs",
        "x_absmax",
        "x_qdq_absmax",
        "cosine",
        "cosine_error",
        "first_step",
        "last_step",
        "act_group_size",
        "act_clip_ratio",
        "scheme",
        "quantize_input",
        "skip_input_quant",
        "smooth_enabled",
        "input_channels_dim",
        "has_branch",
        "quant_method",
    ]
    csv_path = out_dir / "activation_qdq_error.csv"
    jsonl_path = out_dir / "activation_qdq_error.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata["raw_timesteps"] = {
        expert: {str(k): v for k, v in steps.items()} for expert, steps in collector.raw_timesteps.items()
    }
    metadata["num_reported_layers"] = len(rows)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved CSV: %s", csv_path)
    LOGGER.info("Saved JSONL: %s", jsonl_path)
    LOGGER.info("Saved metadata: %s", out_dir / "metadata.json")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")
    if args.sample_steps <= 0:
        raise ValueError("--sample_steps must be > 0")

    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found.")
    pattern = _build_filter_regex(args)
    LOGGER.info("Target filter: %s", "all quantized Linear modules" if pattern is None else pattern.pattern)

    quant_ckpt_dir = args.quant_ckpt_dir or args.ckpt_dir
    LOGGER.info("[1/4] loading quant model skeleton and PTQ states")
    q_pipe = _build_pipe(quant_ckpt_dir, args)
    low_state, high_state, low_ptq_path, high_ptq_path = _load_i2v_ptq_states(args.ptq_dir)
    _apply_ptq_state_to_model(
        q_pipe.low_noise_model,
        low_state,
        "low_noise_model",
        low_ptq_path,
        keep_fp_blocks=_parse_keep_fp_blocks(args.low_keep_fp_blocks),
    )
    _apply_ptq_state_to_model(
        q_pipe.high_noise_model,
        high_state,
        "high_noise_model",
        high_ptq_path,
        keep_fp_blocks=_parse_keep_fp_blocks(args.high_keep_fp_blocks),
    )
    q_pipe.high_noise_model.cpu()
    q_pipe.low_noise_model.cpu()
    if q_pipe.device.type == "cuda":
        torch.cuda.empty_cache()

    q_models = {"high": q_pipe.high_noise_model, "low": q_pipe.low_noise_model}
    target_modules: Dict[str, List[str]] = {}
    module_meta: Dict[str, Dict[str, Dict[str, Any]]] = {"high": {}, "low": {}}
    for expert in _expert_names(args.experts):
        names, _modules, meta = _select_quant_targets(q_models[expert], pattern)
        target_modules[expert] = names
        module_meta[expert] = meta
        LOGGER.info("%s target QuantLinear modules: %d", expert, len(names))
        for name in names:
            LOGGER.info("%s | %s", expert, name)

    if args.list_linears:
        return
    if not any(target_modules.values()):
        raise ValueError("No target quantized Linear modules found.")

    cast_handles: List[Any] = []
    stat_handles: List[Any] = []
    collector = ActivationQDQCollector(module_meta=module_meta)
    try:
        for expert in _expert_names(args.experts):
            cast_handles.extend(_register_linear_input_cast_hooks(q_models[expert]))
            stat_handles.extend(_register_activation_qdq_hooks(q_models[expert], expert, target_modules[expert], collector))
        LOGGER.info("[2/4] registered activation QDQ hooks: %d", len(stat_handles))

        cfg = WAN_CONFIGS["i2v-A14B"]
        frame_num = cfg.frame_num
        shift = cfg.sample_shift if args.shift is None else float(args.shift)
        boundary = float(q_pipe.boundary * q_pipe.num_train_timesteps)
        use_amp = q_pipe.device.type == "cuda" and q_pipe.param_dtype in (torch.float16, torch.bfloat16)

        sample_iter: Iterable[ManifestSample] = samples
        if tqdm is not None and not args.disable_tqdm:
            sample_iter = tqdm(samples, desc="activation QDQ samples", unit="sample", dynamic_ncols=True)

        LOGGER.info("[3/4] running quant denoise trajectory")
        active_expert: Optional[str] = None
        for sample_idx, sample in enumerate(sample_iter):
            torch.manual_seed(int(args.seed) + sample_idx)
            item = _prep_sample(q_pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))
            latent = item["latent"].to(device=q_pipe.device, dtype=q_pipe.param_dtype)
            y = item["y"].to(device=q_pipe.device, dtype=q_pipe.param_dtype)
            context = [t.to(device=q_pipe.device, dtype=q_pipe.param_dtype) for t in item["context"]]
            model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}
            scheduler, timesteps = _build_schedule(
                sample_solver=args.sample_solver,
                sampling_steps=int(args.sample_steps),
                shift=shift,
                num_train_timesteps=q_pipe.num_train_timesteps,
                device=q_pipe.device,
            )
            timesteps_list = [float(t.item()) for t in timesteps]
            step_iter: Iterable[Tuple[int, float]] = list(enumerate(timesteps_list))
            if tqdm is not None and not args.disable_tqdm:
                step_iter = tqdm(step_iter, desc=f"{sample.sample_id} denoise", unit="step", leave=False, dynamic_ncols=True)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=q_pipe.param_dtype, enabled=use_amp):
                for step_idx, t_scalar in step_iter:
                    expert = "high" if t_scalar >= boundary else "low"
                    if args.offload_models and active_expert != expert:
                        for name, model in q_models.items():
                            model.to(q_pipe.device if name == expert else "cpu")
                        active_expert = expert
                        if q_pipe.device.type == "cuda":
                            torch.cuda.empty_cache()
                    elif not args.offload_models:
                        q_models[expert].to(q_pipe.device)

                    collector.begin_step(step_idx=step_idx, raw_timestep=t_scalar, expert=expert)
                    timestep = torch.tensor([int(t_scalar)], device=q_pipe.device, dtype=torch.long)
                    t_for_step = torch.tensor(t_scalar, device=q_pipe.device, dtype=timesteps.dtype)
                    noise_pred = _run_model(q_models[expert], latent, timestep, model_args)
                    latent = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t_for_step,
                        latent.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)
                    del noise_pred

            del item, latent, y, context, model_args, scheduler, timesteps, timesteps_list
            if q_pipe.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        _remove_hooks(stat_handles)
        _remove_hooks(cast_handles)

    metadata = {
        "ckpt_dir": args.ckpt_dir,
        "quant_ckpt_dir": quant_ckpt_dir,
        "ptq_dir": args.ptq_dir,
        "manifest": args.manifest,
        "max_samples": int(args.max_samples),
        "sample_steps": int(args.sample_steps),
        "size": args.size,
        "seed": int(args.seed),
        "experts": _expert_names(args.experts),
        "sample_solver": args.sample_solver,
        "shift": shift,
        "target_regex": "" if pattern is None else pattern.pattern,
        "target_modules": target_modules,
        "low_keep_fp_blocks": args.low_keep_fp_blocks,
        "high_keep_fp_blocks": args.high_keep_fp_blocks,
        "forwards_per_denoise_step": {"quant": 1},
        "definition": "activation_qdq: compare QDQ(x_smooth) vs x_smooth during quantized model real forward",
    }
    LOGGER.info("[4/4] writing activation QDQ error reports")
    _write_outputs(Path(args.out_dir), collector, metadata)


if __name__ == "__main__":
    main()
