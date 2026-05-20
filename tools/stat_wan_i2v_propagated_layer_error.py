#!/usr/bin/env python3
"""Stat propagated Linear input/output error for Wan I2V DiT.

Unlike isolated layer error, this script lets the BF16 and quantized models run
their own full forward paths. For each target module it compares:

    input_error  = x_q - x_ref
    output_error = y_q - y_ref

where x_ref/y_ref come from the BF16 model's real forward, and x_q/y_q come
from the quantized model's real forward after earlier quantization errors have
already propagated.

For each denoising step, the BF16 model is forwarded once and the quantized
model is forwarded once. Target-layer BF16 x/y tensors are held only until the
paired quantized forward finishes, then they are immediately reduced to scalar
stats and discarded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import re
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
    _offload_unused_quant_parts,
    _prep_sample,
    _select_targets,
    _size_to_hw,
)
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger("stat_wan_i2v_propagated_layer_error")


class TensorErrorStats:
    """Streaming scalar error stats for one tensor stream."""

    def __init__(self) -> None:
        self.calls = 0
        self.numel = 0
        self.sum_abs_err = 0.0
        self.sum_sq_err = 0.0
        self.max_abs_err = 0.0
        self.sum_ref_abs = 0.0
        self.sum_q_abs = 0.0
        self.sum_ref_sq = 0.0
        self.sum_q_sq = 0.0
        self.sum_dot = 0.0
        self.ref_absmax = 0.0
        self.q_absmax = 0.0
        self.first_step: Optional[int] = None
        self.last_step: Optional[int] = None

    @torch.no_grad()
    def update(self, ref: torch.Tensor, q: torch.Tensor, step_idx: Optional[int]) -> None:
        if tuple(ref.shape) != tuple(q.shape):
            raise ValueError(f"Shape mismatch while comparing propagated tensors: ref={tuple(ref.shape)} q={tuple(q.shape)}")
        ref32 = ref.detach().to(device="cpu", dtype=torch.float32)
        q32 = q.detach().to(device="cpu", dtype=torch.float32)
        err = q32 - ref32
        n = int(err.numel())
        if n == 0:
            return

        self.calls += 1
        self.numel += n
        abs_err = err.abs()
        self.sum_abs_err += float(abs_err.sum().item())
        self.sum_sq_err += float((err * err).sum().item())
        self.max_abs_err = max(self.max_abs_err, float(abs_err.amax().item()))
        self.sum_ref_abs += float(ref32.abs().sum().item())
        self.sum_q_abs += float(q32.abs().sum().item())
        self.sum_ref_sq += float((ref32 * ref32).sum().item())
        self.sum_q_sq += float((q32 * q32).sum().item())
        self.sum_dot += float((ref32 * q32).sum().item())
        self.ref_absmax = max(self.ref_absmax, float(ref32.abs().amax().item()))
        self.q_absmax = max(self.q_absmax, float(q32.abs().amax().item()))
        if step_idx is not None:
            step_idx = int(step_idx)
            self.first_step = step_idx if self.first_step is None else min(self.first_step, step_idx)
            self.last_step = step_idx if self.last_step is None else max(self.last_step, step_idx)

    def to_row(self, expert: str, module_name: str, tensor_kind: str) -> Dict[str, Any]:
        denom = max(1, self.numel)
        mse = self.sum_sq_err / denom
        cosine = self.sum_dot / math.sqrt(max(self.sum_ref_sq * self.sum_q_sq, 1e-30))
        return {
            "expert": expert,
            "module": module_name,
            "tensor": tensor_kind,
            "calls": int(self.calls),
            "numel": int(self.numel),
            "mae": self.sum_abs_err / denom,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "rel_l2": math.sqrt(self.sum_sq_err / max(self.sum_ref_sq, 1e-30)),
            "max_abs_error": self.max_abs_err,
            "mean_ref_abs": self.sum_ref_abs / denom,
            "mean_q_abs": self.sum_q_abs / denom,
            "ref_absmax": self.ref_absmax,
            "q_absmax": self.q_absmax,
            "cosine": cosine,
            "cosine_error": 1.0 - cosine,
            "first_step": "" if self.first_step is None else int(self.first_step),
            "last_step": "" if self.last_step is None else int(self.last_step),
        }


class PropagatedErrorCollector:
    def __init__(
        self,
        capture_device: str = "cpu",
        capture_dtype: str = "bf16",
        max_tokens_per_layer: int = 2048,
    ) -> None:
        self.current_step: Optional[int] = None
        self.current_expert: Optional[str] = None
        self.raw_timesteps: Dict[str, Dict[int, float]] = {"high": {}, "low": {}}
        self.capture_device = torch.device(capture_device)
        self.capture_dtype = _resolve_capture_dtype(capture_dtype)
        self.max_tokens_per_layer = int(max_tokens_per_layer)
        self.ref_cache: Dict[str, Dict[str, Any]] = {}
        self.input_stats: Dict[str, Dict[str, TensorErrorStats]] = {"high": {}, "low": {}}
        self.output_stats: Dict[str, Dict[str, TensorErrorStats]] = {"high": {}, "low": {}}

    def begin_step(self, step_idx: int, raw_timestep: float, expert: str) -> None:
        self.current_step = int(step_idx)
        self.current_expert = expert
        self.raw_timesteps[expert][int(step_idx)] = float(raw_timestep)

    def clear_cache(self) -> None:
        self.ref_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def capture_ref(self, module_name: str, x_ref: torch.Tensor, y_ref: torch.Tensor) -> None:
        x_ref_sampled, x_token_idx = _sample_tokens_for_cache(x_ref, self.max_tokens_per_layer)
        y_ref_sampled, y_token_idx = _sample_tokens_for_cache(y_ref, self.max_tokens_per_layer)
        self.ref_cache[module_name] = {
            "x": x_ref_sampled.detach().to(device=self.capture_device, dtype=self.capture_dtype).contiguous(),
            "y": y_ref_sampled.detach().to(device=self.capture_device, dtype=self.capture_dtype).contiguous(),
            "x_token_idx": x_token_idx,
            "y_token_idx": y_token_idx,
        }

    @torch.no_grad()
    def compare_quant(self, expert: str, module_name: str, x_q: torch.Tensor, y_q: torch.Tensor) -> None:
        cached = self.ref_cache.get(module_name)
        if cached is None:
            return
        x_q_sampled = _apply_token_index(x_q, cached["x_token_idx"])
        y_q_sampled = _apply_token_index(y_q, cached["y_token_idx"])
        x_q_sampled = x_q_sampled.detach().to(device=self.capture_device, dtype=self.capture_dtype).contiguous()
        y_q_sampled = y_q_sampled.detach().to(device=self.capture_device, dtype=self.capture_dtype).contiguous()
        in_rec = self.input_stats[expert].setdefault(module_name, TensorErrorStats())
        out_rec = self.output_stats[expert].setdefault(module_name, TensorErrorStats())
        in_rec.update(ref=cached["x"], q=x_q_sampled, step_idx=self.current_step)
        out_rec.update(ref=cached["y"], q=y_q_sampled, step_idx=self.current_step)


def _resolve_capture_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if key in ("float32", "fp32", "torch.float32"):
        return torch.float32
    raise ValueError(f"Unsupported --capture_dtype: {name}")


def _token_indices(num_tokens: int, max_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
    if max_tokens <= 0 or num_tokens <= max_tokens:
        return None
    return torch.linspace(0, num_tokens - 1, steps=max_tokens, device=device).round().to(torch.long).unique(sorted=True)


def _sample_tokens_for_cache(x: torch.Tensor, max_tokens: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if x.ndim == 3:
        num_tokens = int(x.shape[1])
        idx = _token_indices(num_tokens, max_tokens, x.device)
        if idx is not None:
            return x.index_select(1, idx), idx.detach().cpu()
    return x, None


def _apply_token_index(x: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
    if idx is None:
        return x
    if x.ndim != 3:
        return x
    return x.index_select(1, idx.to(device=x.device, dtype=torch.long))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stat Wan I2V propagated Linear input/output error")
    p.add_argument("--ckpt_dir", required=True, help="BF16 Wan I2V checkpoint directory")
    p.add_argument("--quant_ckpt_dir", default="", help="Checkpoint directory used to instantiate the quantized model; defaults to --ckpt_dir")
    p.add_argument("--ptq_dir", required=True, help="ptq_stats.pt path/dir or legacy split PTQ directory")
    p.add_argument("--manifest", required=True, help="jsonl manifest with image/used_image/img_paths and prompt")
    p.add_argument("--out_dir", required=True)
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
    p.add_argument("--module_chunk_size", type=int, default=0, help="Deprecated and ignored; each denoise step now runs one BF16 forward and one quant forward.")
    p.add_argument("--list_linears", action="store_true", default=False)
    p.add_argument("--low_keep_fp_blocks", type=str, default="")
    p.add_argument("--high_keep_fp_blocks", type=str, default="")
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--offload_models", action="store_true", default=True)
    p.add_argument("--no_offload_models", action="store_false", dest="offload_models")
    p.add_argument("--capture_device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--capture_dtype", choices=["bf16", "float32"], default="bf16")
    p.add_argument("--max_tokens_per_layer", type=int, default=2048)
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _register_ref_hooks(model: nn.Module, module_names: Sequence[str], collector: PropagatedErrorCollector) -> List[Any]:
    handles = []
    module_map = dict(model.named_modules())
    for name in module_names:
        module = module_map.get(name)
        if not isinstance(module, nn.Linear):
            LOGGER.warning("BF16 target is not nn.Linear: %s", name)
            continue

        def _make_hook(module_name: str):
            def _hook(_module: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
                if not inp or not torch.is_tensor(inp[0]) or not torch.is_tensor(out):
                    return
                collector.capture_ref(module_name, inp[0], out)

            return _hook

        handles.append(module.register_forward_hook(_make_hook(name), with_kwargs=False))
    return handles


def _register_quant_hooks(
    model: nn.Module,
    expert: str,
    module_names: Sequence[str],
    collector: PropagatedErrorCollector,
) -> List[Any]:
    handles = []
    module_map = dict(model.named_modules())
    for name in module_names:
        module = module_map.get(name)
        if not isinstance(module, (QuantLinearWithBranch, QuantLinear)):
            LOGGER.warning("Quant target is not QuantLinear wrapper: %s (%s)", name, type(module).__name__)
            continue

        def _make_hook(module_name: str):
            def _hook(_module: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
                if not inp or not torch.is_tensor(inp[0]) or not torch.is_tensor(out):
                    return
                collector.compare_quant(expert, module_name, inp[0], out)

            return _hook

        handles.append(module.register_forward_hook(_make_hook(name), with_kwargs=False))
    return handles


def _remove_hooks(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


def _run_model(model: nn.Module, latent: torch.Tensor, timestep: torch.Tensor, model_args: Dict[str, Any]) -> torch.Tensor:
    return model([latent], t=timestep, **model_args)[0]


def _write_outputs(out_dir: Path, collector: PropagatedErrorCollector, metadata: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for expert in ("high", "low"):
        for module_name, rec in sorted(collector.input_stats[expert].items()):
            rows.append(rec.to_row(expert, module_name, "input"))
        for module_name, rec in sorted(collector.output_stats[expert].items()):
            rows.append(rec.to_row(expert, module_name, "output"))

    fieldnames = [
        "expert",
        "module",
        "tensor",
        "calls",
        "numel",
        "mae",
        "mse",
        "rmse",
        "rel_l2",
        "max_abs_error",
        "mean_ref_abs",
        "mean_q_abs",
        "ref_absmax",
        "q_absmax",
        "cosine",
        "cosine_error",
        "first_step",
        "last_step",
    ]
    csv_path = out_dir / "propagated_layer_error.csv"
    jsonl_path = out_dir / "propagated_layer_error.jsonl"
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
    metadata["num_reported_rows"] = len(rows)
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
    if args.max_tokens_per_layer < 0:
        raise ValueError("--max_tokens_per_layer must be >= 0")
    if args.capture_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--capture_device=cuda requested but CUDA is not available")

    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found.")
    pattern = _build_filter_regex(args)
    LOGGER.info("Target filter: %s", "all quantized Linear modules" if pattern is None else pattern.pattern)

    quant_ckpt_dir = args.quant_ckpt_dir or args.ckpt_dir
    LOGGER.info("[1/5] loading quant model skeleton and PTQ states")
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
    _offload_unused_quant_parts(q_pipe)

    LOGGER.info("[2/5] loading BF16 reference pipeline")
    ref_pipe = _build_pipe(args.ckpt_dir, args)
    ref_pipe.high_noise_model.cpu()
    ref_pipe.low_noise_model.cpu()

    ref_models = {"high": ref_pipe.high_noise_model, "low": ref_pipe.low_noise_model}
    q_models = {"high": q_pipe.high_noise_model, "low": q_pipe.low_noise_model}
    target_modules: Dict[str, List[str]] = {}
    for expert in _expert_names(args.experts):
        names, _ = _select_targets(ref_models[expert], q_models[expert], pattern)
        target_modules[expert] = names
        LOGGER.info("%s target quantized Linear modules: %d", expert, len(names))
        for name in names:
            LOGGER.info("%s | %s", expert, name)

    if args.list_linears:
        return
    if not any(target_modules.values()):
        raise ValueError("No target quantized Linear modules found.")

    cast_handles = []
    for expert in ("high", "low"):
        cast_handles.extend(_register_linear_input_cast_hooks(ref_models[expert]))
    collector = PropagatedErrorCollector(
        capture_device=args.capture_device,
        capture_dtype=args.capture_dtype,
        max_tokens_per_layer=int(args.max_tokens_per_layer),
    )

    try:
        stat_handles: List[Any] = []
        for expert in _expert_names(args.experts):
            stat_handles.extend(_register_ref_hooks(ref_models[expert], target_modules[expert], collector))
            stat_handles.extend(_register_quant_hooks(q_models[expert], expert, target_modules[expert], collector))
        LOGGER.info("[3/5] registered propagated error hooks: %d", len(stat_handles))

        cfg = WAN_CONFIGS["i2v-A14B"]
        frame_num = cfg.frame_num
        shift = cfg.sample_shift if args.shift is None else float(args.shift)
        boundary = float(ref_pipe.boundary * ref_pipe.num_train_timesteps)
        use_amp = ref_pipe.device.type == "cuda" and ref_pipe.param_dtype in (torch.float16, torch.bfloat16)

        sample_iter: Iterable[ManifestSample] = samples
        if tqdm is not None and not args.disable_tqdm:
            sample_iter = tqdm(samples, desc="propagated error samples", unit="sample", dynamic_ncols=True)

        LOGGER.info("[3/5] running paired BF16/quant denoise trajectories")
        active_expert: Optional[str] = None
        for sample_idx, sample in enumerate(sample_iter):
            torch.manual_seed(int(args.seed) + sample_idx)
            item = _prep_sample(ref_pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))
            ref_latent = item["latent"].to(device=ref_pipe.device, dtype=ref_pipe.param_dtype)
            q_latent = ref_latent.detach().clone()
            y = item["y"].to(device=ref_pipe.device, dtype=ref_pipe.param_dtype)
            context = [t.to(device=ref_pipe.device, dtype=ref_pipe.param_dtype) for t in item["context"]]
            model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}
            ref_scheduler, timesteps = _build_schedule(
                sample_solver=args.sample_solver,
                sampling_steps=int(args.sample_steps),
                shift=shift,
                num_train_timesteps=ref_pipe.num_train_timesteps,
                device=ref_pipe.device,
            )
            q_scheduler, _ = _build_schedule(
                sample_solver=args.sample_solver,
                sampling_steps=int(args.sample_steps),
                shift=shift,
                num_train_timesteps=ref_pipe.num_train_timesteps,
                device=ref_pipe.device,
            )
            timesteps_list = [float(t.item()) for t in timesteps]
            step_iter: Iterable[Tuple[int, float]] = list(enumerate(timesteps_list))
            if tqdm is not None and not args.disable_tqdm:
                step_iter = tqdm(step_iter, desc=f"{sample.sample_id} denoise", unit="step", leave=False, dynamic_ncols=True)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=ref_pipe.param_dtype, enabled=use_amp):
                for step_idx, t_scalar in step_iter:
                    expert = "high" if t_scalar >= boundary else "low"
                    if args.offload_models and active_expert != expert:
                        for name in ("high", "low"):
                            ref_models[name].to(ref_pipe.device if name == expert else "cpu")
                            q_models[name].to(ref_pipe.device if name == expert else "cpu")
                        active_expert = expert
                        if ref_pipe.device.type == "cuda":
                            torch.cuda.empty_cache()
                    elif not args.offload_models:
                        ref_models[expert].to(ref_pipe.device)
                        q_models[expert].to(ref_pipe.device)

                    timestep = torch.tensor([int(t_scalar)], device=ref_pipe.device, dtype=torch.long)
                    t_for_step = torch.tensor(t_scalar, device=ref_pipe.device, dtype=timesteps.dtype)
                    collector.begin_step(step_idx=step_idx, raw_timestep=t_scalar, expert=expert)
                    collector.clear_cache()
                    ref_noise_pred = _run_model(ref_models[expert], ref_latent, timestep, model_args)
                    q_noise_pred = _run_model(q_models[expert], q_latent, timestep, model_args)
                    collector.clear_cache()

                    ref_latent = ref_scheduler.step(
                        ref_noise_pred.unsqueeze(0),
                        t_for_step,
                        ref_latent.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)
                    q_latent = q_scheduler.step(
                        q_noise_pred.unsqueeze(0),
                        t_for_step,
                        q_latent.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)
                    del ref_noise_pred, q_noise_pred

            del item, ref_latent, q_latent, y, context, model_args, ref_scheduler, q_scheduler, timesteps, timesteps_list
            if ref_pipe.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if "stat_handles" in locals():
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
        "module_chunk_size": int(args.module_chunk_size),
        "module_chunk_size_ignored": True,
        "forwards_per_denoise_step": {"bf16": 1, "quant": 1},
        "capture_device": args.capture_device,
        "capture_dtype": args.capture_dtype,
        "max_tokens_per_layer": int(args.max_tokens_per_layer),
        "low_keep_fp_blocks": args.low_keep_fp_blocks,
        "high_keep_fp_blocks": args.high_keep_fp_blocks,
        "definition": "propagated: BF16 and quant models use their own real forward inputs/outputs",
    }
    LOGGER.info("[5/5] writing propagated layer error reports")
    _write_outputs(Path(args.out_dir), collector, metadata)


if __name__ == "__main__":
    main()
