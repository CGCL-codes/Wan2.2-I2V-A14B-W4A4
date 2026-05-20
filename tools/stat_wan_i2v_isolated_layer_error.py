#!/usr/bin/env python3
"""Stat isolated Linear quantization error for Wan I2V DiT.

For each target Linear layer, this script uses the BF16 model's real layer
input as the shared input:

    x_ref = input seen by BF16 Linear
    y_ref = BF16 Linear(x_ref)
    y_q   = quantized_layer(x_ref)

Only scalar reductions are accumulated in hooks. Full activations are never
saved to CPU, disk, or long-lived Python containers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from quant.quant_linear import QuantLinear, QuantLinearWithBranch
from quant.wan_smooth import _build_schedule, _register_linear_input_cast_hooks
from tools.infer_wan_i2v_svdquant import (
    _apply_ptq_state_to_model,
    _load_i2v_ptq_states,
    _parse_keep_fp_blocks,
)
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger("stat_wan_i2v_isolated_layer_error")


@dataclass(frozen=True)
class ManifestSample:
    image: Path
    prompt: str
    sample_id: str


class ErrorStats:
    """Streaming scalar error stats for one isolated layer."""

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
    def update(self, y_ref: torch.Tensor, y_q: torch.Tensor, step_idx: Optional[int]) -> None:
        ref = y_ref.detach().to(torch.float32)
        q = y_q.detach().to(torch.float32)
        err = q - ref
        n = int(err.numel())
        if n == 0:
            return

        self.calls += 1
        self.numel += n
        self.sum_abs_err += float(err.abs().sum().item())
        self.sum_sq_err += float((err * err).sum().item())
        self.max_abs_err = max(self.max_abs_err, float(err.abs().amax().item()))
        self.sum_ref_abs += float(ref.abs().sum().item())
        self.sum_q_abs += float(q.abs().sum().item())
        self.sum_ref_sq += float((ref * ref).sum().item())
        self.sum_q_sq += float((q * q).sum().item())
        self.sum_dot += float((ref * q).sum().item())
        self.ref_absmax = max(self.ref_absmax, float(ref.abs().amax().item()))
        self.q_absmax = max(self.q_absmax, float(q.abs().amax().item()))
        if step_idx is not None:
            step_idx = int(step_idx)
            self.first_step = step_idx if self.first_step is None else min(self.first_step, step_idx)
            self.last_step = step_idx if self.last_step is None else max(self.last_step, step_idx)

    def to_row(self, expert: str, module_name: str) -> Dict[str, Any]:
        denom = max(1, self.numel)
        mse = self.sum_sq_err / denom
        rmse = math.sqrt(mse)
        rel_l2 = math.sqrt(self.sum_sq_err / max(self.sum_ref_sq, 1e-30))
        cosine = self.sum_dot / math.sqrt(max(self.sum_ref_sq * self.sum_q_sq, 1e-30))
        return {
            "expert": expert,
            "module": module_name,
            "calls": int(self.calls),
            "numel": int(self.numel),
            "mae": self.sum_abs_err / denom,
            "mse": mse,
            "rmse": rmse,
            "rel_l2": rel_l2,
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


class IsolatedErrorCollector:
    def __init__(self, q_modules: Dict[str, Dict[str, nn.Module]], device: torch.device) -> None:
        self.q_modules = q_modules
        self.device = device
        self.current_step: Optional[int] = None
        self.current_timestep: Optional[float] = None
        self.current_expert: Optional[str] = None
        self.stats: Dict[str, Dict[str, ErrorStats]] = {"high": {}, "low": {}}
        self.raw_timesteps: Dict[str, Dict[int, float]] = {"high": {}, "low": {}}

    def begin_step(self, step_idx: int, raw_timestep: float, expert: str) -> None:
        self.current_step = int(step_idx)
        self.current_timestep = float(raw_timestep)
        self.current_expert = expert
        self.raw_timesteps[expert][int(step_idx)] = float(raw_timestep)

    @torch.no_grad()
    def compare(self, expert: str, module_name: str, x_ref: torch.Tensor, y_ref: torch.Tensor) -> None:
        if self.current_expert != expert:
            return
        q_layer = self.q_modules.get(expert, {}).get(module_name)
        if q_layer is None:
            return
        if next(q_layer.parameters(), None) is not None:
            param = next(q_layer.parameters())
            if param.device != x_ref.device:
                q_layer.to(x_ref.device)
        else:
            # QuantLinear may store only buffers when bias=False and branch=None.
            q_layer.to(x_ref.device)
        y_q = q_layer(x_ref.detach())
        if not torch.is_tensor(y_q):
            raise TypeError(f"Quant layer output is not tensor for {expert}.{module_name}: {type(y_q).__name__}")
        rec = self.stats[expert].setdefault(module_name, ErrorStats())
        rec.update(y_ref=y_ref, y_q=y_q, step_idx=self.current_step)
        del y_q


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stat Wan I2V isolated Linear quantization error")
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
    p.add_argument("--list_linears", action="store_true", default=False)
    p.add_argument("--low_keep_fp_blocks", type=str, default="")
    p.add_argument("--high_keep_fp_blocks", type=str, default="")
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--offload_ref_models", action="store_true", default=True)
    p.add_argument("--no_offload_ref_models", action="store_false", dest="offload_ref_models")
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _expert_names(arg: str) -> List[str]:
    return ["high", "low"] if arg == "both" else [arg]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def _parse_csv_text(text: str) -> List[str]:
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def _build_filter_regex(args: argparse.Namespace) -> Optional[re.Pattern[str]]:
    if args.target_regex:
        return re.compile(args.target_regex)
    blocks = _parse_csv_ints(args.blocks)
    modules = _parse_csv_text(args.modules)
    if not blocks and not modules:
        return None
    if not blocks or not modules:
        raise ValueError("--blocks and --modules must be provided together when filtering by block/module")
    block_pat = "|".join(str(b) for b in blocks)
    module_pat = "|".join(re.escape(m) for m in modules)
    return re.compile(rf"(^|\.)blocks\.({block_pat})\.({module_pat})$")


def _size_to_hw(size: str) -> Tuple[int, int]:
    w, h = [int(x) for x in size.split("*")]
    return h, w


def _load_manifest(path: Path, max_samples: int) -> List[ManifestSample]:
    samples: List[ManifestSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_samples > 0 and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = str(rec.get("prompt", ""))
            raw_image = rec.get("used_image") or rec.get("image") or rec.get("image_path")
            if raw_image is None and isinstance(rec.get("img_paths"), list) and rec["img_paths"]:
                raw_image = rec["img_paths"][0]
            if raw_image is None:
                LOGGER.warning("manifest line %d missing image path; skipped", line_idx + 1)
                continue
            image = Path(str(raw_image))
            if not image.is_absolute():
                candidates = [path.parent / image, Path(PROJECT_ROOT) / image, Path(PROJECT_ROOT) / "OpenS2V-Eval" / image]
                image = next((c for c in candidates if c.exists()), candidates[0])
            if not image.exists():
                LOGGER.warning("manifest line %d image not found: %s; skipped", line_idx + 1, image)
                continue
            sample_id = str(rec.get("videoid") or rec.get("id") or rec.get("name") or f"sample_{line_idx:06d}")
            samples.append(ManifestSample(image=image.resolve(), prompt=prompt, sample_id=sample_id))
    return samples


def _build_mask(frame_num: int, lat_h: int, lat_w: int, device: torch.device) -> torch.Tensor:
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    return msk.transpose(1, 2)[0]


def _prep_sample(pipe: wan.WanI2V, sample: ManifestSample, frame_num: int, target_hw: Tuple[int, int]) -> Dict[str, Any]:
    img = Image.open(sample.image).convert("RGB")
    cond_image = TF.to_tensor(img)
    cond_image = TF.resize(cond_image, size=list(target_hw), antialias=True)
    cond_image = TF.center_crop(cond_image, list(target_hw))

    c, h, w = int(cond_image.shape[0]), int(cond_image.shape[1]), int(cond_image.shape[2])
    cond_cfthw = torch.zeros((c, frame_num, h, w), device=pipe.device, dtype=torch.float32)
    cond_cfthw[:, 0] = cond_image.to(device=pipe.device, dtype=torch.float32) * 2.0 - 1.0

    cond_latent = pipe.vae.encode([cond_cfthw])[0]
    latent = torch.randn_like(cond_latent)
    _, f_lat, lat_h, lat_w = cond_latent.shape
    msk = _build_mask(frame_num=frame_num, lat_h=lat_h, lat_w=lat_w, device=pipe.device)
    y = torch.cat([msk, cond_latent], dim=0)
    seq_len = f_lat * lat_h * lat_w // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int(math.ceil(seq_len / pipe.sp_size)) * pipe.sp_size

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([sample.prompt], pipe.device)
        pipe.text_encoder.model.cpu()
        context = [t.cpu() for t in context]
    else:
        context = [t.cpu() for t in pipe.text_encoder([sample.prompt], torch.device("cpu"))]

    return {"latent": latent.cpu(), "y": y.cpu(), "seq_len": int(seq_len), "context": context}


def _build_pipe(ckpt_dir: str, args: argparse.Namespace) -> wan.WanI2V:
    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    pipe.high_noise_model.eval().requires_grad_(False)
    pipe.low_noise_model.eval().requires_grad_(False)
    return pipe


def _offload_unused_quant_parts(pipe: wan.WanI2V) -> None:
    if hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder, "model"):
        pipe.text_encoder.model.cpu()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "model"):
        pipe.vae.model.cpu()
        pipe.vae.mean = pipe.vae.mean.cpu()
        pipe.vae.std = pipe.vae.std.cpu()
        pipe.vae.scale = [pipe.vae.mean, 1.0 / pipe.vae.std]
    pipe.high_noise_model.cpu()
    pipe.low_noise_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _quant_module_names(model: nn.Module) -> List[str]:
    out = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinearWithBranch, QuantLinear)):
            out.append(name)
    return out


def _find_module(root: nn.Module, full_name: str) -> Optional[nn.Module]:
    cur: nn.Module = root
    for part in full_name.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _select_targets(
    ref_model: nn.Module,
    q_model: nn.Module,
    pattern: Optional[re.Pattern[str]],
) -> Tuple[List[str], Dict[str, nn.Module]]:
    names: List[str] = []
    q_layers: Dict[str, nn.Module] = {}
    for name in _quant_module_names(q_model):
        if pattern is not None and not pattern.search(name):
            continue
        ref_layer = _find_module(ref_model, name)
        q_layer = _find_module(q_model, name)
        if not isinstance(ref_layer, nn.Linear):
            LOGGER.warning("Skipping %s because BF16 module is not nn.Linear: %s", name, type(ref_layer).__name__)
            continue
        if not isinstance(q_layer, (QuantLinearWithBranch, QuantLinear)):
            continue
        if int(ref_layer.in_features) != int(q_layer.in_features) or int(ref_layer.out_features) != int(q_layer.out_features):
            LOGGER.warning("Skipping %s due to shape mismatch", name)
            continue
        names.append(name)
        q_layers[name] = q_layer
    return names, q_layers


def _register_hooks(
    model: nn.Module,
    expert: str,
    module_names: Sequence[str],
    collector: IsolatedErrorCollector,
) -> List[Any]:
    handles = []
    module_map = dict(model.named_modules())
    for name in module_names:
        module = module_map.get(name)
        if not isinstance(module, nn.Linear):
            LOGGER.warning("%s target is not nn.Linear: %s", expert, name)
            continue

        def _make_hook(module_name: str):
            def _hook(_module: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
                if not inp or not torch.is_tensor(inp[0]) or not torch.is_tensor(out):
                    return
                collector.compare(expert=expert, module_name=module_name, x_ref=inp[0], y_ref=out)

            return _hook

        handles.append(module.register_forward_hook(_make_hook(name), with_kwargs=False))
    return handles


def _write_outputs(out_dir: Path, collector: IsolatedErrorCollector, metadata: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for expert in ("high", "low"):
        for module_name, rec in sorted(collector.stats[expert].items()):
            rows.append(rec.to_row(expert, module_name))

    csv_path = out_dir / "isolated_layer_error.csv"
    jsonl_path = out_dir / "isolated_layer_error.jsonl"
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
        "mean_ref_abs",
        "mean_q_abs",
        "ref_absmax",
        "q_absmax",
        "cosine",
        "cosine_error",
        "first_step",
        "last_step",
    ]
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
    q_modules: Dict[str, Dict[str, nn.Module]] = {"high": {}, "low": {}}
    for expert in _expert_names(args.experts):
        names, qmap = _select_targets(ref_models[expert], q_models[expert], pattern)
        target_modules[expert] = names
        q_modules[expert] = qmap
        LOGGER.info("%s target quantized Linear modules: %d", expert, len(names))
        for name in names:
            LOGGER.info("%s | %s", expert, name)
        for q_layer in qmap.values():
            q_layer.to(ref_pipe.device).eval().requires_grad_(False)

    if args.list_linears:
        return
    if not any(target_modules.values()):
        raise ValueError("No target quantized Linear modules found.")

    collector = IsolatedErrorCollector(q_modules=q_modules, device=ref_pipe.device)
    handles: List[Any] = []
    cast_handles: List[Any] = []
    try:
        for expert in _expert_names(args.experts):
            cast_handles.extend(_register_linear_input_cast_hooks(ref_models[expert]))
            handles.extend(_register_hooks(ref_models[expert], expert, target_modules[expert], collector))
        LOGGER.info("[3/5] registered BF16 forward hooks: %d", len(handles))

        cfg = WAN_CONFIGS["i2v-A14B"]
        frame_num = cfg.frame_num
        shift = cfg.sample_shift if args.shift is None else float(args.shift)
        boundary = float(ref_pipe.boundary * ref_pipe.num_train_timesteps)
        use_amp = ref_pipe.device.type == "cuda" and ref_pipe.param_dtype in (torch.float16, torch.bfloat16)

        sample_iter: Iterable[ManifestSample] = samples
        if tqdm is not None and not args.disable_tqdm:
            sample_iter = tqdm(samples, desc="isolated error samples", unit="sample", dynamic_ncols=True)

        LOGGER.info("[4/5] running BF16 denoise trajectory and isolated layer comparisons")
        active_ref_expert: Optional[str] = None
        for sample_idx, sample in enumerate(sample_iter):
            torch.manual_seed(int(args.seed) + sample_idx)
            item = _prep_sample(ref_pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))
            latent = item["latent"].to(device=ref_pipe.device, dtype=ref_pipe.param_dtype)
            y = item["y"].to(device=ref_pipe.device, dtype=ref_pipe.param_dtype)
            context = [t.to(device=ref_pipe.device, dtype=ref_pipe.param_dtype) for t in item["context"]]
            model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}
            scheduler, timesteps = _build_schedule(
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
                    if args.offload_ref_models and active_ref_expert != expert:
                        for name, model in ref_models.items():
                            model.to(ref_pipe.device if name == expert else "cpu")
                        active_ref_expert = expert
                        if ref_pipe.device.type == "cuda":
                            torch.cuda.empty_cache()
                    elif not args.offload_ref_models:
                        ref_models[expert].to(ref_pipe.device)

                    collector.begin_step(step_idx=step_idx, raw_timestep=t_scalar, expert=expert)
                    timestep = torch.tensor([int(t_scalar)], device=ref_pipe.device, dtype=torch.long)
                    noise_pred = ref_models[expert]([latent], t=timestep, **model_args)[0]
                    latent = scheduler.step(
                        noise_pred.unsqueeze(0),
                        torch.tensor(t_scalar, device=ref_pipe.device, dtype=timesteps.dtype),
                        latent.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)

            del item, latent, y, context, model_args, scheduler, timesteps, timesteps_list
            if ref_pipe.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
        for handle in cast_handles:
            handle.remove()

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
    }
    LOGGER.info("[5/5] writing isolated layer error reports")
    _write_outputs(Path(args.out_dir), collector, metadata)


if __name__ == "__main__":
    main()
