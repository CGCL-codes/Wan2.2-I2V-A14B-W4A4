#!/usr/bin/env python3
"""Collect Wan I2V DiT Linear input activation ranges and plot 3D surfaces.

The script streams per-channel statistics from forward_pre_hooks. It never
stores full activation tensors; hook bodies reduce input[0] to length-C CPU
statistics immediately.
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

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from quant.wan_smooth import _build_schedule, _register_linear_input_cast_hooks
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger("stat_wan_i2v_activation_surface")


@dataclass(frozen=True)
class ManifestSample:
    image: Path
    prompt: str
    sample_id: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stat Wan I2V DiT Linear activation surfaces")
    p.add_argument("-ckpt_dir", "--ckpt_dir", required=True)
    p.add_argument("-manifest", "--manifest", required=True, help="jsonl manifest with image/used_image/img_paths and prompt")
    p.add_argument("-max_samples", "--max_samples", type=int, default=1)
    p.add_argument("-sample_steps", "--sample_steps", type=int, default=5)
    p.add_argument("-size", "--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("-seed", "--seed", type=int, default=42)
    p.add_argument("-out_dir", "--out_dir", required=True)
    p.add_argument("-blocks", "--blocks", type=str, default="0,10,20,30,39")
    p.add_argument(
        "-modules",
        "--modules",
        type=str,
        default="self_attn.q,self_attn.o,cross_attn.q,cross_attn.o,ffn.0,ffn.2",
    )
    p.add_argument("-experts", "--experts", type=str, default="both", choices=["high", "low", "both"])
    p.add_argument("-num_channels", "--num_channels", type=int, default=64)
    p.add_argument("-channel_mode", "--channel_mode", type=str, default="top_absmean", choices=["first", "stride", "top_absmean"])
    p.add_argument("-stat", "--stat", type=str, default="log2_absmax", choices=["range", "absmax", "log2_range", "log2_absmax"])
    p.add_argument("-target_regex", "--target_regex", type=str, default="")
    p.add_argument("-save_npz", "--save_npz", action="store_true", default=False)
    p.add_argument("-plot", "--plot", action="store_true", default=False)
    p.add_argument("-list_linears", "--list_linears", action="store_true", default=False)
    p.add_argument("-device_id", "--device_id", type=int, default=0)
    p.add_argument("-sample_solver", "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("-shift", "--shift", type=float, default=None)
    p.add_argument("-t5_cpu", "--t5_cpu", action="store_true", default=False)
    p.add_argument("-convert_model_dtype", "--convert_model_dtype", action="store_true", default=True)
    p.add_argument("-disable_tqdm", "--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_csv_text(text: str) -> List[str]:
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def _build_target_regex(blocks: Sequence[int], modules: Sequence[str]) -> re.Pattern[str]:
    block_pat = "|".join(str(int(b)) for b in blocks)
    module_pat = "|".join(re.escape(m) for m in modules)
    pattern = rf"(^|\.)blocks\.({block_pat})\.({module_pat})$"
    return re.compile(pattern)


def _size_to_hw(size: str) -> Tuple[int, int]:
    w, h = [int(x) for x in size.split("*")]
    return h, w


def _expected_module_names(blocks: Sequence[int], modules: Sequence[str]) -> List[str]:
    return [f"blocks.{int(block)}.{module}" for block in blocks for module in modules]


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
                candidates = [path.parent / image, PROJECT_ROOT / image, PROJECT_ROOT / "OpenS2V-Eval" / image]
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
    msk = msk.transpose(1, 2)[0]
    return msk


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


class ModuleStepStats:
    def __init__(self, sample_steps: int, channels: int) -> None:
        self.sample_steps = int(sample_steps)
        self.channels = int(channels)
        shape = (self.sample_steps, self.channels)
        self.max_absmax = torch.zeros(shape, dtype=torch.float32)
        self.max_range = torch.zeros(shape, dtype=torch.float32)
        self.sum_absmax = torch.zeros(shape, dtype=torch.float32)
        self.sum_range = torch.zeros(shape, dtype=torch.float32)
        self.sum_rms = torch.zeros(shape, dtype=torch.float32)
        self.sample_count = torch.zeros(self.sample_steps, dtype=torch.int64)

    def update(self, step_idx: int, ch_range: torch.Tensor, ch_absmax: torch.Tensor, ch_rms: torch.Tensor) -> None:
        ch_range = ch_range.to(dtype=torch.float32, device="cpu")
        ch_absmax = ch_absmax.to(dtype=torch.float32, device="cpu")
        ch_rms = ch_rms.to(dtype=torch.float32, device="cpu")
        self.max_absmax[step_idx] = torch.maximum(self.max_absmax[step_idx], ch_absmax)
        self.max_range[step_idx] = torch.maximum(self.max_range[step_idx], ch_range)
        self.sum_absmax[step_idx] += ch_absmax
        self.sum_range[step_idx] += ch_range
        self.sum_rms[step_idx] += ch_rms
        self.sample_count[step_idx] += 1

    @property
    def mean_absmax(self) -> torch.Tensor:
        denom = self.sample_count.clamp_min(1).to(torch.float32).view(-1, 1)
        return self.sum_absmax / denom

    @property
    def mean_range(self) -> torch.Tensor:
        denom = self.sample_count.clamp_min(1).to(torch.float32).view(-1, 1)
        return self.sum_range / denom


class ActivationSurfaceCollector:
    def __init__(self, sample_steps: int) -> None:
        self.sample_steps = int(sample_steps)
        self.current_step: Optional[int] = None
        self.current_raw_timestep: Optional[float] = None
        self.current_expert: Optional[str] = None
        self.raw_timestep_by_step: Dict[int, float] = {}
        self.expert_by_step: Dict[int, str] = {}
        self.raw_timesteps: Dict[str, Dict[int, float]] = {"high": {}, "low": {}}
        self.stats: Dict[str, Dict[str, ModuleStepStats]] = {"high": {}, "low": {}}

    def begin_step(self, step_idx: int, raw_timestep: float, expert_name: str) -> None:
        self.current_step = int(step_idx)
        self.current_raw_timestep = float(raw_timestep)
        self.current_expert = expert_name
        self.raw_timestep_by_step[int(step_idx)] = float(raw_timestep)
        self.expert_by_step[int(step_idx)] = expert_name
        self.raw_timesteps[expert_name][int(step_idx)] = float(raw_timestep)

    @torch.no_grad()
    def add(self, expert: str, module_name: str, x: torch.Tensor) -> None:
        if self.current_step is None or self.current_expert != expert:
            return
        if x.ndim == 0:
            return
        c = int(x.shape[-1])
        x2 = x.detach().reshape(-1, c)
        x32 = x2.to(torch.float32)
        ch_min = x32.amin(dim=0)
        ch_max = x32.amax(dim=0)
        ch_range = ch_max - ch_min
        ch_absmax = x32.abs().amax(dim=0)
        ch_rms = torch.sqrt(torch.mean(x32 * x32, dim=0).clamp_min(0.0))
        if module_name not in self.stats[expert]:
            self.stats[expert][module_name] = ModuleStepStats(self.sample_steps, c)
        self.stats[expert][module_name].update(self.current_step, ch_range, ch_absmax, ch_rms)


def _stat_matrix(rec: ModuleStepStats, stat: str) -> np.ndarray:
    eps = 1e-6
    if stat == "range":
        z = rec.max_range
    elif stat == "absmax":
        z = rec.max_absmax
    elif stat == "log2_range":
        z = torch.log2(rec.max_range + eps)
    elif stat == "log2_absmax":
        z = torch.log2(rec.max_absmax + eps)
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    return z.cpu().numpy()


def _select_channels(rec: ModuleStepStats, num_channels: int, mode: str) -> np.ndarray:
    c = rec.channels
    n = min(int(num_channels), c)
    if n <= 0:
        return np.arange(c, dtype=np.int64)
    if mode == "first":
        return np.arange(n, dtype=np.int64)
    if mode == "stride":
        return np.linspace(0, c - 1, num=n, dtype=np.int64)
    if mode == "top_absmean":
        score = rec.mean_absmax.mean(dim=0)
        return torch.topk(score, k=n).indices.sort().values.cpu().numpy().astype(np.int64)
    raise ValueError(f"Unsupported channel_mode: {mode}")


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _plot_surface(path: Path, expert: str, module_name: str, stat: str, z: np.ndarray, channels: np.ndarray, title_suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(z.shape[0], dtype=np.int64)
    x_grid, y_grid = np.meshgrid(channels, steps)
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, z[:, channels], cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("channel index")
    ax.set_ylabel("denoising step")
    ax.set_zlabel(stat)
    ax.set_title(f"{expert} | {module_name} | {stat} | {title_suffix}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_outputs(
    out_dir: Path,
    collector: ActivationSurfaceCollector,
    args: argparse.Namespace,
    metadata: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    surfaces_dir = out_dir / "surfaces"
    selected_channels: Dict[str, Dict[str, List[int]]] = {"high": {}, "low": {}}
    summary_rows: List[Dict[str, Any]] = []
    npz_payload: Dict[str, np.ndarray] = {}

    for expert, modules in collector.stats.items():
        for module_name, rec in modules.items():
            channels = _select_channels(rec, args.num_channels, args.channel_mode)
            selected_channels[expert][module_name] = channels.tolist()
            z = _stat_matrix(rec, args.stat)
            z_sel = z[:, channels]
            flat_idx = int(np.nanargmax(z_sel)) if z_sel.size else 0
            arg_step_local, arg_ch_local = np.unravel_index(flat_idx, z_sel.shape) if z_sel.size else (0, 0)
            row = {
                "expert": expert,
                "module": module_name,
                "stat": args.stat,
                "max_value": float(np.nanmax(z_sel)) if z_sel.size else 0.0,
                "mean_value": float(np.nanmean(z_sel)) if z_sel.size else 0.0,
                "p99_value": float(np.nanpercentile(z_sel, 99)) if z_sel.size else 0.0,
                "argmax_step": int(arg_step_local),
                "argmax_channel": int(channels[arg_ch_local]) if len(channels) else 0,
                "num_samples": int(args.max_samples),
            }
            summary_rows.append(row)

            key = f"{expert}__{_safe_name(module_name)}"
            npz_payload[f"{key}__max_absmax"] = rec.max_absmax.numpy()
            npz_payload[f"{key}__max_range"] = rec.max_range.numpy()
            npz_payload[f"{key}__mean_absmax"] = rec.mean_absmax.numpy()
            npz_payload[f"{key}__mean_range"] = rec.mean_range.numpy()
            npz_payload[f"{key}__sample_count"] = rec.sample_count.numpy()

            if args.plot:
                png = surfaces_dir / f"{expert}__{_safe_name(module_name)}__{args.stat}.png"
                _plot_surface(
                    png,
                    expert,
                    module_name,
                    args.stat,
                    z,
                    channels,
                    title_suffix=f"num_samples={args.max_samples}, sample_steps={args.sample_steps}",
                )

    np.savez_compressed(out_dir / "stats.npz", **npz_payload)

    metadata["selected_channel_indices"] = selected_channels
    metadata["raw_timesteps"] = {
        expert: {str(k): v for k, v in steps.items()} for expert, steps in collector.raw_timesteps.items()
    }
    metadata["raw_timestep_by_step"] = {str(k): v for k, v in collector.raw_timestep_by_step.items()}
    metadata["expert_by_step"] = {str(k): v for k, v in collector.expert_by_step.items()}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["expert", "module", "stat", "max_value", "mean_value", "p99_value", "argmax_step", "argmax_channel", "num_samples"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _build_pipe(args: argparse.Namespace) -> wan.WanI2V:
    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    return pipe


def _register_hooks(model: nn.Module, expert: str, module_names: Sequence[str], collector: ActivationSurfaceCollector) -> List[Any]:
    handles = []
    module_map = dict(model.named_modules())
    for name in module_names:
        module = module_map.get(name)
        if module is None:
            LOGGER.warning("%s target module not found: %s", expert, name)
            continue
        if not isinstance(module, nn.Linear):
            LOGGER.warning("%s target is not nn.Linear: %s (%s)", expert, name, type(module).__name__)
            continue

        def _make_hook(module_name: str):
            def _hook(_module: nn.Module, inp: Tuple[Any, ...]) -> None:
                if not inp or not torch.is_tensor(inp[0]):
                    return
                collector.add(expert, module_name, inp[0])

            return _hook

        handles.append(module.register_forward_pre_hook(_make_hook(name), with_kwargs=False))
    return handles


def _matched_linears(model: nn.Module, pattern: re.Pattern[str]) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear) and pattern.search(name)]


def _expert_names(arg: str) -> List[str]:
    if arg == "both":
        return ["high", "low"]
    return [arg]


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.sample_steps <= 0:
        raise ValueError("--sample_steps must be > 0")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")

    out_dir = Path(args.out_dir)
    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found.")

    blocks = _parse_csv_ints(args.blocks)
    modules = _parse_csv_text(args.modules)
    pattern = re.compile(args.target_regex) if args.target_regex else _build_target_regex(blocks, modules)
    LOGGER.info("Target regex: %s", pattern.pattern)

    pipe = _build_pipe(args)
    model_by_expert = {"high": pipe.high_noise_model, "low": pipe.low_noise_model}
    target_modules: Dict[str, List[str]] = {}
    for expert in _expert_names(args.experts):
        names = _matched_linears(model_by_expert[expert], pattern)
        target_modules[expert] = names
        LOGGER.info("%s matched Linear modules: %d", expert, len(names))
        for name in names:
            LOGGER.info("%s | %s", expert, name)
        if not args.target_regex:
            matched = set(names)
            for expected in _expected_module_names(blocks, modules):
                if expected not in matched:
                    LOGGER.warning("%s requested Linear not found, continuing: %s", expert, expected)

    if args.list_linears:
        return

    collector = ActivationSurfaceCollector(sample_steps=int(args.sample_steps))
    handles: List[Any] = []
    cast_handles = _register_linear_input_cast_hooks(pipe.high_noise_model) + _register_linear_input_cast_hooks(pipe.low_noise_model)
    try:
        for expert in _expert_names(args.experts):
            handles.extend(_register_hooks(model_by_expert[expert], expert, target_modules[expert], collector))
        LOGGER.info("Registered hooks: %d", len(handles))
        if not handles:
            raise ValueError("No target Linear hooks registered. Use --list_linears to inspect names.")

        cfg = WAN_CONFIGS["i2v-A14B"]
        frame_num = cfg.frame_num
        if args.shift is None:
            args.shift = cfg.sample_shift
        boundary = float(pipe.boundary * pipe.num_train_timesteps)
        use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)

        sample_iter: Iterable[ManifestSample] = samples
        if tqdm is not None and not args.disable_tqdm:
            sample_iter = tqdm(samples, desc="activation surface samples", unit="sample", dynamic_ncols=True)

        for sample_idx, sample in enumerate(sample_iter):
            torch.manual_seed(int(args.seed) + sample_idx)
            item = _prep_sample(
                pipe,
                sample,
                frame_num=frame_num,
                target_hw=_size_to_hw(args.size),
            )
            latent = item["latent"].to(device=pipe.device, dtype=pipe.param_dtype)
            y = item["y"].to(device=pipe.device, dtype=pipe.param_dtype)
            context = [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in item["context"]]
            model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}
            scheduler, timesteps = _build_schedule(
                sample_solver=args.sample_solver,
                sampling_steps=int(args.sample_steps),
                shift=float(args.shift),
                num_train_timesteps=pipe.num_train_timesteps,
                device=pipe.device,
            )
            timesteps_list = [float(t.item()) for t in timesteps]
            step_iter: Iterable[Tuple[int, float]] = list(enumerate(timesteps_list))
            if tqdm is not None and not args.disable_tqdm:
                step_iter = tqdm(step_iter, desc=f"{sample.sample_id} denoise", unit="step", leave=False, dynamic_ncols=True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
                for step_idx, t_scalar in step_iter:
                    expert = "high" if t_scalar >= boundary else "low"
                    model = model_by_expert[expert]
                    collector.begin_step(step_idx=step_idx, raw_timestep=t_scalar, expert_name=expert)
                    timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
                    noise_pred = model([latent], t=timestep, **model_args)[0]
                    latent = scheduler.step(
                        noise_pred.unsqueeze(0),
                        torch.tensor(t_scalar, device=pipe.device, dtype=timesteps.dtype),
                        latent.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)
            del item, latent, y, context, model_args, scheduler, timesteps, timesteps_list
            if pipe.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
        for handle in cast_handles:
            handle.remove()

    metadata = {
        "ckpt_dir": args.ckpt_dir,
        "manifest": args.manifest,
        "sample_steps": int(args.sample_steps),
        "size": args.size,
        "seed": int(args.seed),
        "experts": _expert_names(args.experts),
        "blocks": blocks,
        "modules": modules,
        "target_regex": pattern.pattern,
        "target_modules": target_modules,
        "num_samples": len(samples),
        "stat": args.stat,
        "channel_mode": args.channel_mode,
        "num_channels": int(args.num_channels),
    }
    _write_outputs(out_dir, collector, args, metadata)
    LOGGER.info("Saved stats: %s", out_dir / "stats.npz")
    LOGGER.info("Saved metadata: %s", out_dir / "metadata.json")
    LOGGER.info("Saved summary: %s", out_dir / "summary.csv")


if __name__ == "__main__":
    main()
