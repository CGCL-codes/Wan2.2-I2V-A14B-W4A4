#!/usr/bin/env python3
"""End-to-end Wan I2V DiT SVDQuant PTQ from OpenS2V-Eval calibration samples.

This script folds the old act.pt -> wgt.pt -> ptq_state.pt flow into one run:
1. load OpenS2V-Eval image/prompt calibration samples;
2. collect Linear input activation statistics;
3. for GPTQ-MXFP4, collect each expert's Hessian in memory and immediately
   export that expert's SVDQuant PTQ state;
4. save one merged ptq_stats.pt artifact.

No intermediate act.pt, wgt.pt, or gptq_hessian.pt files are written by the
default online flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from quant.quant_linear import QuantLinearWithBranch
from quant.gptq import GPTQHessianCollector
from quant.quant_linear import _qdq_mxfp4_activation_groupwise
from quant.wan_smooth import (
    WanActStatConfig,
    _build_schedule,
    _build_smoothed_weight,
    _register_linear_input_cast_hooks,
    _smooth_scale_from_act_and_weight,
    _weight_absmax_in_channel,
    calibrate_wan_acts,
    collect_wan_linear_modules,
)
from tools.generate_opens2v_eval import EvalSample, load_eval_samples, select_samples
from wan.configs import WAN_CONFIGS

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


LOGGER = logging.getLogger("ptq_wan_i2v_opens2v_svdquant")


def _find_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    cur = root
    for p in full_name.split("."):
        cur = getattr(cur, p)
    return cur


def _progress_iter(items: Sequence[Any], desc: str, unit: str, disable: bool = False) -> Iterable[Any]:
    if disable or tqdm is None:
        if tqdm is None and not disable:
            LOGGER.info("tqdm is not installed; progress bar disabled for %s", desc)
        return items
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True)


def _progress_iterable(items: Iterable[Any], desc: str, unit: str, total: int | None, disable: bool = False) -> Iterable[Any]:
    if disable or tqdm is None:
        if tqdm is None and not disable:
            LOGGER.info("tqdm is not installed; progress bar disabled for %s", desc)
        return items
    return tqdm(items, desc=desc, unit=unit, total=total, dynamic_ncols=True)


def _build_mask(frame_num: int, lat_h: int, lat_w: int, device: torch.device) -> torch.Tensor:
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk


def _load_single_image_sample(
    image_path: str,
    prompt: str,
    target_frames: int,
    target_size: Tuple[int, int],
) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    cond_image = TF.to_tensor(img)
    cond_image = TF.resize(cond_image, size=list(target_size), antialias=True)
    cond_image = TF.center_crop(cond_image, list(target_size))
    return {
        "cond_image": cond_image,
        "prompt": prompt,
        "meta": {
            "image_path": image_path,
            "target_frames": int(target_frames),
            "target_size": [int(target_size[0]), int(target_size[1])],
            "source": "opens2v_eval",
        },
    }


def _prep_calib_item(pipe: wan.WanI2V, sample: Dict[str, Any]) -> Dict[str, Any]:
    device = pipe.device
    cond_chw = sample["cond_image"]
    frame_num = int(sample.get("meta", {}).get("target_frames", 0))
    if frame_num <= 0:
        raise ValueError("Calibration sample must provide meta.target_frames")

    c, h, w = int(cond_chw.shape[0]), int(cond_chw.shape[1]), int(cond_chw.shape[2])
    cond_cfthw = torch.zeros((c, frame_num, h, w), device=device, dtype=torch.float32)
    cond_cfthw[:, 0] = cond_chw.to(device=device, dtype=torch.float32) * 2.0 - 1.0

    cond_latent = pipe.vae.encode([cond_cfthw])[0]
    latent = torch.randn_like(cond_latent)

    _, f_lat, lat_h, lat_w = cond_latent.shape
    msk = _build_mask(frame_num=frame_num, lat_h=lat_h, lat_w=lat_w, device=device)
    y = torch.cat([msk, cond_latent], dim=0)

    seq_len = f_lat * lat_h * lat_w // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int(math.ceil(seq_len / pipe.sp_size)) * pipe.sp_size

    prompt = str(sample["prompt"])
    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)
        pipe.text_encoder.model.cpu()
        context = [t.cpu() for t in context]
    else:
        context = [t.cpu() for t in pipe.text_encoder([prompt], torch.device("cpu"))]

    return {
        "latent": latent.cpu(),
        "y": y.cpu(),
        "seq_len": int(seq_len),
        "context": context,
    }


def _parse_args() -> argparse.Namespace:
    default_eval_jsons = [
        "OpenS2V-Eval/calib_12_manifest.jsonl"
    ]

    p = argparse.ArgumentParser("End-to-end Wan I2V DiT SVDQuant PTQ from OpenS2V-Eval")
    p.add_argument("--task", type=str, default="i2v-A14B", choices=["i2v-A14B"])
    p.add_argument("--ckpt_dir", type=str, default='../Wan2.2-I2V-A14B-bf16')
    p.add_argument("--output_dir", type=str, default='/home/wjh/Wan2.2/outputs/cal12_ptq')
    p.add_argument("--eval_json", nargs="+", default=default_eval_jsons)
    p.add_argument("--image_root", type=str, default="OpenS2V-Eval")
    p.add_argument("--num_samples", type=int, default=12)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--device_id", type=int, default=1)
    p.add_argument("--target_frames", type=int, default=61)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--sampling_steps", type=int, default=40)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--channels_dim", type=int, default=-1)
    p.add_argument("--act_save_dtype", type=str, default="float32")
    p.add_argument("--wgt_save_dtype", type=str, default="bfloat16")
    p.add_argument("--module_name_filter", type=str, default="")
    p.add_argument("--progress_interval", type=int, default=1)
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    p.add_argument("--smooth_alpha", type=float, default=0.5)
    p.add_argument("--smooth_eps", type=float, default=1e-8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--weight_group_size", type=int, default=32)
    p.add_argument("--act_group_size", type=int, default=32)
    p.add_argument("--weight_clip_ratio", type=float, default=1.0)
    p.add_argument("--act_clip_ratio", type=float, default=1.0)
    p.add_argument("--compute_dtype", type=str, default="bfloat16")
    p.add_argument("--main_quant_method", "--main-quant-method", type=str, default="gptq_mxfp4", choices=["rtn_mxfp4", "gptq_mxfp4"])
    p.add_argument("--gptq_hessian_path", "--gptq-hessian-path", type=str, default="")
    p.add_argument("--high_gptq_hessian_path", "--high-gptq-hessian-path", type=str, default="")
    p.add_argument("--low_gptq_hessian_path", "--low-gptq-hessian-path", type=str, default="")
    p.add_argument("--gptq_damp_percent", "--gptq-damp-percent", type=float, default=0.01)
    p.add_argument("--gptq_block_size", "--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq_dynamic_group_scale", "--gptq-dynamic-group-scale", action="store_true", default=False)
    p.add_argument("--collect_gptq_hessian", "--collect-gptq-hessian", action="store_true", default=False, help="Deprecated compatibility flag; GPTQ Hessian is collected online when needed.")
    p.add_argument("--save_gptq_hessian", "--save-gptq-hessian", type=str, default="", help="Deprecated and ignored; Hessian is no longer saved in the default online GPTQ flow.")
    p.add_argument("--gptq_hessian_device", "--gptq-hessian-device", type=str, default="cpu")
    p.add_argument("--gptq_sample_tokens", "--gptq-sample-tokens", type=int, default=0)
    p.add_argument("--gptq_max_tokens", "--gptq-max-tokens", type=int, default=0)
    p.add_argument("--no_merge_ptq_stats", "--no-merge-ptq-stats", action="store_true", default=False, help="Deprecated; ptq_stats.pt is always saved. When set, split high/low states are also saved for debugging.")
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 0:
        raise ValueError(f"--num_samples must be >= 0, got {args.num_samples}")
    if args.start < 0:
        raise ValueError(f"--start must be >= 0, got {args.start}")
    if int(args.weight_group_size) != 32:
        raise ValueError(f"MXFP4 reference weight_group_size must be 32, got {args.weight_group_size}")
    if int(args.act_group_size) != 32:
        raise ValueError(f"MXFP4 reference act_group_size must be 32, got {args.act_group_size}")
    if not 0.0 <= float(args.smooth_alpha) <= 1.0:
        raise ValueError(f"--smooth_alpha must be in [0, 1], got {args.smooth_alpha}")
    if args.main_quant_method == "gptq_mxfp4":
        if int(args.gptq_block_size) <= 0:
            raise ValueError(f"--gptq_block_size must be > 0, got {args.gptq_block_size}")


def _ensure_paths(args: argparse.Namespace) -> Tuple[List[Path], Path]:
    eval_jsons = [Path(p) for p in args.eval_json]
    missing = [str(p) for p in eval_jsons if not p.exists()]
    if missing:
        raise FileNotFoundError(f"OpenS2V eval json not found: {missing}")
    image_root = Path(args.image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"--image_root not found: {image_root}")
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"--ckpt_dir not found: {ckpt_dir}")
    return [p.resolve() for p in eval_jsons], image_root.resolve()


def _build_pipe(args: argparse.Namespace) -> wan.WanI2V:
    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    return pipe


def _iter_opens2v_calib_items(
    pipe: wan.WanI2V,
    samples: Sequence[EvalSample],
    target_frames: int,
    target_size: Tuple[int, int],
) -> Iterable[Dict[str, Any]]:
    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        LOGGER.info(
            "[stage 2/5] preparing calibration sample %d/%d | id=%s | image=%s",
            idx,
            total,
            sample.videoid,
            sample.used_image,
        )
        raw = _load_single_image_sample(
            image_path=str(sample.used_image),
            prompt=sample.prompt,
            target_frames=target_frames,
            target_size=target_size,
        )
        item = _prep_calib_item(pipe, raw)
        LOGGER.info(
            "[stage 2/5] prepared sample %d/%d | latent_shape=%s | seq_len=%d",
            idx,
            total,
            tuple(item["latent"].shape),
            int(item["seq_len"]),
        )
        yield item
        del raw, item
        if pipe.device.type == "cuda":
            torch.cuda.empty_cache()


def _resolve_svd_device(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.device_id}")
    return torch.device("cpu")


def _load_gptq_hessians(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict GPTQ Hessian artifact at {path}, got {type(obj).__name__}")
    return obj


def _parse_save_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.strip().lower()
    if name in ("float32", "fp32", "torch.float32"):
        return torch.float32
    if name in ("float16", "fp16", "half", "torch.float16"):
        return torch.float16
    if name in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _build_smoothed_record(
    model: nn.Module,
    model_name: str,
    module_name: str,
    act_stats: Dict[str, Any],
    args: argparse.Namespace,
    include_weight: bool,
) -> Dict[str, Any]:
    rec = act_stats[model_name][module_name]
    module = _find_module_by_name(model, module_name)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear at {model_name}.{module_name}, got {type(module).__name__}")
    weight = module.weight.detach().to(torch.float32)
    act_s = rec["s"].detach().to(device=weight.device, dtype=torch.float32)
    w_s = _weight_absmax_in_channel(weight, eps=float(args.smooth_eps))
    scale = _smooth_scale_from_act_and_weight(
        act_s=act_s,
        w_s=w_s,
        alpha=float(args.smooth_alpha),
        eps=float(args.smooth_eps),
    )
    out_dtype = _parse_save_dtype(args.wgt_save_dtype)
    weight_out_dtype = torch.bfloat16 if out_dtype == torch.float32 else out_dtype
    out: Dict[str, Any] = {
        "scale": scale.detach().to(device="cpu", dtype=out_dtype),
        "act_s": act_s.detach().to(device="cpu", dtype=out_dtype),
        "w_s": w_s.detach().to(device="cpu", dtype=out_dtype),
        "weight_channels_dim": 1,
        "input_channels_dim": int(rec.get("channels_dim", -1)),
        "shape": [int(weight.shape[0]), int(weight.shape[1])],
        "orig_dtype": str(module.weight.dtype).replace("torch.", ""),
    }
    if include_weight:
        out["weight"] = _build_smoothed_weight(weight, scale).detach().to(device="cpu", dtype=weight_out_dtype)
    return out


def _smooth_input_for_hessian(x: torch.Tensor, smooth_scale: torch.Tensor, channels_dim: int) -> torch.Tensor:
    cdim = channels_dim if channels_dim >= 0 else x.ndim + channels_dim
    if cdim < 0 or cdim >= x.ndim:
        raise ValueError(f"Invalid input_channels_dim={channels_dim} for input shape={tuple(x.shape)}")
    scale = smooth_scale.to(device=x.device, dtype=x.dtype)
    view_shape = [1] * x.ndim
    view_shape[cdim] = int(scale.numel())
    return x / scale.view(*view_shape)


@torch.inference_mode()
def _collect_gptq_hessians(
    pipe: wan.WanI2V,
    calib_samples: Iterable[Dict[str, Any]],
    act_stats: Dict[str, Any],
    args: argparse.Namespace,
    target_model_name: str,
) -> Dict[str, Any]:
    """Collect main-branch Hessians for GPTQ-MXFP4 using smoothed + QDQ inputs."""
    high_model = pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)
    low_model = pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    model_map = {
        "high_noise_model": high_model,
        "low_noise_model": low_model,
    }

    collectors: Dict[str, Dict[str, GPTQHessianCollector]] = {
        "high_noise_model": {},
        "low_noise_model": {},
    }
    cast_handles = _register_linear_input_cast_hooks(high_model) + _register_linear_input_cast_hooks(low_model)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    sample_tokens = int(args.gptq_sample_tokens) if int(args.gptq_sample_tokens) > 0 else None
    max_tokens = int(args.gptq_max_tokens) if int(args.gptq_max_tokens) > 0 else None

    def _make_hook(model_name: str, module_name: str):
        rec = _build_smoothed_record(
            model=model_map[model_name],
            model_name=model_name,
            module_name=module_name,
            act_stats=act_stats,
            args=args,
            include_weight=False,
        )
        smooth_scale = rec["scale"].detach().to(torch.float32)
        input_channels_dim = int(rec.get("input_channels_dim", -1))

        def _hook(_module: nn.Module, inp: Tuple[Any, ...]) -> None:
            x = None
            for obj in inp:
                if torch.is_tensor(obj):
                    x = obj
                    break
            if x is None:
                return
            x_smooth = _smooth_input_for_hessian(x, smooth_scale, input_channels_dim)
            # This matches QuantLinearWithBranch.forward: main branch consumes
            # activation QDQ output, while the low-rank branch consumes x_smooth.
            x_main = _qdq_mxfp4_activation_groupwise(
                x_smooth,
                group_size=int(args.act_group_size),
                clip_ratio=float(args.act_clip_ratio),
                channels_dim=input_channels_dim,
            )
            collectors[model_name][module_name].add_batch(x_main)

        return _hook

    for model_name, model in model_map.items():
        if model_name != target_model_name:
            continue
        module_names = collect_wan_linear_modules(model, module_name_filter=args.module_name_filter or None)
        LOGGER.info("[stage 3b/5] %s target Linear modules for Hessian: %d", model_name, len(module_names))
        for module_name in module_names:
            if module_name not in act_stats[model_name]:
                continue
            module = _find_module_by_name(model, module_name)
            if not isinstance(module, nn.Linear):
                continue
            collectors[model_name][module_name] = GPTQHessianCollector(
                in_features=module.in_features,
                device=args.gptq_hessian_device,
                sample_tokens=sample_tokens,
                max_tokens=max_tokens,
            )
            handles.append(module.register_forward_pre_hook(_make_hook(model_name, module_name), with_kwargs=False))

    boundary = float(pipe.boundary * pipe.num_train_timesteps)
    use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)
    num_done = 0
    try:
        sample_iter = _progress_iterable(
            calib_samples,
            desc=f"{target_model_name} Hessian samples",
            unit="sample",
            total=int(args.num_samples),
            disable=bool(args.disable_tqdm),
        )
        for sample in sample_iter:
            idx = num_done + 1
            LOGGER.info("[stage 3b/5] %s collecting GPTQ Hessian sample %d/%d", target_model_name, idx, int(args.num_samples))
            latent = sample["latent"].to(device=pipe.device, dtype=pipe.param_dtype)
            y = sample["y"].to(device=pipe.device, dtype=pipe.param_dtype)
            context = [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in sample["context"]]
            seq_len = int(sample["seq_len"])
            model_args = {"context": context, "seq_len": seq_len, "y": [y]}
            scheduler, timesteps = _build_schedule(
                sample_solver=args.sample_solver,
                sampling_steps=int(args.sampling_steps),
                shift=float(args.shift),
                num_train_timesteps=pipe.num_train_timesteps,
                device=pipe.device,
            )
            timesteps_list = [float(t.item()) for t in timesteps]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
                for t_scalar in timesteps_list:
                    model = high_model if t_scalar >= boundary else low_model
                    timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
                    noise_pred = model([latent], t=timestep, **model_args)[0]
                    latent = scheduler.step(
                        noise_pred.unsqueeze(0),
                        torch.tensor(t_scalar, device=pipe.device, dtype=timesteps.dtype),
                        latent.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)
            num_done += 1
            del latent, y, context, model_args, scheduler, timesteps, timesteps_list
            if pipe.device.type == "cuda":
                torch.cuda.empty_cache()
            if num_done >= int(args.num_samples):
                break
    finally:
        for h in handles:
            h.remove()
        for h in cast_handles:
            h.remove()

    out: Dict[str, Any] = {
        "__meta__": {
            "artifact": "wan_gptq_hessian",
            "num_calib_samples": int(num_done),
            "main_input": "smooth_then_mxfp4_activation_qdq",
            "sample_solver": args.sample_solver,
            "sampling_steps": int(args.sampling_steps),
            "shift": float(args.shift),
        },
        "high_noise_model": {},
        "low_noise_model": {},
    }
    for model_name in ("high_noise_model", "low_noise_model"):
        for module_name, collector in collectors[model_name].items():
            out[model_name][module_name] = collector.export(normalize=True)
        LOGGER.info("[stage 3b/5] %s GPTQ Hessian modules=%d", model_name, len(out[model_name]))
    return out


def _build_one_model_ptq_state(
    model: nn.Module,
    model_name: str,
    act_stats: Dict[str, Any],
    args: argparse.Namespace,
    gptq_hessians: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
            "source_act_path": None,
            "source_wgt_path": None,
            "source_pipeline": "opens2v_e2e_no_intermediate_act_wgt",
            "smooth_alpha": float(args.smooth_alpha),
            "smooth_eps": float(args.smooth_eps),
            "main_quant_method": str(args.main_quant_method),
            "gptq_damp_percent": float(args.gptq_damp_percent),
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_dynamic_group_scale": bool(args.gptq_dynamic_group_scale),
        }
    }

    model_acts = act_stats[model_name]
    if not isinstance(model_acts, dict):
        raise TypeError(f"act_stats[{model_name}] must be dict")

    replaced_modules: List[str] = []
    total_weight_params = 0
    total_branch_params = 0
    module_names = list(model_acts.keys())
    total = len(module_names)

    module_iter = _progress_iter(
        module_names,
        desc=f"{model_name} PTQ export",
        unit="module",
        disable=bool(args.disable_tqdm),
    )
    for idx, module_name in enumerate(module_iter, start=1):
        if idx == 1 or idx % max(1, int(args.progress_interval)) == 0 or idx == total:
            LOGGER.info("[stage 4/5] %s SVDQuant module %d/%d | %s", model_name, idx, total, module_name)

        module = _find_module_by_name(model, module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {model_name}.{module_name}, got {type(module).__name__}")
        rec = _build_smoothed_record(
            model=model,
            model_name=model_name,
            module_name=module_name,
            act_stats=act_stats,
            args=args,
            include_weight=True,
        )

        smoothed_weight = rec["weight"]
        smooth_scale = rec.get("scale", None)
        if not torch.is_tensor(smoothed_weight):
            raise TypeError(f"smoothed weight must be tensor at {model_name}.{module_name}")
        if smooth_scale is None or not torch.is_tensor(smooth_scale):
            raise ValueError(f"smooth scale tensor missing at {model_name}.{module_name}")
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
            input_channels_dim=int(rec.get("input_channels_dim", -1)),
            smooth_enabled=True,
            svd_device=svd_device,
            gptq_hessian=gptq_hessian,
            gptq_damp_percent=float(args.gptq_damp_percent),
            gptq_block_size=int(args.gptq_block_size),
            gptq_dynamic_group_scale=bool(args.gptq_dynamic_group_scale),
        )

        model_states[module_name] = qmod.export_state()
        replaced_modules.append(module_name)
        total_weight_params += int(module.weight.numel())
        if qmod.branch is not None and hasattr(qmod.branch, "a") and hasattr(qmod.branch, "b"):
            total_branch_params += int(qmod.branch.a.weight.numel() + qmod.branch.b.weight.numel())
        del qmod

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
        "svd_device": str(svd_device),
        "total_weight_params": int(total_weight_params),
        "total_branch_params": int(total_branch_params),
        "source_act_modules": len(model_acts),
        "source_wgt_modules": len(model_acts),
    }
    LOGGER.info(
        "[stage 4/5] %s complete | replaced_modules=%d | total_weight_params=%d | total_branch_params=%d",
        model_name,
        len(replaced_modules),
        total_weight_params,
        total_branch_params,
    )
    return model_states, summary


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    _validate_args(args)
    eval_jsons, image_root = _ensure_paths(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptq_stats_path = out_dir / "ptq_stats.pt"
    summary_path = out_dir / "summary.json"
    high_hessian_path = Path(args.high_gptq_hessian_path) if args.high_gptq_hessian_path else None
    low_hessian_path = Path(args.low_gptq_hessian_path) if args.low_gptq_hessian_path else None
    high_ptq_state_path = out_dir / "high_ptq_state.pt"
    low_ptq_state_path = out_dir / "low_ptq_state.pt"

    LOGGER.info("[stage 1/5] loading OpenS2V-Eval calibration samples")
    all_samples = load_eval_samples(eval_jsons, image_root=image_root)
    samples = select_samples(all_samples, start=int(args.start), limit=int(args.num_samples))
    if not samples:
        raise ValueError("No calibration samples selected.")
    LOGGER.info(
        "[stage 1/5] loaded=%d selected=%d start=%d requested=%d jsons=%s",
        len(all_samples),
        len(samples),
        int(args.start),
        int(args.num_samples),
        [str(p) for p in eval_jsons],
    )
    LOGGER.info(
        "[config] method=%s rank=%d samples=%d steps=%d size=%dx%d output=%s save_split_debug=%s",
        args.main_quant_method,
        int(args.rank),
        len(samples),
        int(args.sampling_steps),
        int(args.height),
        int(args.width),
        out_dir,
        bool(args.no_merge_ptq_stats),
    )
    if args.main_quant_method == "gptq_mxfp4" and args.save_gptq_hessian:
        LOGGER.warning(
            "--save_gptq_hessian is deprecated and ignored; GPTQ Hessians are kept in memory and only ptq_stats.pt is saved."
        )
    if args.main_quant_method == "gptq_mxfp4" and args.collect_gptq_hessian:
        LOGGER.info("--collect_gptq_hessian is no longer required; online GPTQ collection runs automatically when no Hessian path is provided.")

    LOGGER.info("[stage 2/5] building Wan I2V pipeline and preparing online calibration iterator")
    pipe = _build_pipe(args)
    calib_iter = _iter_opens2v_calib_items(
        pipe=pipe,
        samples=samples,
        target_frames=int(args.target_frames),
        target_size=(int(args.height), int(args.width)),
    )

    LOGGER.info("[stage 3/5] collecting activation statistics")
    stat_cfg = WanActStatConfig(
        max_calib_samples=len(samples),
        save_dtype=args.act_save_dtype,
        channels_dim=int(args.channels_dim),
        module_name_filter=args.module_name_filter or None,
        sample_solver=args.sample_solver,
        sampling_steps=int(args.sampling_steps),
        shift=float(args.shift),
        alpha=float(args.smooth_alpha),
        eps=float(args.smooth_eps),
        progress_interval=int(args.progress_interval),
        show_timestep_progress=(not args.disable_tqdm),
    )
    act_stats = calibrate_wan_acts(pipe, calib_iter, config=stat_cfg)
    LOGGER.info(
        "[stage 3/5] activation stats complete | high_modules=%d | low_modules=%d | calib_samples=%d",
        len(act_stats["high_noise_model"]),
        len(act_stats["low_noise_model"]),
        int(act_stats["__meta__"]["num_calib_samples"]),
    )

    merged_gptq_hessians = None
    if args.main_quant_method == "gptq_mxfp4" and args.gptq_hessian_path:
        LOGGER.info("[stage 3b/5] loading merged GPTQ Hessians: %s", args.gptq_hessian_path)
        merged_gptq_hessians = _load_gptq_hessians(args.gptq_hessian_path)

    LOGGER.info("[stage 4/5] generating high PTQ state")
    high_gptq_hessians = None
    if args.main_quant_method == "gptq_mxfp4":
        if merged_gptq_hessians is not None:
            high_gptq_hessians = merged_gptq_hessians
        elif high_hessian_path is not None:
            LOGGER.info("[stage 3b/5] loading high GPTQ Hessians: %s", high_hessian_path)
            high_gptq_hessians = _load_gptq_hessians(str(high_hessian_path))
        else:
            LOGGER.info("[stage 3b/5] collecting high_noise_model GPTQ Hessians in memory")
            high_iter = _iter_opens2v_calib_items(
                pipe=pipe,
                samples=samples,
                target_frames=int(args.target_frames),
                target_size=(int(args.height), int(args.width)),
            )
            high_gptq_hessians = _collect_gptq_hessians(
                pipe=pipe,
                calib_samples=high_iter,
                act_stats=act_stats,
                args=args,
                target_model_name="high_noise_model",
            )
    high_state, high_summary = _build_one_model_ptq_state(
        model=pipe.high_noise_model,
        model_name="high_noise_model",
        act_stats=act_stats,
        args=args,
        gptq_hessians=high_gptq_hessians,
    )
    if merged_gptq_hessians is None:
        del high_gptq_hessians
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    LOGGER.info("[stage 4/5] generating low PTQ state")
    low_gptq_hessians = None
    if args.main_quant_method == "gptq_mxfp4":
        if merged_gptq_hessians is not None:
            low_gptq_hessians = merged_gptq_hessians
        elif low_hessian_path is not None:
            LOGGER.info("[stage 3b/5] loading low GPTQ Hessians: %s", low_hessian_path)
            low_gptq_hessians = _load_gptq_hessians(str(low_hessian_path))
        else:
            LOGGER.info("[stage 3b/5] collecting low_noise_model GPTQ Hessians in memory")
            low_iter = _iter_opens2v_calib_items(
                pipe=pipe,
                samples=samples,
                target_frames=int(args.target_frames),
                target_size=(int(args.height), int(args.width)),
            )
            low_gptq_hessians = _collect_gptq_hessians(
                pipe=pipe,
                calib_samples=low_iter,
                act_stats=act_stats,
                args=args,
                target_model_name="low_noise_model",
            )
    low_state, low_summary = _build_one_model_ptq_state(
        model=pipe.low_noise_model,
        model_name="low_noise_model",
        act_stats=act_stats,
        args=args,
        gptq_hessians=low_gptq_hessians,
    )
    if merged_gptq_hessians is None:
        del low_gptq_hessians
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.no_merge_ptq_stats:
        torch.save(high_state, high_ptq_state_path)
        torch.save(low_state, low_ptq_state_path)
        LOGGER.info("[stage 5/5] saved split PTQ states: high=%s low=%s", high_ptq_state_path, low_ptq_state_path)

    LOGGER.info("[stage 5/5] writing summary and merged artifact")
    final_summary = {
        "artifact": "wan_dit_ptq_stats",
        "task": args.task,
        "quant_scheme": "mxfp4",
        "weight_format": "fp4_e2m1_packed+e8m0_scale",
        "activation_qdq_format": "mxfp4_reference",
        "weight_block_size": int(args.weight_group_size),
        "activation_block_size": int(args.act_group_size),
        "device_id": int(args.device_id),
        "eval_json": [str(p) for p in eval_jsons],
        "image_root": str(image_root),
        "total_loaded_samples": len(all_samples),
        "selected_samples": len(samples),
        "start": int(args.start),
        "target_frames": int(args.target_frames),
        "height": int(args.height),
        "width": int(args.width),
        "sample_solver": args.sample_solver,
        "sampling_steps": int(args.sampling_steps),
        "shift": float(args.shift),
        "smooth_alpha": float(args.smooth_alpha),
        "smooth_eps": float(args.smooth_eps),
        "rank": int(args.rank),
        "weight_group_size": int(args.weight_group_size),
        "act_group_size": int(args.act_group_size),
        "weight_clip_ratio": float(args.weight_clip_ratio),
        "act_clip_ratio": float(args.act_clip_ratio),
        "compute_dtype": str(args.compute_dtype),
        "main_quant_method": str(args.main_quant_method),
        "gptq_hessian_path": str(args.gptq_hessian_path),
        "high_gptq_hessian_path": "" if high_hessian_path is None else str(high_hessian_path),
        "low_gptq_hessian_path": "" if low_hessian_path is None else str(low_hessian_path),
        "collect_gptq_hessian": bool(args.collect_gptq_hessian),
        "save_gptq_hessian": "",
        "hessian_saved": False,
        "gptq_damp_percent": float(args.gptq_damp_percent),
        "gptq_block_size": int(args.gptq_block_size),
        "gptq_dynamic_group_scale": bool(args.gptq_dynamic_group_scale),
        "high_ptq_state_path": str(high_ptq_state_path) if args.no_merge_ptq_stats else "",
        "low_ptq_state_path": str(low_ptq_state_path) if args.no_merge_ptq_stats else "",
        "ptq_stats_path": str(ptq_stats_path),
        "merge_ptq_stats": True,
        "save_split_ptq_states": bool(args.no_merge_ptq_stats),
        "models": [high_summary, low_summary],
        "sample_ids": [s.videoid for s in samples],
    }
    LOGGER.info("[stage 5/5] saving ptq_stats.pt: %s", ptq_stats_path)
    artifact = {
        "__meta__": final_summary,
        "high_noise_model": high_state,
        "low_noise_model": low_state,
    }
    torch.save(artifact, ptq_stats_path)
    LOGGER.info("[stage 5/5] saved PTQ stats: %s", ptq_stats_path)
    _write_summary(summary_path, final_summary)

    LOGGER.info("[stage 5/5] saved summary: %s", summary_path)
    LOGGER.info("[done] ptq_stats=%s", ptq_stats_path)


if __name__ == "__main__":
    main()
