#!/usr/bin/env python3
"""Trace Wan I2V first-frame latent drift during denoising.

This experiment records per-step metrics for latent[:, 0] against the VAE
condition latent y[4:, 0]. It is intended to compare BF16, W4A4 activation
quantization, and W4A4 with low-expert activation quantization disabled.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from quant.act_policy import (
    build_wan_i2v_timestep_schedule,
    get_timestep_bin_from_schedule,
    load_act_policy_json,
    maybe_patch_model_forward_for_act_context,
    set_quant_modules_act_policy,
)
from quant.quant_linear import QuantLinearWithBranch


LOGGER = logging.getLogger("trace_i2v_first_frame_latent_error")


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Trace Wan I2V first-frame latent drift")
    p.add_argument("--ckpt_dir", type=str, default="../Wan2.2-I2V-A14B-bf16", help="Wan checkpoint directory.")
    p.add_argument("--mode", type=str, default="quant", choices=["bf16", "quant"], help="Run BF16 or SVDQuant PTQ.")
    p.add_argument("--ptq_dir", type=str, default="/home/wjh/Wan2.2/outputs/gptq_ptq", help="PTQ artifact path for quant mode.")
    p.add_argument("--image", type=str, default="examples/5.png", help="Input condition image.")
    p.add_argument("--prompt", type=str, required=True, help="Input prompt.")
    p.add_argument("--negative_prompt", type=str, default="", help="Negative prompt. Empty uses Wan default.")
    p.add_argument("--size", type=str, default="480*832", help="Wan size key.")
    p.add_argument("--frame_num", type=int, default=61, help="Number of video frames.")
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampler.")
    p.add_argument("--sample_steps", type=int, default=40, help="Sampling steps.")
    p.add_argument("--sample_shift", type=float, default=None, help="Sampling shift.")
    p.add_argument("--sample_guide_scale", type=float, default=None, help="CFG scale or Wan tuple default.")
    p.add_argument("--device_id", type=int, default=0, help="CUDA device id.")
    p.add_argument("--offload_model", type=_str2bool, default=True, help="Offload inactive low/high model.")
    p.add_argument("--t5_cpu", action="store_true", default=False, help="Run T5 on CPU.")
    p.add_argument("--convert_model_dtype", action="store_true", default=True, help="Convert DiT dtype.")
    p.add_argument("--base_seed", type=int, default=0, help="Random seed.")
    p.add_argument("--low_keep_fp_blocks", type=str, default="", help="Low model blocks to keep BF16 in quant mode.")
    p.add_argument("--high_keep_fp_blocks", type=str, default="", help="High model blocks to keep BF16 in quant mode.")
    p.add_argument("--act_policy_json", type=str, default="", help="Optional activation policy JSON.")
    p.add_argument("--act_scale_method", type=str, default="", choices=["", "ocp_floor", "safe_ceil"], help="Override act scale method.")
    p.add_argument("--act_exp_offset", type=int, default=0, help="Override act exponent offset when nonzero.")
    p.add_argument("--disable_low_act_quant", action="store_true", default=False, help="Disable activation QDQ in low expert.")
    p.add_argument("--disable_high_act_quant", action="store_true", default=False, help="Disable activation QDQ in high expert.")
    p.add_argument("--output_csv", type=str, required=True, help="Per-step metrics CSV path.")
    p.add_argument("--output_json", type=str, default="", help="Summary JSON path. Defaults to CSV stem + .json.")
    p.add_argument("--output_npz", type=str, default="", help="Optional NPZ path for compact numeric arrays.")
    p.add_argument("--save_video", type=str, default="", help="Optional decoded video path.")
    p.add_argument("--no_decode", action="store_true", default=False, help="Skip VAE decode after tracing.")
    return p.parse_args()


def _resolve_defaults(args: argparse.Namespace) -> None:
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["i2v-A14B"]
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale


def _set_activation_quant_enabled(model: nn.Module, enabled: bool) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, QuantLinearWithBranch):
            module.quant_linear.skip_input_quant = not bool(enabled)
            count += 1
    return count


def _tensor_metrics(x: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    xf = x.detach().to(torch.float32)
    rf = ref.detach().to(device=xf.device, dtype=torch.float32)
    diff = xf - rf
    mse = diff.square().mean()
    ref_mse = rf.square().mean().clamp_min(1e-12)
    x_flat = xf.flatten()
    r_flat = rf.flatten()
    cosine = torch.nn.functional.cosine_similarity(x_flat, r_flat, dim=0, eps=1e-8)
    return {
        "mse": float(mse.detach().cpu().item()),
        "rmse": float(torch.sqrt(mse).detach().cpu().item()),
        "rel_mse": float((mse / ref_mse).detach().cpu().item()),
        "mae": float(diff.abs().mean().detach().cpu().item()),
        "max_abs": float(diff.abs().max().detach().cpu().item()),
        "cosine": float(cosine.detach().cpu().item()),
        "latent_norm": float(xf.norm().detach().cpu().item()),
        "ref_norm": float(rf.norm().detach().cpu().item()),
    }


def _build_scheduler(pipe: Any, sample_solver: str, sampling_steps: int, shift: float):
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    if sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipe.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=pipe.device, shift=shift)
        return scheduler, scheduler.timesteps
    if sample_solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=pipe.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(scheduler, device=pipe.device, sigmas=sampling_sigmas)
        return scheduler, timesteps
    raise NotImplementedError(f"Unsupported solver: {sample_solver}")


def _prepare_quant(pipe: Any, args: argparse.Namespace) -> None:
    if args.mode != "quant":
        return
    from tools.infer_wan_i2v_svdquant import (
        _apply_ptq_state_to_model,
        _load_i2v_ptq_states,
        _parse_keep_fp_blocks,
    )

    low_state, high_state, low_ptq_path, high_ptq_path = _load_i2v_ptq_states(args.ptq_dir)
    _apply_ptq_state_to_model(
        pipe.low_noise_model,
        low_state,
        "low_noise_model",
        low_ptq_path,
        keep_fp_blocks=_parse_keep_fp_blocks(args.low_keep_fp_blocks),
    )
    _apply_ptq_state_to_model(
        pipe.high_noise_model,
        high_state,
        "high_noise_model",
        high_ptq_path,
        keep_fp_blocks=_parse_keep_fp_blocks(args.high_keep_fp_blocks),
    )
    policy = load_act_policy_json(args.act_policy_json) if args.act_policy_json else None
    set_quant_modules_act_policy(
        pipe.low_noise_model,
        "low_noise_model",
        policy,
        act_scale_method=args.act_scale_method,
        act_exp_offset=args.act_exp_offset if args.act_exp_offset != 0 else None,
    )
    set_quant_modules_act_policy(
        pipe.high_noise_model,
        "high_noise_model",
        policy,
        act_scale_method=args.act_scale_method,
        act_exp_offset=args.act_exp_offset if args.act_exp_offset != 0 else None,
    )
    if args.disable_low_act_quant:
        LOGGER.info("Disabled low expert activation QDQ on %d modules", _set_activation_quant_enabled(pipe.low_noise_model, False))
    if args.disable_high_act_quant:
        LOGGER.info("Disabled high expert activation QDQ on %d modules", _set_activation_quant_enabled(pipe.high_noise_model, False))


def _trace_generate(pipe: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], Optional[torch.Tensor]]:
    from wan.configs import MAX_AREA_CONFIGS

    guide_scale = args.sample_guide_scale
    guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
    if isinstance(guide_scale, (int, float)):
        guide_scale = (float(guide_scale), float(guide_scale))

    img = Image.open(args.image).convert("RGB")
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(pipe.device)
    frame_num = int(args.frame_num)
    h, w = img_tensor.shape[1:]
    aspect_ratio = h / w
    max_area = MAX_AREA_CONFIGS[args.size]
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // pipe.vae_stride[1] //
        pipe.patch_size[1] * pipe.patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // pipe.vae_stride[2] //
        pipe.patch_size[2] * pipe.patch_size[2]
    )
    h = lat_h * pipe.vae_stride[1]
    w = lat_w * pipe.vae_stride[2]
    max_seq_len = ((frame_num - 1) // pipe.vae_stride[0] + 1) * lat_h * lat_w // (
        pipe.patch_size[1] * pipe.patch_size[2]
    )
    max_seq_len = int(math.ceil(max_seq_len / pipe.sp_size)) * pipe.sp_size

    seed = int(args.base_seed) if int(args.base_seed) >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16,
        (frame_num - 1) // pipe.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=pipe.device,
    )

    msk = torch.ones(1, frame_num, lat_h, lat_w, device=pipe.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:],
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    n_prompt = args.negative_prompt or pipe.sample_neg_prompt
    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([args.prompt], pipe.device)
        context_null = pipe.text_encoder([n_prompt], pipe.device)
        if args.offload_model:
            pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([args.prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
        context_null = [t.to(pipe.device) for t in context_null]

    cond_video = torch.concat([
        torch.nn.functional.interpolate(
            img_tensor[None].cpu(), size=(h, w), mode="bicubic"
        ).transpose(0, 1),
        torch.zeros(3, frame_num - 1, h, w),
    ], dim=1).to(pipe.device)
    cond_latent_all = pipe.vae.encode([cond_video])[0]
    cond_first = cond_latent_all[:, 0].detach()
    y = torch.concat([msk, cond_latent_all])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low_noise = getattr(pipe.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high_noise = getattr(pipe.high_noise_model, "no_sync", noop_no_sync)
    sample_scheduler, timesteps = _build_scheduler(pipe, args.sample_solver, int(args.sample_steps), float(args.sample_shift))
    boundary = pipe.boundary * pipe.num_train_timesteps

    _, boundary_timestep, expert_timestep_schedule = build_wan_i2v_timestep_schedule(
        num_train_timesteps=pipe.num_train_timesteps,
        boundary=pipe.boundary,
        sample_solver=args.sample_solver,
        sampling_steps=int(args.sample_steps),
        shift=float(args.sample_shift),
        device=pipe.device,
    )
    maybe_patch_model_forward_for_act_context(
        pipe.low_noise_model,
        "low",
        expert_timestep_schedule=expert_timestep_schedule,
    )
    maybe_patch_model_forward_for_act_context(
        pipe.high_noise_model,
        "high",
        expert_timestep_schedule=expert_timestep_schedule,
    )

    records: list[dict[str, Any]] = []
    latent = noise
    x0 = [latent]
    arg_c = {"context": [context[0]], "seq_len": max_seq_len, "y": [y]}
    arg_null = {"context": context_null, "seq_len": max_seq_len, "y": [y]}

    if args.offload_model:
        torch.cuda.empty_cache()

    with (
        torch.amp.autocast("cuda", dtype=pipe.param_dtype),
        torch.no_grad(),
        no_sync_low_noise(),
        no_sync_high_noise(),
    ):
        for step_idx, t in enumerate(tqdm(timesteps)):
            latent_before = latent
            before_metrics = _tensor_metrics(latent_before[:, 0], cond_first)
            latent_model_input = [latent_before.to(pipe.device)]
            timestep = torch.stack([t]).to(pipe.device)
            expert_id = "high" if float(t.item()) >= float(boundary) else "low"
            timestep_bin = get_timestep_bin_from_schedule(expert_id, t, expert_timestep_schedule)
            model = pipe._prepare_model_for_timestep(t, boundary, args.offload_model)
            step_guide_scale = guide_scale[1] if expert_id == "high" else guide_scale[0]

            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
            if args.offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
            if args.offload_model:
                torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + step_guide_scale * (noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent_before.unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latent = temp_x0.squeeze(0)
            x0 = [latent]

            after_metrics = _tensor_metrics(latent[:, 0], cond_first)
            update = (latent[:, 0].detach().to(torch.float32) - latent_before[:, 0].detach().to(torch.float32))
            pred_first = noise_pred[:, 0].detach().to(torch.float32)
            cond_pred_first = noise_pred_cond[:, 0].detach().to(torch.float32)
            uncond_pred_first = noise_pred_uncond[:, 0].detach().to(torch.float32)
            cfg_delta_first = (cond_pred_first - uncond_pred_first).to(torch.float32)

            records.append({
                "step_idx": int(step_idx),
                "timestep": float(t.detach().cpu().item()),
                "expert": expert_id,
                "timestep_bin": timestep_bin,
                "guide_scale": float(step_guide_scale),
                "before_mse": before_metrics["mse"],
                "before_rmse": before_metrics["rmse"],
                "before_rel_mse": before_metrics["rel_mse"],
                "before_mae": before_metrics["mae"],
                "before_max_abs": before_metrics["max_abs"],
                "before_cosine": before_metrics["cosine"],
                "after_mse": after_metrics["mse"],
                "after_rmse": after_metrics["rmse"],
                "after_rel_mse": after_metrics["rel_mse"],
                "after_mae": after_metrics["mae"],
                "after_max_abs": after_metrics["max_abs"],
                "after_cosine": after_metrics["cosine"],
                "delta_mse": after_metrics["mse"] - before_metrics["mse"],
                "update_l2": float(update.norm().detach().cpu().item()),
                "update_mean_abs": float(update.abs().mean().detach().cpu().item()),
                "noise_pred_first_l2": float(pred_first.norm().detach().cpu().item()),
                "noise_pred_first_mean_abs": float(pred_first.abs().mean().detach().cpu().item()),
                "cfg_delta_first_l2": float(cfg_delta_first.norm().detach().cpu().item()),
                "cond_uncond_first_mse": float((cond_pred_first - uncond_pred_first).square().mean().detach().cpu().item()),
            })
            del latent_model_input, timestep, noise_pred_cond, noise_pred_uncond, noise_pred

    if args.offload_model:
        pipe.low_noise_model.cpu()
        pipe.high_noise_model.cpu()
        torch.cuda.empty_cache()

    video = None
    if not args.no_decode and pipe.rank == 0:
        with torch.no_grad():
            video = pipe.vae.decode(x0)[0]

    summary = {
        "mode": args.mode,
        "ptq_dir": args.ptq_dir if args.mode == "quant" else "",
        "act_policy_json": args.act_policy_json,
        "disable_low_act_quant": bool(args.disable_low_act_quant),
        "disable_high_act_quant": bool(args.disable_high_act_quant),
        "image": args.image,
        "prompt": args.prompt,
        "size": args.size,
        "frame_num": int(args.frame_num),
        "sample_solver": args.sample_solver,
        "sample_steps": int(args.sample_steps),
        "sample_shift": float(args.sample_shift),
        "boundary_timestep": float(boundary_timestep),
        "high_steps": len(expert_timestep_schedule["high_timestep_list"]),
        "low_steps": len(expert_timestep_schedule["low_timestep_list"]),
        "latent_shape": list(noise.shape),
        "cond_first_shape": list(cond_first.shape),
        "initial_mse": records[0]["before_mse"] if records else None,
        "final_mse": records[-1]["after_mse"] if records else None,
        "final_rmse": records[-1]["after_rmse"] if records else None,
        "final_cosine": records[-1]["after_cosine"] if records else None,
        "max_positive_delta_mse": max((r["delta_mse"] for r in records), default=None),
        "max_update_l2": max((r["update_l2"] for r in records), default=None),
    }

    del noise, latent, x0, sample_scheduler
    if args.offload_model:
        gc.collect()
        torch.cuda.synchronize()
    return records, summary, video


def _write_outputs(records: list[dict[str, Any]], summary: dict[str, Any], video: Optional[torch.Tensor], args: argparse.Namespace) -> None:
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["step_idx"])
        writer.writeheader()
        writer.writerows(records)

    json_path = Path(args.output_json) if args.output_json else csv_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "records": records}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_npz:
        npz_path = Path(args.output_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        numeric: dict[str, list[float]] = {}
        for key in records[0].keys() if records else []:
            if isinstance(records[0][key], (int, float)):
                numeric[key] = [float(r[key]) for r in records]
        np.savez(npz_path, **{k: np.asarray(v, dtype=np.float64) for k, v in numeric.items()})

    if args.save_video and video is not None:
        from wan.configs import WAN_CONFIGS
        from wan.utils.utils import save_video

        cfg = WAN_CONFIGS["i2v-A14B"]
        save_video(
            tensor=video[None],
            save_file=args.save_video,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    LOGGER.info("Wrote trace CSV: %s", csv_path)
    LOGGER.info("Wrote trace JSON: %s", json_path)
    if args.output_npz:
        LOGGER.info("Wrote trace NPZ: %s", args.output_npz)
    if args.save_video and video is not None:
        LOGGER.info("Wrote decoded video: %s", args.save_video)


def main() -> None:
    args = _parse_args()
    _resolve_defaults(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    import wan
    from wan.configs import WAN_CONFIGS

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
    _prepare_quant(pipe, args)
    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)

    records, summary, video = _trace_generate(pipe, args)
    _write_outputs(records, summary, video, args)
    LOGGER.info(
        "Final first-frame latent drift: mse=%s rmse=%s cosine=%s",
        summary.get("final_mse"),
        summary.get("final_rmse"),
        summary.get("final_cosine"),
    )


if __name__ == "__main__":
    main()
