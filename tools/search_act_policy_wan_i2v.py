#!/usr/bin/env python3
"""Search Wan I2V MXFP4 activation policies by local layer-output error."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from PIL import Image

import wan
from quant.act_policy import (
    build_wan_i2v_timestep_schedule,
    maybe_patch_model_forward_for_act_context,
    set_quant_modules_act_policy,
)
from quant.quant_linear import QuantLinearWithBranch, _qdq_mxfp4_activation_groupwise
from tools.infer_wan_i2v_svdquant import _apply_ptq_state_to_model, _load_i2v_ptq_states
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import str2bool


LOGGER = logging.getLogger("search_act_policy_wan_i2v")


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Search Wan I2V SVDQuant MXFP4 activation policy")
    p.add_argument("--ckpt_dir", type=str, default="../Wan2.2-I2V-A14B-bf16", help="Wan checkpoint directory.")
    p.add_argument("--ptq_dir", type=str, required=True, help="PTQ artifact path or directory.")
    p.add_argument("--image", type=str, required=True, help="Calibration reference image.")
    p.add_argument("--prompt", type=str, required=True, help="Calibration prompt.")
    p.add_argument("--size", type=str, default="480*832", choices=list(SIZE_CONFIGS.keys()), help="Wan size key.")
    p.add_argument("--frame_num", type=int, default=61, help="Number of frames.")
    p.add_argument("--sample_steps", type=int, default=40, help="Sampling steps.")
    p.add_argument("--sample_shift", type=float, default=None, help="Sampling shift.")
    p.add_argument("--sample_guide_scale", type=float, default=None, help="CFG scale.")
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampler.")
    p.add_argument("--device_id", type=int, default=0, help="CUDA device id.")
    p.add_argument("--offload_model", type=str2bool, default=True, help="Offload inactive models during generation.")
    p.add_argument("--t5_cpu", action="store_true", default=False, help="Run T5 on CPU.")
    p.add_argument("--convert_model_dtype", action="store_true", default=True, help="Convert model dtype like Wan defaults.")
    p.add_argument("--base_seed", type=int, default=0, help="Base random seed.")
    p.add_argument("--output_json", type=str, required=True, help="Where to write searched policy JSON.")
    p.add_argument("--search_low_only", action="store_true", default=False, help="Only attach hooks to low_noise_model.")
    p.add_argument("--search_high_only", action="store_true", default=False, help="Only attach hooks to high_noise_model.")
    p.add_argument("--module_regex", type=str, default="self_attn|ffn|ffn.net|modulation", help="Regex for QuantLinearWithBranch module names.")
    p.add_argument("--max_tokens_per_layer", type=int, default=4096, help="Max token rows sampled per layer hook call.")
    p.add_argument("--candidates_clip", type=str, default="0.6,0.7,0.8,0.9,1.0", help="Comma-separated clip_ratio candidates.")
    p.add_argument("--candidates_exp_offset", type=str, default="-1,0,1", help="Comma-separated exp_offset candidates.")
    p.add_argument("--scale_method", type=str, default="safe_ceil", choices=["ocp_floor", "safe_ceil"], help="Scale method to search.")
    p.add_argument("--metric", type=str, default="output_mse", choices=["output_mse", "act_mse"], help="Search metric.")
    p.add_argument("--first_frame_token_weight", type=float, default=1.0, help="Reserved token weighting knob.")
    p.add_argument("--dry_run", action="store_true", default=False, help="List hook targets and exit.")
    return p.parse_args()


def _resolve_defaults(args: argparse.Namespace) -> None:
    cfg = WAN_CONFIGS["i2v-A14B"]
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale


def _sample_activation_rows(x: torch.Tensor, channels_dim: int, max_tokens: int) -> torch.Tensor:
    cdim = channels_dim if channels_dim >= 0 else x.ndim + channels_dim
    if cdim < 0 or cdim >= x.ndim:
        cdim = x.ndim - 1
    if cdim != x.ndim - 1:
        x = x.movedim(cdim, -1)
    c = int(x.shape[-1])
    rows = x.reshape(-1, c)
    if max_tokens > 0 and rows.shape[0] > max_tokens:
        idx = torch.linspace(0, rows.shape[0] - 1, steps=max_tokens, device=rows.device).round().long()
        rows = rows.index_select(0, idx)
    return rows


def _make_hook(
    model_name: str,
    module_name: str,
    stats: dict[tuple[str, str, str, str], dict[tuple[float, int], dict[str, float]]],
    candidates: list[tuple[float, int]],
    scale_method: str,
    metric: str,
    max_tokens: int,
):
    def hook(module: QuantLinearWithBranch, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        qlinear = module.quant_linear
        context = qlinear.act_policy_context
        expert_id = context.get("expert_id") or ("low" if model_name.startswith("low") else "high")
        timestep_bin = context.get("timestep_bin") or "default"
        key = (model_name, module_name, str(expert_id), str(timestep_bin))

        with torch.no_grad():
            x = inputs[0].detach()
            x_smooth = qlinear.smooth_input_only(x)
            rows = _sample_activation_rows(
                x_smooth,
                channels_dim=int(qlinear.input_channels_dim.item()),
                max_tokens=max_tokens,
            )
            if rows.numel() == 0:
                return
            rows = rows.contiguous()

            y_ref = None
            branch_ref = None
            if metric == "output_mse":
                y_ref = qlinear.forward_preprocessed(rows, out_dtype=torch.float32)
                if module.branch is not None:
                    branch_ref = module.branch(rows).to(torch.float32)
                    y_ref = y_ref + branch_ref

            for clip_ratio, exp_offset in candidates:
                x_q = _qdq_mxfp4_activation_groupwise(
                    rows,
                    group_size=int(qlinear.act_group_size),
                    clip_ratio=float(clip_ratio),
                    channels_dim=-1,
                    exp_offset=int(exp_offset),
                    scale_method=scale_method,
                )
                if metric == "output_mse":
                    y_q = qlinear.forward_preprocessed(x_q, out_dtype=torch.float32)
                    if branch_ref is not None:
                        y_q = y_q + branch_ref
                    diff = y_q - y_ref
                    denom = y_ref.square().sum().clamp_min(1e-12)
                else:
                    diff = x_q.to(torch.float32) - rows.to(torch.float32)
                    denom = rows.to(torch.float32).square().sum().clamp_min(1e-12)
                num = diff.to(torch.float32).square().sum()
                bucket = stats[key][(float(clip_ratio), int(exp_offset))]
                bucket["num"] += float(num.detach().cpu().item())
                bucket["den"] += float(denom.detach().cpu().item())
                bucket["samples"] += 1.0
    return hook


def _register_hooks(
    model: torch.nn.Module,
    model_name: str,
    regex: str,
    stats: dict[tuple[str, str, str, str], dict[tuple[float, int], dict[str, float]]],
    candidates: list[tuple[float, int]],
    args: argparse.Namespace,
) -> list[Any]:
    pattern = re.compile(regex)
    handles = []
    matched = 0
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinearWithBranch):
            continue
        module.set_module_name(name)
        if not pattern.search(name):
            continue
        matched += 1
        handles.append(
            module.register_forward_pre_hook(
                _make_hook(
                    model_name=model_name,
                    module_name=name,
                    stats=stats,
                    candidates=candidates,
                    scale_method=args.scale_method,
                    metric=args.metric,
                    max_tokens=int(args.max_tokens_per_layer),
                )
            )
        )
    LOGGER.info("%s activation search hook targets: %d", model_name, matched)
    if matched == 0:
        LOGGER.warning("%s had no QuantLinearWithBranch modules matching regex: %s", model_name, regex)
    return handles


def _build_policy(
    stats: dict[tuple[str, str, str, str], dict[tuple[float, int], dict[str, float]]],
    candidates_clip: list[float],
    candidates_exp_offset: list[int],
    scale_method: str,
    metric: str,
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "__meta__": {
            "type": "wan_i2v_mxfp4_act_policy",
            "candidates_clip": candidates_clip,
            "candidates_exp_offset": candidates_exp_offset,
            "scale_method": scale_method,
            "metric": metric,
        },
        "low_noise_model": {"__default__": {"clip_ratio": 1.0, "exp_offset": 0, "scale_method": scale_method}},
        "high_noise_model": {"__default__": {"clip_ratio": 1.0, "exp_offset": 0, "scale_method": scale_method}},
    }
    for (model_name, module_name, expert_id, timestep_bin), cand_stats in sorted(stats.items()):
        best = None
        for (clip_ratio, exp_offset), values in cand_stats.items():
            samples = int(values.get("samples", 0.0))
            if samples <= 0:
                continue
            err = float(values["num"]) / max(float(values["den"]), 1e-12)
            if best is None or err < best[0]:
                best = (err, clip_ratio, exp_offset, samples)
        if best is None:
            continue
        err, clip_ratio, exp_offset, samples = best
        model_policy = policy.setdefault(model_name, {})
        module_policy = model_policy.setdefault(module_name, {})
        expert_policy = module_policy.setdefault(expert_id, {})
        expert_policy[timestep_bin] = {
            "clip_ratio": float(clip_ratio),
            "exp_offset": int(exp_offset),
            "scale_method": scale_method,
            "search_err": float(err),
            "samples": int(samples),
        }
    return policy


def _print_summary(policy: dict[str, Any]) -> None:
    worst = []
    for model_name in ("low_noise_model", "high_noise_model"):
        model_policy = policy.get(model_name, {})
        counts: Counter[tuple[str, float, int]] = Counter()
        for module_name, module_policy in model_policy.items():
            if module_name.startswith("__") or not isinstance(module_policy, dict):
                continue
            for expert_id, expert_policy in module_policy.items():
                if not isinstance(expert_policy, dict):
                    continue
                for bin_id, entry in expert_policy.items():
                    if not isinstance(entry, dict):
                        continue
                    counts[(str(bin_id), float(entry["clip_ratio"]), int(entry["exp_offset"]))] += 1
                    worst.append((float(entry.get("search_err", 0.0)), model_name, module_name, expert_id, bin_id, entry))
        for (bin_id, clip, exp), count in sorted(counts.items()):
            LOGGER.info("%s bin=%s clip=%s exp_offset=%s count=%d", model_name, bin_id, clip, exp, count)
    for item in sorted(worst, reverse=True)[:20]:
        err, model_name, module_name, expert_id, bin_id, entry = item
        LOGGER.info(
            "top_err err=%.6g %s %s %s/%s clip=%s exp=%s samples=%s",
            err,
            model_name,
            module_name,
            expert_id,
            bin_id,
            entry.get("clip_ratio"),
            entry.get("exp_offset"),
            entry.get("samples"),
        )


def main() -> None:
    args = _parse_args()
    _resolve_defaults(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.search_low_only and args.search_high_only:
        raise ValueError("--search_low_only and --search_high_only are mutually exclusive")
    _ = float(args.first_frame_token_weight)

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

    low_state, high_state, low_ptq_path, high_ptq_path = _load_i2v_ptq_states(args.ptq_dir)
    if not args.search_high_only:
        _apply_ptq_state_to_model(pipe.low_noise_model, low_state, "low_noise_model", low_ptq_path)
    if not args.search_low_only:
        _apply_ptq_state_to_model(pipe.high_noise_model, high_state, "high_noise_model", high_ptq_path)

    set_quant_modules_act_policy(pipe.low_noise_model, "low_noise_model", None, act_scale_method=args.scale_method)
    set_quant_modules_act_policy(pipe.high_noise_model, "high_noise_model", None, act_scale_method=args.scale_method)
    timesteps, boundary_timestep, expert_timestep_schedule = build_wan_i2v_timestep_schedule(
        num_train_timesteps=pipe.num_train_timesteps,
        boundary=pipe.boundary,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        shift=args.sample_shift,
        device=pipe.device,
    )
    LOGGER.info(
        "Activation policy timestep schedule: solver=%s steps=%d boundary=%.3f high_steps=%d low_steps=%d first_t=%.3f last_t=%.3f",
        args.sample_solver,
        int(args.sample_steps),
        float(boundary_timestep),
        len(expert_timestep_schedule["high_timestep_list"]),
        len(expert_timestep_schedule["low_timestep_list"]),
        float(timesteps[0].detach().cpu().item()) if len(timesteps) else float("nan"),
        float(timesteps[-1].detach().cpu().item()) if len(timesteps) else float("nan"),
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

    candidates_clip = _parse_csv_floats(args.candidates_clip)
    candidates_exp_offset = _parse_csv_ints(args.candidates_exp_offset)
    candidates = [(clip, exp) for clip in candidates_clip for exp in candidates_exp_offset]
    if not candidates:
        raise ValueError("No activation policy candidates were provided")

    stats = defaultdict(lambda: defaultdict(lambda: {"num": 0.0, "den": 0.0, "samples": 0.0}))
    handles = []
    if not args.search_high_only:
        handles.extend(_register_hooks(pipe.low_noise_model, "low_noise_model", args.module_regex, stats, candidates, args))
    if not args.search_low_only:
        handles.extend(_register_hooks(pipe.high_noise_model, "high_noise_model", args.module_regex, stats, candidates, args))
    if args.dry_run:
        LOGGER.info("Dry run complete; hooks that would be registered: %d", len(handles))
        for handle in handles:
            handle.remove()
        return

    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)
    img = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        _ = pipe.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )
    for handle in handles:
        handle.remove()

    policy = _build_policy(stats, candidates_clip, candidates_exp_offset, args.scale_method, args.metric)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote activation policy JSON: %s", out_path)
    _print_summary(policy)


if __name__ == "__main__":
    main()
