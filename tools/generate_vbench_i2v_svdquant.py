#!/usr/bin/env python3
"""Generate Wan I2V videos for VBench-I2V evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOGGER = logging.getLogger("generate_vbench_i2v_svdquant")


DEFAULT_DIMENSIONS = [
    "i2v_subject",
    "i2v_background",
    "camera_motion",
]


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in ("yes", "true", "t", "1", "y"):
        return True
    if text in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate VBench-I2V samples with Wan I2V SVDQuant")
    p.add_argument("--full_info_json", type=str, default="VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json", help="VBench-I2V full info JSON.")
    p.add_argument("--vbench_root", type=str, default="VBench/vbench2_beta_i2v", help="VBench-I2V root directory.")
    p.add_argument("--ratio", type=str, default="16-9", help="VBench crop ratio folder, e.g. 1-1, 8-5, 7-4, 16-9.")
    p.add_argument("--image_folder", type=str, default="", help="Override image folder. Defaults to <vbench_root>/data/crop/<ratio>.")
    p.add_argument("--output_dir", type=str, default="outputs/vbench_quant", help="Directory for generated VBench videos.")
    p.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMENSIONS, help="VBench dimensions to generate.")
    p.add_argument("--samples_per_prompt", type=int, default=5, help="Number of videos per prompt.")
    p.add_argument("--start", type=int, default=0, help="Start offset after filtering prompts.")
    p.add_argument("--limit", type=int, default=None, help="Maximum number of filtered prompts to process.")
    p.add_argument("--seed", type=int, default=0, help="Base seed. Actual seed is seed + prompt_index * samples_per_prompt + sample_index.")
    p.add_argument("--overwrite", action="store_true", default=False, help="Regenerate existing mp4 files.")
    p.add_argument("--dry_run", action="store_true", default=False, help="List planned samples without loading Wan.")
    p.add_argument("--manifest", type=str, default="", help="JSONL manifest path. Defaults to <output_dir>/generation_manifest.jsonl.")

    p.add_argument("--mode", type=str, default="quant", choices=["bf16", "quant"], help="Run full precision/bf16 or SVDQuant PTQ.")
    p.add_argument("--ckpt_dir", type=str, default="../Wan2.2-I2V-A14B-bf16", help="Wan checkpoint directory.")
    p.add_argument("--ptq_dir", type=str, default="/home/wjh/Wan2.2/outputs/gptq_ptq", help="PTQ artifact path for quant mode.")
    p.add_argument("--size", type=str, default="480*832", help="Wan size key.")
    p.add_argument("--frame_num", type=int, default=61, help="Number of video frames.")
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampler.")
    p.add_argument("--sample_steps", type=int, default=40, help="Sampling steps.")
    p.add_argument("--sample_shift", type=float, default=None, help="Sampling shift.")
    p.add_argument("--sample_guide_scale", type=float, default=None, help="CFG scale.")
    p.add_argument("--device_id", type=int, default=1, help="CUDA device id.")
    p.add_argument("--offload_model", type=_str2bool, default=True, help="Offload model between submodules.")
    p.add_argument("--t5_cpu", action="store_true", default=False, help="Run T5 on CPU.")
    p.add_argument("--convert_model_dtype", action="store_true", default=True, help="Convert model dtype like Wan defaults.")
    p.add_argument("--low_keep_fp_blocks", type=str, default="", help="Low expert block indices to keep BF16.")
    p.add_argument("--high_keep_fp_blocks", type=str, default="", help="High expert block indices to keep BF16.")
    p.add_argument("--low_keep_fp_modules", type=str, default="", help="Comma-separated low expert module names to keep BF16.")
    p.add_argument("--high_keep_fp_modules", type=str, default="", help="Comma-separated high expert module names to keep BF16.")
    p.add_argument("--keep_fp_module_regex", type=str, default="", help="Regex for modules to keep BF16 in both experts.")
    p.add_argument("--low_keep_fp_module_regex", type=str, default="", help="Regex for low expert modules to keep BF16.")
    p.add_argument("--high_keep_fp_module_regex", type=str, default="", help="Regex for high expert modules to keep BF16.")
    p.add_argument("--print_replaced_modules", action="store_true", default=False, help="Log final quant/BF16 module lists.")
    p.add_argument("--act_policy_json", type=str, default="outputs/act_policy.json", help="Optional MXFP4 activation policy JSON.")
    p.add_argument("--act_scale_method", type=str, default="", choices=["", "ocp_floor", "safe_ceil"], help="Override activation MXFP4 scale method.")
    p.add_argument("--act_exp_offset", type=int, default=0, help="Override activation exponent offset when nonzero.")
    p.add_argument("--act_policy_print", action="store_true", default=False, help="Print activation policy setup summary.")
    p.add_argument("--freeze_condition_latent", action="store_true", default=True, help="Use infer script experiment that resets condition latent only after the final step.")
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


def _load_vbench_items(path: Path, dimensions: Set[str]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data).__name__}")
    items = []
    seen = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"{path}[{idx}] must be dict, got {type(item).__name__}")
        item_dims = set(str(x) for x in item.get("dimension", []))
        if dimensions and not (item_dims & dimensions):
            continue
        prompt = str(item.get("prompt_en", "")).strip()
        image_name = str(item.get("image_name", "")).strip()
        if not prompt or not image_name:
            LOGGER.warning("Skipping invalid VBench item %d: prompt=%r image_name=%r", idx, prompt, image_name)
            continue
        key = (prompt, image_name)
        if key in seen:
            continue
        seen.add(key)
        item = dict(item)
        item["_source_index"] = idx
        items.append(item)
    return items


def _append_manifest(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _setup_quant_if_needed(pipe: Any, args: argparse.Namespace) -> None:
    if args.mode != "quant":
        return
    from quant.act_policy import (
        build_wan_i2v_timestep_schedule,
        load_act_policy_json,
        maybe_patch_model_forward_for_act_context,
        set_quant_modules_act_policy,
    )
    from tools.infer_wan_i2v_svdquant import (
        _apply_ptq_state_to_model,
        _load_i2v_ptq_states,
        _parse_keep_fp_blocks,
        _parse_keep_fp_modules,
        _print_quant_fp_module_summary,
        _regex_matched_modules,
        _restore_bf16_modules,
        _snapshot_bf16_modules,
    )

    low_state, high_state, low_ptq_path, high_ptq_path = _load_i2v_ptq_states(args.ptq_dir)
    low_keep_fp_blocks = _parse_keep_fp_blocks(args.low_keep_fp_blocks)
    high_keep_fp_blocks = _parse_keep_fp_blocks(args.high_keep_fp_blocks)
    low_keep_fp_modules = _parse_keep_fp_modules(args.low_keep_fp_modules, pipe.low_noise_model)
    high_keep_fp_modules = _parse_keep_fp_modules(args.high_keep_fp_modules, pipe.high_noise_model)
    low_keep_fp_modules.update(_regex_matched_modules(pipe.low_noise_model, args.keep_fp_module_regex))
    high_keep_fp_modules.update(_regex_matched_modules(pipe.high_noise_model, args.keep_fp_module_regex))
    low_keep_fp_modules.update(_regex_matched_modules(pipe.low_noise_model, args.low_keep_fp_module_regex))
    high_keep_fp_modules.update(_regex_matched_modules(pipe.high_noise_model, args.high_keep_fp_module_regex))
    LOGGER.info("low_noise_model keep-fp module requests expanded to %d modules", len(low_keep_fp_modules))
    LOGGER.info("high_noise_model keep-fp module requests expanded to %d modules", len(high_keep_fp_modules))

    low_bf16_snapshot = _snapshot_bf16_modules(pipe.low_noise_model, low_keep_fp_modules, "low_noise_model")
    high_bf16_snapshot = _snapshot_bf16_modules(pipe.high_noise_model, high_keep_fp_modules, "high_noise_model")
    _apply_ptq_state_to_model(
        pipe.low_noise_model,
        low_state,
        "low_noise_model",
        low_ptq_path,
        keep_fp_blocks=low_keep_fp_blocks,
    )
    _apply_ptq_state_to_model(
        pipe.high_noise_model,
        high_state,
        "high_noise_model",
        high_ptq_path,
        keep_fp_blocks=high_keep_fp_blocks,
    )
    _restore_bf16_modules(pipe.low_noise_model, low_bf16_snapshot, "low_noise_model")
    _restore_bf16_modules(pipe.high_noise_model, high_bf16_snapshot, "high_noise_model")

    act_policy = load_act_policy_json(args.act_policy_json) if args.act_policy_json else None
    low_policy_modules = set_quant_modules_act_policy(
        pipe.low_noise_model,
        "low_noise_model",
        act_policy,
        act_scale_method=args.act_scale_method,
        act_exp_offset=args.act_exp_offset if args.act_exp_offset != 0 else None,
    )
    high_policy_modules = set_quant_modules_act_policy(
        pipe.high_noise_model,
        "high_noise_model",
        act_policy,
        act_scale_method=args.act_scale_method,
        act_exp_offset=args.act_exp_offset if args.act_exp_offset != 0 else None,
    )
    if args.act_policy_print:
        LOGGER.info(
            "Activation policy configured: json=%s low_modules=%d high_modules=%d override_scale_method=%s exp_offset=%d",
            args.act_policy_json or "<none>",
            low_policy_modules,
            high_policy_modules,
            args.act_scale_method or "<state/default>",
            int(args.act_exp_offset),
        )

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

    if args.print_replaced_modules:
        _print_quant_fp_module_summary(pipe.low_noise_model, "low_noise_model")
        _print_quant_fp_module_summary(pipe.high_noise_model, "high_noise_model")


def _generate_one(pipe: Any, args: argparse.Namespace, image_path: Path, prompt: str, seed: int) -> Any:
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS

    img = Image.open(image_path).convert("RGB")
    if args.freeze_condition_latent:
        from tools.infer_wan_i2v_svdquant import _generate_i2v_freezing_condition_latent

        local_args = argparse.Namespace(**vars(args))
        local_args.prompt = prompt
        local_args.base_seed = seed
        return _generate_i2v_freezing_condition_latent(pipe, local_args, img)
    return pipe.generate(
        prompt,
        img,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=seed,
        offload_model=args.offload_model,
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    dimensions = set(str(x) for x in args.dimensions)
    full_info_path = Path(args.full_info_json)
    image_folder = Path(args.image_folder) if args.image_folder else Path(args.vbench_root) / "data" / "crop" / args.ratio
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "generation_manifest.jsonl"

    items = _load_vbench_items(full_info_path, dimensions)
    selected = items[int(args.start):]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    LOGGER.info(
        "Loaded VBench items: total_filtered=%d selected=%d dimensions=%s image_folder=%s output=%s",
        len(items),
        len(selected),
        sorted(dimensions),
        image_folder,
        output_dir,
    )
    if args.dry_run:
        for idx, item in enumerate(selected[:20]):
            LOGGER.info("dry-run[%d] image=%s prompt=%s", idx, image_folder / item["image_name"], item["prompt_en"])
        LOGGER.info("Dry run complete. Planned videos: %d", len(selected) * int(args.samples_per_prompt))
        return

    _resolve_defaults(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

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
    _setup_quant_if_needed(pipe, args)
    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)

    generated = 0
    skipped = 0
    failed = 0
    for prompt_idx, item in enumerate(selected, start=int(args.start)):
        prompt = str(item["prompt_en"])
        image_path = image_folder / str(item["image_name"])
        if not image_path.exists():
            LOGGER.error("Missing VBench image: %s", image_path)
            failed += int(args.samples_per_prompt)
            continue
        for sample_idx in range(int(args.samples_per_prompt)):
            save_path = output_dir / f"{prompt}-{sample_idx}.mp4"
            seed = int(args.seed) + prompt_idx * int(args.samples_per_prompt) + sample_idx
            record: Dict[str, Any] = {
                "prompt": prompt,
                "sample_index": sample_idx,
                "seed": seed,
                "image_path": str(image_path),
                "image_name": item["image_name"],
                "dimensions": item.get("dimension", []),
                "image_type": item.get("image_type", ""),
                "source_index": item.get("_source_index"),
                "save_path": str(save_path),
                "mode": args.mode,
                "size": args.size,
                "frame_num": int(args.frame_num),
                "sample_steps": int(args.sample_steps),
                "sample_solver": args.sample_solver,
                "sample_shift": float(args.sample_shift),
                "sample_guide_scale": _jsonable(args.sample_guide_scale),
                "ratio": args.ratio,
            }
            if save_path.exists() and not args.overwrite:
                LOGGER.info("Skipping existing video: %s", save_path)
                record["status"] = "skipped_existing"
                _append_manifest(manifest_path, record)
                skipped += 1
                continue
            try:
                LOGGER.info(
                    "Generating [%d/%d] sample=%d seed=%d prompt=%s",
                    prompt_idx - int(args.start) + 1,
                    len(selected),
                    sample_idx,
                    seed,
                    prompt,
                )
                video = _generate_one(pipe, args, image_path, prompt, seed)
                save_video(
                    tensor=video[None],
                    save_file=str(save_path),
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                record["status"] = "ok"
                generated += 1
            except Exception as exc:
                LOGGER.exception("Failed generating prompt=%r sample=%d", prompt, sample_idx)
                record["status"] = "failed"
                record["error"] = repr(exc)
                failed += 1
            _append_manifest(manifest_path, record)

    LOGGER.info("Finished VBench generation: generated=%d skipped=%d failed=%d manifest=%s", generated, skipped, failed, manifest_path)


if __name__ == "__main__":
    main()
