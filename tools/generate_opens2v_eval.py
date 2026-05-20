#!/usr/bin/env python3
"""Batch-generate Wan I2V videos for OpenS2V-Eval samples.

This script reads one or more OpenS2V-Eval json files, resolves each sample's
reference image and prompt, then runs Wan I2V generation in-process.

Output layout:
    output_dir/
      <videoid>.mp4
      generation_manifest.jsonl
      missing.txt
      failed.txt
      _logs/
        <videoid>.log

Notes:
- OpenS2V `img_paths` is a list and may contain multiple reference images.
  The current Wan I2V entry uses a single image, so we default to `img_paths[0]`.
  If Wan supports multi-image input in the future, this is the natural place
  to extend.
"""

from __future__ import annotations

import argparse
import traceback
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

import wan
from tools.infer_wan_i2v_svdquant import (
    _apply_ptq_state_to_model,
    _load_i2v_ptq_states,
    _parse_keep_fp_blocks,
)
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video, str2bool

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


LOGGER = logging.getLogger("generate_opens2v_eval")


@dataclass(frozen=True)
class EvalSample:
    videoid: str
    prompt: str
    img_paths: List[str]
    used_image: Path
    source_json: Path
    source_index: int
    raw_sample: Dict[str, Any]
    derived_id: bool


def parse_args() -> argparse.Namespace:
    default_eval_root = PROJECT_ROOT / "OpenS2V-Eval"
    default_eval_jsons = [
        default_eval_root / "Single-Domain_Eval.json",
        default_eval_root / "Human-Domain_Eval.json",
        default_eval_root / "Open-Domain_Eval.json",
    ]
    parser = argparse.ArgumentParser(
        description="Batch-generate Wan I2V videos for OpenS2V-Eval JSON files in-process."
    )
    parser.add_argument("--eval-json", "--eval_json", nargs="+", default=[str(p) for p in default_eval_jsons], help="One or more OpenS2V-Eval json files.")
    parser.add_argument("--image-root", "--image_root", default=str(default_eval_root), help="OpenS2V-Eval root directory, typically containing Images/.")
    parser.add_argument("--output-dir", "--output_dir", required=True, help="Directory to save generated mp4 outputs and logs.")
    parser.add_argument("--ckpt-dir", "--ckpt_dir", default='../Wan2.2-I2V-A14B-bf16', help="Wan checkpoint directory.")
    parser.add_argument("--task", default="i2v-A14B", help="Wan task name passed to generate.py.")
    parser.add_argument("--mode", choices=["bf16", "quant"], default="quant", help="Run full precision/bf16 or SVDQuant inference.")
    parser.add_argument("--ptq-dir", "--ptq_dir", default="", help="Quant mode PTQ artifact. Supports ptq_stats.pt, a dir with ptq_stats.pt, or legacy split ptq_state.pt files.")
    parser.add_argument("--low-keep-fp-blocks", "--low_keep_fp_blocks", type=str, default="", help="Quant mode low-noise blocks to keep as original Linear, e.g. 0,1,38-40.")
    parser.add_argument("--high-keep-fp-blocks", "--high_keep_fp_blocks", type=str, default="", help="Quant mode high-noise blocks to keep as original Linear, e.g. 0,1,38-40.")
    parser.add_argument("--size", default="832*480", choices=list(SIZE_CONFIGS.keys()), help="Wan I2V size key.")
    parser.add_argument("--sample-steps", "--sample_steps", type=int, default=None, help="Optional sampling steps passed as --sample_steps.")
    parser.add_argument("--num-frames", "--num_frames", "--frame_num", type=int, default=None, help="Optional frame count.")
    parser.add_argument("--sample-shift", "--sample_shift", type=float, default=None, help="Sampling shift factor.")
    parser.add_argument("--sample-guide-scale", "--sample_guide_scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--sample-solver", "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"], help="Sampler.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed passed to generate.py as --base_seed.")
    parser.add_argument("--num-samples", "--num_samples", type=int, default=None, help="Only process the first N samples after --start.")
    parser.add_argument("--limit", type=int, default=180, help="Alias for --num-samples.")
    parser.add_argument("--start", type=int, default=0, help="Start from flattened sample index N.")
    parser.add_argument("--device-id", "--device_id", type=int, default=1)
    parser.add_argument("--offload-model", "--offload_model", type=str2bool, default=True)
    parser.add_argument("--t5-cpu", "--t5_cpu", action="store_true", default=False)
    parser.add_argument("--convert-model-dtype", "--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False, help="Regenerate outputs even if the mp4 already exists.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print commands without executing them.")
    return parser.parse_args()


def load_json_file(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise json.JSONDecodeError(
                        f"{path}:{line_no}: {exc.msg}",
                        exc.doc,
                        exc.pos,
                    ) from exc
        return records

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_samples(obj: Any, source_json: Path) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                raise TypeError(f"{source_json}: expected sample dict inside list, got {type(item).__name__}")
            yield item
        return

    if isinstance(obj, dict):
        for key in ("data", "samples", "items", "annotations"):
            value = obj.get(key)
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        raise TypeError(f"{source_json}: expected sample dict inside '{key}', got {type(item).__name__}")
                    yield item
                return

        # OpenS2V-Eval json commonly uses a top-level mapping:
        # {"singlehuman_1": {...}, "singlehuman_2": {...}, ...}
        # In that case the mapping key itself is the stable sample id.
        if obj and all(isinstance(v, dict) for v in obj.values()):
            for key, value in obj.items():
                sample = dict(value)
                if "videoid" not in sample and "video_id" not in sample and "id" not in sample and "name" not in sample:
                    sample["videoid"] = str(key)
                yield sample
            return

    raise TypeError(f"{source_json}: unsupported json top-level structure {type(obj).__name__}")


def resolve_image_path(image_root: Path, raw_path: str) -> Optional[Path]:
    raw = Path(str(raw_path))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            image_root / raw,
            image_root / "Images" / raw,
            image_root / raw.name,
            image_root / "Images" / raw.name,
        ]
    )
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def derive_videoid(sample: Dict[str, Any], source_json: Path, source_index: int, img_paths: Sequence[str]) -> tuple[str, bool]:
    for key in ("videoid", "video_id", "id", "name"):
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, False

    stem = ""
    if img_paths:
        stem = Path(img_paths[0]).stem.strip()
    if not stem:
        stem = source_json.stem
    stable_blob = json.dumps(sample, ensure_ascii=False, sort_keys=True)
    suffix = hashlib.sha1(f"{source_json}:{source_index}:{stable_blob}".encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{suffix}", True


def parse_one_sample(sample: Dict[str, Any], source_json: Path, source_index: int, image_root: Path) -> EvalSample:
    prompt = str(sample.get("prompt", "")).strip()
    img_paths_raw = sample.get("img_paths", None)
    if img_paths_raw is None:
        raise KeyError(f"{source_json}[{source_index}] missing 'img_paths'")
    if not isinstance(img_paths_raw, list) or not img_paths_raw:
        raise ValueError(f"{source_json}[{source_index}] expects non-empty list in 'img_paths'")

    img_paths = [str(x) for x in img_paths_raw]
    used_image = resolve_image_path(image_root, img_paths[0])
    if used_image is None:
        raise FileNotFoundError(
            f"{source_json}[{source_index}] cannot resolve first image path: {img_paths[0]!r}"
        )

    videoid, derived_id = derive_videoid(sample, source_json, source_index, img_paths)
    return EvalSample(
        videoid=videoid,
        prompt=prompt,
        img_paths=img_paths,
        used_image=used_image,
        source_json=source_json,
        source_index=source_index,
        raw_sample=sample,
        derived_id=derived_id,
    )


def load_eval_samples(eval_jsons: Sequence[Path], image_root: Path) -> List[EvalSample]:
    samples: List[EvalSample] = []
    seen_videoids: Dict[str, EvalSample] = {}
    for source_json in eval_jsons:
        obj = load_json_file(source_json)
        for idx, sample in enumerate(iter_json_samples(obj, source_json)):
            parsed = parse_one_sample(sample, source_json, idx, image_root)
            existing = seen_videoids.get(parsed.videoid)
            if existing is not None:
                LOGGER.warning(
                    "Duplicate videoid %r detected at %s[%d]; keeping first occurrence from %s[%d] and skipping later one. "
                    "first_image=%s later_image=%s",
                    parsed.videoid,
                    source_json,
                    idx,
                    existing.source_json,
                    existing.source_index,
                    existing.used_image,
                    parsed.used_image,
                )
                continue
            seen_videoids[parsed.videoid] = parsed
            samples.append(parsed)
    return samples


def select_samples(samples: Sequence[EvalSample], start: int, limit: Optional[int]) -> List[EvalSample]:
    if start < 0:
        raise ValueError(f"--start must be >= 0, got {start}")
    subset = list(samples[start:])
    if limit is not None:
        if limit < 0:
            raise ValueError(f"--limit must be >= 0, got {limit}")
        subset = subset[:limit]
    return subset


def append_manifest_record(manifest_path: Path, record: Dict[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def jsonable_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable_value(x) for x in value]
    if isinstance(value, list):
        return [jsonable_value(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_generation_defaults(args: argparse.Namespace) -> None:
    cfg = WAN_CONFIGS[args.task]
    if args.num_frames is None:
        args.num_frames = cfg.frame_num
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale


def build_pipeline(args: argparse.Namespace) -> wan.WanI2V:
    cfg = WAN_CONFIGS[args.task]
    LOGGER.info("Creating WanI2V pipeline: mode=%s ckpt=%s", args.mode, args.ckpt_dir)
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
    if args.mode == "quant":
        if not args.ptq_dir:
            raise ValueError("--ptq-dir is required when --mode quant")
        LOGGER.info("Loading SVDQuant PTQ states: %s", args.ptq_dir)
        low_state, high_state, low_path, high_path = _load_i2v_ptq_states(args.ptq_dir)
        _apply_ptq_state_to_model(
            pipe.low_noise_model,
            low_state,
            "low_noise_model",
            low_path,
            keep_fp_blocks=_parse_keep_fp_blocks(args.low_keep_fp_blocks),
        )
        _apply_ptq_state_to_model(
            pipe.high_noise_model,
            high_state,
            "high_noise_model",
            high_path,
            keep_fp_blocks=_parse_keep_fp_blocks(args.high_keep_fp_blocks),
        )
        pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
        pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)
    return pipe


def run_one_sample(
    args: argparse.Namespace,
    pipe: Optional[wan.WanI2V],
    sample: EvalSample,
    output_dir: Path,
    logs_dir: Path,
    manifest_path: Path,
) -> tuple[bool, bool]:
    save_file = output_dir / f"{sample.videoid}.mp4"
    log_file = logs_dir / f"{sample.videoid}.log"
    timestamp = datetime.now().isoformat(timespec="seconds")

    skipped_existing = save_file.exists() and save_file.stat().st_size > 0 and not args.overwrite
    record = {
        "videoid": sample.videoid,
        "prompt": sample.prompt,
        "img_paths": sample.img_paths,
        "used_image": str(sample.used_image),
        "save_file": str(save_file),
        "mode": args.mode,
        "ptq_dir": args.ptq_dir if args.mode == "quant" else "",
        "low_keep_fp_blocks": args.low_keep_fp_blocks if args.mode == "quant" else "",
        "high_keep_fp_blocks": args.high_keep_fp_blocks if args.mode == "quant" else "",
        "size": args.size,
        "sample_steps": int(args.sample_steps),
        "frame_num": int(args.num_frames),
        "sample_shift": jsonable_value(args.sample_shift),
        "sample_guide_scale": jsonable_value(args.sample_guide_scale),
        "sample_solver": args.sample_solver,
        "seed": int(args.seed),
        "source_json": str(sample.source_json),
        "source_index": int(sample.source_index),
        "generated_at": timestamp,
        "derived_id": bool(sample.derived_id),
        "status": "skipped_existing" if skipped_existing else ("dry_run" if args.dry_run else "pending"),
    }

    if skipped_existing:
        append_manifest_record(manifest_path, record)
        return True, False

    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False))
        append_manifest_record(manifest_path, record)
        return True, False

    try:
        if pipe is None:
            raise RuntimeError("Internal error: pipeline is not initialized outside dry-run mode")
        LOGGER.info("Generating %s | image=%s", sample.videoid, sample.used_image)
        img = Image.open(sample.used_image).convert("RGB")
        video = pipe.generate(
            sample.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.num_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.seed,
            offload_model=args.offload_model,
        )
        save_video(
            tensor=video[None],
            save_file=str(save_file),
            fps=WAN_CONFIGS[args.task].sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        record["status"] = "ok"
        log_file.write_text(
            json.dumps({"status": "ok", "record": record}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        del video, img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ok = True
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        log_file.write_text(traceback.format_exc(), encoding="utf-8")
        LOGGER.exception("Failed to generate %s", sample.videoid)
        ok = False
    append_manifest_record(manifest_path, record)
    return ok, True


def write_line_file(path: Path, lines: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def iter_with_progress(samples: Sequence[EvalSample], dry_run: bool) -> Iterable[EvalSample]:
    desc = "Dry-run OpenS2V generation" if dry_run else "OpenS2V generation"
    if tqdm is None:
        LOGGER.info("tqdm is not installed; continuing without a progress bar.")
        return samples
    return tqdm(samples, desc=desc, unit="video")


def ensure_paths(args: argparse.Namespace) -> None:
    eval_jsons = [Path(p) for p in args.eval_json]
    for path in eval_jsons:
        if not path.exists():
            raise FileNotFoundError(f"--eval-json not found: {path}")
    for label, path_str in (
        ("--image-root", args.image_root),
        ("--ckpt-dir", args.ckpt_dir),
    ):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.task != "i2v-A14B":
        raise ValueError("This in-process OpenS2V script currently supports only --task i2v-A14B.")
    if args.mode == "quant" and not args.ptq_dir:
        raise ValueError("--ptq-dir is required when --mode quant")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_paths(args)
    resolve_generation_defaults(args)

    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    logs_dir = output_dir / "_logs"
    manifest_path = output_dir / "generation_manifest.jsonl"
    missing_path = output_dir / "missing.txt"
    failed_path = output_dir / "failed.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    eval_jsons = [Path(p).resolve() for p in args.eval_json]
    all_samples = load_eval_samples(eval_jsons, image_root=image_root)
    sample_limit = args.num_samples if args.num_samples is not None else args.limit
    samples = select_samples(all_samples, start=int(args.start), limit=sample_limit)

    missing: List[str] = []
    failed: List[str] = []

    pipe = None if args.dry_run else build_pipeline(args)

    for sample in iter_with_progress(samples, dry_run=bool(args.dry_run)):
        ok, attempted = run_one_sample(
            args=args,
            pipe=pipe,
            sample=sample,
            output_dir=output_dir,
            logs_dir=logs_dir,
            manifest_path=manifest_path,
        )
        save_file = output_dir / f"{sample.videoid}.mp4"
        file_ok = save_file.exists() and save_file.stat().st_size > 0
        if attempted and not ok:
            failed.append(sample.videoid)
        if not file_ok:
            missing.append(sample.videoid)

    write_line_file(missing_path, missing)
    write_line_file(failed_path, failed)

    print(
        json.dumps(
            {
                "total_loaded": len(all_samples),
                "selected": len(samples),
                "missing_count": len(missing),
                "failed_count": len(failed),
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "missing_txt": str(missing_path),
                "failed_txt": str(failed_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
