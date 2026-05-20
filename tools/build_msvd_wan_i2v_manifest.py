#!/usr/bin/env python3
"""Build a minimal Wan2.2-I2V DiT calibration manifest from MSVD."""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MSVD -> Wan2.2 I2V manifest (jsonl) for DiT calibration"
    )
    parser.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="Root dir of MSVD videos, e.g. /home/wjh/MSVD/videos",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        required=True,
        help="Annotation file or directory. Supports .json and .jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output manifest jsonl path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=128,
        help="Max number of samples to randomly select",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Comma separated splits, e.g. train or train,val",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def _normalize_splits(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _infer_split(record: Dict[str, Any], source_name: str) -> str:
    for k in ("split", "subset", "set"):
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()

    low = source_name.lower()
    for name in ("train", "val", "valid", "validation", "test"):
        if name in low:
            return "val" if name in ("valid", "validation") else name
    return "train"


def _extract_captions(record: Dict[str, Any]) -> List[str]:
    for k in ("caption", "captions", "sentence", "sentences", "text", "texts"):
        v = record.get(k)
        if isinstance(v, str):
            c = v.strip()
            return [c] if c else []
        if isinstance(v, list):
            out = []
            for item in v:
                s = _safe_str(item).strip()
                if s:
                    out.append(s)
            return out
    return []


def _extract_video_name(record: Dict[str, Any]) -> Optional[str]:
    for k in ("video", "video_path", "video_file", "filename", "file_name"):
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return Path(v.strip()).name

    vid = record.get("video_id")
    if isinstance(vid, str) and vid.strip():
        name = vid.strip()
        if Path(name).suffix:
            return Path(name).name
        return f"{name}.avi"
    return None


def _resolve_video_path(video_root: Path, video_name: str) -> Optional[Path]:
    candidates = [
        video_root / video_name,
        video_root / "YouTubeClips" / video_name,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def _read_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    logging.warning("Bad JSONL line %d in %s: %s", i, path, e)
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    logging.warning("Skip non-dict JSONL line %d in %s", i, path)
        return records

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                records.append(item)
            else:
                logging.warning("Skip non-dict item idx=%d in %s", i, path)
        return records

    if isinstance(obj, dict):
        # Common cases: {'data': [...]} or split->list
        if "data" in obj and isinstance(obj["data"], list):
            for i, item in enumerate(obj["data"]):
                if isinstance(item, dict):
                    records.append(item)
                else:
                    logging.warning("Skip non-dict data item idx=%d in %s", i, path)
            return records

        found_any = False
        for _, v in obj.items():
            if isinstance(v, list):
                found_any = True
                for item in v:
                    if isinstance(item, dict):
                        records.append(item)
        if found_any:
            return records

    logging.warning("Unsupported annotation format in %s", path)
    return records


def _read_video_meta(video_path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int], Optional[str]]:
    if cv2 is None:
        return None, None, None, None, "cv2_not_available"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None, None, "open_failed"

    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    def _cast_num(v: float, is_float: bool) -> Optional[Any]:
        if v is None:
            return None
        try:
            if is_float:
                return float(v) if v > 0 else None
            iv = int(round(v))
            return iv if iv > 0 else None
        except Exception:
            return None

    return (
        _cast_num(fps, True),
        _cast_num(nframes, False),
        _cast_num(width, False),
        _cast_num(height, False),
        None,
    )


def _iter_annotation_files(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted([x for x in p.rglob("*") if x.suffix.lower() in {".json", ".jsonl"}])
        return files
    return []


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    rng = random.Random(args.seed)
    wanted_splits = set(_normalize_splits(args.splits))
    video_root = Path(args.video_root)
    anno_path = Path(args.annotation_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    anno_files = _iter_annotation_files(anno_path)
    if not anno_files:
        raise FileNotFoundError(f"No annotation file found in: {anno_path}")

    logging.info("Found %d annotation files", len(anno_files))

    missing_video = 0
    empty_caption = 0
    bad_video = 0
    split_filtered = 0
    malformed = 0
    bad_video_log_count = 0
    bad_video_log_limit = 20

    candidates: List[Dict[str, Any]] = []

    for ap in anno_files:
        records = _read_records(ap)
        logging.info("Load %d records from %s", len(records), ap)
        for i, rec in enumerate(records):
            split = _infer_split(rec, ap.name)
            if wanted_splits and split not in wanted_splits:
                split_filtered += 1
                continue

            video_name = _extract_video_name(rec)
            if not video_name:
                malformed += 1
                logging.warning("Skip record missing video name: file=%s idx=%d", ap, i)
                continue

            video_path = _resolve_video_path(video_root, video_name)
            if video_path is None:
                missing_video += 1
                logging.warning("Missing video file: %s (from %s idx=%d)", video_name, ap, i)
                continue

            caps = _extract_captions(rec)
            if not caps:
                empty_caption += 1
                logging.warning("Empty caption list: %s (from %s idx=%d)", video_name, ap, i)
                continue

            caption_index = rng.randrange(len(caps))
            prompt = caps[caption_index]

            fps, num_frames, width, height, err = _read_video_meta(video_path)
            if err is not None:
                bad_video += 1
                if bad_video_log_count < bad_video_log_limit:
                    logging.warning("Bad video metadata (%s): %s", err, video_path)
                    bad_video_log_count += 1

            sample_id = rec.get("video_id") or Path(video_name).stem
            item = {
                "id": _safe_str(sample_id),
                "split": split,
                "video_path": str(video_path),
                "prompt": prompt,
                "caption_index": caption_index,
                "fps": fps,
                "num_frames": num_frames,
                "width": width,
                "height": height,
            }
            candidates.append(item)

    if not candidates:
        raise RuntimeError("No valid candidates after filtering. Please check paths/splits/annotations.")

    if len(candidates) > args.max_samples:
        selected = rng.sample(candidates, args.max_samples)
    else:
        selected = candidates

    with output.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info("Manifest written: %s", output)
    logging.info("Selected: %d / Candidates: %d", len(selected), len(candidates))
    if bad_video > bad_video_log_limit:
        logging.info(
            "Bad video metadata warnings are truncated: shown=%d total=%d",
            bad_video_log_limit,
            bad_video,
        )
    logging.info(
        "Stats: split_filtered=%d missing_video=%d empty_caption=%d bad_video=%d malformed=%d",
        split_filtered,
        missing_video,
        empty_caption,
        bad_video,
        malformed,
    )


if __name__ == "__main__":
    main()
