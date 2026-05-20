#!/usr/bin/env python3
"""Calibrate Wan DiT activation stats (acts.pt) from MSVD manifest dataset."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

import wan
from quant.dit_ptq import (
    DiTActivationQuantConfig,
    DiTQuantConfig,
    DiTWeightQuantConfig,
    ptq,
)
from tools.msvd_wan_i2v_dataset import MSVDWanI2VDataset
from wan.configs import WAN_CONFIGS


def _build_mask(frame_num: int, lat_h: int, lat_w: int, device: torch.device) -> torch.Tensor:
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk


def _prep_latents_for_sample(wan_i2v: wan.WanI2V, sample: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int]:
    device = wan_i2v.device

    video_tchw = sample["video_frames"]  # [T,C,H,W], float in [0,1]
    cond_chw = sample["cond_image"]  # [C,H,W], float in [0,1]
    frame_num = int(video_tchw.shape[0])

    target_cfthw = video_tchw.permute(1, 0, 2, 3).contiguous().to(device)
    target_cfthw = target_cfthw * 2.0 - 1.0

    cond_cfthw = torch.zeros_like(target_cfthw)
    cond_cfthw[:, 0] = cond_chw.to(device) * 2.0 - 1.0

    # VAE encode to DiT latent space. We only use VAE for data preparation.
    target_latent = wan_i2v.vae.encode([target_cfthw])[0]
    cond_latent = wan_i2v.vae.encode([cond_cfthw])[0]

    _, f_lat, lat_h, lat_w = target_latent.shape
    msk = _build_mask(frame_num=frame_num, lat_h=lat_h, lat_w=lat_w, device=device)
    y = torch.cat([msk, cond_latent], dim=0)

    seq_len = f_lat * lat_h * lat_w // (wan_i2v.patch_size[1] * wan_i2v.patch_size[2])
    seq_len = int(math.ceil(seq_len / wan_i2v.sp_size)) * wan_i2v.sp_size
    return target_latent, y, seq_len


def _resolve_manifest(msvd_root: str, manifest: str) -> str:
    if manifest:
        return manifest
    root = Path(msvd_root)
    candidates = [
        root / "msvd_wan_i2v_manifest.jsonl",
        root / "manifest.jsonl",
        root / "manifests" / "msvd_wan_i2v_manifest.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Cannot find manifest under {root}. Please set --manifest explicitly."
    )


def _build_calib_inputs_from_msvd(
    pipe: wan.WanI2V,
    manifest_path: str,
    calib_samples: int,
    timestep: int,
    target_frames: int,
    target_size: tuple[int, int],
    pad_mode: str,
) -> List[Dict[str, Any]]:
    ds = MSVDWanI2VDataset(
        manifest_path=manifest_path,
        target_frames=target_frames,
        target_size=target_size,
        pad_mode=pad_mode,  # type: ignore[arg-type]
        return_uint8=False,
    )
    n = min(calib_samples, len(ds))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        sample = ds[i]
        x, y, seq_len = _prep_latents_for_sample(pipe, sample)
        prompt = str(sample["prompt"])
        if not pipe.t5_cpu:
            pipe.text_encoder.model.to(pipe.device)
            context = pipe.text_encoder([prompt], pipe.device)
            pipe.text_encoder.model.cpu()
        else:
            context_cpu = pipe.text_encoder([prompt], torch.device("cpu"))
            context = [t.to(pipe.device) for t in context_cpu]
        out.append(
            {
                "x": [x],
                "t": torch.tensor([int(timestep)], device=pipe.device, dtype=torch.long),
                "context": context,
                "seq_len": int(seq_len),
                "y": [y],
                "meta": sample.get("meta", {}),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Calibrate Wan DiT activation stats cache from MSVD manifest")
    p.add_argument("--ckpt_dir", type=str, required=True, help="Wan2.2-I2V-A14B checkpoint root")
    p.add_argument("--subfolder", type=str, default="low_noise_model", choices=["low_noise_model", "high_noise_model"])
    p.add_argument("--msvd_root", type=str, default="/home/wjh/MSVD", help="MSVD root directory")
    p.add_argument("--manifest", type=str, default="", help="MSVD manifest jsonl (if empty, auto-resolve from msvd_root)")
    p.add_argument("--save_dir", type=str, required=True, help="Output dir for acts.pt/wgts.pt/branch.pt/scale.pt.")
    p.add_argument("--load_dir", type=str, default="", help="Optional cache dir to load existing acts.pt and skip recollect.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_samples", type=int, default=8)
    p.add_argument("--target_frames", type=int, default=61)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--pad_mode", type=str, default="repeat_last", choices=["repeat_last", "loop"])
    p.add_argument("--timestep", type=int, default=100, help="Calibration timestep for DiT forward")
    p.add_argument("--enable_wgts", action="store_true", default=False, help="Also run weight skeleton flow.")
    p.add_argument("--save_model", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = torch.device(args.device)

    manifest_path = _resolve_manifest(args.msvd_root, args.manifest)
    logging.info("Using MSVD manifest: %s", manifest_path)

    cfg_model = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg_model,
        checkpoint_dir=args.ckpt_dir,
        device_id=(device.index if device.index is not None else 0),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    model = pipe.low_noise_model if args.subfolder == "low_noise_model" else pipe.high_noise_model
    model.to(device).eval().requires_grad_(False)

    samples = _build_calib_inputs_from_msvd(
        pipe=pipe,
        manifest_path=manifest_path,
        calib_samples=args.max_samples,
        timestep=args.timestep,
        target_frames=args.target_frames,
        target_size=(args.height, args.width),
        pad_mode=args.pad_mode,
    )
    logging.info("Prepared calibration inputs: %d samples", len(samples))

    cfg = DiTQuantConfig(
        wgts=DiTWeightQuantConfig(enabled=args.enable_wgts),
        ipts=DiTActivationQuantConfig(enabled=True, calib_max_samples=args.max_samples, collect_per_channel=True),
    )

    _ = ptq(
        model,
        cfg,
        calib_samples=samples,
        load_dirpath=args.load_dir,
        save_dirpath=args.save_dir,
        save_model=args.save_model,
    )
    print(f"Done. Cache saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
