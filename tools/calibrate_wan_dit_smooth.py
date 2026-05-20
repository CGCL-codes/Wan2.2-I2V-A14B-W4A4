#!/usr/bin/env python3
"""Collect Wan I2V DiT Linear input activation stats and save to act.pt."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from quant.wan_smooth import (
    WanActStatConfig,
    build_wan_smoothed_weights,
    calibrate_wan_acts,
    save_wan_act_stats,
    save_wan_smoothed_weights,
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


def _resolve_manifest(msvd_root: str, manifest: str) -> str:
    if manifest:
        return manifest
    root = Path(msvd_root)
    candidates = [
        root / "msvd_wan_i2v_calib_manifest.jsonl",
        root / "msvd_wan_i2v_manifest.jsonl",
        root / "manifest.jsonl",
        root / "manifests" / "msvd_wan_i2v_manifest.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"Cannot find manifest under {root}, please set --manifest.")


def _load_single_image_sample(
    image_path: str,
    prompt: str,
    target_frames: int,
    target_size: tuple[int, int],
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
            "source": "single_image",
        },
    }


def _prep_calib_item(pipe: wan.WanI2V, sample: Dict[str, Any]) -> Dict[str, Any]:
    device = pipe.device
    cond_chw = sample["cond_image"]  # [C,H,W], in [0,1]
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


def _iter_calib_items(
    pipe: wan.WanI2V,
    manifest_path: str,
    num_samples: int,
    target_frames: int,
    target_size: tuple[int, int],
    pad_mode: str,
    image_path: str = "",
    prompt: str = "",
) -> Iterable[Dict[str, Any]]:
    if image_path:
        sample = _load_single_image_sample(
            image_path=image_path,
            prompt=prompt,
            target_frames=target_frames,
            target_size=target_size,
        )
        item = _prep_calib_item(pipe, sample)
        logging.info(
            "Prepared single-image calibration sample: image=%s latent_shape=%s seq_len=%d",
            image_path,
            tuple(item["latent"].shape),
            int(item["seq_len"]),
        )
        yield item
        del sample, item
        if pipe.device.type == "cuda":
            torch.cuda.empty_cache()
        return

    ds = MSVDWanI2VDataset(
        manifest_path=manifest_path,
        target_frames=target_frames,
        target_size=target_size,
        pad_mode=pad_mode,  # type: ignore[arg-type]
        return_uint8=False,
        load_video_frames=False,
    )
    n = min(num_samples, len(ds))
    for i in range(n):
        logging.info("Preparing calibration sample %d/%d", i + 1, n)
        sample = ds[i]
        item = _prep_calib_item(pipe, sample)
        logging.info(
            "Prepared calibration sample %d/%d: latent_shape=%s seq_len=%d",
            i + 1,
            n,
            tuple(item["latent"].shape),
            int(item["seq_len"]),
        )
        yield item
        del sample, item
        if pipe.device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Calibrate Wan I2V DiT activation stats (act.pt)")
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--msvd_root", type=str, default="/home/wjh/MSVD")
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--image", type=str, default="")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--target_frames", type=int, default=61)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--pad_mode", type=str, default="repeat_last", choices=["repeat_last", "loop"])
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--sampling_steps", type=int, default=40)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--channels_dim", type=int, default=-1)
    p.add_argument("--save_dtype", type=str, default="float32")
    p.add_argument("--module_name_filter", type=str, default="")
    p.add_argument("--progress_interval", type=int, default=1)
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    p.add_argument("--export_wgt", action="store_true", default=False)
    p.add_argument("--wgt_alpha", type=float, default=0.5)
    p.add_argument("--wgt_eps", type=float, default=1e-8)
    p.add_argument("--wgt_save_dtype", type=str, default="bfloat16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = ""
    if args.image:
        logging.info("Single-image calibration: image=%s", args.image)
        if args.prompt:
            logging.info("Single-image prompt provided.")
        else:
            logging.info("Single-image prompt not provided; using empty prompt.")
    else:
        manifest = _resolve_manifest(args.msvd_root, args.manifest)
        logging.info("Manifest: %s", manifest)

    cfg = WAN_CONFIGS["i2v-A14B"]
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

    calib_iter = _iter_calib_items(
        pipe=pipe,
        manifest_path=manifest,
        num_samples=args.num_samples,
        target_frames=args.target_frames,
        target_size=(args.height, args.width),
        pad_mode=args.pad_mode,
        image_path=args.image,
        prompt=args.prompt,
    )

    stat_cfg = WanActStatConfig(
        max_calib_samples=args.num_samples,
        save_dtype=args.save_dtype,
        channels_dim=args.channels_dim,
        module_name_filter=args.module_name_filter or None,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        progress_interval=args.progress_interval,
        show_timestep_progress=(not args.disable_tqdm),
    )
    stats = calibrate_wan_acts(pipe, calib_iter, config=stat_cfg)

    act_path = out_dir / "act.pt"
    save_wan_act_stats(str(act_path), stats)
    logging.info("Saved activation statistics: %s", act_path)

    if args.export_wgt:
        wgt_payload = build_wan_smoothed_weights(
            pipe=pipe,
            act_stats=stats,
            alpha=args.wgt_alpha,
            eps=args.wgt_eps,
            save_dtype=args.wgt_save_dtype,
        )
        wgt_path = out_dir / "wgt.pt"
        save_wan_smoothed_weights(str(wgt_path), wgt_payload)
        logging.info("Saved smoothed weights: %s", wgt_path)


if __name__ == "__main__":
    main()
