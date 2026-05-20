"""Calibration runner for Wan2.2 I2V-A14B DiT W4A4 MXFP4."""

from __future__ import annotations
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch

import wan
from quant.eval_hooks import compare_wan_models
from quant.replace import QuantReplaceConfig, load_quant_config, replace_wan_dit_linear_with_quant
from tools.msvd_wan_i2v_dataset import MSVDWanI2VDataset
from wan.configs import WAN_CONFIGS


def _build_mask(frame_num: int, lat_h: int, lat_w: int, device: torch.device) -> torch.Tensor:
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk


def _prep_latents_for_sample(wan_i2v: wan.WanI2V, sample: Dict[str, Any]):
    device = wan_i2v.device

    video_tchw = sample["video_frames"]  # [T,C,H,W], float in [0,1]
    cond_chw = sample["cond_image"]      # [C,H,W], float in [0,1]
    frame_num = int(video_tchw.shape[0])

    target_cfthw = video_tchw.permute(1, 0, 2, 3).contiguous().to(device)
    target_cfthw = target_cfthw * 2.0 - 1.0

    cond_cfthw = torch.zeros_like(target_cfthw)
    cond_cfthw[:, 0] = cond_chw.to(device) * 2.0 - 1.0

    # VAE encode to DiT latent space
    target_latent = wan_i2v.vae.encode([target_cfthw])[0]
    cond_latent = wan_i2v.vae.encode([cond_cfthw])[0]

    _, f_lat, lat_h, lat_w = target_latent.shape
    msk = _build_mask(frame_num=frame_num, lat_h=lat_h, lat_w=lat_w, device=device)
    y = torch.cat([msk, cond_latent], dim=0)

    seq_len = f_lat * lat_h * lat_w // (wan_i2v.patch_size[1] * wan_i2v.patch_size[2])
    return target_latent, y, seq_len


def _chunks(lst: List[Any], bs: int):
    for i in range(0, len(lst), bs):
        yield lst[i : i + bs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Calibrate Wan2.2 I2V DiT W4A4 MXFP4")
    parser.add_argument("--model_path", type=str, required=True, help="Wan2.2-I2V-A14B ckpt dir")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest jsonl path")
    parser.add_argument("--output_dir", type=str, default="./quant_out")
    parser.add_argument("--quant_config", type=str, default="configs/wan_i2v_w4a4_mxfp4.yaml")
    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--target_frames", type=int, default=61)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--pad_mode", type=str, default="repeat_last", choices=["repeat_last", "loop"])
    parser.add_argument("--weight_group_size", type=int, default=None)
    parser.add_argument("--act_group_size", type=int, default=None)
    parser.add_argument("--save_quant_model", action="store_true", default=False)
    parser.add_argument("--run_eval_hooks", action="store_true", default=False)
    parser.add_argument("--eval_level", type=str, default="block", choices=["block", "linear"])
    parser.add_argument("--device_id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = WAN_CONFIGS["i2v-A14B"]
    qcfg = load_quant_config(args.quant_config)
    if args.weight_group_size is not None:
        qcfg.weight_group_size = args.weight_group_size
    if args.act_group_size is not None:
        qcfg.act_group_size = args.act_group_size

    (out_dir / "quant_config_resolved.json").write_text(
        json.dumps(qcfg.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logging.info("Loading WanI2V pipeline from %s", args.model_path)
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.model_path,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )

    # Keep reference copies optionally for error reports.
    ref_low = copy.deepcopy(pipe.low_noise_model).to(pipe.device).eval() if args.run_eval_hooks else None
    ref_high = copy.deepcopy(pipe.high_noise_model).to(pipe.device).eval() if args.run_eval_hooks else None

    # Replace DiT linears with QuantLinear.
    low_summary = replace_wan_dit_linear_with_quant(
        pipe.low_noise_model,
        config=qcfg,
        log_dir=str(out_dir / "replace_logs"),
        model_tag="low_noise_model",
    )
    high_summary = replace_wan_dit_linear_with_quant(
        pipe.high_noise_model,
        config=qcfg,
        log_dir=str(out_dir / "replace_logs"),
        model_tag="high_noise_model",
    )

    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)

    ds = MSVDWanI2VDataset(
        manifest_path=args.manifest,
        target_frames=args.target_frames,
        target_size=(args.height, args.width),
        pad_mode=args.pad_mode,
        return_uint8=False,
    )

    n = min(args.calib_samples, len(ds))
    samples = [ds[i] for i in range(n)]
    logging.info("Calibration samples: %d", n)

    num_steps = 0
    for batch_idx, batch in enumerate(_chunks(samples, args.batch_size)):
        x_list = []
        y_list = []
        seq_lens = []
        prompts = []

        for sample in batch:
            x, y, seq_len = _prep_latents_for_sample(pipe, sample)
            x_list.append(x)
            y_list.append(y)
            seq_lens.append(seq_len)
            prompts.append(sample["prompt"])

        if not pipe.t5_cpu:
            pipe.text_encoder.model.to(pipe.device)
            context = pipe.text_encoder(prompts, pipe.device)
            pipe.text_encoder.model.cpu()
        else:
            context_cpu = pipe.text_encoder(prompts, torch.device("cpu"))
            context = [t.to(pipe.device) for t in context_cpu]

        seq_len = int(max(seq_lens))
        seq_len = int((seq_len + pipe.sp_size - 1) // pipe.sp_size * pipe.sp_size)

        t_low = torch.full((len(batch),), 100, device=pipe.device, dtype=torch.long)
        t_high = torch.full((len(batch),), 950, device=pipe.device, dtype=torch.long)

        model_inputs_low = {"x": x_list, "t": t_low, "context": context, "seq_len": seq_len, "y": y_list}
        model_inputs_high = {"x": x_list, "t": t_high, "context": context, "seq_len": seq_len, "y": y_list}

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype):
            pipe.low_noise_model(**model_inputs_low)
            pipe.high_noise_model(**model_inputs_high)

        if args.run_eval_hooks and batch_idx == 0 and ref_low is not None and ref_high is not None:
            low_report = out_dir / "eval_low_noise_report.json"
            high_report = out_dir / "eval_high_noise_report.json"
            compare_wan_models(ref_low, pipe.low_noise_model, model_inputs_low, str(low_report), level=args.eval_level)
            compare_wan_models(ref_high, pipe.high_noise_model, model_inputs_high, str(high_report), level=args.eval_level)
            logging.info("Eval hook reports saved: %s , %s", low_report, high_report)

        num_steps += 1
        logging.info("Calibration step %d done (batch=%d)", num_steps, len(batch))

    summary = {
        "calib_samples": n,
        "batch_size": args.batch_size,
        "num_steps": num_steps,
        "low_noise_replace_summary": low_summary,
        "high_noise_replace_summary": high_summary,
    }
    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.save_quant_model:
        qout = out_dir / "quantized_ckpt"
        (qout / "low_noise_model").mkdir(parents=True, exist_ok=True)
        (qout / "high_noise_model").mkdir(parents=True, exist_ok=True)
        torch.save(pipe.low_noise_model.state_dict(), qout / "low_noise_model" / "mxfp4_state.pt")
        torch.save(pipe.high_noise_model.state_dict(), qout / "high_noise_model" / "mxfp4_state.pt")
        logging.info("Saved quantized model states to %s", qout)

    logging.info("Calibration complete.")


if __name__ == "__main__":
    main()
