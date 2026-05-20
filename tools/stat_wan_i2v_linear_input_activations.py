#!/usr/bin/env python3
"""Stat Wan2.2 I2V WanModel block-linear input activations with forward_pre_hook.

Targets:
- blocks.*.self_attn.q/k/v/o
- blocks.*.cross_attn.q/k/v/o
- blocks.*.ffn.0/2

For each target linear input activation x:
- compute per-channel stats on abs(x) over last dim: max / p99 / p50

Outputs (for low_noise_model and high_noise_model separately):
- *_activation_stats.json
- *_activation_stats.png
"""

from __future__ import annotations
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group

# non-interactive backend for servers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_SUFFIX_RE = re.compile(
    r"(?:^|\.)blocks\.(\d+)\.(self_attn|cross_attn)\.(q|k|v|o)$|(?:^|\.)blocks\.(\d+)\.ffn\.(0|2)$"
)


@dataclass
class ModuleStat:
    name: str
    block_index: int
    p50: torch.Tensor
    p99: torch.Tensor
    vmax: torch.Tensor


class HookCollector:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.stats: Dict[str, ModuleStat] = {}

    @staticmethod
    def _parse_block_index(name: str) -> int:
        m = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
        if m is None:
            return -1
        return int(m.group(1))

    @staticmethod
    def _is_target(name: str, module: torch.nn.Module) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        return TARGET_SUFFIX_RE.search(name) is not None

    @staticmethod
    def _pick_input_tensor(inp_tuple: Tuple[object, ...]) -> Optional[torch.Tensor]:
        for obj in inp_tuple:
            if torch.is_tensor(obj):
                return obj
        return None

    def _hook_fn(self, name: str):
        def _hook(module: torch.nn.Module, inp: Tuple[object, ...]):
            x = self._pick_input_tensor(inp)
            if x is None:
                return
            if x.numel() == 0:
                return

            # per-channel stats on abs(x), along last dim C
            xa = x.detach().to(torch.float32).abs()
            c = xa.shape[-1]
            xa2 = xa.reshape(-1, c)

            # single forward call expected; if multiple calls, merge by max and recompute quantile on concat is costly.
            # Here we keep first occurrence per module for deterministic one-shot stat.
            if name in self.stats:
                return

            q = torch.quantile(xa2, torch.tensor([0.50, 0.99], device=xa2.device), dim=0)
            p50 = q[0].cpu()
            p99 = q[1].cpu()
            vmax = xa2.max(dim=0).values.cpu()

            self.stats[name] = ModuleStat(
                name=name,
                block_index=self._parse_block_index(name),
                p50=p50,
                p99=p99,
                vmax=vmax,
            )

        return _hook

    def register(self) -> List[str]:
        targets = []
        for name, module in self.model.named_modules():
            if self._is_target(name, module):
                h = module.register_forward_pre_hook(self._hook_fn(name), with_kwargs=False)
                self.handles.append(h)
                targets.append(name)
        return targets

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _module_order_key(name: str) -> Tuple[int, int, str]:
    # enforce stable order like q,k,v,o then ffn.0,ffn.2 per block
    block_m = re.search(r"(?:^|\.)blocks\.(\d+)\.(.*)$", name)
    if block_m is None:
        return (10**9, 10**9, name)
    b = int(block_m.group(1))
    rest = block_m.group(2)

    order_map = {
        "self_attn.q": 0,
        "self_attn.k": 1,
        "self_attn.v": 2,
        "self_attn.o": 3,
        "cross_attn.q": 4,
        "cross_attn.k": 5,
        "cross_attn.v": 6,
        "cross_attn.o": 7,
        "ffn.0": 8,
        "ffn.2": 9,
    }
    return (b, order_map.get(rest, 10**6), name)


def _concat_for_plot(stats: Dict[str, ModuleStat]):
    names = sorted(stats.keys(), key=_module_order_key)
    p50_all = []
    p99_all = []
    vmax_all = []
    boundaries = []

    offset = 0
    for n in names:
        s = stats[n]
        p50_all.append(s.p50.numpy())
        p99_all.append(s.p99.numpy())
        vmax_all.append(s.vmax.numpy())
        offset += s.p50.numel()
        boundaries.append(offset)

    if not names:
        return names, np.array([]), np.array([]), np.array([]), []

    p50 = np.concatenate(p50_all, axis=0)
    p99 = np.concatenate(p99_all, axis=0)
    vmax = np.concatenate(vmax_all, axis=0)
    return names, p50, p99, vmax, boundaries


def _plot_stats(stats: Dict[str, ModuleStat], title: str, save_path: Path) -> None:
    names, p50, p99, vmax, boundaries = _concat_for_plot(stats)
    if p50.size == 0:
        raise RuntimeError(f"No stats to plot for {title}")

    x = np.arange(p50.size)

    fig = plt.figure(figsize=(12, 4.5))
    ax = fig.add_subplot(111)

    # layered fill (style close to provided figure)
    ax.fill_between(x, 0.0, p50, color="#6b8e23", alpha=0.85, label="50% Percentile")
    ax.fill_between(x, p50, p99, color="#f6b21a", alpha=0.9, label="99% Percentile")
    ax.fill_between(x, p99, vmax, color="#b1123b", alpha=0.82, label="Max")

    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", linewidth=0.35, alpha=0.5)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Weight Channel", fontsize=14)
    ax.set_ylabel("|activation|", fontsize=12)
    ax.grid(True, alpha=0.25)

    # right-side stacked legend panel
    leg = ax.legend(loc="upper right", framealpha=0.92, fontsize=12)
    leg.get_frame().set_facecolor("#f1f1f1")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _stats_to_jsonable(stats: Dict[str, ModuleStat]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "num_modules": len(stats),
        "modules": {},
    }
    for name in sorted(stats.keys(), key=_module_order_key):
        s = stats[name]
        out["modules"][name] = {
            "block_index": s.block_index,
            "num_channels": int(s.p50.numel()),
            "p50": s.p50.tolist(),
            "p99": s.p99.tolist(),
            "max": s.vmax.tolist(),
            "summary": {
                "p50_mean": float(s.p50.mean().item()),
                "p99_mean": float(s.p99.mean().item()),
                "max_mean": float(s.vmax.mean().item()),
                "max_max": float(s.vmax.max().item()),
            },
        }
    return out


def _prepare_inputs_for_one_forward(
    pipe: wan.WanI2V,
    prompt: str,
    image_path: str,
    max_area: int,
    frame_num: int,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], int, List[torch.Tensor]]:
    """Prepare one-shot I2V DiT inputs, aligned with wan/image2video.py logic."""
    device = pipe.device

    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w

    lat_h = round(
        math.sqrt(max_area * aspect_ratio)
        // pipe.vae_stride[1]
        // pipe.patch_size[1]
        * pipe.patch_size[1]
    )
    lat_w = round(
        math.sqrt(max_area / aspect_ratio)
        // pipe.vae_stride[2]
        // pipe.patch_size[2]
        * pipe.patch_size[2]
    )
    h = lat_h * pipe.vae_stride[1]
    w = lat_w * pipe.vae_stride[2]

    seq_len = ((F - 1) // pipe.vae_stride[0] + 1) * lat_h * lat_w // (
        pipe.patch_size[1] * pipe.patch_size[2]
    )
    seq_len = int(math.ceil(seq_len / pipe.sp_size)) * pipe.sp_size

    noise = torch.randn(
        16,
        (F - 1) // pipe.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        device=device,
    )

    msk = torch.ones(1, F, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(device)
        context = pipe.text_encoder([prompt], device)
        pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([prompt], torch.device("cpu"))
        context = [t.to(device) for t in context]

    y = pipe.vae.encode(
        [
            torch.concat(
                [
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, F - 1, h, w),
                ],
                dim=1,
            ).to(device)
        ]
    )[0]
    y = torch.concat([msk, y])

    x_list = [noise]
    y_list = [y]
    return x_list, context, y_list, seq_len


def _run_one_model_stat(
    model: torch.nn.Module,
    model_name: str,
    x_list: List[torch.Tensor],
    context: List[torch.Tensor],
    seq_len: int,
    y_list: List[torch.Tensor],
    timestep: int,
    dtype: torch.dtype,
) -> Dict[str, ModuleStat]:
    collector = HookCollector(model)
    targets = collector.register()
    if not targets:
        collector.remove()
        raise RuntimeError(f"No target modules matched in {model_name}")

    print(f"[{model_name}] hook targets: {len(targets)}")

    t = torch.tensor([timestep], device=x_list[0].device, dtype=torch.long)

    model.eval().requires_grad_(False)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        _ = model(x_list, t=t, context=context, seq_len=seq_len, y=y_list)

    collector.remove()
    return collector.stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stat Wan2.2 I2V linear input activations")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--frame_num", type=int, default=61)
    parser.add_argument("--out_dir", type=str, default="./activation_stats")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--ulysses_size", type=int, default=1, help="Sequence parallel size")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Enable FSDP for T5")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Enable FSDP for DiT")
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--low_timestep", type=int, default=100)
    parser.add_argument("--high_timestep", type=int, default=950)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        device_id = local_rank
    else:
        device_id = args.device_id
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError("t5_fsdp/dit_fsdp require distributed launch (torchrun).")
        if args.ulysses_size > 1:
            raise ValueError("ulysses_size > 1 requires distributed launch (torchrun).")

    if args.ulysses_size > 1:
        if args.ulysses_size != world_size:
            raise ValueError(
                f"ulysses_size ({args.ulysses_size}) must equal WORLD_SIZE ({world_size})."
            )
        init_distributed_group()

    try:
        out_dir = Path(args.out_dir)
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)

        cfg = WAN_CONFIGS["i2v-A14B"]
        max_area = MAX_AREA_CONFIGS[args.size]

        pipe = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=False,
            convert_model_dtype=args.convert_model_dtype,
        )

        x_list, context, y_list, seq_len = _prepare_inputs_for_one_forward(
            pipe=pipe,
            prompt=args.prompt,
            image_path=args.image,
            max_area=max_area,
            frame_num=args.frame_num,
        )

        low_stats = _run_one_model_stat(
            model=pipe.low_noise_model,
            model_name="low_noise_model",
            x_list=x_list,
            context=context,
            seq_len=seq_len,
            y_list=y_list,
            timestep=args.low_timestep,
            dtype=pipe.param_dtype,
        )

        high_stats = _run_one_model_stat(
            model=pipe.high_noise_model,
            model_name="high_noise_model",
            x_list=x_list,
            context=context,
            seq_len=seq_len,
            y_list=y_list,
            timestep=args.high_timestep,
            dtype=pipe.param_dtype,
        )

        if rank == 0:
            low_json = out_dir / "low_noise_model_activation_stats.json"
            high_json = out_dir / "high_noise_model_activation_stats.json"
            low_png = out_dir / "low_noise_model_activation_stats.png"
            high_png = out_dir / "high_noise_model_activation_stats.png"

            low_json.write_text(
                json.dumps(_stats_to_jsonable(low_stats), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            high_json.write_text(
                json.dumps(_stats_to_jsonable(high_stats), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            _plot_stats(low_stats, "Low Noise Model", low_png)
            _plot_stats(high_stats, "High Noise Model", high_png)

            print(f"Saved: {low_json}")
            print(f"Saved: {high_json}")
            print(f"Saved: {low_png}")
            print(f"Saved: {high_png}")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
