#!/usr/bin/env python3
"""Plot raw vs smoothed activation group curves for Wan I2V QuantLinear layers.

MXFP4 activation quantization shares one scale per activation group (default
32 elements along the input-channel dimension). This tool keeps the existing
Wan I2V quantized denoise path, but the hook is intentionally lightweight:

    x_raw    = input before SmoothQuant-style activation scaling
    x_smooth = x_raw / s, the actual input consumed by activation QDQ

For each selected QuantLinear / QuantLinearWithBranch, it records per-group
absolute activation curves (p50, p99, max) before and after smoothing and
writes compact CSV/NPZ plus paper-style PNG plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn

from quant.quant_linear import QuantLinear, QuantLinearWithBranch
from quant.wan_smooth import _build_schedule, _register_linear_input_cast_hooks
from tools.infer_wan_i2v_svdquant import (
    _apply_ptq_state_to_model,
    _load_i2v_ptq_states,
    _parse_keep_fp_blocks,
)
from tools.stat_wan_i2v_isolated_layer_error import (
    ManifestSample,
    _build_filter_regex,
    _build_pipe,
    _expert_names,
    _load_manifest,
    _prep_sample,
    _size_to_hw,
)
from tools.stat_wan_i2v_propagated_layer_error import _remove_hooks, _run_model
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger("stat_wan_i2v_smooth_activation_distribution")


class GroupCurveStats:
    """Streaming per-group activation curves for one raw/smooth domain."""

    CURVES = ("p50", "p95", "p99", "p999", "max")

    def __init__(self, max_group_rows: int) -> None:
        self.max_group_rows = int(max_group_rows)
        self.calls = 0
        self.num_groups = 0
        self.group_size = 0
        self.sampled_rows = 0
        self.total_rows = 0
        self.weight = 0.0
        self.sum_curves: Dict[str, Optional[np.ndarray]] = {name: None for name in self.CURVES}
        self.max_curves: Dict[str, Optional[np.ndarray]] = {name: None for name in self.CURVES}

    @torch.no_grad()
    def update(self, x: torch.Tensor, channels_dim: int, group_size: int) -> None:
        grouped = _activation_groups(x.detach(), channels_dim=channels_dim, group_size=group_size)
        if grouped is None:
            return
        total_rows = int(grouped.shape[0])
        num_groups = int(grouped.shape[1])
        if total_rows == 0 or num_groups == 0:
            return

        sampled = _sample_group_rows(grouped, self.max_group_rows)
        curves = _group_abs_curves(sampled)
        weight = float(sampled.shape[0] * sampled.shape[-1])

        self.calls += 1
        self.num_groups = num_groups
        self.group_size = int(group_size)
        self.total_rows += total_rows
        self.sampled_rows += int(sampled.shape[0])
        self.weight += weight

        for name, values in curves.items():
            values64 = values.astype(np.float64, copy=False)
            if self.sum_curves[name] is None:
                self.sum_curves[name] = values64 * weight
                self.max_curves[name] = values.astype(np.float32, copy=True)
            else:
                self.sum_curves[name] += values64 * weight
                self.max_curves[name] = np.maximum(self.max_curves[name], values)

    def mean_curve(self, name: str) -> np.ndarray:
        values = self.sum_curves[name]
        if values is None or self.weight <= 0:
            return np.zeros((0,), dtype=np.float32)
        return (values / self.weight).astype(np.float32)

    def max_curve(self, name: str) -> np.ndarray:
        values = self.max_curves[name]
        if values is None:
            return np.zeros((0,), dtype=np.float32)
        return values.astype(np.float32, copy=False)


class ModuleGroupCurves:
    def __init__(self, max_group_rows: int) -> None:
        self.raw = GroupCurveStats(max_group_rows=max_group_rows)
        self.smooth = GroupCurveStats(max_group_rows=max_group_rows)
        self.first_step: Optional[int] = None
        self.last_step: Optional[int] = None

    def update(self, x_raw: torch.Tensor, x_smooth: torch.Tensor, channels_dim: int, group_size: int, step_idx: Optional[int]) -> None:
        self.raw.update(x_raw, channels_dim=channels_dim, group_size=group_size)
        self.smooth.update(x_smooth, channels_dim=channels_dim, group_size=group_size)
        if step_idx is not None:
            step_idx = int(step_idx)
            self.first_step = step_idx if self.first_step is None else min(self.first_step, step_idx)
            self.last_step = step_idx if self.last_step is None else max(self.last_step, step_idx)


class GroupCurveCollector:
    def __init__(self, module_meta: Dict[str, Dict[str, Dict[str, Any]]], max_group_rows: int) -> None:
        self.current_step: Optional[int] = None
        self.current_expert: Optional[str] = None
        self.raw_timesteps: Dict[str, Dict[int, float]] = {"high": {}, "low": {}}
        self.module_meta = module_meta
        self.max_group_rows = int(max_group_rows)
        self.stats: Dict[str, Dict[str, ModuleGroupCurves]] = {"high": {}, "low": {}}

    def begin_step(self, step_idx: int, raw_timestep: float, expert: str) -> None:
        self.current_step = int(step_idx)
        self.current_expert = expert
        self.raw_timesteps[expert][int(step_idx)] = float(raw_timestep)

    @torch.no_grad()
    def observe(self, expert: str, module_name: str, module: nn.Module, x_raw: torch.Tensor) -> None:
        if self.current_expert != expert:
            return
        qlinear = _inner_quant_linear(module)
        channels_dim = int(qlinear.input_channels_dim.item())
        group_size = int(qlinear.act_group_size)
        x_raw_detached = x_raw.detach()
        x_smooth = qlinear.smooth_input_only(x_raw_detached)

        rec = self.stats[expert].setdefault(module_name, ModuleGroupCurves(max_group_rows=self.max_group_rows))
        rec.update(
            x_raw=x_raw_detached,
            x_smooth=x_smooth,
            channels_dim=channels_dim,
            group_size=group_size,
            step_idx=self.current_step,
        )
        del x_smooth


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot Wan I2V raw/smoothed activation group curves")
    p.add_argument("--ckpt_dir", default="../Wan2.2-I2V-A14B-bf16", help="Wan I2V checkpoint directory")
    p.add_argument("--quant_ckpt_dir", default="", help="Checkpoint directory used to instantiate the quantized model; defaults to --ckpt_dir")
    p.add_argument("--ptq_dir", default="outputs/gptq_ptq", help="ptq_stats.pt path/dir or legacy split PTQ directory")
    p.add_argument("--manifest", default="/home/wjh/Wan2.2/opens2v_outputs/wan_bf16/generation_manifest.jsonl", help="jsonl manifest with image/used_image/img_paths and prompt")
    p.add_argument("--out_dir", default="outputs/activation_distribution")
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=5)
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--experts", type=str, default="both", choices=["high", "low", "both"])
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--target_regex", type=str, default="", help="Optional regex over QuantLinear module names")
    p.add_argument("--blocks", type=str, default="0", help="Comma-separated block ids used with --modules when --target_regex is empty")
    p.add_argument("--modules", type=str, default="self_attn.q", help="Comma-separated module suffixes, e.g. self_attn.q,ffn.2")
    p.add_argument("--max_targets", type=int, default=8, help="Safety cap for plotted target modules; set 0 to disable")
    p.add_argument("--max_group_rows", type=int, default=2048, help="Max token/row samples per module call for group percentile curves")
    p.add_argument("--plot_curve", type=str, default="p99", choices=["p50", "p95", "p99", "p999", "max"], help="Primary curve used for summary ratios")
    p.add_argument("--list_linears", action="store_true", default=False)
    p.add_argument("--low_keep_fp_blocks", type=str, default="")
    p.add_argument("--high_keep_fp_blocks", type=str, default="")
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--offload_models", action="store_true", default=True)
    p.add_argument("--no_offload_models", action="store_false", dest="offload_models")
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _inner_quant_linear(module: nn.Module) -> QuantLinear:
    if isinstance(module, QuantLinearWithBranch):
        return module.quant_linear
    if isinstance(module, QuantLinear):
        return module
    raise TypeError(f"Expected QuantLinear or QuantLinearWithBranch, got {type(module).__name__}")


def _module_meta(module: nn.Module) -> Dict[str, Any]:
    qlinear = _inner_quant_linear(module)
    return {
        "act_group_size": int(qlinear.act_group_size),
        "act_clip_ratio": float(qlinear.act_clip_ratio),
        "scheme": str(qlinear.scheme),
        "smooth_enabled": bool(qlinear.smooth_enabled.item()),
        "input_channels_dim": int(qlinear.input_channels_dim.item()),
        "has_branch": bool(isinstance(module, QuantLinearWithBranch) and module.branch is not None),
        "quant_method": str(qlinear.quant_method),
    }


def _select_quant_targets(
    model: nn.Module,
    pattern: Optional[Any],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    names: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinearWithBranch):
            target = True
        elif isinstance(module, QuantLinear):
            target = not name.endswith(".quant_linear")
        else:
            target = False
        if not target:
            continue
        if pattern is not None and not pattern.search(name):
            continue
        names.append(name)
        meta[name] = _module_meta(module)
    return names, meta


def _apply_max_targets(
    target_modules: Dict[str, List[str]],
    module_meta: Dict[str, Dict[str, Dict[str, Any]]],
    max_targets: int,
) -> None:
    if max_targets <= 0:
        return
    total = sum(len(v) for v in target_modules.values())
    if total <= max_targets:
        return
    remaining = int(max_targets)
    for expert in ("high", "low"):
        keep = target_modules.get(expert, [])[:remaining]
        removed = set(target_modules.get(expert, [])) - set(keep)
        target_modules[expert] = keep
        module_meta[expert] = {name: module_meta[expert][name] for name in keep}
        remaining = max(0, remaining - len(keep))
        for name in sorted(removed):
            LOGGER.warning("Skipping target due to --max_targets=%d: %s | %s", max_targets, expert, name)


def _register_hooks(
    model: nn.Module,
    expert: str,
    module_names: Sequence[str],
    collector: GroupCurveCollector,
) -> List[Any]:
    handles = []
    module_map = dict(model.named_modules())
    for name in module_names:
        module = module_map.get(name)
        if not isinstance(module, (QuantLinearWithBranch, QuantLinear)):
            LOGGER.warning("Quant target is not QuantLinear wrapper: %s (%s)", name, type(module).__name__)
            continue

        def _make_hook(module_name: str):
            def _hook(mod: nn.Module, inp: Tuple[Any, ...]) -> None:
                if not inp or not torch.is_tensor(inp[0]):
                    return
                collector.observe(expert=expert, module_name=module_name, module=mod, x_raw=inp[0])

            return _hook

        handles.append(module.register_forward_pre_hook(_make_hook(name), with_kwargs=False))
    return handles


def _activation_groups(x: torch.Tensor, channels_dim: int, group_size: int) -> Optional[torch.Tensor]:
    if x.ndim == 0:
        return None
    group_size = int(group_size)
    if group_size <= 0:
        return None
    cdim = channels_dim if channels_dim >= 0 else x.ndim + channels_dim
    if cdim < 0 or cdim >= x.ndim:
        cdim = x.ndim - 1
    xg = x.movedim(cdim, -1) if cdim != x.ndim - 1 else x
    c = int(xg.shape[-1])
    if c <= 0:
        return None
    x2 = xg.reshape(-1, c)
    pad = (group_size - (c % group_size)) % group_size
    if pad > 0:
        x2 = torch.nn.functional.pad(x2, (0, pad))
    return x2.view(-1, x2.shape[-1] // group_size, group_size)


def _sample_group_rows(grouped: torch.Tensor, max_rows: int) -> torch.Tensor:
    if max_rows <= 0 or int(grouped.shape[0]) <= max_rows:
        return grouped
    stride = int(math.ceil(int(grouped.shape[0]) / int(max_rows)))
    return grouped[::stride]


def _group_abs_curves(grouped: torch.Tensor) -> Dict[str, np.ndarray]:
    # Shape: [rows, groups, group_size] -> [rows * group_size, groups].
    values = grouped.detach().abs().to(device="cpu", dtype=torch.float32, non_blocking=False)
    values = values.permute(0, 2, 1).reshape(-1, int(values.shape[1]))
    if values.numel() == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {"p50": empty, "p95": empty, "p99": empty, "p999": empty, "max": empty}
    sorted_values = values.sort(dim=0).values
    n = int(sorted_values.shape[0])

    def _take(q: float) -> np.ndarray:
        idx = min(n - 1, max(0, int(math.ceil(q * n) - 1)))
        return sorted_values[idx].numpy().astype(np.float32, copy=False)

    return {
        "p50": _take(0.50),
        "p95": _take(0.95),
        "p99": _take(0.99),
        "p999": _take(0.999),
        "max": sorted_values[-1].numpy().astype(np.float32, copy=False),
    }


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _plot_module_curves(
    path: Path,
    expert: str,
    module_name: str,
    rec: ModuleGroupCurves,
    meta: Dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_max = rec.raw.mean_curve("max")
    x = np.arange(raw_max.shape[0], dtype=np.int64)
    if x.size == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, domain, stats in (
        (axes[0], "raw", rec.raw),
        (axes[1], "smooth", rec.smooth),
    ):
        _draw_percentile_band(ax, x, stats, domain)
    axes[1].set_xlabel("Input Activation Group Index")
    _format_group_xaxis(axes[1], int(x.size))
    fig.suptitle(
        f"{expert} | {module_name} | group_size={meta['act_group_size']} | calls={rec.raw.calls}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_percentile_band(ax: Any, x: np.ndarray, stats: GroupCurveStats, domain: str) -> None:
    p50 = stats.mean_curve("p50")
    p99 = stats.mean_curve("p99")
    max_curve = stats.mean_curve("max")
    p99 = np.maximum(p99, p50)
    max_curve = np.maximum(max_curve, p99)

    green = "#7f9c49"
    yellow = "#f6bb2f"
    red = "#c12d47"
    ax.fill_between(x, 0, max_curve, color=red, linewidth=0.0, alpha=0.92, label="Max")
    ax.fill_between(x, 0, p99, color=yellow, linewidth=0.0, alpha=0.96, label="99% Percentile")
    ax.fill_between(x, 0, p50, color=green, linewidth=0.0, alpha=0.96, label="50% Percentile")
    ax.set_ylabel(f"{domain}\nInput Activation Value")
    ax.grid(True, alpha=0.22)
    ax.margins(x=0)
    _draw_band_legend(ax, colors=(red, yellow, green))


def _draw_band_legend(ax: Any, colors: Tuple[str, str, str]) -> None:
    from matplotlib.patches import Rectangle

    red, yellow, green = colors
    box_x = 1.03
    box_y = 0.05
    box_w = 0.22
    box_h = 0.88
    frame = Rectangle(
        (box_x, box_y),
        box_w,
        box_h,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#cfcfcf",
        linewidth=1.0,
        clip_on=False,
    )
    ax.add_patch(frame)
    bar_x = box_x + 0.08
    bar_w = 0.08
    ax.add_patch(Rectangle((bar_x, box_y + 0.58), bar_w, 0.30, transform=ax.transAxes, color=red, clip_on=False))
    ax.add_patch(Rectangle((bar_x, box_y + 0.26), bar_w, 0.32, transform=ax.transAxes, color=yellow, clip_on=False))
    ax.add_patch(Rectangle((bar_x, box_y + 0.04), bar_w, 0.22, transform=ax.transAxes, color=green, clip_on=False))
    ax.text(box_x + 0.11, box_y + 0.90, "Max", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    ax.text(box_x + 0.11, box_y + 0.58, "99% Percentile", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    ax.text(box_x + 0.11, box_y + 0.26, "50% Percentile", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


def _format_group_xaxis(ax: Any, num_groups: int) -> None:
    if num_groups <= 1:
        return
    ticks = np.linspace(0, num_groups - 1, num=4)
    ax.set_xticks(ticks)
    labels = []
    for value in ticks:
        value_i = int(round(float(value)))
        if value_i >= 1000:
            labels.append(f"{value_i // 1000}k")
        else:
            labels.append(str(value_i))
    ax.set_xticklabels(labels)


def _write_outputs(out_dir: Path, collector: GroupCurveCollector, args: argparse.Namespace, metadata: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "group_curves"
    rows: List[Dict[str, Any]] = []
    npz_payload: Dict[str, np.ndarray] = {}
    primary = str(args.plot_curve)

    for expert in ("high", "low"):
        for module_name, rec in sorted(collector.stats[expert].items()):
            meta = collector.module_meta[expert][module_name]
            key = f"{expert}__{_safe_name(module_name)}"
            png_path = plots_dir / f"{key}.png"
            _plot_module_curves(png_path, expert, module_name, rec, meta)

            for domain_name, stats in (("raw", rec.raw), ("smooth", rec.smooth)):
                for curve_name in GroupCurveStats.CURVES:
                    npz_payload[f"{key}__{domain_name}_{curve_name}_mean"] = stats.mean_curve(curve_name)
                    npz_payload[f"{key}__{domain_name}_{curve_name}_max"] = stats.max_curve(curve_name)

            raw_primary = rec.raw.mean_curve(primary)
            smooth_primary = rec.smooth.mean_curve(primary)
            raw_max = rec.raw.max_curve("max")
            smooth_max = rec.smooth.max_curve("max")
            raw_primary_peak = float(raw_primary.max()) if raw_primary.size else 0.0
            smooth_primary_peak = float(smooth_primary.max()) if smooth_primary.size else 0.0
            raw_max_peak = float(raw_max.max()) if raw_max.size else 0.0
            smooth_max_peak = float(smooth_max.max()) if smooth_max.size else 0.0
            rows.append(
                {
                    "expert": expert,
                    "module": module_name,
                    "calls": int(rec.raw.calls),
                    "num_groups": int(rec.raw.num_groups),
                    "group_size": int(rec.raw.group_size),
                    "first_step": "" if rec.first_step is None else int(rec.first_step),
                    "last_step": "" if rec.last_step is None else int(rec.last_step),
                    "primary_curve": primary,
                    "raw_primary_peak": raw_primary_peak,
                    "smooth_primary_peak": smooth_primary_peak,
                    "smooth_primary_peak_ratio": smooth_primary_peak / max(raw_primary_peak, 1e-30),
                    "raw_max_peak": raw_max_peak,
                    "smooth_max_peak": smooth_max_peak,
                    "smooth_max_peak_ratio": smooth_max_peak / max(raw_max_peak, 1e-30),
                    "plot": str(png_path),
                    **meta,
                }
            )

    csv_path = out_dir / "group_curves_summary.csv"
    fieldnames = [
        "expert",
        "module",
        "calls",
        "num_groups",
        "group_size",
        "first_step",
        "last_step",
        "primary_curve",
        "raw_primary_peak",
        "smooth_primary_peak",
        "smooth_primary_peak_ratio",
        "raw_max_peak",
        "smooth_max_peak",
        "smooth_max_peak_ratio",
        "plot",
        "act_group_size",
        "act_clip_ratio",
        "scheme",
        "smooth_enabled",
        "input_channels_dim",
        "has_branch",
        "quant_method",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "group_curves.npz", **npz_payload)

    metadata["raw_timesteps"] = {
        expert: {str(k): v for k, v in steps.items()} for expert, steps in collector.raw_timesteps.items()
    }
    metadata["num_reported_layers"] = len(rows)
    metadata["outputs"] = {
        "summary_csv": str(csv_path),
        "curves_npz": str(out_dir / "group_curves.npz"),
        "plots_dir": str(plots_dir),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved summary: %s", csv_path)
    LOGGER.info("Saved curves: %s", out_dir / "group_curves.npz")
    LOGGER.info("Saved plots: %s", plots_dir)
    LOGGER.info("Saved metadata: %s", out_dir / "metadata.json")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")
    if args.sample_steps <= 0:
        raise ValueError("--sample_steps must be > 0")
    if args.max_group_rows < 0:
        raise ValueError("--max_group_rows must be >= 0")

    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found.")
    pattern = _build_filter_regex(args)
    LOGGER.info("Target filter: %s", "all quantized Linear modules" if pattern is None else pattern.pattern)

    quant_ckpt_dir = args.quant_ckpt_dir or args.ckpt_dir
    LOGGER.info("[1/4] loading quant model skeleton and PTQ states")
    q_pipe = _build_pipe(quant_ckpt_dir, args)
    low_state, high_state, low_ptq_path, high_ptq_path = _load_i2v_ptq_states(args.ptq_dir)
    _apply_ptq_state_to_model(
        q_pipe.low_noise_model,
        low_state,
        "low_noise_model",
        low_ptq_path,
        keep_fp_blocks=_parse_keep_fp_blocks(args.low_keep_fp_blocks),
    )
    _apply_ptq_state_to_model(
        q_pipe.high_noise_model,
        high_state,
        "high_noise_model",
        high_ptq_path,
        keep_fp_blocks=_parse_keep_fp_blocks(args.high_keep_fp_blocks),
    )
    q_pipe.high_noise_model.cpu()
    q_pipe.low_noise_model.cpu()
    if q_pipe.device.type == "cuda":
        torch.cuda.empty_cache()

    q_models = {"high": q_pipe.high_noise_model, "low": q_pipe.low_noise_model}
    target_modules: Dict[str, List[str]] = {}
    module_meta: Dict[str, Dict[str, Dict[str, Any]]] = {"high": {}, "low": {}}
    for expert in _expert_names(args.experts):
        names, meta = _select_quant_targets(q_models[expert], pattern)
        target_modules[expert] = names
        module_meta[expert] = meta
        LOGGER.info("%s matched QuantLinear modules before cap: %d", expert, len(names))
        for name in names:
            LOGGER.info("%s | %s", expert, name)
    if args.list_linears:
        return
    _apply_max_targets(target_modules, module_meta, max_targets=int(args.max_targets))
    if not any(target_modules.values()):
        raise ValueError("No target quantized Linear modules found.")

    collector = GroupCurveCollector(module_meta=module_meta, max_group_rows=int(args.max_group_rows))
    cast_handles: List[Any] = []
    stat_handles: List[Any] = []
    try:
        for expert in _expert_names(args.experts):
            cast_handles.extend(_register_linear_input_cast_hooks(q_models[expert]))
            stat_handles.extend(_register_hooks(q_models[expert], expert, target_modules[expert], collector))
        LOGGER.info("[2/4] registered group-curve hooks: %d", len(stat_handles))

        cfg = WAN_CONFIGS["i2v-A14B"]
        frame_num = cfg.frame_num
        shift = cfg.sample_shift if args.shift is None else float(args.shift)
        boundary = float(q_pipe.boundary * q_pipe.num_train_timesteps)
        use_amp = q_pipe.device.type == "cuda" and q_pipe.param_dtype in (torch.float16, torch.bfloat16)

        sample_iter: Iterable[ManifestSample] = samples
        if tqdm is not None and not args.disable_tqdm:
            sample_iter = tqdm(samples, desc="activation group curve samples", unit="sample", dynamic_ncols=True)

        LOGGER.info("[3/4] running quant denoise trajectory")
        active_expert: Optional[str] = None
        for sample_idx, sample in enumerate(sample_iter):
            torch.manual_seed(int(args.seed) + sample_idx)
            item = _prep_sample(q_pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))
            latent = item["latent"].to(device=q_pipe.device, dtype=q_pipe.param_dtype)
            y = item["y"].to(device=q_pipe.device, dtype=q_pipe.param_dtype)
            context = [t.to(device=q_pipe.device, dtype=q_pipe.param_dtype) for t in item["context"]]
            model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}
            scheduler, timesteps = _build_schedule(
                sample_solver=args.sample_solver,
                sampling_steps=int(args.sample_steps),
                shift=shift,
                num_train_timesteps=q_pipe.num_train_timesteps,
                device=q_pipe.device,
            )
            timesteps_list = [float(t.item()) for t in timesteps]
            step_iter: Iterable[Tuple[int, float]] = list(enumerate(timesteps_list))
            if tqdm is not None and not args.disable_tqdm:
                step_iter = tqdm(step_iter, desc=f"{sample.sample_id} denoise", unit="step", leave=False, dynamic_ncols=True)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=q_pipe.param_dtype, enabled=use_amp):
                for step_idx, t_scalar in step_iter:
                    expert = "high" if t_scalar >= boundary else "low"
                    if args.offload_models and active_expert != expert:
                        for name, model in q_models.items():
                            model.to(q_pipe.device if name == expert else "cpu")
                        active_expert = expert
                        if q_pipe.device.type == "cuda":
                            torch.cuda.empty_cache()
                    elif not args.offload_models:
                        q_models[expert].to(q_pipe.device)

                    collector.begin_step(step_idx=step_idx, raw_timestep=t_scalar, expert=expert)
                    timestep = torch.tensor([int(t_scalar)], device=q_pipe.device, dtype=torch.long)
                    t_for_step = torch.tensor(t_scalar, device=q_pipe.device, dtype=timesteps.dtype)
                    noise_pred = _run_model(q_models[expert], latent, timestep, model_args)
                    latent = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t_for_step,
                        latent.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)
                    del noise_pred

            del item, latent, y, context, model_args, scheduler, timesteps, timesteps_list
            if q_pipe.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        _remove_hooks(stat_handles)
        _remove_hooks(cast_handles)

    metadata = {
        "ckpt_dir": args.ckpt_dir,
        "quant_ckpt_dir": quant_ckpt_dir,
        "ptq_dir": args.ptq_dir,
        "manifest": args.manifest,
        "max_samples": int(args.max_samples),
        "sample_steps": int(args.sample_steps),
        "size": args.size,
        "seed": int(args.seed),
        "experts": _expert_names(args.experts),
        "sample_solver": args.sample_solver,
        "shift": shift,
        "target_regex": "" if pattern is None else pattern.pattern,
        "target_modules": target_modules,
        "max_targets": int(args.max_targets),
        "max_group_rows": int(args.max_group_rows),
        "plot_curve": args.plot_curve,
        "low_keep_fp_blocks": args.low_keep_fp_blocks,
        "high_keep_fp_blocks": args.high_keep_fp_blocks,
        "definition": "group-only: raw vs smooth per activation group abs-value p95/p99/p999/max curves during quantized model real forward",
        "group_note": "Groups use QuantLinear.act_group_size along input_channels_dim; final partial groups are zero-padded like the reference activation QDQ path.",
    }
    LOGGER.info("[4/4] writing group curve plots and compact reports")
    _write_outputs(Path(args.out_dir), collector, args, metadata)


if __name__ == "__main__":
    main()
