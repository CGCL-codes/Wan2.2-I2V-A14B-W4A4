"""Wan I2V DiT activation statistics collection (act.pt only).

This module only implements online per-channel absmax collection for Linear
input activations from conditional denoise calibration forwards.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

LOGGER = logging.getLogger(__name__)


@dataclass
class WanActStatConfig:
    max_calib_samples: int = 8
    save_dtype: str = "float32"
    channels_dim: int = -1
    keep_stats_on_gpu: bool = False
    module_name_filter: Optional[str] = None
    sample_solver: str = "unipc"
    sampling_steps: int = 40
    shift: float = 5.0
    alpha: float = 0.5
    eps: float = 1e-8
    progress_interval: int = 1
    show_timestep_progress: bool = True


def _find_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    cur = root
    for p in full_name.split("."):
        cur = getattr(cur, p)
    return cur


def _parse_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.strip().lower()
    if name in ("float32", "fp32", "torch.float32"):
        return torch.float32
    if name in ("float16", "fp16", "half", "torch.float16"):
        return torch.float16
    if name in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported save_dtype: {dtype_name}")


def _build_schedule(
    sample_solver: str,
    sampling_steps: int,
    shift: float,
    num_train_timesteps: int,
    device: torch.device,
):
    if sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        return scheduler, scheduler.timesteps
    if sample_solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(
            scheduler,
            device=device,
            sigmas=sampling_sigmas,
        )
        return scheduler, timesteps
    raise ValueError(f"Unsupported sample_solver: {sample_solver}")


def collect_wan_linear_modules(
    model: nn.Module,
    module_name_filter: Optional[str] = None,
) -> List[str]:
    """Collect target DiT nn.Linear modules for activation statistics."""
    out: List[str] = []
    filter_lower = module_name_filter.lower() if module_name_filter else None
    skip_tokens = ("norm", "modulation", "modulate", "embed", "embedding")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Only DiT blocks, skip non-transformer paths.
        if not name.startswith("blocks."):
            continue
        low_name = name.lower()
        if any(tok in low_name for tok in skip_tokens):
            continue
        if filter_lower and filter_lower not in low_name:
            continue
        out.append(name)
    return sorted(out)


class LinearActStatCollector:
    """Online running absmax collector for Linear input activations."""

    def __init__(
        self,
        model: nn.Module,
        module_names: Sequence[str],
        channels_dim: int = -1,
        keep_stats_on_gpu: bool = False,
    ) -> None:
        self.model = model
        self.module_names = list(module_names)
        self.channels_dim = int(channels_dim)
        self.keep_stats_on_gpu = bool(keep_stats_on_gpu)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.stats: Dict[str, Dict[str, Any]] = {
            name: {
                "s": None,  # Tensor[C]
                "channels_dim": int(channels_dim),
                "num_calls": 0,
            }
            for name in self.module_names
        }

    def _hook(self, module_name: str):
        def _inner(_module: nn.Module, inp: Tuple[Any, ...]) -> None:
            x = None
            for obj in inp:
                if torch.is_tensor(obj):
                    x = obj
                    break
            if x is None or x.ndim == 0:
                return

            cdim = self.channels_dim if self.channels_dim >= 0 else x.ndim + self.channels_dim
            if cdim < 0 or cdim >= x.ndim:
                return
            x2 = x.detach()
            if cdim != x.ndim - 1:
                x2 = x2.movedim(cdim, -1)
            if x2.ndim == 1:
                cur_absmax = x2.abs()
            else:
                reduce_dims = tuple(range(x2.ndim - 1))
                cur_absmax = x2.abs().amax(dim=reduce_dims)
            if cur_absmax.dtype != torch.float32:
                cur_absmax = cur_absmax.to(torch.float32)

            rec = self.stats[module_name]
            if not self.keep_stats_on_gpu and cur_absmax.device.type != "cpu":
                cur_absmax = cur_absmax.to(device="cpu", non_blocking=False)
            if rec["s"] is None:
                rec["s"] = cur_absmax
            else:
                rec["s"] = torch.maximum(rec["s"], cur_absmax)
            rec["num_calls"] += 1

        return _inner

    def register(self) -> None:
        for name in self.module_names:
            mod = _find_module_by_name(self.model, name)
            self.handles.append(mod.register_forward_pre_hook(self._hook(name), with_kwargs=False))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def export(self, save_dtype: torch.dtype) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for module_name, rec in self.stats.items():
            s = rec["s"]
            if s is None:
                continue
            out[module_name] = {
                "s": s.detach().to(device="cpu", dtype=save_dtype),
                "channels_dim": int(rec["channels_dim"]),
                "num_calls": int(rec["num_calls"]),
            }
        return out


def _register_linear_input_cast_hooks(model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """Align Linear input dtype to Linear weight dtype before stat hooks run."""
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _cast_hook(m: nn.Module, inp: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if not inp:
            return inp
        x0 = inp[0]
        if not torch.is_tensor(x0):
            return inp
        assert isinstance(m, nn.Linear)
        target_dtype = m.weight.dtype
        if x0.dtype == target_dtype:
            return inp
        x0 = x0.to(dtype=target_dtype)
        if len(inp) == 1:
            return (x0,)
        return (x0, *inp[1:])

    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(_cast_hook, with_kwargs=False))
    return handles


@torch.inference_mode()
def calibrate_wan_acts(
    pipe: Any,
    calib_samples: Iterable[Dict[str, Any]],
    config: Optional[WanActStatConfig] = None,
) -> Dict[str, Any]:
    """Calibrate Wan I2V Linear input activation stats on a conditional denoise trajectory."""
    cfg = config or WanActStatConfig()
    save_dtype = _parse_dtype(cfg.save_dtype)

    high_model = pipe.high_noise_model
    low_model = pipe.low_noise_model
    high_model.eval().requires_grad_(False).to(pipe.device)
    low_model.eval().requires_grad_(False).to(pipe.device)

    high_modules = collect_wan_linear_modules(high_model, module_name_filter=cfg.module_name_filter)
    low_modules = collect_wan_linear_modules(low_model, module_name_filter=cfg.module_name_filter)
    LOGGER.info("High model target Linear modules: %d", len(high_modules))
    LOGGER.info("Low model target Linear modules: %d", len(low_modules))

    cast_handles = _register_linear_input_cast_hooks(high_model) + _register_linear_input_cast_hooks(low_model)
    high_collector = LinearActStatCollector(
        high_model,
        high_modules,
        channels_dim=cfg.channels_dim,
        keep_stats_on_gpu=cfg.keep_stats_on_gpu,
    )
    low_collector = LinearActStatCollector(
        low_model,
        low_modules,
        channels_dim=cfg.channels_dim,
        keep_stats_on_gpu=cfg.keep_stats_on_gpu,
    )
    high_collector.register()
    low_collector.register()

    boundary = float(pipe.boundary * pipe.num_train_timesteps)
    use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)
    num_done = 0

    try:
        for sample in calib_samples:
            idx = num_done + 1
            if cfg.progress_interval > 0 and (idx == 1 or idx % cfg.progress_interval == 0):
                LOGGER.info("calibrate_wan_acts sample %d/%d", idx, int(cfg.max_calib_samples))

            latent = sample["latent"].to(device=pipe.device, dtype=pipe.param_dtype)
            y = sample["y"].to(device=pipe.device, dtype=pipe.param_dtype)
            context = [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in sample["context"]]
            seq_len = int(sample["seq_len"])

            model_args = {"context": context, "seq_len": seq_len, "y": [y]}
            scheduler, timesteps = _build_schedule(
                sample_solver=cfg.sample_solver,
                sampling_steps=cfg.sampling_steps,
                shift=cfg.shift,
                num_train_timesteps=pipe.num_train_timesteps,
                device=pipe.device,
            )
            timesteps_list = [float(t.item()) for t in timesteps]
            LOGGER.info("sample %d timestep_count=%d", idx, len(timesteps_list))

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
                if cfg.show_timestep_progress:
                    t_iter = enumerate(
                        tqdm(
                            timesteps_list,
                            total=len(timesteps_list),
                            desc=f"denoise sample {idx}",
                            leave=True,
                            dynamic_ncols=True,
                            mininterval=0.1,
                        )
                    )
                else:
                    t_iter = enumerate(timesteps_list)

                for step_idx, t_scalar in t_iter:
                    model = high_model if t_scalar >= boundary else low_model
                    timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
                    noise_pred = model([latent], t=timestep, **model_args)[0]
                    latent = scheduler.step(
                        noise_pred.unsqueeze(0),
                        torch.tensor(t_scalar, device=pipe.device, dtype=timesteps.dtype),
                        latent.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)
                    if step_idx == 0:
                        LOGGER.info("sample %d first denoise step done (t=%.3f)", idx, t_scalar)

            num_done += 1
            del latent, y, context, model_args, scheduler, timesteps, timesteps_list
            if pipe.device.type == "cuda":
                torch.cuda.empty_cache()
            if num_done >= cfg.max_calib_samples:
                break
    finally:
        high_collector.remove()
        low_collector.remove()
        for h in cast_handles:
            h.remove()

    high_stats = high_collector.export(save_dtype=save_dtype)
    low_stats = low_collector.export(save_dtype=save_dtype)

    # Per-module debug logs.
    for model_name, model_stats in (("high_noise_model", high_stats), ("low_noise_model", low_stats)):
        LOGGER.info("%s collected modules: %d", model_name, len(model_stats))
        for module_name, rec in model_stats.items():
            s = rec["s"]
            LOGGER.info(
                "%s | %s | channel_count=%d | num_calls=%d | s.min=%.6e | s.max=%.6e",
                model_name,
                module_name,
                int(s.numel()),
                int(rec["num_calls"]),
                float(s.min().item()),
                float(s.max().item()),
            )

    return {
        "__meta__": {
            "version": 1,
            "model": "wan_i2v_dit",
            "stat_type": "per_channel_absmax",
            "channels_dim": int(cfg.channels_dim),
            "num_calib_samples": int(num_done),
            "sample_solver": cfg.sample_solver,
            "sampling_steps": int(cfg.sampling_steps),
            "shift": float(cfg.shift),
        },
        "high_noise_model": high_stats,
        "low_noise_model": low_stats,
    }


def save_wan_act_stats(path: str, stats: Dict[str, Any]) -> None:
    """Save activation statistics to act.pt."""
    _robust_torch_save(stats, path)


def _robust_torch_save(obj: Any, path: str) -> None:
    """Save directly to target path and fallback to legacy serialization on zip writer failure."""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    try:
        try:
            torch.save(obj, path)
        except RuntimeError as e:
            msg = str(e)
            if "unexpected pos" not in msg:
                raise
            LOGGER.warning("torch.save zip serialization failed for %s, retrying with legacy format", path)
            if os.path.exists(path):
                os.remove(path)
            torch.save(obj, path, _use_new_zipfile_serialization=False)
    finally:
        if os.path.exists(path) and os.path.getsize(path) == 0:
            os.remove(path)


def _weight_absmax_in_channel(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute per-input-channel absmax for Linear weight [out_features, in_features]."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape={tuple(weight.shape)}")
    return weight.detach().to(torch.float32).abs().amax(dim=0).clamp_min(float(eps))


def _smooth_scale_from_act_and_weight(
    act_s: torch.Tensor,
    w_s: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SmoothQuant-like scale: scale = act_absmax^alpha / weight_absmax^(1-alpha)."""
    target_device = w_s.device
    act_s = act_s.detach().to(device=target_device, dtype=torch.float32).clamp_min(float(eps))
    w_s = w_s.detach().to(device=target_device, dtype=torch.float32).clamp_min(float(eps))
    if act_s.shape != w_s.shape:
        raise ValueError(f"act_s and w_s shape mismatch: {tuple(act_s.shape)} vs {tuple(w_s.shape)}")
    beta = 1.0 - float(alpha)
    scale = act_s.pow(float(alpha)) / w_s.pow(beta)
    return scale.clamp_min(float(eps))


def _build_smoothed_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply in-channel scaling: W_s[:, j] = W[:, j] * scale[j]."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape={tuple(weight.shape)}")
    if scale.ndim != 1:
        raise ValueError(f"Expected 1D scale, got shape={tuple(scale.shape)}")
    if weight.shape[1] != scale.shape[0]:
        raise ValueError(
            f"Scale size mismatch for weight shape={tuple(weight.shape)} and scale shape={tuple(scale.shape)}"
        )
    return weight.detach().to(torch.float32) * scale.detach().to(torch.float32).view(1, -1)


def build_wan_smoothed_weights(
    pipe: Any,
    act_stats: Dict[str, Any],
    alpha: float = 0.5,
    eps: float = 1e-8,
    save_dtype: str = "bfloat16",
) -> Dict[str, Any]:
    """Build smoothed weights payload from act.pt stats and current model weights."""
    out_dtype = _parse_dtype(save_dtype)
    weight_out_dtype = torch.bfloat16 if out_dtype == torch.float32 else out_dtype

    if "high_noise_model" not in act_stats or "low_noise_model" not in act_stats:
        raise ValueError("act_stats must contain both 'high_noise_model' and 'low_noise_model'")

    model_map = {
        "high_noise_model": pipe.high_noise_model,
        "low_noise_model": pipe.low_noise_model,
    }
    payload: Dict[str, Any] = {
        "__meta__": {
            "version": 1,
            "artifact": "wan_smoothed_weights",
            "model": "wan_i2v_dit",
            "source_artifact": "act.pt",
            "source_stat_type": "per_channel_absmax",
            "smooth_method": "smoothquant_like",
            "alpha": float(alpha),
            "save_dtype": str(out_dtype).replace("torch.", ""),
            "weight_save_dtype": str(weight_out_dtype).replace("torch.", ""),
            "scale_formula": "scale = act_absmax^alpha / weight_absmax^(1-alpha)",
            "weight_transform": "W_s[:, j] = W[:, j] * scale[j]",
            "input_compensation_required": True,
        },
        "high_noise_model": {},
        "low_noise_model": {},
    }

    for model_name in ("high_noise_model", "low_noise_model"):
        model = model_map[model_name]
        model_stats = act_stats[model_name]
        if not isinstance(model_stats, dict):
            raise TypeError(f"act_stats[{model_name}] must be dict, got {type(model_stats).__name__}")
        module_count = 0
        total_params = 0
        scale_min = float("inf")
        scale_max = 0.0
        weight_absmax_before_max = 0.0
        weight_absmax_after_max = 0.0
        for module_name, rec in model_stats.items():
            if not isinstance(rec, dict) or "s" not in rec:
                raise ValueError(f"Invalid stat record for {model_name}.{module_name}")

            try:
                module = _find_module_by_name(model, module_name)
            except Exception as e:
                raise RuntimeError(f"Module not found: {model_name}.{module_name}") from e
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {model_name}.{module_name}, got {type(module).__name__}")

            weight = module.weight.detach().to(torch.float32)
            act_s = rec["s"].detach().to(device=weight.device, dtype=torch.float32)
            if act_s.ndim != 1:
                raise ValueError(f"act_s must be 1D at {model_name}.{module_name}, got shape={tuple(act_s.shape)}")
            if weight.shape[1] != act_s.shape[0]:
                raise ValueError(
                    f"act_s size mismatch at {model_name}.{module_name}: in_features={weight.shape[1]} vs act_s={act_s.shape[0]}"
                )

            w_s = _weight_absmax_in_channel(weight, eps=eps)
            scale = _smooth_scale_from_act_and_weight(act_s=act_s, w_s=w_s, alpha=alpha, eps=eps)
            smoothed_weight = _build_smoothed_weight(weight, scale)
            before_absmax = float(weight.abs().amax().item())
            after_absmax = float(smoothed_weight.abs().amax().item())
            module_count += 1
            total_params += int(weight.numel())
            scale_min = min(scale_min, float(scale.min().item()))
            scale_max = max(scale_max, float(scale.max().item()))
            weight_absmax_before_max = max(weight_absmax_before_max, before_absmax)
            weight_absmax_after_max = max(weight_absmax_after_max, after_absmax)

            payload[model_name][module_name] = {
                "weight": smoothed_weight.detach().to(device="cpu", dtype=weight_out_dtype),
                "scale": scale.detach().to(device="cpu", dtype=out_dtype),
                "act_s": act_s.detach().to(device="cpu", dtype=out_dtype),
                "w_s": w_s.detach().to(device="cpu", dtype=out_dtype),
                "weight_channels_dim": 1,
                "input_channels_dim": int(rec.get("channels_dim", -1)),
                "shape": [int(weight.shape[0]), int(weight.shape[1])],
                "orig_dtype": str(module.weight.dtype).replace("torch.", ""),
            }
        LOGGER.info(
            "%s smoothed modules=%d total_params=%d alpha=%.4f weight_dtype=%s aux_dtype=%s scale.min=%.6e scale.max=%.6e w.absmax.before.max=%.6e w.absmax.after.max=%.6e",
            model_name,
            module_count,
            total_params,
            float(alpha),
            str(weight_out_dtype).replace("torch.", ""),
            str(out_dtype).replace("torch.", ""),
            0.0 if module_count == 0 else scale_min,
            scale_max,
            weight_absmax_before_max,
            weight_absmax_after_max,
        )
    return payload


def save_wan_smoothed_weights(path: str, payload: Dict[str, Any]) -> None:
    """Save smoothed weight payload to wgt.pt."""
    _robust_torch_save(payload, path)
