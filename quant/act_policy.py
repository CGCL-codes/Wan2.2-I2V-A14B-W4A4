"""Activation quantization policy helpers for Wan SVDQuant inference/search."""

from __future__ import annotations

import json
import logging
import math
import types
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from .quant_linear import QuantLinearWithBranch


LOGGER = logging.getLogger(__name__)


def load_act_policy_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        policy = json.load(f)
    if not isinstance(policy, dict):
        raise TypeError(f"Activation policy JSON must contain a dict, got {type(policy).__name__}")
    return policy


def extract_model_policy(policy: Optional[dict[str, Any]], model_name: str) -> Optional[dict[str, Any]]:
    if policy is None:
        return None
    value = policy.get(model_name)
    if isinstance(value, dict):
        return value
    return policy


def set_quant_modules_act_policy(
    model: nn.Module,
    model_name: str,
    policy: Optional[dict[str, Any]],
    act_scale_method: str = "",
    act_exp_offset: Optional[int] = None,
) -> int:
    model_policy = extract_model_policy(policy, model_name)
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinearWithBranch):
            continue
        module.set_module_name(name)
        module.set_act_policy(model_policy)
        if act_scale_method:
            module.quant_linear.act_scale_method = str(act_scale_method)
        if act_exp_offset is not None:
            module.quant_linear.act_exp_offset = int(act_exp_offset)
        count += 1
    return count


def set_quant_context(model: nn.Module, expert_id: str, timestep_bin: str) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, QuantLinearWithBranch):
            module.set_act_policy_context(expert_id, timestep_bin)
            count += 1
    return count


def get_timestep_bin_from_step_index(expert_id: str, step_idx_in_expert: int, num_steps_in_expert: int) -> str:
    if num_steps_in_expert <= 0:
        return "default"
    r = float(step_idx_in_expert) / float(max(1, num_steps_in_expert))
    if expert_id == "high":
        if r < 0.33:
            return "high_early"
        if r < 0.66:
            return "high_mid"
        return "boundary_pre"
    if expert_id == "low":
        if r < 0.20:
            return "boundary_post"
        if r < 0.70:
            return "low_mid"
        return "low_late"
    return "default"


def _timestep_key(value: float) -> str:
    return f"{float(value):.6f}"


def _scalar_timestep_value(timestep: Any) -> Optional[float]:
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return None
        return float(timestep.detach().flatten()[0].item())
    try:
        return float(timestep)
    except (TypeError, ValueError):
        return None


def split_timesteps_by_boundary(timesteps: Any, boundary_timestep: float) -> dict[str, list[float]]:
    if isinstance(timesteps, torch.Tensor):
        values = [float(x.item()) for x in timesteps.detach().flatten().cpu()]
    else:
        values = [_scalar_timestep_value(x) for x in timesteps]
        values = [float(x) for x in values if x is not None]
    boundary = float(boundary_timestep)
    return {
        "high": [t for t in values if t >= boundary],
        "low": [t for t in values if t < boundary],
    }


def build_expert_timestep_schedule(timesteps: Any, boundary_timestep: float) -> dict[str, Any]:
    split = split_timesteps_by_boundary(timesteps, boundary_timestep)
    schedule: dict[str, Any] = {
        "boundary_timestep": float(boundary_timestep),
        "high": {},
        "low": {},
        "high_timestep_list": split["high"],
        "low_timestep_list": split["low"],
    }
    for expert_id in ("high", "low"):
        expert_ts = split[expert_id]
        num_steps = len(expert_ts)
        for idx, value in enumerate(expert_ts):
            schedule[expert_id][_timestep_key(value)] = {
                "index": idx,
                "num_steps": num_steps,
                "bin": get_timestep_bin_from_step_index(expert_id, idx, num_steps),
                "timestep": float(value),
            }
    return schedule


def build_wan_i2v_timestep_schedule(
    num_train_timesteps: int,
    boundary: float,
    sample_solver: str,
    sampling_steps: int,
    shift: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    boundary_timestep = float(boundary) * float(num_train_timesteps)
    if sample_solver == "unipc":
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=int(num_train_timesteps),
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(int(sampling_steps), device=device, shift=float(shift))
        timesteps = sample_scheduler.timesteps
    elif sample_solver == "dpm++":
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=int(num_train_timesteps),
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(int(sampling_steps), float(shift))
        timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
    else:
        raise NotImplementedError(f"Unsupported solver: {sample_solver}")
    return timesteps, boundary_timestep, build_expert_timestep_schedule(timesteps, boundary_timestep)


def get_timestep_bin_from_schedule(
    expert_id: str,
    timestep: Any,
    expert_timestep_schedule: Optional[dict[str, Any]],
) -> str:
    if not expert_timestep_schedule:
        return "default"
    value = _scalar_timestep_value(timestep)
    if value is None:
        return "default"
    expert_schedule = expert_timestep_schedule.get(expert_id)
    if not isinstance(expert_schedule, dict):
        return "default"
    entry = expert_schedule.get(_timestep_key(value))
    if entry is None:
        best_key = None
        best_delta = math.inf
        for key in expert_schedule:
            try:
                delta = abs(float(key) - float(value))
            except ValueError:
                continue
            if delta < best_delta:
                best_key = key
                best_delta = delta
        if best_key is not None and best_delta <= 1e-3:
            entry = expert_schedule.get(best_key)
    if isinstance(entry, dict):
        return str(entry.get("bin", "default"))
    return "default"


def get_timestep_bin_from_value(
    expert_id: str,
    timestep: Any,
    boundary_timestep: Optional[float] = None,
    max_timestep: Optional[float] = None,
) -> str:
    value = _scalar_timestep_value(timestep)
    if value is None or boundary_timestep is None or max_timestep is None:
        return "default"
    if expert_id == "high":
        denom = max(float(max_timestep) - float(boundary_timestep), 1e-6)
        r = (float(max_timestep) - value) / denom
        if r < 0.33:
            return "high_early"
        if r < 0.66:
            return "high_mid"
        return "boundary_pre"
    if expert_id == "low":
        denom = max(float(boundary_timestep), 1e-6)
        r = (float(boundary_timestep) - value) / denom
        if r < 0.20:
            return "boundary_post"
        if r < 0.70:
            return "low_mid"
        return "low_late"
    return "default"


def _extract_timestep_from_forward(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for key in ("t", "timestep", "timesteps"):
        if key in kwargs:
            return kwargs[key]
    if len(args) >= 2:
        return args[1]
    return None


def maybe_patch_model_forward_for_act_context(
    model: nn.Module,
    expert_id: str,
    total_steps: Optional[int] = None,
    boundary_timestep: Optional[float] = None,
    max_timestep: Optional[float] = None,
    expert_timestep_schedule: Optional[dict[str, Any]] = None,
) -> nn.Module:
    if getattr(model, "_act_policy_forward_patched", False):
        return model

    original_forward = model.forward
    state = {
        "last_timestep": None,
        "step_idx": -1,
        "printed": False,
    }

    def wrapped_forward(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
        timestep = _extract_timestep_from_forward(args, kwargs)
        value = _scalar_timestep_value(timestep)
        if value is not None and value != state["last_timestep"]:
            state["step_idx"] += 1
            state["last_timestep"] = value
        elif value is None and state["step_idx"] < 0:
            state["step_idx"] = 0

        bin_id = get_timestep_bin_from_schedule(expert_id, timestep, expert_timestep_schedule)
        if bin_id == "default":
            bin_id = get_timestep_bin_from_value(
                expert_id,
                timestep,
                boundary_timestep=boundary_timestep,
                max_timestep=max_timestep,
            )
        if bin_id == "default" and total_steps is not None:
            bin_id = get_timestep_bin_from_step_index(expert_id, max(0, state["step_idx"]), int(total_steps))
        set_quant_context(self, expert_id, bin_id)

        if not state["printed"]:
            shape = tuple(timestep.shape) if isinstance(timestep, torch.Tensor) else None
            LOGGER.info(
                "Activation policy context patched for %s: first_timestep_value=%s shape=%s bin=%s",
                expert_id,
                value,
                shape,
                bin_id,
            )
            state["printed"] = True
        return original_forward(*args, **kwargs)

    model._act_policy_original_forward = original_forward  # type: ignore[attr-defined]
    model.forward = types.MethodType(wrapped_forward, model)
    model._act_policy_forward_patched = True  # type: ignore[attr-defined]
    return model
