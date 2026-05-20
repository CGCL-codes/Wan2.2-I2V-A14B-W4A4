#!/usr/bin/env python3
"""Run Wan I2V inference from exported SVDQuant PTQ states."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from PIL import Image

import wan
from quant.act_policy import (
    build_wan_i2v_timestep_schedule,
    load_act_policy_json,
    maybe_patch_model_forward_for_act_context,
    set_quant_modules_act_policy,
)
from quant.quant_linear import QuantLinear, QuantLinearWithBranch
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video, str2bool


DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, "
    "and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, "
    "as if savoring the sea breeze and warm sunlight."
)
DEFAULT_PROMPT1 = (
    "A man gently clutching a bouquet of vibrant flowers, his eyes radiating a serene contentment as he glances at the camera. His slightly upturned lips convey a sense of calm joy, accompanied by a faint twinkle in his eye. The scene is set in a lush garden, brimming with colorful blooms and verdant foliage, creating a tranquil haven. The shot captures him from the waist up, emphasizing his relaxed stance and the natural harmony of his surroundings."
)



LOGGER = logging.getLogger("infer_wan_i2v_svdquant")
EXPECTED_QUANT_SCHEME = "mxfp4"
EXPECTED_WEIGHT_BLOCK_SIZE = 32
EXPECTED_ACTIVATION_BLOCK_SIZE = 32


def _find_parent_and_leaf(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    if not parts:
        raise ValueError("empty module path")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _set_module_by_name(root: nn.Module, full_name: str, module: nn.Module) -> None:
    parent, leaf = _find_parent_and_leaf(root, full_name)
    parent._modules[leaf] = module


def _get_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    parent, leaf = _find_parent_and_leaf(root, full_name)
    return parent._modules[leaf]


def _validate_ptq_meta(meta: Dict[str, Any], path: str, expected_model_name: str) -> None:
    if not isinstance(meta, dict):
        raise TypeError(f"PTQ state __meta__ must be dict at {path}")

    model_name = str(meta.get("model_name", ""))
    if model_name != expected_model_name:
        raise ValueError(f"PTQ state model_name mismatch at {path}: expected {expected_model_name}, got {model_name}")

    quant_scheme = str(meta.get("quant_scheme", ""))
    if quant_scheme != EXPECTED_QUANT_SCHEME:
        raise ValueError(
            f"PTQ state quant_scheme mismatch at {path}: expected {EXPECTED_QUANT_SCHEME}, got {quant_scheme}"
        )

    weight_format = str(meta.get("weight_format", ""))
    if weight_format and weight_format != "fp4_e2m1_packed+e8m0_scale":
        raise ValueError(f"Unsupported weight_format at {path}: {weight_format}")

    activation_qdq_format = str(meta.get("activation_qdq_format", ""))
    if activation_qdq_format and activation_qdq_format != "mxfp4_reference":
        raise ValueError(f"Unsupported activation_qdq_format at {path}: {activation_qdq_format}")

    weight_block_size = int(meta.get("weight_block_size", meta.get("weight_group_size", -1)))
    act_block_size = int(meta.get("activation_block_size", meta.get("act_group_size", -1)))
    if weight_block_size != EXPECTED_WEIGHT_BLOCK_SIZE:
        raise ValueError(
            f"PTQ state weight block mismatch at {path}: expected {EXPECTED_WEIGHT_BLOCK_SIZE}, got {weight_block_size}"
        )
    if act_block_size != EXPECTED_ACTIVATION_BLOCK_SIZE:
        raise ValueError(
            f"PTQ state activation block mismatch at {path}: expected {EXPECTED_ACTIVATION_BLOCK_SIZE}, got {act_block_size}"
        )


def _validate_quant_module_state(state: Dict[str, Any], path: str, model_name: str, module_name: str) -> None:
    qstate = state.get("quant_linear", None)
    if not isinstance(qstate, dict):
        raise ValueError(f"Missing quant_linear state at {path}:{model_name}.{module_name}")

    scheme = str(qstate.get("scheme", ""))
    if scheme != EXPECTED_QUANT_SCHEME:
        raise ValueError(
            f"Module quant scheme mismatch at {path}:{model_name}.{module_name}: expected {EXPECTED_QUANT_SCHEME}, got {scheme}"
        )

    weight_block_size = int(qstate.get("w_group_size", qstate.get("weight_group_size", -1)))
    act_block_size = int(qstate.get("act_group_size", -1))
    if weight_block_size != EXPECTED_WEIGHT_BLOCK_SIZE:
        raise ValueError(
            f"Module weight block mismatch at {path}:{model_name}.{module_name}: expected {EXPECTED_WEIGHT_BLOCK_SIZE}, got {weight_block_size}"
        )
    if act_block_size != EXPECTED_ACTIVATION_BLOCK_SIZE:
        raise ValueError(
            f"Module activation block mismatch at {path}:{model_name}.{module_name}: expected {EXPECTED_ACTIVATION_BLOCK_SIZE}, got {act_block_size}"
        )


def _build_quant_module_from_state(state: Dict[str, Any]) -> QuantLinearWithBranch:
    qstate = state["quant_linear"]
    bias = qstate.get("bias", None)
    qmod = QuantLinear(
        in_features=int(qstate["in_features"]),
        out_features=int(qstate["out_features"]),
        bias=bias is not None,
        weight_group_size=int(qstate.get("w_group_size", qstate.get("weight_group_size", 64))),
        act_group_size=int(qstate.get("act_group_size", 64)),
        weight_clip_ratio=float(qstate.get("weight_clip_ratio", 1.0)),
        act_clip_ratio=float(qstate.get("act_clip_ratio", 1.0)),
        quantize_input=bool(qstate.get("quantize_input", True)),
        skip_input_quant=bool(qstate.get("skip_input_quant", False)),
        compute_dtype=qstate.get("compute_dtype", "bfloat16"),
        scheme=qstate.get("scheme", "sym_int4"),
    )
    qmod.act_exp_offset = int(qstate.get("act_exp_offset", qmod.act_exp_offset))
    qmod.act_scale_method = str(qstate.get("act_scale_method", qmod.act_scale_method))
    out = QuantLinearWithBranch(quant_linear=qmod, branch=None)
    out.load_state(state)
    out.eval().requires_grad_(False)
    return out


def _load_ptq_state(path: str) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected dict PTQ state at {path}, got {type(state).__name__}")
    return state


def _load_i2v_ptq_states(ptq_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    ptq_root = Path(ptq_dir)
    combined_path = ptq_root if ptq_root.is_file() else ptq_root / "ptq_stats.pt"
    if combined_path.exists():
        payload = _load_ptq_state(str(combined_path))
        low_state = payload.get("low_noise_model", None)
        high_state = payload.get("high_noise_model", None)
        if not isinstance(low_state, dict):
            raise KeyError(f"Combined PTQ stats missing low_noise_model dict: {combined_path}")
        if not isinstance(high_state, dict):
            raise KeyError(f"Combined PTQ stats missing high_noise_model dict: {combined_path}")
        LOGGER.info("Loaded combined PTQ stats: %s", combined_path)
        return (
            low_state,
            high_state,
            f"{combined_path}:low_noise_model",
            f"{combined_path}:high_noise_model",
        )

    low_ptq_path = ptq_root / "low_noise_model" / "ptq_state.pt"
    high_ptq_path = ptq_root / "high_noise_model" / "ptq_state.pt"
    if not low_ptq_path.exists():
        raise FileNotFoundError(f"PTQ state not found: {low_ptq_path}")
    if not high_ptq_path.exists():
        raise FileNotFoundError(f"PTQ state not found: {high_ptq_path}")

    LOGGER.info("Loaded legacy split PTQ states from: %s", ptq_root)
    return (
        _load_ptq_state(str(low_ptq_path)),
        _load_ptq_state(str(high_ptq_path)),
        str(low_ptq_path),
        str(high_ptq_path),
    )


def _parse_keep_fp_blocks(value: str) -> Set[int]:
    blocks: Set[int] = set()
    value = value.strip()
    if not value:
        return blocks

    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"Invalid empty block token in keep_fp_blocks expression: {value!r}")
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2 or not bounds[0].strip() or not bounds[1].strip():
                raise ValueError(f"Invalid block range token: {token!r}")
            start = int(bounds[0])
            end = int(bounds[1])
            if start < 0 or end < 0:
                raise ValueError(f"Block indices must be non-negative: {token!r}")
            if start > end:
                raise ValueError(f"Invalid descending block range: {token!r}")
            blocks.update(range(start, end + 1))
        else:
            block = int(token)
            if block < 0:
                raise ValueError(f"Block indices must be non-negative: {token!r}")
            blocks.add(block)
    return blocks


def _module_block_index(module_name: str) -> int | None:
    parts = module_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def _all_block_indices(model: nn.Module) -> Set[int]:
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, (nn.ModuleList, list, tuple)):
        return set(range(len(blocks)))
    out: Set[int] = set()
    for name, _module in model.named_modules():
        idx = _module_block_index(name)
        if idx is not None:
            out.add(idx)
    return out


def _expand_keep_fp_module_expr(expr: str, block_indices: Iterable[int]) -> Set[str]:
    expr = expr.strip()
    if not expr:
        return set()
    parts = expr.split(".")
    if len(parts) < 3 or parts[0] != "blocks":
        return {expr}

    block_spec = parts[1]
    suffix = ".".join(parts[2:])
    if block_spec == "*":
        return {f"blocks.{idx}.{suffix}" for idx in sorted(block_indices)}
    if "-" in block_spec:
        bounds = block_spec.split("-")
        if len(bounds) != 2 or not bounds[0].strip() or not bounds[1].strip():
            raise ValueError(f"Invalid module block range expression: {expr!r}")
        start = int(bounds[0])
        end = int(bounds[1])
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"Invalid module block range expression: {expr!r}")
        return {f"blocks.{idx}.{suffix}" for idx in range(start, end + 1)}
    if block_spec.isdigit():
        return {expr}
    return {expr}


def _parse_keep_fp_modules(value: str, model: nn.Module) -> Set[str]:
    modules: Set[str] = set()
    value = value.strip()
    if not value:
        return modules
    block_indices = _all_block_indices(model)
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"Invalid empty module token in keep_fp_modules expression: {value!r}")
        modules.update(_expand_keep_fp_module_expr(token, block_indices))
    return modules


def _regex_matched_modules(model: nn.Module, regex: str) -> Set[str]:
    regex = regex.strip()
    if not regex:
        return set()
    pattern = re.compile(regex)
    return {name for name, module in model.named_modules() if isinstance(module, nn.Linear) and pattern.search(name)}


def _snapshot_bf16_modules(model: nn.Module, module_names: Set[str], model_name: str) -> Dict[str, nn.Module]:
    snapshot: Dict[str, nn.Module] = {}
    for module_name in sorted(module_names):
        try:
            module = _get_module_by_name(model, module_name)
        except (AttributeError, KeyError):
            LOGGER.warning("%s keep-fp module not found before PTQ replacement: %s", model_name, module_name)
            continue
        if not isinstance(module, nn.Linear):
            LOGGER.warning("%s keep-fp target is not BF16 nn.Linear before PTQ replacement: %s (%s)", model_name, module_name, type(module).__name__)
            continue
        snapshot[module_name] = module
    return snapshot


def _restore_bf16_modules(model: nn.Module, snapshot: Dict[str, nn.Module], model_name: str) -> int:
    restored = 0
    for module_name, module in sorted(snapshot.items()):
        _set_module_by_name(model, module_name, module)
        restored += 1
    if restored:
        LOGGER.info("%s restored BF16 modules after PTQ replacement: %d | modules=%s", model_name, restored, sorted(snapshot))
    return restored


def _print_quant_fp_module_summary(model: nn.Module, model_name: str) -> None:
    quant_names = []
    fp_names = []
    for name, module in model.named_modules():
        if isinstance(module, QuantLinearWithBranch):
            quant_names.append(name)
        elif isinstance(module, nn.Linear):
            # Keep the report focused on DiT block linears.
            if _module_block_index(name) is not None:
                fp_names.append(name)
    LOGGER.info("%s final QuantLinearWithBranch modules: %d", model_name, len(quant_names))
    for name in quant_names:
        LOGGER.info("%s | quant | %s", model_name, name)
    LOGGER.info("%s final BF16 Linear modules in blocks: %d", model_name, len(fp_names))
    for name in fp_names:
        LOGGER.info("%s | bf16 | %s", model_name, name)


def _apply_ptq_state_to_model(
    model: nn.Module,
    ptq_state: Dict[str, Any],
    model_name: str,
    path: str,
    keep_fp_blocks: Set[int] | None = None,
) -> int:
    meta = ptq_state.get("__meta__", None)
    if meta is None:
        raise KeyError(f"PTQ state missing __meta__ at {path}")
    _validate_ptq_meta(meta, path, expected_model_name=model_name)

    keep_fp_blocks = keep_fp_blocks or set()
    replaced = 0
    skipped = 0
    for module_name, state in ptq_state.items():
        if module_name == "__meta__":
            continue
        if not isinstance(state, dict) or "quant_linear" not in state:
            raise ValueError(f"Invalid PTQ state entry at {model_name}.{module_name}")
        block_index = _module_block_index(module_name)
        if block_index in keep_fp_blocks:
            skipped += 1
            continue
        _validate_quant_module_state(state, path=path, model_name=model_name, module_name=module_name)
        qmod = _build_quant_module_from_state(state)
        qmod.set_module_name(module_name)
        _set_module_by_name(model, module_name, qmod)
        replaced += 1
    LOGGER.info(
        "%s loaded PTQ modules: %d | kept_fp_modules=%d | keep_fp_blocks=%s | scheme=%s | weight_block=%d | act_block=%d",
        model_name,
        replaced,
        skipped,
        sorted(keep_fp_blocks),
        EXPECTED_QUANT_SCHEME,
        EXPECTED_WEIGHT_BLOCK_SIZE,
        EXPECTED_ACTIVATION_BLOCK_SIZE,
    )
    return replaced


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run Wan I2V SVDQuant PTQ inference")
    p.add_argument("--ckpt_dir", type=str, default='../Wan2.2-I2V-A14B-bf16')
    p.add_argument(
        "--ptq_dir",
        type=str,
        default='/home/wjh/Wan2.2/outputs/cal12_ptq',
        help=(
            "PTQ artifact path. Supports a combined ptq_stats.pt file, a directory containing ptq_stats.pt, "
            "or the legacy high_noise_model/ptq_state.pt + low_noise_model/ptq_state.pt layout."
        ),
    )
    p.add_argument("--image", type=str, default="examples/5.png")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT1)
    p.add_argument("--size", type=str, default="480*832", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--frame_num", type=int, default=61)
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--sample_steps", type=int, default=40)
    p.add_argument("--sample_shift", type=float, default=None)
    p.add_argument("--sample_guide_scale", type=float, default=None)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--offload_model", type=str2bool, default=True)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--save_file", type=str, default="")
    p.add_argument("--low_keep_fp_blocks", type=str, default="")
    p.add_argument("--high_keep_fp_blocks", type=str, default="")
    p.add_argument("--low_keep_fp_modules", type=str, default="")
    p.add_argument("--high_keep_fp_modules", type=str, default="")
    p.add_argument("--keep_fp_module_regex", type=str, default="")
    p.add_argument("--low_keep_fp_module_regex", type=str, default="")
    p.add_argument("--high_keep_fp_module_regex", type=str, default="")
    p.add_argument("--print_replaced_modules", action="store_true", default=False)
    p.add_argument("--act_policy_json", type=str, default="outputs/act_policy.json", help="Optional MXFP4 activation policy JSON.")
    p.add_argument(
        "--act_scale_method",
        type=str,
        default="",
        choices=["", "ocp_floor", "safe_ceil"],
        help="Override activation MXFP4 scale method for all quant modules.",
    )
    p.add_argument("--act_exp_offset", type=int, default=0, help="Override activation MXFP4 exponent offset.")
    p.add_argument("--act_policy_print", action="store_true", default=False, help="Log activation policy attachment summary.")
    p.add_argument(
        "--freeze_condition_latent",
        action="store_true",
        default=True,
        help="Experiment: reset latent[:, 0] to the VAE-encoded I2V condition frame latent only after the final scheduler step.",
    )
    return p.parse_args()


def _resolve_defaults(args: argparse.Namespace) -> None:
    cfg = WAN_CONFIGS["i2v-A14B"]
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale


def _generate_i2v_freezing_condition_latent(pipe: wan.WanI2V, args: argparse.Namespace, img: Image.Image) -> torch.Tensor:
    """Wan I2V generate variant that resets latent[:, 0] to condition latent at the final step."""
    import gc
    import math
    import random
    from contextlib import contextmanager

    import numpy as np
    import torchvision.transforms.functional as TF
    from tqdm import tqdm

    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    guide_scale = args.sample_guide_scale
    guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale

    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(pipe.device)
    frame_num = int(args.frame_num)
    h, w = img_tensor.shape[1:]
    aspect_ratio = h / w
    max_area = MAX_AREA_CONFIGS[args.size]
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // pipe.vae_stride[1] //
        pipe.patch_size[1] * pipe.patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // pipe.vae_stride[2] //
        pipe.patch_size[2] * pipe.patch_size[2]
    )
    h = lat_h * pipe.vae_stride[1]
    w = lat_w * pipe.vae_stride[2]
    max_seq_len = ((frame_num - 1) // pipe.vae_stride[0] + 1) * lat_h * lat_w // (
        pipe.patch_size[1] * pipe.patch_size[2]
    )
    max_seq_len = int(math.ceil(max_seq_len / pipe.sp_size)) * pipe.sp_size

    seed = int(args.base_seed) if int(args.base_seed) >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16,
        (frame_num - 1) // pipe.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=pipe.device,
    )

    msk = torch.ones(1, frame_num, lat_h, lat_w, device=pipe.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:],
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([args.prompt], pipe.device)
        context_null = pipe.text_encoder([pipe.sample_neg_prompt], pipe.device)
        if args.offload_model:
            pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([args.prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([pipe.sample_neg_prompt], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
        context_null = [t.to(pipe.device) for t in context_null]

    cond_video = torch.concat([
        torch.nn.functional.interpolate(
            img_tensor[None].cpu(), size=(h, w), mode="bicubic"
        ).transpose(0, 1),
        torch.zeros(3, frame_num - 1, h, w),
    ], dim=1).to(pipe.device)
    cond_latent = pipe.vae.encode([cond_video])[0]
    cond_first_latent = cond_latent[:, 0].detach()
    y = torch.concat([msk, cond_latent])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low_noise = getattr(pipe.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high_noise = getattr(pipe.high_noise_model, "no_sync", noop_no_sync)

    with (
        torch.amp.autocast("cuda", dtype=pipe.param_dtype),
        torch.no_grad(),
        no_sync_low_noise(),
        no_sync_high_noise(),
    ):
        boundary = pipe.boundary * pipe.num_train_timesteps
        if args.sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(args.sample_steps, device=pipe.device, shift=args.sample_shift)
            timesteps = sample_scheduler.timesteps
        elif args.sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(args.sample_steps, args.sample_shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=pipe.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        latent = noise
        arg_c = {"context": [context[0]], "seq_len": max_seq_len, "y": [y]}
        arg_null = {"context": context_null, "seq_len": max_seq_len, "y": [y]}

        if args.offload_model:
            torch.cuda.empty_cache()

        x0 = [latent]
        for step_idx, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(pipe.device)]
            timestep = torch.stack([t]).to(pipe.device)
            model = pipe._prepare_model_for_timestep(t, boundary, args.offload_model)
            sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
            if args.offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
            if args.offload_model:
                torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latent = temp_x0.squeeze(0)
            if step_idx == len(timesteps) - 1:
                latent[:, 0].copy_(cond_first_latent.to(device=latent.device, dtype=latent.dtype))
            x0 = [latent]
            del latent_model_input, timestep

        if args.offload_model:
            pipe.low_noise_model.cpu()
            pipe.high_noise_model.cpu()
            torch.cuda.empty_cache()

        videos = pipe.vae.decode(x0)

    del noise, latent, x0, sample_scheduler
    if args.offload_model:
        gc.collect()
        torch.cuda.synchronize()
    return videos[0]


def main() -> None:
    args = _parse_args()
    _resolve_defaults(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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

    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)

    img = Image.open(args.image).convert("RGB")
    LOGGER.info("Running I2V PTQ inference: image=%s", args.image)
    LOGGER.info("Prompt: %s", args.prompt)
    if args.freeze_condition_latent:
        LOGGER.info("Experiment enabled: resetting I2V condition frame latent at latent[:, 0] after the final scheduler step.")
        video = _generate_i2v_freezing_condition_latent(pipe, args, img)
    else:
        video = pipe.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

    if not args.save_file:
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_file = f"i2v_svdquant_{args.size}_{formatted_time}.mp4"
    LOGGER.info("Saving generated video to %s", args.save_file)
    save_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    LOGGER.info("Finished.")


if __name__ == "__main__":
    main()
