"""Model traversal and replacement for Wan DiT W4A4 MXFP4 QuantLinear."""

from __future__ import annotations

import json
import logging
from fnmatch import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .quant_linear import QuantLinear, QuantLinearWithBranch, resolve_dtype


@dataclass
class QuantReplaceConfig:
    weight_group_size: int = 64
    act_group_size: int = 64
    weight_clip_ratio: float = 1.0
    act_clip_ratio: float = 1.0
    quantize_input: bool = False
    compute_dtype: str = "bfloat16"
    keep_blocks_policy: str = "none"
    head_keep: int = 0
    tail_keep: int = 0
    max_keep_blocks: int = 5
    skip_module_keywords: List[str] = field(default_factory=lambda: [
        "time_embedding",
        "time_projection",
        "patch_embedding",
        "text_embedding",
        "head",
        "modulation",
    ])
    target_linear_name_keywords: List[str] = field(default_factory=lambda: [
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q",
        "k",
        "v",
        "o",
        "fc1",
        "fc2",
        "up_proj",
        "down_proj",
        "gate_proj",
        "ffn",
        "mlp",
        "expert",
    ])
    # New SVDQuant-style options (single-layer prototype integration).
    enabled_low_rank: bool = True
    rank: int = 16
    weight_bits: int = 4
    act_bits: int = 4
    group_size: int = 64
    target_module_patterns: List[str] = field(default_factory=lambda: [
        "blocks.*.self_attn.q",
        "blocks.*.self_attn.k",
        "blocks.*.self_attn.v",
        "blocks.*.self_attn.o",
        "blocks.*.cross_attn.q",
        "blocks.*.cross_attn.k",
        "blocks.*.cross_attn.v",
        "blocks.*.cross_attn.o",
        "blocks.*.ffn.0",
        "blocks.*.ffn.2",
    ])
    skip_module_patterns: List[str] = field(default_factory=lambda: [
        "*modulation*",
        "head",
        "head.*",
        "time_embedding*",
        "time_projection*",
        "patch_embedding*",
        "text_embedding*",
    ])


def load_quant_config(path: Optional[str]) -> QuantReplaceConfig:
    if path is None:
        return QuantReplaceConfig()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Quant config not found: {p}")

    text = p.read_text(encoding="utf-8")
    data: Dict[str, Any]

    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
        except Exception:
            # Very small fallback parser for simple key: value yaml.
            data = {}
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                data[k.strip()] = _auto_cast_yaml_scalar(v.strip())

    cfg = QuantReplaceConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _auto_cast_yaml_scalar(v: str) -> Any:
    vl = v.lower()
    if vl in ("true", "false"):
        return vl == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        pass
    if v.startswith("[") and v.endswith("]"):
        items = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        return items
    return v.strip("'\"")


def _find_transformer_blocks(model: nn.Module) -> List[Tuple[int, str, nn.Module]]:
    if hasattr(model, "blocks") and isinstance(getattr(model, "blocks"), nn.ModuleList):
        blocks = getattr(model, "blocks")
        return [(i, f"blocks.{i}", b) for i, b in enumerate(blocks)]

    # fallback: find the largest ModuleList containing nn.Module entries
    best_name = None
    best_list = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0 and all(isinstance(x, nn.Module) for x in module):
            if best_list is None or len(module) > len(best_list):
                best_name = name
                best_list = module

    if best_list is None:
        return []

    prefix = best_name + "." if best_name else ""
    return [(i, f"{prefix}{i}", b) for i, b in enumerate(best_list)]


def _select_keep_block_indices(num_blocks: int, cfg: QuantReplaceConfig) -> List[int]:
    if num_blocks <= 0:
        return []

    policy = str(cfg.keep_blocks_policy).lower()
    if policy in ("none", "all_quant", "allquant", "full_quant", "fullquant"):
        return []
    if policy != "head2_tail3":
        raise ValueError(f"Unsupported keep_blocks_policy: {cfg.keep_blocks_policy}")

    head = min(cfg.head_keep, num_blocks)
    tail = min(cfg.tail_keep, max(num_blocks - head, 0))

    keep = list(range(head)) + list(range(max(num_blocks - tail, head), num_blocks))
    keep = sorted(set(keep))
    if len(keep) > cfg.max_keep_blocks:
        keep = keep[: cfg.max_keep_blocks]
    return keep


def _iter_named_modules_with_parent(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        yield module, name, full, child
        yield from _iter_named_modules_with_parent(child, full)


def _in_block(module_path: str, block_paths: Sequence[str]) -> Tuple[bool, Optional[int]]:
    for i, bp in enumerate(block_paths):
        if module_path == bp or module_path.startswith(bp + "."):
            return True, i
    return False, None


def _should_skip_by_keyword(path: str, keywords: Sequence[str]) -> bool:
    lp = path.lower()
    return any(k.lower() in lp for k in keywords)


def _module_is_target_linear(path: str, module: nn.Module, cfg: QuantReplaceConfig) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if _should_skip_by_keyword(path, cfg.skip_module_keywords):
        return False

    # Primary rule: nn.Linear inside transformer blocks are candidates.
    # Name keywords are a compatibility assist; not the only criterion.
    leaf = path.split(".")[-1].lower()
    if any(k.lower() == leaf or k.lower() in leaf for k in cfg.target_linear_name_keywords):
        return True

    # Also allow numeric ffns like blocks.X.ffn.0 / .2
    if ".ffn." in path or ".mlp." in path or ".expert" in path:
        return True

    if ".self_attn." in path or ".cross_attn." in path:
        return True

    return True


def _matches_any_pattern(path: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    return any(fnmatch(path, p) for p in patterns)


def _validate_bits(cfg: QuantReplaceConfig) -> None:
    # Current backend only supports int4 engineering quantization in QuantLinear.
    if int(cfg.weight_bits) != 4:
        raise ValueError(f"Unsupported weight_bits={cfg.weight_bits}. Current prototype only supports 4-bit.")
    if int(cfg.act_bits) != 4:
        logging.getLogger(__name__).warning(
            "act_bits=%s is configured but activation quantization path is not implemented; "
            "running with weight-only prototype.",
            cfg.act_bits,
        )


def _effective_group_size(cfg: QuantReplaceConfig) -> int:
    return int(cfg.group_size if cfg.group_size > 0 else cfg.weight_group_size)


def replace_wan_dit_linear_with_quant(
    model: nn.Module,
    config: QuantReplaceConfig,
    log_dir: Optional[str] = None,
    model_tag: str = "model",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or logging.getLogger(__name__)

    blocks = _find_transformer_blocks(model)
    block_paths = [p for _, p, _ in blocks]
    keep_indices = _select_keep_block_indices(len(blocks), config)

    block_map = [{"index": i, "path": p} for i, p, _ in blocks]
    logger.info("[%s] detected %d transformer blocks", model_tag, len(blocks))
    for b in block_map:
        logger.info("[%s] block[%d] -> %s", model_tag, b["index"], b["path"])
    logger.info("[%s] keep high-precision blocks: %s", model_tag, keep_indices)

    quantized_modules: List[str] = []
    skipped_modules: List[Dict[str, Any]] = []

    for parent, child_name, full_name, child in list(_iter_named_modules_with_parent(model)):
        in_block, block_idx = _in_block(full_name, block_paths)
        if not in_block:
            continue
        if not isinstance(child, nn.Linear):
            continue

        if block_idx in keep_indices:
            skipped_modules.append({"module": full_name, "reason": "kept_high_precision_block"})
            continue

        if not _module_is_target_linear(full_name, child, config):
            skipped_modules.append({"module": full_name, "reason": "not_target_linear"})
            continue

        qmod = QuantLinear.from_linear(
            child,
            weight_group_size=config.weight_group_size,
            act_group_size=config.act_group_size,
            weight_clip_ratio=config.weight_clip_ratio,
            act_clip_ratio=config.act_clip_ratio,
            quantize_input=config.quantize_input,
            skip_input_quant=False,
            compute_dtype=config.compute_dtype,
            scheme="sym_int4",
        )
        setattr(parent, child_name, qmod)
        quantized_modules.append(full_name)

    logger.info("[%s] quantized linear modules: %d", model_tag, len(quantized_modules))
    logger.info("[%s] skipped modules: %d", model_tag, len(skipped_modules))

    summary = {
        "model_tag": model_tag,
        "num_blocks": len(blocks),
        "block_map": block_map,
        "keep_block_indices": keep_indices,
        "quantized_modules": quantized_modules,
        "skipped_modules": skipped_modules,
        "config": {
            "weight_group_size": config.weight_group_size,
            "act_group_size": config.act_group_size,
            "weight_clip_ratio": config.weight_clip_ratio,
            "act_clip_ratio": config.act_clip_ratio,
            "quantize_input": config.quantize_input,
            "compute_dtype": config.compute_dtype,
            "keep_blocks_policy": config.keep_blocks_policy,
        },
    }

    if log_dir is not None:
        out = Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{model_tag}_block_map.json").write_text(
            json.dumps(block_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / f"{model_tag}_quantized_modules.json").write_text(
            json.dumps(quantized_modules, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / f"{model_tag}_skipped_modules.json").write_text(
            json.dumps(skipped_modules, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / f"{model_tag}_replace_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return summary


def replace_wan_dit_linear_with_branch_quant(
    model: nn.Module,
    config: QuantReplaceConfig,
    dry_run: bool = True,
    cache_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    model_tag: str = "model",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Replace selected Wan DiT Linear layers with QuantLinearWithBranch.

    - dry_run=True: only report selected modules.
    - dry_run=False: perform in-place replacement and optionally save branch/wgts caches.
    """
    logger = logger or logging.getLogger(__name__)
    _validate_bits(config)

    blocks = _find_transformer_blocks(model)
    block_paths = [p for _, p, _ in blocks]
    keep_indices = _select_keep_block_indices(len(blocks), config)
    group_size = _effective_group_size(config)

    selected_modules: List[str] = []
    replaced_modules: List[str] = []
    skipped_modules: List[Dict[str, Any]] = []
    wgts_cache: Dict[str, Dict[str, Any]] = {}
    branch_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    for parent, child_name, full_name, child in list(_iter_named_modules_with_parent(model)):
        in_block, block_idx = _in_block(full_name, block_paths)
        if not in_block:
            continue
        if not isinstance(child, nn.Linear):
            continue

        if block_idx in keep_indices:
            skipped_modules.append({"module": full_name, "reason": "kept_high_precision_block"})
            continue
        if _matches_any_pattern(full_name, config.skip_module_patterns):
            skipped_modules.append({"module": full_name, "reason": "skip_module_patterns"})
            continue
        if config.target_module_patterns and not _matches_any_pattern(full_name, config.target_module_patterns):
            skipped_modules.append({"module": full_name, "reason": "not_in_target_module_patterns"})
            continue

        selected_modules.append(full_name)
        if dry_run:
            continue

        qmod = QuantLinearWithBranch.from_linear_svdquant(
            child,
            rank=(int(config.rank) if config.enabled_low_rank else 0),
            weight_group_size=group_size,
            act_group_size=group_size,
            weight_clip_ratio=config.weight_clip_ratio,
            act_clip_ratio=config.act_clip_ratio,
            compute_dtype=resolve_dtype(config.compute_dtype),
        )
        setattr(parent, child_name, qmod)
        replaced_modules.append(full_name)

        state = qmod.export_state()
        wgts_cache[full_name] = state["quant_linear"]
        branch_cache[full_name] = state["branch"]

    summary = {
        "model_tag": model_tag,
        "dry_run": dry_run,
        "num_blocks": len(blocks),
        "keep_block_indices": keep_indices,
        "enabled_low_rank": config.enabled_low_rank,
        "rank": config.rank,
        "weight_bits": config.weight_bits,
        "act_bits": config.act_bits,
        "group_size": group_size,
        "target_module_patterns": config.target_module_patterns,
        "skip_module_patterns": config.skip_module_patterns,
        "selected_modules": selected_modules,
        "replaced_modules": replaced_modules,
        "skipped_modules": skipped_modules,
    }

    logger.info("[%s] dry_run=%s selected=%d replaced=%d skipped=%d", model_tag, dry_run, len(selected_modules), len(replaced_modules), len(skipped_modules))
    if dry_run:
        for name in selected_modules:
            logger.info("[%s] would replace: %s", model_tag, name)

    cache_paths: Dict[str, str] = {}
    if (not dry_run) and cache_dir is not None:
        out = Path(cache_dir)
        out.mkdir(parents=True, exist_ok=True)
        wgts_path = out / f"{model_tag}_wgts.pt"
        branch_path = out / f"{model_tag}_branch.pt"
        torch.save(wgts_cache, wgts_path)
        torch.save(branch_cache, branch_path)
        cache_paths["wgts"] = str(wgts_path)
        cache_paths["branch"] = str(branch_path)
        summary["cache_paths"] = cache_paths

    if log_dir is not None:
        out = Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{model_tag}_branch_quant_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / f"{model_tag}_selected_modules.json").write_text(
            json.dumps(selected_modules, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / f"{model_tag}_skipped_modules.json").write_text(
            json.dumps(skipped_modules, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def load_quant_state_dict_into_model(model: nn.Module, quant_state_path: str, strict: bool = True) -> None:
    state = torch.load(quant_state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"load_state_dict mismatch for quant model:\nmissing={missing}\nunexpected={unexpected}"
        )
