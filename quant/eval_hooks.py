"""Evaluation hooks for BF16 vs quantized Wan DiT output comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .quant_linear import QuantLinear


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().to(torch.float32).reshape(-1)
    b = b.detach().to(torch.float32).reshape(-1)
    mse = torch.mean((a - b) ** 2).item()
    max_abs = torch.max(torch.abs(a - b)).item()
    denom = (torch.norm(a) * torch.norm(b)).item()
    if denom == 0:
        cosine = 1.0 if torch.norm(a - b).item() == 0 else 0.0
    else:
        cosine = torch.dot(a, b).item() / denom
    return {"mse": float(mse), "max_abs": float(max_abs), "cosine": float(cosine)}


def _register_output_hooks(
    model: nn.Module,
    module_names: Sequence[str],
    out_dict: Dict[str, torch.Tensor],
):
    handles = []
    lookup = dict(model.named_modules())
    for name in module_names:
        mod = lookup[name]

        def _make_hook(key: str):
            def _hook(_m, _inp, out):
                if isinstance(out, (list, tuple)):
                    if len(out) == 0:
                        return
                    out_t = out[0]
                else:
                    out_t = out
                if torch.is_tensor(out_t):
                    out_dict[key] = out_t.detach().to(torch.float32).cpu()

            return _hook

        handles.append(mod.register_forward_hook(_make_hook(name)))
    return handles


def _select_block_module_names(model: nn.Module) -> List[str]:
    names: List[str] = []
    if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
        for i in range(len(model.blocks)):
            names.append(f"blocks.{i}")
    return names


def _select_linear_module_names(model: nn.Module) -> List[str]:
    names = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, QuantLinear)):
            if ".blocks." in f".{n}." or n.startswith("blocks."):
                names.append(n)
    return names


def compare_wan_models(
    ref_model: nn.Module,
    quant_model: nn.Module,
    model_inputs: Dict[str, Any],
    report_path: str | None = None,
    level: str = "block",
) -> List[Dict[str, Any]]:
    """Run same inputs through ref/quant models and report error metrics."""
    if level not in ("block", "linear"):
        raise ValueError("level must be 'block' or 'linear'")

    if level == "block":
        names = _select_block_module_names(ref_model)
    else:
        names = _select_linear_module_names(ref_model)

    if not names:
        raise RuntimeError(f"No modules selected for level={level}")

    ref_outs: Dict[str, torch.Tensor] = {}
    quant_outs: Dict[str, torch.Tensor] = {}

    h1 = _register_output_hooks(ref_model, names, ref_outs)
    h2 = _register_output_hooks(quant_model, names, quant_outs)

    with torch.no_grad():
        ref_model(**model_inputs)
        quant_model(**model_inputs)

    for h in h1 + h2:
        h.remove()

    rows: List[Dict[str, Any]] = []
    for n in names:
        if n not in ref_outs or n not in quant_outs:
            continue
        m = _tensor_metrics(ref_outs[n], quant_outs[n])
        rows.append({"module": n, **m})

    rows.sort(key=lambda x: x["mse"], reverse=True)

    if report_path is not None:
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "level": level,
            "num_modules": len(rows),
            "rows": rows,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return rows
