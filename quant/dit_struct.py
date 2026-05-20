"""DiT structure abstraction for Wan2.2 I2V DiT quantization workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import torch.nn as nn


@dataclass
class DiTLinearRef:
    module_key: str
    module_name: str
    module: nn.Linear
    parent: nn.Module
    field_name: str
    block_index: int


@dataclass
class DiTBlockStruct:
    index: int
    name: str
    module: nn.Module
    linear_refs: List[DiTLinearRef] = field(default_factory=list)

    def named_key_modules(self) -> Generator[Tuple[str, str, nn.Linear, nn.Module, str], None, None]:
        for ref in self.linear_refs:
            yield ref.module_key, ref.module_name, ref.module, ref.parent, ref.field_name


@dataclass
class DiTModelStruct:
    module: nn.Module
    block_structs: List[DiTBlockStruct]

    @property
    def num_blocks(self) -> int:
        return len(self.block_structs)

    @classmethod
    def construct(cls, model: nn.Module) -> "DiTModelStruct":
        if not hasattr(model, "blocks") or not isinstance(getattr(model, "blocks"), nn.ModuleList):
            raise ValueError("Expected model.blocks to be nn.ModuleList for Wan DiT.")

        blocks = getattr(model, "blocks")
        block_structs: List[DiTBlockStruct] = []
        for idx, block in enumerate(blocks):
            bname = f"blocks.{idx}"
            refs = list(_collect_block_linear_refs(block, bname, idx))
            block_structs.append(DiTBlockStruct(index=idx, name=bname, module=block, linear_refs=refs))
        return cls(module=model, block_structs=block_structs)

    def named_key_modules(self) -> Generator[Tuple[str, str, nn.Linear, nn.Module, str], None, None]:
        for block in self.block_structs:
            yield from block.named_key_modules()

    def get_named_layers(
        self,
        skip_pre_modules: bool = True,
        skip_post_modules: bool = True,
        skip_blocks: bool = False,
    ) -> Dict[str, DiTBlockStruct]:
        if skip_blocks:
            return {}
        return {b.name: b for b in self.block_structs}

    def get_prev_module_keys(self) -> Tuple[str, ...]:
        # Pre-modules are intentionally skipped in this first DiT-only framework.
        return tuple()

    def get_post_module_keys(self) -> Tuple[str, ...]:
        # Post-modules are intentionally skipped in this first DiT-only framework.
        return tuple()


def _iter_named_modules_with_parent(module: nn.Module, prefix: str = "") -> Iterable[Tuple[nn.Module, str, str, nn.Module]]:
    for child_name, child in module.named_children():
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        yield module, child_name, full_name, child
        yield from _iter_named_modules_with_parent(child, full_name)


def _module_key_from_path(path: str) -> Tuple[str, str]:
    if ".self_attn." in path:
        leaf = path.split(".")[-1]
        if leaf in ("q", "k", "v"):
            return "self_attn_qkv_proj", "qkv_proj"
        if leaf == "o":
            return "self_attn_out_proj", "o_proj"
        return "self_attn", leaf
    if ".cross_attn." in path:
        leaf = path.split(".")[-1]
        if leaf in ("q", "k", "v"):
            return "cross_attn_qkv_proj", "qkv_proj"
        if leaf == "o":
            return "cross_attn_out_proj", "o_proj"
        return "cross_attn", leaf
    if ".ffn." in path:
        leaf = path.split(".")[-1]
        if leaf == "0":
            return "ffn_up_proj", "up_proj"
        if leaf == "2":
            return "ffn_down_proj", "down_proj"
        return "ffn", leaf
    return "other_linear", path.split(".")[-1]


def _collect_block_linear_refs(
    block: nn.Module,
    block_path: str,
    block_index: int,
) -> Generator[DiTLinearRef, None, None]:
    for parent, child_name, rel_path, child in _iter_named_modules_with_parent(block):
        if not isinstance(child, nn.Linear):
            continue
        full_path = f"{block_path}.{rel_path}"
        module_key, field_name = _module_key_from_path(full_path)
        yield DiTLinearRef(
            module_key=module_key,
            module_name=full_path,
            module=child,
            parent=parent,
            field_name=field_name,
            block_index=block_index,
        )
