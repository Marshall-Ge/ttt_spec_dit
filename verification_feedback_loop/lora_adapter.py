# -*- coding: utf-8 -*-
"""M4: LoRA Adapter — 低秩 adapter 挂载与管理.

在 DiT backbone 的 attention 和 FF 线性层上挂载 LoRA adapter。
严格遵循约束:

  * B 矩阵零初始化 — 挂载后初始 forward 等同于原模型 (no-op 起点)
  * backbone 全程 ``requires_grad=False`` — 只更新 LoRA 参数
  * 先采集后决策 — 用 M1/M2 的 reject 频率统计选 top-K 层
  * 版本隔离 — adapter 带 base_model_version 标签

DiT block 线性层结构::

    attn1.to_q     Linear[1152, 1152]
    attn1.to_k     Linear[1152, 1152]
    attn1.to_v     Linear[1152, 1152]
    attn1.to_out.0 Linear[1152, 1152]
    ff.net[0].proj Linear[4608, 1152]
    ff.net[2]      Linear[1152, 4608]

每 block 6 个 Linear, rank=8 时每 block ~221K 参数。
Top-K=3 → ~663K 参数。
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# LoRA Linear wrapper
# ===========================================================================


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear``.

    Forward::

        y = W·x + (α/r) · (B @ A) @ x

    where A ∈ R^{r×in}, B ∈ R^{out×r}, B 零初始化。
    ``α`` is the scaling factor (default = rank).
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        out_features, in_features = base.weight.shape

        # Determine device and dtype from base layer
        device = base.weight.device
        dtype = base.weight.dtype

        # A: (rank, in_features) — Kaiming uniform init
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math_sqrt(5))

        # B: (out_features, rank) — ZERO init → no-op at start
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))

        # Freeze base
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # LoRA correction: x @ A^T @ B^T  (x: ... × in, A: r×in, B: out×r)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling

    @property
    def weight(self):
        """Compatibility: expose effective weight for FLOPs profiling."""
        return self.base.weight

    def merge_to_base(self) -> nn.Linear:
        """Merge LoRA into base weights, returning a plain nn.Linear.

        Used when deploying a validated adapter: eliminates the LoRA overhead.
        """
        merged_weight = self.base.weight.data + (
            self.lora_B.data @ self.lora_A.data
        ) * self.scaling
        merged = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
        )
        merged.weight.data = merged_weight
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data.clone()
        return merged

    def reset_lora(self):
        """Re-zero the B matrix, resetting adapter to no-op."""
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.lora_A, a=math_sqrt(5))


def math_sqrt(x: float) -> float:
    return x ** 0.5


# ===========================================================================
# Layer path resolution
# ===========================================================================

# DiT block 中可挂载 LoRA 的线性层路径 (相对于 block)
_LORA_TARGET_PATHS = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ff.net.0.proj",   # GELU inner projection
    "ff.net.2",         # FF output projection
]


def _resolve_path(block: nn.Module, path: str) -> nn.Linear:
    """Resolve dotted path relative to a block. Raises AttributeError if not found."""
    obj = block
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, nn.Linear):
        raise TypeError(f"Path {path!r} resolved to {type(obj).__name__}, not Linear")
    return obj


# ===========================================================================
# Attach / detach
# ===========================================================================


def attach_lora_to_block(block: nn.Module, rank: int = 8, alpha: int = 16
                         ) -> Dict[str, LoRALinear]:
    """Attach LoRA to a single transformer block.

    Returns a dict mapping path → LoRALinear for later management.
    """
    wrappers: Dict[str, LoRALinear] = {}
    for path in _LORA_TARGET_PATHS:
        try:
            linear = _resolve_path(block, path)
            lora = LoRALinear(linear, rank=rank, alpha=alpha)
            # Replace in-place
            *parent_path, attr = path.split(".")
            parent = block
            for part in parent_path:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            if attr.isdigit():
                parent[int(attr)] = lora
            else:
                setattr(parent, attr, lora)
            wrappers[path] = lora
        except (AttributeError, TypeError):
            pass  # path doesn't exist on this block variant
    return wrappers


def attach_lora(transformer, layer_ids: List[int],
                rank: int = 8, alpha: int = 16) -> Dict[int, Dict[str, LoRALinear]]:
    """Attach LoRA to selected layers of the transformer.

    Parameters
    ----------
    transformer : DiTTransformer2D
    layer_ids : list of int
        Which block indices to attach LoRA to (e.g. [18, 19, 20]).
    rank : int
    alpha : int

    Returns
    -------
    layer_wrappers : dict
        {layer_id: {path: LoRALinear}} for later management/checkpointing.
    """
    all_wrappers: Dict[int, Dict[str, LoRALinear]] = {}
    for layer_id in layer_ids:
        block = transformer.transformer_blocks[layer_id]
        wrappers = attach_lora_to_block(block, rank=rank, alpha=alpha)
        all_wrappers[layer_id] = wrappers
    return all_wrappers


def detach_lora(transformer, layer_wrappers: Dict[int, Dict[str, LoRALinear]]):
    """Remove LoRA wrappers, restoring original Linear layers."""
    for layer_id, wrappers in layer_wrappers.items():
        block = transformer.transformer_blocks[layer_id]
        for path, lora in wrappers.items():
            *parent_path, attr = path.split(".")
            parent = block
            for part in parent_path:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            if attr.isdigit():
                parent[int(attr)] = lora.base
            else:
                setattr(parent, attr, lora.base)


# ===========================================================================
# Parameter collection
# ===========================================================================


def get_lora_params(transformer) -> List[nn.Parameter]:
    """Collect all LoRA parameters from the transformer."""
    params: List[nn.Parameter] = []
    for mod in transformer.modules():
        if isinstance(mod, LoRALinear):
            params.append(mod.lora_A)
            params.append(mod.lora_B)
    return params


def freeze_backbone(transformer):
    """Ensure backbone weights require no grad; only LoRA params train."""
    for name, param in transformer.named_parameters():
        param.requires_grad_(False)
    # Re-enable LoRA params
    for p in get_lora_params(transformer):
        p.requires_grad_(True)


# ===========================================================================
# Layer selection by reject frequency
# ===========================================================================


def select_top_k_layers(buffer, k: int = 3) -> List[int]:
    """Select top-K layer IDs by reject event frequency in the buffer.

    Parameters
    ----------
    buffer : StratifiedReplayBuffer
        Must have been collecting events for some time (Phase 1).
    k : int
        Number of layers to select.

    Returns
    -------
    layer_ids : list of int, sorted ascending
    """
    stats = buffer.stats()
    per_layer = stats.get("per_layer_hard_negative", {})

    # Sort layers by reject count descending
    sorted_layers = sorted(
        per_layer.items(), key=lambda x: x[1], reverse=True)

    # Filter out sentinel layer_id=-1 (TeaCache step-level events)
    valid = [(lid, cnt) for lid, cnt in sorted_layers if lid >= 0]

    selected = [lid for lid, _ in valid[:k]]
    return sorted(selected)


# ===========================================================================
# Checkpoint management
# ===========================================================================


def save_lora_checkpoint(layer_wrappers: Dict[int, Dict[str, LoRALinear]],
                         path: str,
                         version: str = "v1",
                         base_model_version: str = "unknown",
                         metadata: Optional[Dict] = None):
    """Save LoRA weights to a checkpoint file.

    Only saves lora_A and lora_B — not the full backbone.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    state: Dict = {
        "version": version,
        "base_model_version": base_model_version,
        "rank": None,
        "alpha": None,
        "layers": {},
        "metadata": metadata or {},
    }

    for layer_id, wrappers in layer_wrappers.items():
        layer_state = {}
        for path_str, lora in wrappers.items():
            if state["rank"] is None:
                state["rank"] = lora.rank
                state["alpha"] = lora.alpha
            layer_state[path_str] = {
                "lora_A": lora.lora_A.data.detach().cpu().clone(),
                "lora_B": lora.lora_B.data.detach().cpu().clone(),
            }
        state["layers"][str(layer_id)] = layer_state

    torch.save(state, path)


def load_lora_checkpoint(transformer, path: str
                         ) -> Tuple[Dict[int, Dict[str, LoRALinear]], Dict]:
    """Load LoRA checkpoint and attach to transformer.

    Returns (layer_wrappers, metadata).
    """
    state = torch.load(path, map_location="cpu")

    rank = state["rank"]
    alpha = state["alpha"]
    version = state.get("version", "unknown")
    base_model_version = state.get("base_model_version", "unknown")
    metadata = state.get("metadata", {})

    layer_ids = sorted(int(k) for k in state["layers"].keys())
    layer_wrappers = attach_lora(transformer, layer_ids, rank=rank, alpha=alpha)

    # Load weights
    for layer_id_str, layer_state in state["layers"].items():
        layer_id = int(layer_id_str)
        wrappers = layer_wrappers[layer_id]
        for path_str, tensors in layer_state.items():
            if path_str in wrappers:
                wrappers[path_str].lora_A.data.copy_(
                    tensors["lora_A"].to(wrappers[path_str].lora_A.device))
                wrappers[path_str].lora_B.data.copy_(
                    tensors["lora_B"].to(wrappers[path_str].lora_B.device))

    metadata["checkpoint_version"] = version
    metadata["checkpoint_base_model"] = base_model_version

    return layer_wrappers, metadata


def count_lora_params(layer_wrappers: Dict[int, Dict[str, LoRALinear]]) -> int:
    """Count total LoRA parameters."""
    total = 0
    for wrappers in layer_wrappers.values():
        for lora in wrappers.values():
            total += lora.lora_A.numel() + lora.lora_B.numel()
    return total
