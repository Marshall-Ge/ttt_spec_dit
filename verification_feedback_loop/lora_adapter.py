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

AdaLN-LoRA (时间步调制, 默认开启)::
    在 A→B 瓶颈层注入时间步 embedding 做 AdaLN 风格调制。
    h = Ax                              shape: (B, L, r)
    γ, β = t_proj(t_emb)               shape: (B, r) each
    h' = h ⊙ (1 + γ) + β              broadcast: (B, 1, r)
    y = Wx + (α/r) · B h'
    两条零初始化: lora_B=0 && t_proj[-1]=0 → 起点 h'=h, delta=0
    → 第一次 forward 与原模型严格相等。
    相比旧标量门控 γ(t)·(BA)x, AdaLN-LoRA 可改变修正方向
    (β 允许平移), 且 per-sample 调制天然支持 batch 训练。
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
# Global t_emb cache (复用 vfl_state.py 单例模式)
# ===========================================================================
#
# 由 denoise loop / training forward 在 transformer forward 之前 set,
# LoRALinear.forward 只读不写。每次 forward 算 168 次 (28 layers × 6 Linears)
# 太贵, 所以全局缓存一次。
#
# None → time-conditioning 失效, LoRA 走纯 ΔW(x) 路径
#        (用于 --vfl-no-time-lora 模式或 fallback)

_current_t_emb: "Optional[torch.Tensor]" = None


def set_lora_t_emb(t_emb: "Optional[torch.Tensor]"):
    """Set the global timestep embedding used by LoRALinear.forward.

    Called once per transformer forward (推理循环 / 训练 forward).
    Passing None effectively disables time-conditioning for this forward.
    """
    global _current_t_emb
    _current_t_emb = t_emb


def clear_lora_t_emb():
    """Reset the global t_emb cache. Idempotent."""
    global _current_t_emb
    _current_t_emb = None


def get_lora_t_emb() -> "Optional[torch.Tensor]":
    """Return the cached t_emb (or None if not set)."""
    return _current_t_emb


# ===========================================================================
# t_emb 派发 helper — DiT vs PixArt
# ===========================================================================


def _infer_t_emb_dim_from_block(block: nn.Module) -> "Optional[int]":
    """Try to read DiT's t_emb dim from block.norm1.emb.

    DiT block.norm1 is an AdaLNModulation wrapping a TimestepEmbedder whose
    ``mlp[0].in_features`` equals the conditioning dim (1152 for DiT-2-256).
    Returns None for PixArt-style blocks (no emb on the block).
    """
    norm1 = getattr(block, "norm1", None)
    if norm1 is None:
        return None
    emb = getattr(norm1, "emb", None)
    if emb is None:
        return None
    mlp = getattr(emb, "mlp", None)
    if mlp is None or len(mlp) == 0:
        return None
    first = mlp[0]
    return getattr(first, "in_features", None)


def _infer_t_emb_dim_from_transformer(transformer: nn.Module) -> "Optional[int]":
    """Infer the timestep embedding dim from DiT or PixArt transformer.

    DiT:  transformer.transformer_blocks[0].norm1.emb.mlp[0].in_features
    PixArt: transformer.adaln_single.timestep_embedder.mlp[0].in_features
    """
    try:
        first_block = transformer.transformer_blocks[0]
    except (AttributeError, IndexError):
        return None
    dit_dim = _infer_t_emb_dim_from_block(first_block)
    if dit_dim is not None:
        return dit_dim
    # PixArt path
    adaln = getattr(transformer, "adaln_single", None)
    if adaln is None:
        return None
    ts = getattr(adaln, "timestep_embedder", None)
    if ts is None:
        return None
    mlp = getattr(ts, "mlp", None)
    if mlp is None or len(mlp) == 0:
        return None
    return getattr(mlp[0], "in_features", None)


def compute_timestep_emb_for_transformer(transformer,
                                         timestep: torch.Tensor,
                                         class_labels: "Optional[torch.Tensor]" = None,
                                         hidden_dtype=None
                                         ) -> "Optional[torch.Tensor]":
    """Run DiT or PixArt's own timestep embedding once, return the result.

    Re-uses the model's existing embedding path so we don't invent a new one
    (which would diverge from the conditioning the backbone actually sees).

    Returns None if neither DiT's norm1.emb nor PixArt's adaln_single is
    present (e.g. stub transformers in unit tests).
    """
    try:
        first_block = transformer.transformer_blocks[0]
    except (AttributeError, IndexError):
        return None

    emb = getattr(getattr(first_block, "norm1", None), "emb", None)
    if emb is not None and callable(emb):
        try:
            return emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        except TypeError:
            # Some TimestepEmbedder variants don't take class_labels
            return emb(timestep, hidden_dtype=hidden_dtype)

    adaln = getattr(transformer, "adaln_single", None)
    if adaln is not None and callable(adaln):
        batch_size = timestep.shape[0] if hasattr(timestep, "shape") else 1
        # PixArt adaln_single wants added_cond_kwargs; pass the canonical
        # None-pair (use_additional_conditions defaults to False).
        added = {"resolution": None, "aspect_ratio": None}
        try:
            timestep_emb, _embedded = adaln(
                timestep, added, batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            return timestep_emb
        except Exception:
            return None

    return None


# ===========================================================================
# LoRA Linear wrapper
# ===========================================================================


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear``.

    Forward (vanilla)::

        y = W·x + (α/r) · (B @ A) @ x

    Forward (AdaLN-LoRA, time-conditioned)::

        h   = A·x              (B, L, r)
        γ,β = t_proj(t_emb)    (B, r) each
        h'  = h ⊙ (1+γ) + β   broadcast (B, 1, r)
        y   = W·x + (α/r) · B·h'

    Design mirrors DiT's adaLN-Zero: ``x * (1+scale) + shift``.
    Zero-init invariant: lora_B=0 AND t_proj[-1]=0 → h'=h, delta=0
    → first forward equals base(x) exactly.

    where A ∈ R^{r×in}, B ∈ R^{out×r}, B 零初始化。
    ``α`` is the scaling factor (default = rank).
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16,
                 time_conditioned: bool = False,
                 t_emb_dim: "Optional[int]" = None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.time_conditioned = bool(time_conditioned) and t_emb_dim is not None

        out_features, in_features = base.weight.shape

        # Determine device and dtype from base layer
        device = base.weight.device
        dtype = base.weight.dtype

        # A: (rank, in_features) — Kaiming uniform init
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math_sqrt(5))

        # B: (out_features, rank) — ZERO init → no-op at start
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))

        if self.time_conditioned:
            # AdaLN-LoRA: t_proj outputs (2*rank) → split into γ, β.
            # h' = h * (1 + γ) + β  (mirrors DiT adaLN-Zero).
            # Two-zero-init invariant: lora_B=0 AND t_proj[-1].weight=0
            # guarantees that the very first forward equals base(x).
            self.t_proj = nn.Sequential(
                nn.Linear(t_emb_dim, rank * 2, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(rank * 2, rank * 2, device=device, dtype=dtype),
            )
            nn.init.zeros_(self.t_proj[-1].weight)
            nn.init.zeros_(self.t_proj[-1].bias)
        else:
            # Keep attribute absent so state_dict doesn't carry phantom keys.
            # Callers should check .time_conditioned before touching t_proj.
            pass

        # Freeze base
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # h: (..., r) — LoRA bottleneck
        h = x @ self.lora_A.T

        if self.time_conditioned:
            t_emb = get_lora_t_emb()
            if t_emb is not None:
                # gb: (B, 2*r) → split into γ, β each (B, r)
                gb = self.t_proj(t_emb)
                gamma, beta = gb.chunk(2, dim=-1)
                # Match dtype defensively
                if gamma.dtype != h.dtype:
                    gamma = gamma.to(h.dtype)
                    beta = beta.to(h.dtype)
                # Reshape for broadcast over sequence dim
                if h.ndim == 3:
                    # (B, L, r): unsqueeze to (B, 1, r)
                    gamma = gamma.unsqueeze(1)
                    beta = beta.unsqueeze(1)
                # AdaLN modulation: h' = h * (1 + γ) + β
                h = h * (1.0 + gamma) + beta

        delta = h @ self.lora_B.T
        return base_out + delta * self.scaling

    @property
    def weight(self):
        """Compatibility: expose effective weight for FLOPs profiling."""
        return self.base.weight

    def merge_to_base(self) -> nn.Linear:
        """Merge LoRA into base weights, returning a plain nn.Linear.

        Used when deploying a validated adapter: eliminates the LoRA overhead.
        Time-conditioned LoRA cannot be cleanly merged (γ/β depend on
        runtime t_emb); fall back to the mean ΔW with γ=0,β=0 (no-op).
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
        """Re-zero the B matrix (and t_proj), resetting adapter to no-op."""
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.lora_A, a=math_sqrt(5))
        if self.time_conditioned:
            nn.init.zeros_(self.t_proj[-1].weight)
            nn.init.zeros_(self.t_proj[-1].bias)


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


def attach_lora_to_block(block: nn.Module, rank: int = 8, alpha: int = 16,
                         time_conditioned: bool = False,
                         t_emb_dim: "Optional[int]" = None,
                         ) -> Dict[str, LoRALinear]:
    """Attach LoRA to a single transformer block.

    Parameters
    ----------
    block
        A DiT or PixArt transformer block.
    rank, alpha
        Standard LoRA hyperparams.
    time_conditioned
        If True, each LoRA gets a t_proj MLP that modulates ΔW by γ(t_emb).
        Default False for back-compat with the sync-path (top-K) callers.
    t_emb_dim
        Required when time_conditioned=True. Caller usually gets it from
        ``_infer_t_emb_dim_from_block`` (DiT) or
        ``_infer_t_emb_dim_from_transformer`` (PixArt).

    Returns a dict mapping path → LoRALinear for later management.
    """
    if time_conditioned and t_emb_dim is None:
        # Last-chance inference (works for DiT).
        t_emb_dim = _infer_t_emb_dim_from_block(block)
        if t_emb_dim is None:
            # PixArt-style block (no emb on it) — silently downgrade.
            time_conditioned = False

    wrappers: Dict[str, LoRALinear] = {}
    for path in _LORA_TARGET_PATHS:
        try:
            linear = _resolve_path(block, path)
            lora = LoRALinear(
                linear, rank=rank, alpha=alpha,
                time_conditioned=time_conditioned,
                t_emb_dim=t_emb_dim,
            )
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
                rank: int = 8, alpha: int = 16,
                time_conditioned: bool = False,
                t_emb_dim: "Optional[int]" = None,
                ) -> Dict[int, Dict[str, LoRALinear]]:
    """Attach LoRA to selected layers of the transformer.

    Parameters
    ----------
    transformer : DiTTransformer2D
    layer_ids : list of int
        Which block indices to attach LoRA to (e.g. [18, 19, 20]).
    rank : int
    alpha : int
    time_conditioned : bool
        Enable time-step γ modulation. If True and t_emb_dim is None, will
        be inferred from the transformer (DiT via block.norm1.emb, PixArt
        via adaln_single.timestep_embedder).
    t_emb_dim : int, optional
        Explicit override for the t_emb conditioning dim.

    Returns
    -------
    layer_wrappers : dict
        {layer_id: {path: LoRALinear}} for later management/checkpointing.
    """
    if time_conditioned and t_emb_dim is None:
        t_emb_dim = _infer_t_emb_dim_from_transformer(transformer)
        if t_emb_dim is None:
            # Couldn't infer — fall back to vanilla LoRA to stay safe.
            time_conditioned = False

    all_wrappers: Dict[int, Dict[str, LoRALinear]] = {}
    for layer_id in layer_ids:
        block = transformer.transformer_blocks[layer_id]
        wrappers = attach_lora_to_block(
            block, rank=rank, alpha=alpha,
            time_conditioned=time_conditioned,
            t_emb_dim=t_emb_dim,
        )
        all_wrappers[layer_id] = wrappers
    return all_wrappers


def attach_lora_all_layers(transformer, rank: int = 4, alpha: int = 1.0,
                           time_conditioned: bool = True,
                           t_emb_dim: "Optional[int]" = None,
                           ) -> Dict[int, Dict[str, LoRALinear]]:
    """Attach LoRA to **every** transformer block (Phase 2 async trainer).

    Default is ``time_conditioned=True`` — the time-conditioned variant is
    the production path. Callers pass ``time_conditioned=False`` only when
    the user opted into ``--vfl-no-time-lora``.

    Replaces the old ``select_top_k_layers`` + ``attach_lora`` two-step. In the
    async regime the buffer is sparse when the first training cycle triggers,
    so per-layer reject frequency is an unreliable selection signal; training
    also shifts the error distribution over time, so a fixed top-K cannot
    adapt. Hanging LoRA on all 28 blocks removes the selection problem entirely
    at the cost of a larger adapter (~5.2M params for DiT at rank=4 vs ~220K
    at rank=8 top-3, both tiny compared to the 675M backbone).

    Parameters
    ----------
    transformer : DiTTransformer2D / PixArtTransformer2D
        Must expose ``transformer_blocks`` (ModuleList).
    rank : int
        LoRA rank (default 4 — smaller than the sync path's 8 because we're
        spreading across 28 layers instead of concentrating on 3).
    alpha : int
        LoRA scaling factor.
    time_conditioned : bool
        Enable time-step γ modulation (default True).
    t_emb_dim : int, optional
        Explicit t_emb dim override. Auto-inferred from transformer otherwise.

    Returns
    -------
    layer_wrappers : dict
        ``{layer_id: {path: LoRALinear}}`` for every block. Layers whose
        block is missing any of ``_LORA_TARGET_PATHS`` simply have fewer
        entries in their inner dict.
    """
    if time_conditioned and t_emb_dim is None:
        t_emb_dim = _infer_t_emb_dim_from_transformer(transformer)
        if t_emb_dim is None:
            time_conditioned = False

    all_wrappers: Dict[int, Dict[str, LoRALinear]] = {}
    for layer_idx, block in enumerate(transformer.transformer_blocks):
        wrappers = attach_lora_to_block(
            block, rank=rank, alpha=alpha,
            time_conditioned=time_conditioned,
            t_emb_dim=t_emb_dim,
        )
        all_wrappers[layer_idx] = wrappers
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
    """Collect all LoRA parameters from the transformer.

    Includes lora_A, lora_B and (if present) t_proj weights/biases.
    Used to drive the optimizer and ``freeze_backbone`` re-enable pass.
    """
    params: List[nn.Parameter] = []
    for mod in transformer.modules():
        if isinstance(mod, LoRALinear):
            params.append(mod.lora_A)
            params.append(mod.lora_B)
            if mod.time_conditioned:
                for p in mod.t_proj.parameters():
                    params.append(p)
    return params


def freeze_backbone(transformer):
    """Ensure backbone weights require no grad; only LoRA params train.

    This also re-enables grad on every LoRA-managed parameter, including
    the t_proj MLP (time-conditioned variant).
    """
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

    Saves lora_A, lora_B and (if time-conditioned) t_proj.state_dict per
    layer. Vanilla LoRA checkpoints omit the ``t_proj`` key — old loaders
    simply don't see it.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    state: Dict = {
        "version": version,
        "base_model_version": base_model_version,
        "rank": None,
        "alpha": None,
        "time_conditioned": None,
        "t_emb_dim": None,
        "layers": {},
        "metadata": metadata or {},
    }

    for layer_id, wrappers in layer_wrappers.items():
        layer_state = {}
        for path_str, lora in wrappers.items():
            if state["rank"] is None:
                state["rank"] = lora.rank
                state["alpha"] = lora.alpha
                state["time_conditioned"] = lora.time_conditioned
                if lora.time_conditioned:
                    # t_emb_dim = first Linear's in_features on t_proj
                    state["t_emb_dim"] = lora.t_proj[0].in_features
            entry: Dict[str, torch.Tensor] = {
                "lora_A": lora.lora_A.data.detach().cpu().clone(),
                "lora_B": lora.lora_B.data.detach().cpu().clone(),
            }
            if lora.time_conditioned:
                entry["t_proj"] = {
                    k: v.detach().cpu().clone()
                    for k, v in lora.t_proj.state_dict().items()
                }
            layer_state[path_str] = entry
        state["layers"][str(layer_id)] = layer_state

    torch.save(state, path)


def load_lora_checkpoint(transformer, path: str,
                         time_conditioned: "Optional[bool]" = None,
                         t_emb_dim: "Optional[int]" = None,
                         ) -> Tuple[Dict[int, Dict[str, LoRALinear]], Dict]:
    """Load LoRA checkpoint and attach to transformer.

    Parameters
    ----------
    transformer
        Target transformer (will have LoRA attached in-place).
    path
        Checkpoint path produced by ``save_lora_checkpoint``.
    time_conditioned
        Optional override. If None, follows the checkpoint's stored setting
        (defaulting to vanilla for legacy checkpoints without the field).
    t_emb_dim
        Optional override for the time-conditioning dim. Auto-inferred
        from the transformer otherwise.

    Backward compatibility:
        Legacy checkpoints (pre time-conditioning) have no ``t_proj`` key
        per layer and no top-level ``time_conditioned`` field. Loader
        handles both: missing t_proj → newly-initialised zero t_proj,
        equivalent to no modulation (γ=0 → factor 1.0, but B is also 0
        so the whole adapter is a no-op regardless).

    Returns (layer_wrappers, metadata).
    """
    state = torch.load(path, map_location="cpu")

    rank = state.get("rank")
    alpha = state.get("alpha")
    version = state.get("version", "unknown")
    base_model_version = state.get("base_model_version", "unknown")
    metadata = state.get("metadata", {})

    # Guard against corrupted / empty checkpoints (rank=None means no actual
    # LoRA weights were saved — likely from a buggy training run).
    if rank is None or alpha is None:
        print(f"  [VFL] Skipping empty/corrupted checkpoint: {path} "
              f"(rank={rank}, alpha={alpha}) — removing from disk")
        try:
            os.remove(path)
        except OSError:
            pass
        return {}, metadata

    # Decide time-conditioning for the freshly attached LoRA.
    ckpt_tc = bool(state.get("time_conditioned", False))
    if time_conditioned is None:
        time_conditioned = ckpt_tc
    if time_conditioned and t_emb_dim is None:
        t_emb_dim = state.get("t_emb_dim") or \
            _infer_t_emb_dim_from_transformer(transformer)
        if t_emb_dim is None:
            time_conditioned = False

    layer_ids = sorted(int(k) for k in state["layers"].keys())
    # Filter out layers with no weight data
    layer_ids = [lid for lid in layer_ids
                 if len(state["layers"].get(str(lid), {})) > 0]
    if not layer_ids:
        print(f"  [VFL] Skipping empty checkpoint: {path} "
              f"(all {len(state['layers'])} layers are empty)")
        return {}, metadata

    layer_wrappers = attach_lora(
        transformer, layer_ids,
        rank=rank, alpha=alpha,
        time_conditioned=time_conditioned,
        t_emb_dim=t_emb_dim,
    )

    # Load weights
    for layer_id_str, layer_state in state["layers"].items():
        layer_id = int(layer_id_str)
        wrappers = layer_wrappers[layer_id]
        for path_str, tensors in layer_state.items():
            if path_str not in wrappers:
                continue
            lora = wrappers[path_str]
            lora.lora_A.data.copy_(
                tensors["lora_A"].to(lora.lora_A.device))
            lora.lora_B.data.copy_(
                tensors["lora_B"].to(lora.lora_B.device))

            ckpt_has_t = "t_proj" in tensors and tensors["t_proj"] is not None
            if lora.time_conditioned:
                if ckpt_has_t:
                    # Check for shape mismatch between old Scalar Gate
                    # t_proj (last layer outputs 1) and new AdaLN-LoRA
                    # t_proj (last layer outputs 2*rank).
                    ckpt_state = tensors["t_proj"]
                    live_state = lora.t_proj.state_dict()
                    shape_mismatch = False
                    for k, v in ckpt_state.items():
                        if k not in live_state:
                            continue
                        if v.shape != live_state[k].shape:
                            shape_mismatch = True
                            break
                    if shape_mismatch:
                        # Old Scalar Gate checkpoint — skip t_proj loading.
                        # New t_proj stays zero-init → no-op, forward safe.
                        print(f"  [VFL] WARNING: t_proj shape mismatch in "
                              f"layer {layer_id}/{path_str} — old Scalar Gate "
                              f"checkpoint, skipping t_proj load (keeping "
                              f"zero-init)")
                    else:
                        # Shapes match — safe to load.
                        converted = {
                            k: v.to(lora.t_proj[0].weight.device)
                            for k, v in ckpt_state.items()
                        }
                        lora.t_proj.load_state_dict(converted, strict=False)
                # else: legacy ckpt without t_proj, leave at zero-init (no-op).
            elif ckpt_has_t:
                # Ckpt has t_proj but caller asked for vanilla LoRA — drop it.
                # Already handled by attach_lora not creating a t_proj.
                pass

    metadata["checkpoint_version"] = version
    metadata["checkpoint_base_model"] = base_model_version
    metadata["time_conditioned"] = lora_tc_flag(layer_wrappers)

    return layer_wrappers, metadata


def lora_tc_flag(layer_wrappers: Dict[int, Dict[str, LoRALinear]]) -> bool:
    """Return True if any wrapper in the dict is time-conditioned."""
    for wrappers in layer_wrappers.values():
        for lora in wrappers.values():
            if lora.time_conditioned:
                return True
    return False


def count_lora_params(layer_wrappers: Dict[int, Dict[str, LoRALinear]]) -> int:
    """Count total LoRA parameters.

    Includes lora_A, lora_B and t_proj weights/biases when present.
    """
    total = 0
    for wrappers in layer_wrappers.values():
        for lora in wrappers.values():
            total += lora.lora_A.numel() + lora.lora_B.numel()
            if lora.time_conditioned:
                for p in lora.t_proj.parameters():
                    total += p.numel()
    return total


# ===========================================================================
# Checkpoint discovery (Phase 2 async worker)
# ===========================================================================


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the path of the most recently saved LoRA checkpoint in ``output_dir``.

    Recognises both the canonical ``lora_candidate_vNNN.pt`` naming scheme
    produced by ``save_lora_checkpoint`` and any other ``*.pt`` file, picking
    the one with the largest mtime. Returns ``None`` when the directory is
    absent, empty, or contains no ``.pt`` files.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return None
    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(output_dir):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            candidates.append((os.path.getmtime(path), path))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ===========================================================================
# Mid-run reload: swap weights between train and inference LoRA wrappers
# ===========================================================================


def _swap_lora_weights(inference_wrappers: Dict[int, Dict[str, LoRALinear]],
                       new_state: Dict,
                       device: torch.device,
                       dtype: torch.dtype):
    """Copy LoRA weights from a training snapshot into inference wrappers.

    Handles fp32 (train) → fp16 (inference) conversion.  Missing layers or
    paths in *new_state* are silently skipped (the inference wrapper keeps its
    current weights).  This is the hot-path for mid-run reload — called once
    per image boundary when a new training cycle has completed.

    Raises ``RuntimeError`` on shape mismatch so the caller can fall back.
    """
    for layer_id, layer_wrappers in inference_wrappers.items():
        if str(layer_id) not in new_state.get("layers", {}):
            continue
        layer_state = new_state["layers"][str(layer_id)]
        for path, inf_lora in layer_wrappers.items():
            if path not in layer_state:
                continue
            entry = layer_state[path]
            src_A = entry["lora_A"]
            src_B = entry["lora_B"]
            if src_A.shape != inf_lora.lora_A.shape:
                raise RuntimeError(
                    f"lora_A shape mismatch layer {layer_id}/{path}: "
                    f"src {src_A.shape} vs dst {inf_lora.lora_A.shape}")
            if src_B.shape != inf_lora.lora_B.shape:
                raise RuntimeError(
                    f"lora_B shape mismatch layer {layer_id}/{path}: "
                    f"src {src_B.shape} vs dst {inf_lora.lora_B.shape}")
            inf_lora.lora_A.data.copy_(src_A.to(device=device, dtype=dtype))
            inf_lora.lora_B.data.copy_(src_B.to(device=device, dtype=dtype))
            if inf_lora.time_conditioned and "t_proj" in entry:
                src_t = entry["t_proj"]
                converted = {k: v.to(device=device, dtype=dtype)
                             for k, v in src_t.items()}
                inf_lora.t_proj.load_state_dict(converted, strict=False)


def _load_state_into_wrappers(wrappers: Dict[int, Dict[str, LoRALinear]],
                              ckpt_path: str,
                              device: torch.device,
                              dtype: torch.dtype):
    """Load a checkpoint file into already-attached LoRA wrappers (no re-attach).

    Unlike ``load_lora_checkpoint`` this does **not** call ``attach_lora`` —
    the wrappers must already exist on the inference model.  Used during VFL
    init when the inference model has been unconditionally LoRA-attached.

    Legacy checkpoints without ``t_proj`` are handled gracefully: the
    time-conditioned wrapper's t_proj stays at zero-init (no-op).
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if state.get("rank") is None:
        raise ValueError(f"corrupted checkpoint: {ckpt_path}")

    for layer_id_str, layer_state in state.get("layers", {}).items():
        layer_id = int(layer_id_str)
        if layer_id not in wrappers:
            continue
        for path, tensors in layer_state.items():
            if path not in wrappers[layer_id]:
                continue
            lora = wrappers[layer_id][path]
            lora.lora_A.data.copy_(
                tensors["lora_A"].to(device=device, dtype=dtype))
            lora.lora_B.data.copy_(
                tensors["lora_B"].to(device=device, dtype=dtype))
            if "t_proj" in tensors and lora.time_conditioned:
                src_t = tensors["t_proj"]
                converted = {k: v.to(device=device, dtype=dtype)
                             for k, v in src_t.items()}
                # Check for shape mismatch (old Scalar Gate → new AdaLN)
                live = lora.t_proj.state_dict()
                shape_ok = all(
                    k not in live or v.shape == live[k].shape
                    for k, v in converted.items())
                if shape_ok:
                    lora.t_proj.load_state_dict(converted, strict=False)
                # else: skip t_proj, leave at zero-init
