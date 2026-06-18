# -*- coding: utf-8 -*-
"""
Task 2.1: SOTA-Inspired Feature Cache & Adaptive Control (TeaCache Style)

Implements an adaptive feature caching mechanism based on input relative residual
thresholds, replacing the naive linear extrapolation from Phase 1.

Core mechanics:
  1. Dynamic Residual Monitor — capture early-layer features at each step,
     compute relative L1 residual vs last key-step reference.
  2. Adaptive Skip Execution — if residual < gamma AND skip_count < max_skip,
     bypass heavy blocks and reuse/warp cached intermediate features.
  3. Parallel Verification Simulation — gather drafted features, concatenate
     into macro-batch, pass through tail layers for ground-truth MSE.

Designed to be fully decoupled: no model weights modified, operates via hooks
and split-forward helpers.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
import os


# ---------------------------------------------------------------------------
# Forward hooks for feature capture (early layer + probe point)
# ---------------------------------------------------------------------------

class EarlyFeatureHook:
    """Captures the output of an early transformer block (e.g. block 2)
    for lightweight residual monitoring."""

    def __init__(self, buffer: dict):
        self.buffer = buffer

    def __call__(self, module, args, kwargs, output):
        self.buffer["early_features"] = output.detach()


class ProbeFeatureHook:
    """Captures the output at the probe point (e.g. block 14) for caching
    and draft generation."""

    def __init__(self, buffer: dict):
        self.buffer = buffer

    def __call__(self, module, args, kwargs, output):
        self.buffer["probe_features"] = output.detach()
        self.buffer["timestep_emb"] = kwargs.get("timestep", None)
        if self.buffer["timestep_emb"] is not None:
            self.buffer["timestep_emb"] = self.buffer["timestep_emb"].detach()
        self.buffer["text_emb"] = kwargs.get("encoder_hidden_states", None)
        if self.buffer["text_emb"] is not None:
            self.buffer["text_emb"] = self.buffer["text_emb"].detach()
        self.buffer["attention_mask"] = kwargs.get("attention_mask", None)
        self.buffer["encoder_attention_mask"] = kwargs.get("encoder_attention_mask", None)


class AdalnCaptureHook:
    """Captures embedded_timestep from adaln_single for tail modulation."""

    def __init__(self, buffer: dict):
        self.buffer = buffer

    def __call__(self, module, args, kwargs, output):
        # output = (modulated_emb, raw_emb) — we need raw_emb for tail scale/shift
        self.buffer["embedded_timestep"] = output[1].detach()


# ---------------------------------------------------------------------------
# TeaCache Controller
# ---------------------------------------------------------------------------

class TeaCacheController:
    """SOTA-inspired adaptive feature cache controller.

    Parameters
    ----------
    gamma : float
        Relative L1 residual threshold. Features with residual < gamma are
        considered "close enough" to skip re-computation.
    max_skip : int
        Maximum consecutive skip steps before forcing a full forward pass
        (re-calibration). Prevents unbounded drift.
    """

    def __init__(self, gamma: float = 0.1, max_skip: int = 3):
        self.gamma = gamma
        self.max_skip = max_skip

        # ---- Reference caches (updated at each key/rejection step) ----
        self.early_features_ref: Optional[torch.Tensor] = None
        self.probe_features_ref: Optional[torch.Tensor] = None
        self.step_meta_ref: Optional[Tuple] = None  # (timestep_emb, embedded_timestep, text_emb, attn_mask, enc_attn_mask)

        # ---- State ----
        self.skip_count: int = 0
        self.key_step_idx: int = -1
        self.total_full_forwards: int = 0
        self.total_skips: int = 0

        # ---- Rejection log ----
        self.rejections: List[Dict] = []

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------
    def compute_residual(self, features: torch.Tensor,
                         ref_features: torch.Tensor) -> float:
        """Compute relative L1 residual between current and reference features.

        residual = ||X_t - X_{t_ref}||_1 / (||X_{t_ref}||_1 + eps)
        """
        diff = (features.float() - ref_features.float()).abs().sum()
        norm = ref_features.float().abs().sum() + 1e-8
        return (diff / norm).item()

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    def should_skip(self, early_features: torch.Tensor) -> bool:
        """Decide whether to skip the heavy blocks.

        Returns True if both:
          - Relative L1 residual < gamma
          - Consecutive skip count < max_skip
        """
        if self.early_features_ref is None:
            return False

        residual = self.compute_residual(early_features, self.early_features_ref)

        if residual < self.gamma and self.skip_count < self.max_skip:
            return True
        return False

    # ------------------------------------------------------------------
    # Cache update (after full forward / rejection)
    # ------------------------------------------------------------------
    def update(self, step_idx: int, timestep: int,
               early_features: torch.Tensor,
               probe_features: torch.Tensor,
               step_meta: Tuple) -> None:
        """Store reference features after a full forward pass (key step or rejection)."""
        self.early_features_ref = early_features.clone()
        self.probe_features_ref = probe_features.clone()
        self.step_meta_ref = tuple(
            t.clone() if isinstance(t, torch.Tensor) else t
            for t in step_meta
        )
        self.key_step_idx = step_idx
        self.skip_count = 0
        self.total_full_forwards += 1

    # ------------------------------------------------------------------
    # Draft retrieval
    # ------------------------------------------------------------------
    def get_draft(self) -> Optional[torch.Tensor]:
        """Return cached probe-point features as the draft for skipped steps."""
        if self.probe_features_ref is None:
            return None
        return self.probe_features_ref.clone()

    # ------------------------------------------------------------------
    # Rejection recording
    # ------------------------------------------------------------------
    def record_rejection(self, step_idx: int, timestep: float,
                         reason: str, residual: float) -> None:
        """Log a rejection event for later bottleneck analysis."""
        self.rejections.append({
            "step": int(step_idx),
            "timestep": int(timestep),
            "reason": reason,
            "residual": float(residual),
            "consecutive_skips_before_reject": self.skip_count,
        })
        self.skip_count = 0

    def record_skip(self) -> None:
        """Increment the skip counter after a successful skip."""
        self.skip_count += 1
        self.total_skips += 1

    # ------------------------------------------------------------------
    # Statistics & export
    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        """Return summary statistics."""
        return {
            "gamma": self.gamma,
            "max_skip": self.max_skip,
            "total_full_forwards": self.total_full_forwards,
            "total_skips": self.total_skips,
            "total_rejections": len(self.rejections),
            "skip_ratio": (
                self.total_skips / (self.total_skips + self.total_full_forwards)
                if (self.total_skips + self.total_full_forwards) > 0 else 0.0
            ),
        }

    def export_rejections(self, path: str) -> None:
        """Export rejection log to JSON for bottleneck analysis."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "config": self.stats(),
                "rejections": self.rejections,
            }, f, indent=2)


# ---------------------------------------------------------------------------
# Split-forward helpers (for TeaCache acceleration)
# ---------------------------------------------------------------------------

def run_early_blocks(transformer, hidden_states, timestep_emb,
                     encoder_hidden_states,
                     attention_mask=None, encoder_attention_mask=None,
                     early_layer_idx: int = 2):
    """Run transformer blocks 0 .. early_layer_idx (inclusive).

    Returns the hidden states after the early blocks, which serve as the
    lightweight feature for residual monitoring.
    """
    for block in transformer.transformer_blocks[:early_layer_idx + 1]:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep_emb,
            cross_attention_kwargs=None,
            class_labels=None,
        )
    return hidden_states


def run_heavy_blocks(transformer, hidden_states, timestep_emb,
                     encoder_hidden_states,
                     attention_mask=None, encoder_attention_mask=None,
                     early_layer_idx: int = 2, probe_layer_idx: int = 14):
    """Run transformer blocks from early_layer_idx+1 through probe_layer_idx (inclusive).

    This is the "heavy core" that can be skipped when features are stable.
    Returns hidden states at the probe point.
    """
    for block in transformer.transformer_blocks[early_layer_idx + 1:probe_layer_idx + 1]:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep_emb,
            cross_attention_kwargs=None,
            class_labels=None,
        )
    return hidden_states


def run_tail(transformer, hidden_states, timestep_emb, embedded_timestep,
             encoder_hidden_states,
             attention_mask=None, encoder_attention_mask=None,
             probe_layer_idx: int = 14):
    """Forward from probe_layer_idx+1 through norm_out, proj_out, unpatchify.

    hidden_states: [B, Seq, Hidden] — probe-point features.
    Returns full model output [B, out_channels, H, W].
    """
    p = transformer.config.patch_size
    out_channels = transformer.config.out_channels
    H = W = transformer.config.sample_size // p

    # Remaining transformer blocks
    for block in transformer.transformer_blocks[probe_layer_idx + 1:]:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep_emb,
            cross_attention_kwargs=None,
            class_labels=None,
        )

    # adaLN output modulation
    shift, scale = (
        transformer.scale_shift_table[None]
        + embedded_timestep[:, None].to(transformer.scale_shift_table.device)
    ).chunk(2, dim=1)
    hidden_states = transformer.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
    hidden_states = transformer.proj_out(hidden_states)
    hidden_states = hidden_states.squeeze(1)

    # Unpatchify
    hidden_states = hidden_states.reshape(shape=(-1, H, W, p, p, out_channels))
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(shape=(-1, out_channels, H * p, W * p))
    return output


def run_full_blocks_range(transformer, hidden_states, timestep_emb,
                          encoder_hidden_states,
                          attention_mask=None, encoder_attention_mask=None,
                          start_block: int = 0, end_block: int = None):
    """Run transformer blocks from start_block (inclusive) to end_block (exclusive).

    Useful for running specific segments of the transformer.
    """
    blocks = transformer.transformer_blocks
    if end_block is None:
        end_block = len(blocks)
    for block in blocks[start_block:end_block]:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep_emb,
            cross_attention_kwargs=None,
            class_labels=None,
        )
    return hidden_states
