# -*- coding: utf-8 -*-
"""
TeaCache for PixArt-Alpha — faithful re-implementation of Liu et al., CVPR 2025.

Core mechanics (verbatim from the paper / official repo):
  1. modulated_input = block0.norm1(h) * (1+scale_msa) + shift_msa   [timestep-modulated]
  2. diff = ||modulated_input - prev||_1.mean / prev.abs.mean          [relative L1]
  3. rescaled = poly4(diff)                                            [model-specific]
  4. accumulated += rescaled
  5. should_calc = (cnt in {0, num_steps-1}) or (accumulated >= thresh)
  6. if should_calc: out = blocks(h); residual = out - h; accumulated = 0
     else:            out = h + residual                              [cache hit]

No model weights are modified; we monkeypatch ``transformer.forward``.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from config import load_coefficients


# ---------------------------------------------------------------------------
# PixArt block-0 modulation (timestep-aware signal used as the cache trigger)
# ---------------------------------------------------------------------------

def compute_modulated_input(
    transformer,
    hidden_states: torch.Tensor,
    timestep_emb: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the modulation applied at the *entrance* of block 0.

    Mirrors ``BasicTransformerBlock.forward`` for ``norm_type == "ada_norm_single"``
    (diffusers/models/attention.py:450-455):

        shift_msa, scale_msa, ... = (block0.scale_shift_table[None]
                                     + timestep.reshape(B, 6, -1)).chunk(6, dim=1)
        modulated = block0.norm1(hidden_states) * (1 + scale_msa) + shift_msa

    This is the exact analogue of the FLUX ``block.norm1(inp, emb=temb_)`` call
    used by the official TeaCache to derive ``modulated_inp``.

    Parameters
    ----------
    hidden_states : Tensor [B, Seq, Hidden]  -- output of ``pos_embed`` (patch embed)
    timestep_emb : Tensor [B, 6*Hidden]      -- first return of ``adaln_single``
        (the linear(silu(emb)) projection), i.e. the same tensor the blocks
        receive as their ``timestep`` argument.
    """
    block0 = transformer.transformer_blocks[0]
    batch_size = hidden_states.shape[0]
    proj = block0.scale_shift_table[None] + timestep_emb.reshape(batch_size, 6, -1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = proj.chunk(6, dim=1)
    norm_hidden = block0.norm1(hidden_states)
    modulated = norm_hidden * (1 + scale_msa) + shift_msa
    return modulated


# ---------------------------------------------------------------------------
# TeaCache controller
# ---------------------------------------------------------------------------

class PixArtTeaCache:
    """Stateful TeaCache controller matching the official implementation.

    Parameters
    ----------
    num_steps : int
        Total denoising steps (first and last are always recomputed).
    rel_l1_thresh : float
        Accumulated relative-L1 threshold; higher = more aggressive caching.
    coefficients : list of 5 floats, optional
        4th-order polynomial (highest degree first) for distance rescaling.
        Defaults to calibrated PixArt coefficients (see config.load_coefficients).
    """

    def __init__(self,
                 num_steps: int,
                 rel_l1_thresh: float = 0.25,
                 coefficients: Optional[List[float]] = None):
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.coefficients = coefficients if coefficients is not None else load_coefficients()
        self.rescale_func = np.poly1d(self.coefficients)

        # ---- runtime state ----
        self.cnt: int = 0
        self.accumulated_rel_l1_distance: float = 0.0
        self.previous_modulated_input: Optional[torch.Tensor] = None
        self.previous_residual: Optional[torch.Tensor] = None

        # ---- telemetry ----
        self.decisions: List[str] = []          # "calc" / "skip" per step
        self.accum_history: List[float] = []     # accumulated distance at each step
        self.raw_diff_history: List[float] = []  # raw (pre-rescale) relative L1
        self.rescaled_diff_history: List[float] = []

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def decide(self, modulated_input: torch.Tensor) -> Tuple[bool, float]:
        """Decide whether the current step must run the full block stack.

        Returns (should_calc, raw_rel_l1_diff).
        Faithful to official logic: first & last step always compute.
        """
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
            raw_diff = 0.0
        else:
            prev = self.previous_modulated_input
            # relative L1 distance (mean over all elements)
            raw_diff = (
                (modulated_input - prev).abs().mean()
                / prev.abs().mean()
            ).detach().float().cpu().item()
            rescaled = float(self.rescale_func(raw_diff))
            self.accumulated_rel_l1_distance += rescaled
            should_calc = self.accumulated_rel_l1_distance >= self.rel_l1_thresh
            if should_calc:
                self.accumulated_rel_l1_distance = 0.0

        # bookkeeping
        self.accum_history.append(self.accumulated_rel_l1_distance)
        self.raw_diff_history.append(raw_diff)
        self.rescaled_diff_history.append(
            float(self.rescale_func(raw_diff)) if raw_diff > 0 else 0.0
        )
        self.decisions.append("calc" if should_calc else "skip")
        self.previous_modulated_input = modulated_input.detach()
        return should_calc, raw_diff

    # ------------------------------------------------------------------
    # Residual management
    # ------------------------------------------------------------------
    def cache_residual(self, out: torch.Tensor, ori: torch.Tensor) -> None:
        """Store the residual of the full block stack: out - ori."""
        self.previous_residual = (out - ori).detach()

    def apply_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fast path: reuse cached residual."""
        if self.previous_residual is None:
            return hidden_states
        return hidden_states + self.previous_residual

    def step(self) -> None:
        """Advance the step counter (call once per denoising step)."""
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        n_calc = sum(1 for d in self.decisions if d == "calc")
        n_skip = sum(1 for d in self.decisions if d == "skip")
        total = n_calc + n_skip
        return {
            "rel_l1_thresh": self.rel_l1_thresh,
            "coefficients": list(self.coefficients),
            "num_steps": self.num_steps,
            "total_calc": n_calc,
            "total_skip": n_skip,
            "skip_ratio": n_skip / total if total > 0 else 0.0,
        }

    def export_trace(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "config": self.stats(),
                "per_step": [
                    {
                        "step": i,
                        "decision": self.decisions[i],
                        "raw_rel_l1": self.raw_diff_history[i],
                        "rescaled": self.rescaled_diff_history[i],
                        "accumulated": self.accum_history[i],
                    }
                    for i in range(len(self.decisions))
                ],
            }, f, indent=2)


# ---------------------------------------------------------------------------
# TeaCache-augmented forward (monkeypatch target for PixArtTransformer2DModel)
# ---------------------------------------------------------------------------

def make_teacache_forward(teacache: PixArtTeaCache):
    """Build a forward function that replaces ``transformer.forward``.

    The body is a faithful copy of ``PixArtTransformer2DModel.forward`` with
    the TeaCache branch inserted right after ``pos_embed`` (matching the FLUX
    official implementation, which inserts after ``x_embedder``).
    """

    def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )

        # --- attention mask -> bias (identical to stock forward) ---
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        # ===============================================================
        # TeaCache branch (inserted here, after pos_embed + adaln)
        # ===============================================================
        modulated_input = compute_modulated_input(self, hidden_states, timestep)
        should_calc, _raw_diff = teacache.decide(modulated_input)

        if should_calc:
            ori_hidden_states = hidden_states.clone()
            # 2. full transformer blocks
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )
            # cache the residual of the *block stack* (before norm_out/proj_out)
            teacache.cache_residual(hidden_states, ori_hidden_states)
        else:
            # fast path: reuse cached residual, skip all blocks
            hidden_states = teacache.apply_residual(hidden_states)

        teacache.step()
        # ===============================================================

        # 3. Output (norm_out + proj_out + unpatchify) — always runs
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size,
                   self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels,
                   height * self.config.patch_size, width * self.config.patch_size)
        )

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)

    return teacache_forward


def install_teacache(transformer, teacache: PixArtTeaCache):
    """Monkeypatch ``transformer.forward`` with the TeaCache-augmented version.

    Returns the original forward so it can be restored later.
    """
    original_forward = transformer.forward
    transformer.forward = make_teacache_forward(teacache).__get__(transformer, type(transformer))
    return original_forward


def uninstall_teacache(transformer, original_forward):
    """Restore the original ``transformer.forward``."""
    transformer.forward = original_forward


# ---------------------------------------------------------------------------
# TeaCacheAccelerator: implements the Accelerator interface
# ---------------------------------------------------------------------------

class TeaCacheAccelerator:
    """Accelerator wrapper for TeaCache, implementing the project interface.

    Parameters
    ----------
    num_steps : int
    rel_l1_thresh : float
    coefficients : list of 5 floats, optional
    """

    def __init__(self, num_steps: int = 20, rel_l1_thresh: float = 0.25,
                 coefficients: Optional[List[float]] = None):
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.coefficients = coefficients if coefficients is not None else load_coefficients()
        self._teacache = None
        self._original_forward = None
        self._generator = None

    @property
    def teacache(self) -> PixArtTeaCache:
        if self._teacache is None:
            self._teacache = PixArtTeaCache(
                num_steps=self.num_steps,
                rel_l1_thresh=self.rel_l1_thresh,
                coefficients=self.coefficients,
            )
        return self._teacache

    def install(self, generator):
        """Install TeaCache onto a PixArtGenerator."""
        self._generator = generator
        self._original_forward = install_teacache(generator.transformer, self.teacache)

    def uninstall(self):
        """Restore original transformer.forward."""
        if self._generator is not None and self._original_forward is not None:
            uninstall_teacache(self._generator.transformer, self._original_forward)
            self._original_forward = None

    @property
    def stats(self) -> dict:
        if self._teacache is None:
            return {}
        return self._teacache.stats()

    def reset(self):
        """Reset TeaCache state for a new generation."""
        self._teacache = None
