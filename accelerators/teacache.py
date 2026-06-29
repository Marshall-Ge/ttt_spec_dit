# -*- coding: utf-8 -*-
"""TeaCache accelerator — pure functions, plain-dict state.

Core mechanics (Liu et al., CVPR 2025):
  1. modulated_input = block0.norm1(h) * (1+scale_msa) + shift_msa   [timestep-modulated]
  2. diff = ||modulated_input - prev||_1.mean / prev.abs.mean            [relative L1]
  3. rescaled = poly4(diff)                                            [model-specific]
  4. accumulated += rescaled
  5. should_calc = (cnt in {0, num_steps-1}) or (accumulated >= thresh)
  6. if should_calc: out = blocks(h); residual = out - h; accumulated = 0
     else:            out = h + residual                              [cache hit]

The state dict is owned by the caller (sampling loop in ``run_dit.py`` /
``run_pixart.py``). The model itself is agnostic to TeaCache.

State dict keys (created by ``teacache_init``):
  - cnt, accumulated, previous_modulated_input, previous_residual
  - decisions, accum_history, raw_diff_history, rescaled_diff_history
  - num_steps, rel_l1_thresh, coefficients, rescale_func
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import load_coefficients


# ===========================================================================
# Block-0 modulation (cache trigger signal) — pure, no state
# ===========================================================================


def compute_modulated_input(
    transformer,
    hidden_states: torch.Tensor,
    timestep_emb: torch.Tensor,
) -> torch.Tensor:
    """PixArt-α modulated input at block 0 entrance.

    Mirrors ``BasicTransformerBlock.forward`` for ``norm_type == "ada_norm_single"``:
        shift_msa, scale_msa, ... = (block0.scale_shift_table[None]
                                     + timestep.reshape(B, 6, -1)).chunk(6, dim=1)
        modulated = block0.norm1(hidden_states) * (1 + scale_msa) + shift_msa
    """
    block0 = transformer.transformer_blocks[0]
    batch_size = hidden_states.shape[0]
    proj = block0.scale_shift_table[None] + timestep_emb.reshape(batch_size, 6, -1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = proj.chunk(6, dim=1)
    norm_hidden = block0.norm1(hidden_states)
    modulated = norm_hidden * (1 + scale_msa) + shift_msa
    return modulated


def compute_modulated_input_dit(
    transformer,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    class_labels: torch.Tensor,
) -> torch.Tensor:
    """DiT-2-256 modulated input at block 0 entrance.

    Uses ``AdaLayerNormZero.forward``::
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block0.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=...)
    The first return value is already modulated:
        norm(h) * (1+scale_msa) + shift_msa
    """
    block0 = transformer.transformer_blocks[0]
    modulated = block0.norm1(
        hidden_states, timestep=timestep, class_labels=class_labels,
        hidden_dtype=hidden_states.dtype,
    )[0]
    return modulated


# ===========================================================================
# Pure-function state API
# ===========================================================================


def teacache_init(
    num_steps: int,
    rel_l1_thresh: float = 0.25,
    coefficients: Optional[List[float]] = None,
) -> Dict:
    """Allocate TeaCache state dict.

    Parameters
    ----------
    num_steps : int
        Total denoising steps (first and last are always recomputed).
    rel_l1_thresh : float
        Accumulated relative-L1 threshold; higher = more aggressive caching.
    coefficients : list of 5 floats, optional
        4th-order polynomial (highest degree first) for distance rescaling.
        Defaults to ``config.load_coefficients()``.

    Returns
    -------
    state : dict
        Plain dict with all TeaCache runtime state. Pass this to every
        other ``teacache_*`` function.
    """
    if coefficients is None:
        coefficients = load_coefficients()

    return {
        # ---- config (immutable per generation) ----
        "num_steps": num_steps,
        "rel_l1_thresh": rel_l1_thresh,
        "coefficients": list(coefficients),
        "rescale_func": np.poly1d(coefficients),

        # ---- runtime state ----
        "cnt": 0,
        "accumulated": 0.0,
        "previous_modulated_input": None,
        "previous_residual": None,

        # ---- telemetry (also read by FLOPsMetric via .decisions) ----
        "decisions": [],
        "accum_history": [],
        "raw_diff_history": [],
        "rescaled_diff_history": [],
    }


def teacache_decide(state: Dict, modulated_input: torch.Tensor,
                    calibrator=None, probe_layer: int = -1) -> Tuple[bool, float]:
    """Decide whether the current step must run the full block stack.

    Parameters
    ----------
    state : dict
        TeaCache state (from ``teacache_init``).
    modulated_input : Tensor
        Output of ``compute_modulated_input()`` or ``compute_modulated_input_dit()``.
    calibrator : OnlineCalibrator, optional
        VFL M2 online calibrator. When provided and ready for the current bucket,
        its ``get_rescale_fn()`` and ``get_threshold()`` override the state's
        static ``rescale_func`` and ``rel_l1_thresh`` respectively.
    probe_layer : int
        Layer ID for ``(layer, bucket)`` key. Default -1 = TeaCache's global
        step-level sentinel. Set to the VFL probe layer (e.g. 20 for DiT) if
        per-layer calibration is desired.

    Returns
    -------
    should_calc : bool
    raw_rel_l1_diff : float
    """
    # Compute current timestep bucket (0/1/2) from step counter
    cnt = state["cnt"]
    num_steps = state["num_steps"]
    timestep_bucket = int(cnt * 3 / num_steps) if num_steps > 0 else 0
    timestep_bucket = min(timestep_bucket, 2)

    # ---- Resolve rescale function and threshold ----
    # TeaCache uses the offline-calibrated poly4 rescale function (state["rescale_func"])
    # because the RLS-based online rescale predicts true errors which are too small
    # for the accumulate-vs-threshold mechanism.  The calibrator only adjusts the
    # threshold (dynamic EMA), making it more or less conservative over time.
    if calibrator is not None:
        rescale_fn = state["rescale_func"]
        threshold = calibrator.get_threshold(probe_layer, timestep_bucket,
                                              default=state["rel_l1_thresh"])
    else:
        rescale_fn = state["rescale_func"]
        threshold = state["rel_l1_thresh"]

    if state["cnt"] == 0 or state["cnt"] == state["num_steps"] - 1:
        should_calc = True
        state["accumulated"] = 0.0
        raw_diff = 0.0
    else:
        prev = state["previous_modulated_input"]
        raw_diff = (
            (modulated_input - prev).abs().mean()
            / prev.abs().mean()
        ).detach().float().cpu().item()
        rescaled = max(0.0, float(rescale_fn(raw_diff)))
        state["accumulated"] += rescaled
        should_calc = state["accumulated"] >= threshold
        if should_calc:
            state["accumulated"] = 0.0

    # Telemetry
    state["accum_history"].append(state["accumulated"])
    state["raw_diff_history"].append(raw_diff)
    state["rescaled_diff_history"].append(
        float(rescale_fn(raw_diff)) if raw_diff > 0 else 0.0
    )
    state["decisions"].append("calc" if should_calc else "skip")
    state["previous_modulated_input"] = modulated_input.detach()
    # Stash for VFL calibrator: allows update_with_proxy(raw_diff, error_value)
    state["last_raw_diff"] = raw_diff
    return should_calc, raw_diff


def teacache_cache_residual(state: Dict, out: torch.Tensor, ori: torch.Tensor) -> None:
    """Store the residual of the full block stack: out - ori."""
    state["previous_residual"] = (out - ori).detach()


def teacache_apply_residual(state: Dict, hidden_states: torch.Tensor) -> torch.Tensor:
    """Fast path: reuse cached residual."""
    if state["previous_residual"] is None:
        return hidden_states
    return hidden_states + state["previous_residual"]


def teacache_step(state: Dict) -> None:
    """Advance the step counter (call once per denoising step)."""
    state["cnt"] += 1
    if state["cnt"] == state["num_steps"]:
        state["cnt"] = 0


def teacache_reset(state: Dict) -> None:
    """Reset runtime state for a new generation (keep config)."""
    state["cnt"] = 0
    state["accumulated"] = 0.0
    state["previous_modulated_input"] = None
    state["previous_residual"] = None
    state["decisions"] = []
    state["accum_history"] = []
    state["raw_diff_history"] = []
    state["rescaled_diff_history"] = []


# ===========================================================================
# Stats helper (standalone — not part of the state API)
# ===========================================================================

def teacache_stats(state: Dict) -> Dict:
    """Compute aggregate statistics from a TeaCache state dict."""
    n_calc = sum(1 for d in state["decisions"] if d == "calc")
    n_skip = sum(1 for d in state["decisions"] if d == "skip")
    total = n_calc + n_skip
    return {
        "rel_l1_thresh": state["rel_l1_thresh"],
        "coefficients": list(state["coefficients"]),
        "num_steps": state["num_steps"],
        "total_calc": n_calc,
        "total_skip": n_skip,
        "skip_ratio": n_skip / total if total > 0 else 0.0,
    }


def teacache_export_trace(state: Dict, path: str) -> None:
    """Export per-step trace to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "config": teacache_stats(state),
            "per_step": [
                {
                    "step": i,
                    "decision": state["decisions"][i],
                    "raw_rel_l1": state["raw_diff_history"][i],
                    "rescaled": state["rescaled_diff_history"][i],
                    "accumulated": state["accum_history"][i],
                }
                for i in range(len(state["decisions"]))
            ],
        }, f, indent=2)
