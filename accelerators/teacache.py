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
  - num_steps, rel_l1_thresh, coefficients, rescale_func, refresh_mask
  - boundary_probe (dict, optional) — opt-in telemetry for COVR-v2 viability

Boundary probe lifecycle (opt-in, decision-neutral):
  - ``teacache_init(probe_prefix_steps=N)`` → ``state["boundary_probe"]`` dict
  - ``teacache_decide`` writes per-step causal scalars (modulation distances,
    shadow accumulator, cached residual norms) into the probe without changing
    decisions or the real accumulator.
  - ``teacache_reset`` clears the per-trajectory probe rows but keeps config.
  - ``teacache_boundary_snapshot(state)`` → last probe row dict or None.
  - When ``probe_prefix_steps`` is None (default), no probe key is created
    and no telemetry is computed — the hot path is unchanged.
"""

import json
import math
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
    refresh_mask: Optional[Tuple[bool, ...]] = None,
    probe_prefix_steps: Optional[int] = None,
    gamma_schedule = None,
    residual_decay_lambda: float = 0.0,
    residual_mode: str = "plain",
) -> Dict:
    """Allocate TeaCache state dict.

    Parameters
    ----------
    num_steps : int
        Total denoising steps (first and last are always recomputed).
    rel_l1_thresh : float
        Accumulated relative-L1 threshold; higher = more aggressive caching.
        Ignored per-step when ``gamma_schedule`` is provided.
    coefficients : list of 5 floats, optional
        4th-order polynomial (highest degree first) for distance rescaling.
        Defaults to ``config.load_coefficients()``.
    refresh_mask : tuple of bool, optional
        Forced-schedule mode (used by the COVR bandit). When provided, the
        per-step calc/skip decision follows the mask verbatim.
    gamma_schedule : callable, optional
        Per-step γ(t) schedule: ``gamma_schedule(step_idx, num_steps) -> float``.
        When provided, overrides ``rel_l1_thresh`` for every step, enabling
        step-asymmetric caching (e.g. aggressive early, conservative late).
        Default ``None`` uses constant ``rel_l1_thresh`` throughout.
    probe_prefix_steps : int, optional
        Opt-in boundary telemetry for COVR-v2 viability. When not None,
        creates ``state["boundary_probe"]`` and records causal scalars
        (modulation distances, shadow accumulator, cached residual norms)
        on the first ``probe_prefix_steps`` forced-schedule steps. Does not
        alter decisions, the real accumulator, or any history list. Default
        ``None`` keeps the hot path byte-identical to the no-probe path.
    residual_mode : str, optional
        Skip-step residual application: 'plain' (x + res), 'linear'
        (x + res + (res - res_prev), full residual-drift extrapolation),
        or 'damped' (Hermite-scheduled drift, u = min(streak/10, 1)).
        'linear' validated to cut latent MSE 53-66% at equal skip rate
        (2026-08-27); TeaCache's accumulate-vs-threshold keeps actual streaks
        at 2-3 steps, squarely inside linear extrapolation's best window.

    Returns
    -------
    state : dict
        Plain dict with all TeaCache runtime state. Pass this to every
        other ``teacache_*`` function.
    """
    if residual_mode not in ("plain", "linear", "damped"):
        raise ValueError(
            f"residual_mode must be 'plain'/'linear'/'damped', got {residual_mode!r}")
    if coefficients is None:
        coefficients = load_coefficients()

    validated_mask: Optional[Tuple[bool, ...]] = None
    if refresh_mask is not None:
        validated_mask = tuple(refresh_mask)
        if len(validated_mask) != num_steps:
            raise ValueError("refresh_mask length must match num_steps")
        if any(type(value) is not bool for value in validated_mask):
            raise ValueError("refresh_mask values must be booleans")
        if not validated_mask[0]:
            raise ValueError("refresh_mask must refresh the first step")

    state = {
        # ---- config (immutable per generation) ----
        "num_steps": num_steps,
        "rel_l1_thresh": rel_l1_thresh,
        "coefficients": list(coefficients),
        "rescale_func": np.poly1d(coefficients),
        "refresh_mask": validated_mask,
        "gamma_schedule": gamma_schedule,  # callable or None

        # ---- runtime state ----
        "cnt": 0,
        "accumulated": 0.0,
        "previous_modulated_input": None,
        "previous_residual": None,
        "previous_residual2": None,  # residual from the calc before last
        "skip_streak": 0,  # consecutive skips (for staleness-aware decay)
        "residual_decay_lambda": residual_decay_lambda,
        "residual_mode": residual_mode,

        # ---- telemetry (also read by FLOPsMetric via .decisions) ----
        "decisions": [],
        "accum_history": [],
        "raw_diff_history": [],
        "rescaled_diff_history": [],
    }

    if probe_prefix_steps is not None:
        if probe_prefix_steps <= 0:
            raise ValueError("probe_prefix_steps must be positive")
        if validated_mask is None:
            raise ValueError("boundary probe requires a forced refresh_mask")
        state["boundary_probe"] = {
            "rows": [],
            "shadow_accum": 0.0,
            "prefix_steps": int(probe_prefix_steps),
        }

    return state


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
        its ``get_threshold()`` overrides the static ``rel_l1_thresh``.
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

    # Forced-schedule mode (COVR bandit): follow the mask verbatim, bypassing
    # the dynamic accumulate-vs-threshold logic. previous_modulated_input is
    # still updated so state stays consistent, but no diff/threshold is used.
    refresh_mask = state.get("refresh_mask")
    if refresh_mask is not None:
        if not 0 <= cnt < len(refresh_mask):
            raise ValueError("TeaCache step is outside the refresh mask")
        should_calc = bool(refresh_mask[cnt])
        state["accumulated"] = 0.0
        state["accum_history"].append(0.0)
        state["raw_diff_history"].append(0.0)
        state["rescaled_diff_history"].append(0.0)
        state["decisions"].append("calc" if should_calc else "skip")
        state["skip_streak"] = 0 if should_calc else state.get("skip_streak", 0) + 1
        state["previous_modulated_input"] = modulated_input.detach()
        state["last_raw_diff"] = 0.0

        # ---- Boundary telemetry (opt-in, decision-neutral) ----
        # Runs only when the COVR-v2 viability probe is active.  Records
        # causal scalars that the static dynamic-threshold path would have
        # computed at this point — modulation distances, shadow accumulator,
        # and cached-residual norms.  Never mutates state["accumulated"],
        # decision lists, or the forced return value.
        probe = state.get("boundary_probe")
        if probe is not None and cnt < probe["prefix_steps"]:
            row = {"cnt": int(cnt),
                   "timestep_bucket": int(timestep_bucket),
                   "step_frac": float(cnt / max(num_steps - 1, 1))}
            if cnt == 0:
                # No predecessor — all diff/residual fields are missing.
                row["raw_diff"] = float("nan")
                row["rescaled"] = float("nan")
                row["shadow_accum_before"] = 0.0
                row["shadow_accum_after"] = 0.0
                row["dynamic_would_calc"] = True  # cnt==0 always calc in dynamic
                row["prev_mod_l1"] = float("nan")
                row["prev_mod_l2"] = float("nan")
                row["prev_residual_l1"] = float("nan")
                row["prev_residual_l2"] = float("nan")
                row["residual_modulated_ratio"] = float("nan")
            else:
                prev = state["previous_modulated_input"]
                if prev is not None and modulated_input is not None:
                    raw_diff = float(
                        ((modulated_input - prev).abs().mean()
                         / prev.abs().mean().clamp_min(1e-8))
                        .detach().float().cpu().item())
                    if not math.isfinite(raw_diff):
                        raw_diff = float("nan")
                else:
                    raw_diff = float("nan")
                rescaled = (
                    max(0.0, float(state["rescale_func"](raw_diff)))
                    if math.isfinite(raw_diff) else float("nan"))
                shadow_before = probe["shadow_accum"]
                shadow_after = shadow_before + rescaled if math.isfinite(rescaled) else shadow_before
                dynamic_would_calc = (
                    True if cnt == num_steps - 1
                    else (shadow_after >= state["rel_l1_thresh"]))
                if dynamic_would_calc:
                    shadow_after = 0.0

                row["raw_diff"] = raw_diff
                row["rescaled"] = rescaled
                row["shadow_accum_before"] = shadow_before
                row["shadow_accum_after"] = shadow_after
                row["dynamic_would_calc"] = dynamic_would_calc
                # previous_modulated_input statistics
                row["prev_mod_l1"] = float(
                    prev.detach().float().abs().mean().cpu().item())
                row["prev_mod_l2"] = float(
                    prev.detach().float().square().mean().sqrt().cpu().item())
                # previous_residual (cached at end of step cnt-1's calc)
                res = state.get("previous_residual")
                if res is not None:
                    res_f = res.detach().float()
                    row["prev_residual_l1"] = float(res_f.abs().mean().cpu().item())
                    row["prev_residual_l2"] = float(
                        res_f.square().mean().sqrt().cpu().item())
                    row["residual_modulated_ratio"] = (
                        row["prev_residual_l1"] / row["prev_mod_l1"]
                        if row["prev_mod_l1"] > 1e-12 else float("nan"))
                else:
                    row["prev_residual_l1"] = float("nan")
                    row["prev_residual_l2"] = float("nan")
                    row["residual_modulated_ratio"] = float("nan")

                probe["shadow_accum"] = shadow_after

            probe["rows"].append(row)

        return should_calc, 0.0

    # Rescale: always use offline poly4 (RLS-based online rescale was removed —
    # it predicted values too small for the accumulate-vs-threshold mechanism).
    rescale_fn = state["rescale_func"]

    # Threshold: γ(t) schedule > calibrator > static constant
    gamma_schedule = state.get("gamma_schedule")
    if gamma_schedule is not None:
        threshold = gamma_schedule(state["cnt"], state["num_steps"])
    elif calibrator is not None:
        threshold = calibrator.get_threshold(probe_layer, timestep_bucket,
                                              default=state["rel_l1_thresh"])
    else:
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
    state["skip_streak"] = 0 if should_calc else state.get("skip_streak", 0) + 1
    state["previous_modulated_input"] = modulated_input.detach()
    state["last_raw_diff"] = raw_diff
    return should_calc, raw_diff


def teacache_cache_residual(state: Dict, out: torch.Tensor, ori: torch.Tensor) -> None:
    """Store the residual of the full block stack: out - ori.

    Keeps the previous residual too (``previous_residual2``) so the skip path
    can extrapolate residual drift (linear/damped modes).
    """
    state["previous_residual2"] = state["previous_residual"]
    state["previous_residual"] = (out - ori).detach()


def teacache_apply_residual(state: Dict, hidden_states: torch.Tensor) -> torch.Tensor:
    """Fast path: reuse cached residual with mode + staleness-aware handling.

    Modes:
      plain  : x + res                                   (vanilla TeaCache)
      linear : x + res + (res - res_prev)                (full drift)
      damped : x + res + alpha * (res - res_prev)        (Hermite-scheduled)
    When ``residual_decay_lambda > 0`` the base residual is additionally
    decayed by ``exp(-λ*(streak-1))`` (staleness-aware; applied before the
    mode-specific drift term).
    """
    if state["previous_residual"] is None:
        return hidden_states
    residual = state["previous_residual"]
    lam = state.get("residual_decay_lambda", 0.0)
    streak = state.get("skip_streak", 0)
    if lam > 0 and streak > 1:
        residual = residual * math.exp(-lam * (streak - 1))

    mode = state.get("residual_mode", "plain")
    if mode != "plain":
        prev2 = state.get("previous_residual2")
        if prev2 is not None:
            drift = residual - prev2
            if mode == "linear":
                residual = residual + drift
            elif mode == "damped":
                u = min(streak / 10.0, 1.0)
                alpha = 3.0 * u * u - 2.0 * u * u * u
                residual = residual + alpha * drift
    return hidden_states + residual


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
    state["previous_residual2"] = None
    state["skip_streak"] = 0
    state["decisions"] = []
    state["accum_history"] = []
    state["raw_diff_history"] = []
    state["rescaled_diff_history"] = []
    probe = state.get("boundary_probe")
    if probe is not None:
        probe["rows"] = []
        probe["shadow_accum"] = 0.0


def teacache_boundary_snapshot(state: Dict) -> Optional[Dict]:
    """Return the last probe row dict, or None if the probe is absent.

    The caller (``run_dit.py`` viability hook) reads this immediately after
    the transformer forward but before ``scheduler.step``, so the row was
    populated by the ``teacache_decide`` call at the top of the block stack.
    All scalars are host-native floats; no tensors remain.
    """
    probe = state.get("boundary_probe")
    if probe is None or not probe["rows"]:
        return None
    return probe["rows"][-1]


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
