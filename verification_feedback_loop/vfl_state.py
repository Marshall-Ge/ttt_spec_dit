# -*- coding: utf-8 -*-
"""Shared VFL (Verification Feedback Loop) module-level state and hooks.

These globals are process-wide singletons — set once before generation,
read during ``forward()``.  ``None`` → all VFL code is a no-op.

Originally duplicated identically in ``models/dit.py`` and
``models/pixart.py`` (~50 lines per file).  Extracted here so the model
files only keep model-specific constants (``_VFL_PROBE_LAYER``) and thin
wrapping helpers.
"""

from typing import Optional

# ===========================================================================
# Module-level globals (process-wide singletons)
# ===========================================================================

_vfl_buffer = None          # StratifiedReplayBuffer or None
_vfl_calibrator = None      # OnlineCalibrator or None
_vfl_model_version = "unknown"
_vfl_step_idx = 0           # set by denoising loop before each forward call
_vfl_num_steps = 50
_vfl_sample_id = 0          # identifies the current denoising trajectory (per-image)
_vfl_timestep_actual = 0    # real diffusion timestep (e.g. 981), NOT step_idx

# Behavior mode is derived from which globals the runner registers — no
# separate mode flag:
#   * cal=None, buf=None       → baseline (no --vfl): record_* is a no-op.
#   * cal set, buf=None        → --vfl --vfl-no-train: record_* feeds the
#                                calibrator a single scalar (no GPU→CPU
#                                tensor transfer, no VerificationEvent).
#   * cal set, buf set         → --vfl (with training): record_* builds a
#                                full VerificationEvent, writes to buffer,
#                                and updates the calibrator.


# ===========================================================================
# Setters / getters
# ===========================================================================

def set_vfl_buffer(buffer, model_version: str = "unknown"):
    """Register the global VFL buffer for this process."""
    global _vfl_buffer, _vfl_model_version
    _vfl_buffer = buffer
    _vfl_model_version = model_version


def set_vfl_calibrator(calibrator):
    """Register the global VFL calibrator for this process (M2)."""
    global _vfl_calibrator
    _vfl_calibrator = calibrator


def get_vfl_calibrator():
    """Return the current VFL calibrator (None if VFL / M2 is disabled)."""
    return _vfl_calibrator


def get_vfl_buffer():
    """Return the current VFL buffer (None if VFL is disabled)."""
    return _vfl_buffer


def set_vfl_step_info(step_idx: int,
                      num_steps: int,
                      timestep_actual: Optional[int] = None):
    """Set per-step tracking info for VFL hooks (called by denoising loop).

    Parameters
    ----------
    step_idx : int
        Index of the current denoising step (0..N-1).
    num_steps : int
        Total number of denoising steps.
    timestep_actual : int, optional
        The REAL diffusion timestep value (e.g. 981, 947, ...) — NOT the
        step index. Required for correct L3 training: when compute_training_loss
        re-runs forward, it must pass the actual timestep so adaLN modulation
        matches the original recording. If None, falls back to step_idx
        (legacy behaviour, slightly noisier supervision).
    """
    global _vfl_step_idx, _vfl_num_steps, _vfl_timestep_actual
    _vfl_step_idx = step_idx
    _vfl_num_steps = num_steps
    if timestep_actual is not None:
        _vfl_timestep_actual = int(timestep_actual)


def get_vfl_step_idx() -> int:
    """Return the current denoising step index (set by set_vfl_step_info)."""
    return _vfl_step_idx


def get_vfl_num_steps() -> int:
    """Return the total number of denoising steps (set by set_vfl_step_info)."""
    return _vfl_num_steps


def set_vfl_sample_id(sample_id: int):
    """Set the current denoising trajectory id (called before each image).

    Used by curvature loss to group events from the same trajectory.
    Default 0 — fine for single-image or batched-all-same-id scenarios.
    """
    global _vfl_sample_id
    _vfl_sample_id = sample_id


# ===========================================================================
# Internal helpers — mode-aware fast paths
# ===========================================================================

def _make_calibrator_speca_update(cal,
                                  layer_id: int,
                                  step_idx: int,
                                  num_steps: int,
                                  error_value: float):
    """Fast path: feed a single scalar to the calibrator only.

    Avoids constructing a VerificationEvent (no torch tensors touched, no
    GPU→CPU sync).  Used by ``record_speca_event`` when no buffer is
    registered (``--vfl --vfl-no-train`` mode).
    """
    if num_steps <= 0:
        return
    bucket = int(step_idx * 3 / num_steps)
    if bucket > 2:
        bucket = 2
    # Build a lightweight stand-in object so we don't have to special-case
    # OnlineCalibrator.update — it only reads (layer_id, timestep_bucket,
    # error_value).  Avoids the per-event torch tensor allocation entirely.
    cal.update(_ScalarCalibEvent(layer_id, bucket, error_value))


def _make_calibrator_teacache_update(cal,
                                     layer_id: int,
                                     step_idx: int,
                                     num_steps: int,
                                     error_value: float):
    """Fast path for TeaCache probe events (mirror of the SpecA version)."""
    if num_steps <= 0:
        return
    bucket = int(step_idx * 3 / num_steps)
    if bucket > 2:
        bucket = 2
    cal.update(_ScalarCalibEvent(layer_id, bucket, error_value))


class _ScalarCalibEvent:
    """Minimal stand-in for VerificationEvent consumed by OnlineCalibrator.

    OnlineCalibrator.update only reads ``layer_id``, ``timestep_bucket`` and
    ``error_value``.  Feeding this lightweight object instead of a fully
    built VerificationEvent avoids the per-event ``.detach().half().cpu()``
    tensor transfer on the hot path.
    """

    __slots__ = ("layer_id", "timestep_bucket", "error_value")

    def __init__(self, layer_id: int, timestep_bucket: int, error_value: float):
        self.layer_id = layer_id
        self.timestep_bucket = timestep_bucket
        self.error_value = error_value


# ===========================================================================
# Event-recording hooks (called from model forward passes)
# ===========================================================================

def record_speca_event(layer_id: int,
                       timestep_val: int,
                       step_idx: int,
                       num_steps: int,
                       predicted_hidden: "torch.Tensor",
                       full_hidden: "torch.Tensor",
                       error_value: float,
                       model: str,
                       module_name: str = "",
                       latent_input: "Optional[torch.Tensor]" = None,
                       class_labels: "Optional[torch.Tensor]" = None,
                       encoder_hidden_states: "Optional[torch.Tensor]" = None):
    """Record a SpecA verification event to the global VFL buffer + calibrator.

    Called from within the SpecA Taylor-path error probe (check_layer).

    Behavior is derived from which globals the runner registered:
      * cal=None, buf=None  → no-op (baseline, no --vfl).
      * cal set, buf=None   → scalar calibrator update only; no
                              VerificationEvent built, no GPU→CPU tensor
                              transfer (--vfl --vfl-no-train).
      * cal set, buf set    → full VerificationEvent + buffer write +
                              calibrator update (--vfl with training).
    """
    cal = _vfl_calibrator
    buf = _vfl_buffer

    if cal is None and buf is None:
        return

    if buf is None:
        # No buffer → no point building a VerificationEvent (it would be
        # discarded).  Feed the calibrator a single scalar instead.
        _make_calibrator_speca_update(
            cal, layer_id, step_idx, num_steps, error_value)
        return

    # Buffer registered → build full VerificationEvent + write to buffer.
    import verification_feedback_loop.verification_hook as vh
    event = vh.make_speca_event(
        layer_id=layer_id, timestep_val=timestep_val,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=predicted_hidden, full_hidden=full_hidden,
        error_value=error_value, error_metric="cosine_similarity",
        model=model, base_model_version=_vfl_model_version,
        module=module_name,
        latent_input=latent_input,
        class_labels=class_labels,
        encoder_hidden_states=encoder_hidden_states,
        sample_id=_vfl_sample_id,
        timestep_actual=_vfl_timestep_actual,
    )
    vh.record_event(event, buffer=buf)
    if cal is not None:
        cal.update(event)


def record_teacache_event(layer_id: int,
                          timestep_val: int,
                          step_idx: int,
                          num_steps: int,
                          predicted_hidden: "torch.Tensor",
                          true_hidden: "torch.Tensor",
                          model: str,
                          raw_diff: float = 0.0,
                          latent_input: "Optional[torch.Tensor]" = None,
                          class_labels: "Optional[torch.Tensor]" = None,
                          encoder_hidden_states: "Optional[torch.Tensor]" = None):
    """Record a TeaCache probe event to the global VFL buffer + calibrator.

    Called from the TeaCache calc-step path after the block stack runs,
    comparing the skip-path prediction against the just-computed ground truth.

    Same mode derivation as ``record_speca_event``: buf=None → scalar
    calibrator-only update; buf set → full VerificationEvent + buffer write.
    """
    cal = _vfl_calibrator
    buf = _vfl_buffer

    if cal is None and buf is None:
        return

    if buf is None:
        if cal is not None:
            _make_calibrator_teacache_update(
                cal, layer_id, step_idx, num_steps, float(raw_diff))
        return

    import verification_feedback_loop.verification_hook as vh
    event = vh.make_teacache_probe_event(
        layer_id=layer_id, timestep_val=timestep_val,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=predicted_hidden, true_hidden=true_hidden,
        model=model, base_model_version=_vfl_model_version,
        latent_input=latent_input,
        class_labels=class_labels,
        encoder_hidden_states=encoder_hidden_states,
        sample_id=_vfl_sample_id,
        timestep_actual=_vfl_timestep_actual,
    )
    vh.record_event(event, buffer=buf)
    if cal is not None:
        cal.update(event)
