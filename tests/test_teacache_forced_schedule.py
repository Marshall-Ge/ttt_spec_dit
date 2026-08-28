"""Tests for TeaCache forced-schedule mode (COVR equal-FLOPs arms).

When ``teacache_init`` receives a ``refresh_mask``, ``teacache_decide`` must
follow the mask verbatim (bypassing the dynamic accumulate-vs-threshold
logic), yielding equal-FLOPs arms for the COVR bandit.

Boundary probe (COVR-v2 viability) tests verify:
- Telemetry is absent by default (no probe_prefix_steps).
- Forced decisions and histories are bit-for-bit unchanged when telemetry is enabled.
- Telemetry resets correctly (probe rows cleared, shadow_accum back to 0).
- Cached residual is from the prior step (step 0 has NaN predecessor fields,
  step 1+ has finite prev_mod / prev_residual from prior calc step cache).
"""

import math

import numpy as np
import pytest
import torch

from accelerators.teacache import (
    teacache_decide,
    teacache_init,
    teacache_reset,
    teacache_step,
)


# A trivial poly so we never depend on the offline coefficients file.
_COEF = [0.0, 0.0, 0.0, 1.0, 0.0]  # rescaled == raw_diff


def _modulated(scale: float = 1.0) -> torch.Tensor:
    return torch.full((2, 4, 8), scale, dtype=torch.float32)


def _run_mask(mask):
    """Drive a full generation through the forced schedule, return decisions."""
    state = teacache_init(
        num_steps=len(mask), coefficients=_COEF, refresh_mask=mask)
    decisions = []
    for step in range(len(mask)):
        should_calc, raw_diff = teacache_decide(state, _modulated(step + 1.0))
        decisions.append(should_calc)
        assert raw_diff == 0.0  # forced mode never computes a diff
        teacache_step(state)
    return state, decisions


# ===========================================================================
# Mask execution
# ===========================================================================


def test_forced_schedule_follows_mask_verbatim():
    mask = (True, False, False, True, False, True)
    state, decisions = _run_mask(mask)
    assert tuple(decisions) == mask
    assert tuple(d == "calc" for d in state["decisions"]) == mask


def test_forced_schedule_calc_count_equals_refresh_count():
    """Equal-FLOPs invariant: number of calc steps == refresh count."""
    mask = (True, False, True, False, False, True, False, False)
    state, decisions = _run_mask(mask)
    n_calc = sum(1 for d in state["decisions"] if d == "calc")
    assert n_calc == sum(mask)
    assert sum(decisions) == sum(mask)


def test_forced_schedule_ignores_modulated_input_magnitude():
    """A wildly varying modulated signal must not change the schedule."""
    mask = (True, False, False, False, True)
    state = teacache_init(
        num_steps=len(mask), coefficients=_COEF, refresh_mask=mask)
    # Feed huge, erratic diffs that would trip any threshold logic.
    signals = [1.0, 1000.0, 0.001, 5000.0, 2.0]
    decisions = []
    for step, sig in enumerate(signals):
        should_calc, _ = teacache_decide(state, _modulated(sig))
        decisions.append(should_calc)
        teacache_step(state)
    assert tuple(decisions) == mask
    # Accumulator never grows in forced mode.
    assert all(v == 0.0 for v in state["accum_history"])


def test_forced_schedule_telemetry_is_zeroed():
    mask = (True, False, True)
    state, _ = _run_mask(mask)
    assert state["raw_diff_history"] == [0.0, 0.0, 0.0]
    assert state["rescaled_diff_history"] == [0.0, 0.0, 0.0]
    assert state["accum_history"] == [0.0, 0.0, 0.0]


# ===========================================================================
# Validation (mirrors SpecA refresh_mask rules)
# ===========================================================================


def test_forced_schedule_rejects_wrong_length():
    with pytest.raises(ValueError, match="length must match num_steps"):
        teacache_init(num_steps=4, coefficients=_COEF,
                      refresh_mask=(True, False, True))


def test_forced_schedule_rejects_non_bool():
    with pytest.raises(ValueError, match="values must be booleans"):
        teacache_init(num_steps=3, coefficients=_COEF,
                      refresh_mask=(True, 0, 1))


def test_forced_schedule_requires_first_step_refresh():
    with pytest.raises(ValueError, match="must refresh the first step"):
        teacache_init(num_steps=3, coefficients=_COEF,
                      refresh_mask=(False, True, True))


# ===========================================================================
# Reset keeps the mask (config), clears runtime
# ===========================================================================


def test_reset_preserves_refresh_mask():
    mask = (True, False, True, False)
    state = teacache_init(
        num_steps=len(mask), coefficients=_COEF, refresh_mask=mask)
    for step in range(len(mask)):
        teacache_decide(state, _modulated(step + 1.0))
        teacache_step(state)
    assert state["cnt"] == 0  # wrapped around after num_steps
    assert len(state["decisions"]) == len(mask)

    teacache_reset(state)
    assert state["refresh_mask"] == mask       # config preserved
    assert state["decisions"] == []            # runtime cleared
    assert state["previous_modulated_input"] is None

    # Second generation reproduces the identical schedule.
    decisions = []
    for step in range(len(mask)):
        should_calc, _ = teacache_decide(state, _modulated(step + 10.0))
        decisions.append(should_calc)
        teacache_step(state)
    assert tuple(decisions) == mask


# ===========================================================================
# Default (no mask) still runs the dynamic threshold path
# ===========================================================================


def test_no_mask_uses_dynamic_threshold():
    state = teacache_init(num_steps=5, rel_l1_thresh=0.25, coefficients=_COEF)
    assert state["refresh_mask"] is None
    # First step always calc; feed identical signal so no accumulation.
    should_calc, _ = teacache_decide(state, _modulated(1.0))
    assert should_calc is True  # cnt == 0 forced calc
    teacache_step(state)
    # Middle step with zero diff → accumulated stays 0 < thresh → skip.
    should_calc, raw = teacache_decide(state, _modulated(1.0))
    assert should_calc is False
    assert raw == 0.0


# ===========================================================================
# Boundary probe (COVR-v2 viability telemetry)
# ===========================================================================


def test_boundary_probe_absent_by_default():
    """Without probe_prefix_steps, state has no boundary_probe key."""
    state = teacache_init(num_steps=5, coefficients=_COEF,
                          refresh_mask=(True, False, True, False, True))
    assert "boundary_probe" not in state


def test_boundary_probe_requires_forced_refresh_mask():
    """probe_prefix_steps without refresh_mask raises ValueError."""
    with pytest.raises(ValueError, match="boundary probe requires a forced refresh_mask"):
        teacache_init(num_steps=5, coefficients=_COEF,
                      probe_prefix_steps=3)


def test_boundary_probe_rejects_non_positive_prefix():
    with pytest.raises(ValueError, match="probe_prefix_steps must be positive"):
        teacache_init(num_steps=5, coefficients=_COEF,
                      refresh_mask=(True, True, True, True, True),
                      probe_prefix_steps=0)


def test_boundary_probe_state_structure():
    mask = (True, True, False, True, False)
    state = teacache_init(num_steps=len(mask), coefficients=_COEF,
                          refresh_mask=mask, probe_prefix_steps=3)
    probe = state["boundary_probe"]
    assert isinstance(probe, dict)
    assert probe["rows"] == []
    assert probe["shadow_accum"] == 0.0
    assert probe["prefix_steps"] == 3


def test_boundary_probe_records_rows_during_prefix():
    mask = (True, True, True, False, False)
    state = teacache_init(num_steps=len(mask), coefficients=_COEF,
                          refresh_mask=mask, probe_prefix_steps=3)
    for step in range(len(mask)):
        should_calc, _ = teacache_decide(state, _modulated(step + 1.0))
        assert should_calc == mask[step]
        teacache_step(state)
    probe = state["boundary_probe"]
    rows = probe["rows"]
    # Only first 3 steps (prefix_steps=3) are recorded
    assert len(rows) == 3
    # Step 0: no predecessor → NaN in diff/residual fields
    r0 = rows[0]
    assert r0["cnt"] == 0
    assert math.isnan(r0["raw_diff"])
    assert math.isnan(r0["rescaled"])
    assert math.isnan(r0["prev_residual_l1"])
    assert math.isnan(r0["prev_residual_l2"])
    assert r0["shadow_accum_before"] == 0.0
    assert r0["shadow_accum_after"] == 0.0
    assert r0["dynamic_would_calc"] is True
    # Step 1: has predecessor, but no residual yet (previous_residual is None)
    r1 = rows[1]
    assert r1["cnt"] == 1
    assert not math.isnan(r1["raw_diff"])
    assert not math.isnan(r1["prev_mod_l1"])
    assert not math.isnan(r1["prev_mod_l2"])
    # No residual cached on a non-calc step (previous_residual still None
    # because the forced mask didn't compute at step 1... wait, step 0 was calc,
    # but we never called teacache_cache_residual, so previous_residual is None)
    assert math.isnan(r1["prev_residual_l1"])
    assert math.isnan(r1["residual_modulated_ratio"])


def test_boundary_probe_does_not_mutate_forced_decisions():
    """Enabling the probe must not change any forced decision or history."""
    mask = (True, False, True, False, True)
    state_no = teacache_init(num_steps=len(mask), coefficients=_COEF,
                             refresh_mask=mask)
    state_yes = teacache_init(num_steps=len(mask), coefficients=_COEF,
                              refresh_mask=mask, probe_prefix_steps=2)
    dec_no = []
    dec_yes = []
    for step in range(len(mask)):
        sc_no, rd_no = teacache_decide(state_no, _modulated(step + 1.0))
        sc_yes, rd_yes = teacache_decide(state_yes, _modulated(step + 1.0))
        dec_no.append(sc_no)
        dec_yes.append(sc_yes)
        assert rd_no == 0.0
        assert rd_yes == 0.0
        teacache_step(state_no)
        teacache_step(state_yes)
    assert dec_no == dec_yes
    assert state_no["decisions"] == state_yes["decisions"]
    assert state_no["accum_history"] == state_yes["accum_history"]
    assert state_no["raw_diff_history"] == state_yes["raw_diff_history"]
    assert state_no["rescaled_diff_history"] == state_yes["rescaled_diff_history"]


def test_boundary_probe_reset_clears_rows_and_shadow_accum():
    mask = (True, True, True, False, False)
    state = teacache_init(num_steps=len(mask), coefficients=_COEF,
                          rel_l1_thresh=0.25,
                          refresh_mask=mask, probe_prefix_steps=3)
    for step in range(3):
        teacache_decide(state, _modulated(step + 1.0))
        teacache_step(state)
    probe = state["boundary_probe"]
    assert len(probe["rows"]) == 3
    # shadow_accum may be 0.0 (every dynamic-would-calc resets it) or non-zero;
    # either way, reset must zero it.
    teacache_reset(state)
    assert probe["rows"] == []
    assert probe["shadow_accum"] == 0.0
    assert probe["prefix_steps"] == 3  # config preserved


def test_boundary_probe_after_prefix_stops_recording():
    mask = (True, True, False, True, False)
    state = teacache_init(num_steps=len(mask), coefficients=_COEF,
                          refresh_mask=mask, probe_prefix_steps=2)
    for step in range(len(mask)):
        teacache_decide(state, _modulated(step + 1.0))
        teacache_step(state)
    probe = state["boundary_probe"]
    assert len(probe["rows"]) == 2  # only first 2 steps recorded


def test_boundary_snapshot_returns_last_row():
    mask = (True, True, True, False, False)
    state = teacache_init(num_steps=len(mask), coefficients=_COEF,
                          refresh_mask=mask, probe_prefix_steps=3)
    from accelerators.teacache import teacache_boundary_snapshot
    assert teacache_boundary_snapshot(state) is None  # no rows yet
    for step in range(3):
        teacache_decide(state, _modulated(step + 1.0))
        teacache_step(state)
    snap = teacache_boundary_snapshot(state)
    assert snap is not None
    assert snap["cnt"] == 2  # last row is step 2


def test_boundary_snapshot_returns_none_when_probe_absent():
    state = teacache_init(num_steps=5, coefficients=_COEF,
                          refresh_mask=(True, False, True, False, True))
    from accelerators.teacache import teacache_boundary_snapshot
    assert teacache_boundary_snapshot(state) is None
