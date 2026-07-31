"""Tests for TeaCache forced-schedule mode (COVR equal-FLOPs arms).

When ``teacache_init`` receives a ``refresh_mask``, ``teacache_decide`` must
follow the mask verbatim (bypassing the dynamic accumulate-vs-threshold
logic), yielding equal-FLOPs arms for the COVR bandit.
"""

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
