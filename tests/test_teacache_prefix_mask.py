"""TeaCache forced-calc prefix mask + mid-trajectory commit transition.

The contextual bandit initializes TeaCache with ``refresh_mask=(True,)*num_steps``
so the prefix recomputes the full block stack every step. At commit time the
loop drops the mask (``refresh_mask=None``) and sets the chosen arm's
``rel_l1_thresh``; the dynamic accumulate-vs-threshold path must resume cleanly
from step ``prefix_steps`` using the ``previous_modulated_input`` cached by the
last prefix step.

Also documents the ``teacache_reset`` gamma-leak hazard (the runner restores the
baseline gamma explicitly because reset does not).
"""

import pytest
import torch

from accelerators.teacache import (
    teacache_decide,
    teacache_init,
    teacache_reset,
    teacache_step,
)


def _modulated(seed: int, dim: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, dim, generator=generator)


def test_all_true_mask_forces_calc_each_prefix_step():
    num_steps = 6
    state = teacache_init(
        num_steps, rel_l1_thresh=0.25,
        refresh_mask=tuple([True] * num_steps))
    for step in range(3):  # forced-calc prefix of 3
        should_calc, raw = teacache_decide(state, _modulated(step))
        assert should_calc is True
        assert raw == 0.0
        assert state["accumulated"] == 0.0
        teacache_step(state)
    assert state["decisions"][:3] == ["calc", "calc", "calc"]
    # previous_modulated_input stays consistent (set to the last prefix value).
    assert state["previous_modulated_input"] is not None


def test_dropping_mask_resumes_dynamic_from_step_k():
    num_steps = 6
    state = teacache_init(
        num_steps, rel_l1_thresh=0.25,
        refresh_mask=tuple([True] * num_steps))
    # Run a 3-step forced-calc prefix.
    for step in range(3):
        teacache_decide(state, _modulated(step))
        teacache_step(state)
    # Commit mid-trajectory: drop the mask, set the chosen arm's gamma.
    state["refresh_mask"] = None
    state["rel_l1_thresh"] = 0.35
    # Suffix: dynamic branch resumes from previous_modulated_input (step 2).
    for step in range(3, num_steps):
        teacache_decide(state, _modulated(100 + step))  # distinct inputs
        teacache_step(state)
    assert len(state["decisions"]) == num_steps
    # Suffix decisions are valid calc/skip strings, driven by the dynamic path.
    assert all(d in ("calc", "skip") for d in state["decisions"][3:])
    # The last step is always forced calc (cnt == num_steps - 1).
    assert state["decisions"][-1] == "calc"
    # previous_modulated_input tracks the final suffix input, not a stale prefix.
    assert torch.allclose(
        state["previous_modulated_input"].squeeze(0),
        _modulated(100 + (num_steps - 1)).squeeze(0))


def test_decisions_count_equals_num_steps_after_full_run():
    num_steps = 8
    state = teacache_init(num_steps, refresh_mask=tuple([True] * num_steps))
    state["refresh_mask"] = None  # commit at step 0 -> immediate dynamic
    for step in range(num_steps):
        teacache_decide(state, _modulated(step))
        teacache_step(state)
    assert len(state["decisions"]) == num_steps
    # First (cnt==0) and last (cnt==num_steps-1) steps are always forced calc.
    assert state["decisions"][0] == "calc"
    assert state["decisions"][-1] == "calc"


def test_gamma_leaks_across_reset_documented_hazard():
    num_steps = 6
    state = teacache_init(
        num_steps, rel_l1_thresh=0.25,
        refresh_mask=tuple([True] * num_steps))
    # Simulate a commit that mutates gamma to an aggressive arm.
    state["refresh_mask"] = None
    state["rel_l1_thresh"] = 0.60
    teacache_reset(state)
    # HAZARD: reset does NOT restore rel_l1_thresh (it is "config", kept), so a
    # mutated gamma would leak into the next trajectory. The runner must restore
    # the baseline gamma explicitly after teacache_reset.
    assert state["rel_l1_thresh"] == 0.60
    state["rel_l1_thresh"] = 0.25  # the documented runner-level fix
    assert state["rel_l1_thresh"] == 0.25
    # Runtime state IS cleared by reset.
    assert state["cnt"] == 0
    assert state["accumulated"] == 0.0
    assert state["previous_modulated_input"] is None
    assert state["decisions"] == []


def test_mask_length_must_match_num_steps():
    with pytest.raises(ValueError, match="length must match"):
        teacache_init(6, refresh_mask=(True, True, True))  # len 3 != 6


def test_mask_must_refresh_first_step():
    with pytest.raises(ValueError, match="refresh the first step"):
        teacache_init(4, refresh_mask=(False, True, True, True))


def test_mask_must_contain_booleans():
    with pytest.raises(ValueError, match="booleans"):
        # 1 is an int, not a bool -> rejected (prevents silent truthiness bugs).
        teacache_init(4, refresh_mask=(True, 1, True, True))


def test_dynamic_branch_does_not_read_mask_once_dropped():
    # After commit, refresh_mask is None; the dynamic branch must not index it.
    num_steps = 4
    state = teacache_init(num_steps, refresh_mask=tuple([True] * num_steps))
    teacache_decide(state, _modulated(0))
    teacache_step(state)
    state["refresh_mask"] = None  # commit
    # A short dynamic suffix must not raise (no mask indexing, no KeyError).
    for step in range(1, num_steps):
        should_calc, _ = teacache_decide(state, _modulated(step))
        assert isinstance(should_calc, bool)
        teacache_step(state)
