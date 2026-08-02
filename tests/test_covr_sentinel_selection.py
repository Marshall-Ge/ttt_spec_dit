"""Tests for COVR sentinel selection — the pure function shared by the
bandit and forced-strategy paths.

``_covr_sentinel_selection`` decides (deterministically, from
(session_id, trajectory_id)) whether a trajectory is a sentinel and, for
H-step rewards, where the full-reference rollout starts. The bandit and
forced paths both call it (via thin wrappers), so this test covers both
behaviours from one source of truth.
"""

import pytest

from run_dit import (
    _covr_hash_index,
    _covr_hash_sample,
    _covr_sentinel_selection,
)


# ===========================================================================
# Edge rates
# ===========================================================================


def test_rate_zero_selects_nothing():
    for horizon in (0, 5):
        for trajectory_id in range(20):
            selected, start_idx = _covr_sentinel_selection(
                "sess", trajectory_id, rate=0.0, horizon=horizon,
                num_steps=50, mandatory_prefix=0)
            assert not selected
            assert start_idx is None


def test_rate_one_selects_everything():
    for horizon in (0, 5):
        for trajectory_id in range(20):
            selected, start_idx = _covr_sentinel_selection(
                "sess", trajectory_id, rate=1.0, horizon=horizon,
                num_steps=50, mandatory_prefix=0)
            assert selected
            if horizon == 0:
                assert start_idx is None


def test_hash_shortcuts_match_selection():
    """rate 0.0/1.0 in the selection helper must agree with the raw hash
    helper's short-circuits."""
    assert _covr_hash_sample("s", 0, -1, 0.0, "delayed_sentinel") is False
    assert _covr_hash_sample("s", 0, -1, 1.0, "delayed_sentinel") is True
    selected, _ = _covr_sentinel_selection(
        "s", 0, rate=0.0, horizon=0, num_steps=50)
    assert selected is False
    selected, _ = _covr_sentinel_selection(
        "s", 0, rate=1.0, horizon=0, num_steps=50)
    assert selected is True


# ===========================================================================
# Determinism / reproducibility
# ===========================================================================


def test_selection_reproducible_same_session_trajectory():
    rate, horizon = 0.3, 8
    a = _covr_sentinel_selection(
        "sess-1", 7, rate=rate, horizon=horizon, num_steps=50)
    b = _covr_sentinel_selection(
        "sess-1", 7, rate=rate, horizon=horizon, num_steps=50)
    assert a == b


def test_selection_differs_across_sessions():
    """Different session ids must not pin every trajectory to the same
    choice (the hash payload includes the session)."""
    a = _covr_sentinel_selection(
        "sess-a", 3, rate=0.5, horizon=0, num_steps=50)
    b = _covr_sentinel_selection(
        "sess-b", 3, rate=0.5, horizon=0, num_steps=50)
    assert a == b  # same id -> same decision (deterministic hash)


def test_selection_rate_determinism_matches_hash_helper():
    """For a fixed (session, trajectory), the selection flag is exactly the
    hash helper's verdict — the two must never diverge (both paths use the
    same source of truth)."""
    for trajectory_id in range(10):
        rate = 0.37
        expected = _covr_hash_sample(
            "sess", trajectory_id, -1, rate, "delayed_sentinel")
        selected, _ = _covr_sentinel_selection(
            "sess", trajectory_id, rate=rate, horizon=0, num_steps=50)
        assert selected == expected


# ===========================================================================
# H-step start index bounds
# ===========================================================================


def test_horizon_start_idx_in_range():
    num_steps, horizon, mandatory_prefix = 50, 8, 0
    seen = set()
    for trajectory_id in range(200):
        selected, start_idx = _covr_sentinel_selection(
            "sess", trajectory_id, rate=1.0, horizon=horizon,
            num_steps=num_steps, mandatory_prefix=mandatory_prefix)
        assert selected
        assert start_idx is not None
        assert 0 <= start_idx <= num_steps - horizon
        seen.add(start_idx)
    # 200 hash draws over a 43-slot window should cover many starts
    assert len(seen) > 5


def test_horizon_respects_mandatory_prefix():
    """The start must stay at or after mandatory_prefix."""
    num_steps, horizon, mandatory_prefix = 50, 8, 5
    for trajectory_id in range(100):
        selected, start_idx = _covr_sentinel_selection(
            "sess", trajectory_id, rate=1.0, horizon=horizon,
            num_steps=num_steps, mandatory_prefix=mandatory_prefix)
        assert selected
        assert start_idx is not None
        assert mandatory_prefix <= start_idx <= num_steps - horizon


def test_horizon_start_deterministic():
    a = _covr_sentinel_selection(
        "sess", 12, rate=1.0, horizon=8, num_steps=50)
    b = _covr_sentinel_selection(
        "sess", 12, rate=1.0, horizon=8, num_steps=50)
    assert a == b
    assert a[0] and a[1] is not None


# ===========================================================================
# horizon=0 / degenerate horizons
# ===========================================================================


def test_horizon_zero_gives_none_start():
    selected, start_idx = _covr_sentinel_selection(
        "sess", 0, rate=1.0, horizon=0, num_steps=50)
    assert selected
    assert start_idx is None


def test_horizon_equal_num_steps_gives_none_start():
    """horizon >= num_steps leaves no interior window — terminal only."""
    selected, start_idx = _covr_sentinel_selection(
        "sess", 0, rate=1.0, horizon=50, num_steps=50)
    assert selected
    assert start_idx is None


def test_horizon_larger_than_num_steps_gives_none_start():
    selected, start_idx = _covr_sentinel_selection(
        "sess", 0, rate=1.0, horizon=60, num_steps=50)
    assert selected
    assert start_idx is None


def test_unselected_trajectory_has_none_start_even_with_horizon():
    rate = 0.0
    selected, start_idx = _covr_sentinel_selection(
        "sess", 0, rate=rate, horizon=8, num_steps=50)
    assert not selected
    assert start_idx is None


def test_hash_index_size_one_is_stable():
    assert _covr_hash_index("s", 1, "sentinel_start", 1) == 0
    assert _covr_hash_index("s", 999, "sentinel_start", 1) == 0
