# -*- coding: utf-8 -*-
"""Tests for M3: StratifiedReplayBuffer — stratification, capacity, sampling."""

import pytest
import torch

from verification_feedback_loop.replay_buffer import (
    StratifiedReplayBuffer,
    AnchorSample,
    _Stratum,
)
from verification_feedback_loop.verification_hook import (
    VerificationEvent,
    make_timestep_bucket,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_event(layer_id=5, timestep_bucket=1, decision="reject"):
    """Create a minimal VerificationEvent for testing."""
    return VerificationEvent(
        layer_id=layer_id,
        timestep=500,
        timestep_bucket=timestep_bucket,
        predicted_feature=torch.randn(1, 256, 1152),
        true_feature=torch.randn(1, 256, 1152),
        error_value=0.05,
        decision=decision,
        model="dit",
        base_model_version="v1.0",
        step_idx=10,
    )


def _make_anchor():
    """Create a minimal AnchorSample for testing."""
    return AnchorSample(
        prompt=207,  # golden retriever
        latent=torch.randn(1, 4, 32, 32),
        timestep=torch.tensor([500]),
        target=torch.randn(1, 4, 32, 32),
        model="dit",
        base_model_version="v1.0",
    )


# ===========================================================================
# _Stratum tests
# ===========================================================================


class TestStratum:
    def test_add_and_size(self):
        s = _Stratum(capacity=10)
        for i in range(5):
            s.add(f"event_{i}", "hard_negative")
        assert s.size == 5

    def test_reservoir_overflow(self):
        s = _Stratum(capacity=10)
        for i in range(100):
            s.add(f"event_{i}", "hard_negative")
        assert s.size == 10  # capped at capacity
        assert s._wraps > 0   # wrapped at least once

    def test_sample_by_kind(self):
        s = _Stratum(capacity=100)
        for i in range(50):
            s.add(f"hn_{i}", "hard_negative")
        for i in range(30):
            s.add(f"norm_{i}", "normal")

        samples = s.sample(10, {"hard_negative": 0.7, "normal": 0.3, "anchor": 0.0})
        assert len(samples) <= 10

    def test_empty_sample(self):
        s = _Stratum(capacity=10)
        samples = s.sample(5, {"hard_negative": 0.5, "normal": 0.5, "anchor": 0.0})
        assert samples == []

    def test_stats(self):
        s = _Stratum(capacity=100)
        s.add("e1", "hard_negative")
        s.add("e2", "hard_negative")
        s.add("e3", "normal")
        st = s.stats()
        assert st["size"] == 3
        assert st["by_kind"]["hard_negative"] == 2
        assert st["by_kind"]["normal"] == 1


# ===========================================================================
# StratifiedReplayBuffer tests
# ===========================================================================


class TestStratifiedReplayBuffer:
    def test_add_creates_strata(self):
        buf = StratifiedReplayBuffer(capacity_per_stratum=50)
        event = _make_event(layer_id=5, timestep_bucket=1)
        buf.add(event, kind="hard_negative")
        assert buf.total_samples == 1

    def test_multi_stratum(self):
        buf = StratifiedReplayBuffer(capacity_per_stratum=50)
        for layer in [0, 5, 10]:
            for bucket in [0, 1, 2]:
                for _ in range(5):
                    event = _make_event(layer_id=layer, timestep_bucket=bucket)
                    buf.add(event, kind="hard_negative")
        # 3 layers × 3 buckets × 5 events = 45
        assert buf.total_samples == 45

    def test_capacity_per_stratum(self):
        buf = StratifiedReplayBuffer(capacity_per_stratum=10)
        for i in range(200):
            event = _make_event(layer_id=0, timestep_bucket=0)
            buf.add(event, kind="hard_negative")
        # Should be capped at 10 per stratum
        assert buf.total_samples == 10

    def test_add_anchor(self):
        buf = StratifiedReplayBuffer(capacity_per_stratum=50)
        for _ in range(5):
            buf.add_anchor(_make_anchor())
        assert len(buf._anchor_samples) == 5

    def test_anchor_capacity_protection(self):
        buf = StratifiedReplayBuffer(capacity_per_stratum=50)
        for i in range(600):
            buf.add_anchor(_make_anchor())
        assert len(buf._anchor_samples) <= 500

    def test_sample_training_batch(self):
        buf = StratifiedReplayBuffer(capacity_per_stratum=100)

        # Populate with hard_negative events across strata
        for layer in range(28):
            for bucket in range(3):
                for i in range(5):
                    event = _make_event(layer_id=layer, timestep_bucket=bucket,
                                        decision="reject")
                    buf.add(event, kind="hard_negative")

        # Add some normal events
        for layer in [0, 5, 10]:
            for bucket in range(3):
                event = _make_event(layer_id=layer, timestep_bucket=bucket,
                                    decision="accept")
                buf.add(event, kind="normal")

        # Add anchors
        for _ in range(20):
            buf.add_anchor(_make_anchor())

        events, anchors = buf.sample_training_batch(32)

        # Should return some events and some anchors
        assert len(events) + len(anchors) > 0
        assert len(anchors) <= 32

    def test_stats(self):
        buf = StratifiedReplayBuffer(capacity_per_stratum=10)
        for layer in [0, 1]:
            for bucket in [0, 1]:
                event = _make_event(layer_id=layer, timestep_bucket=bucket)
                buf.add(event, kind="hard_negative")

        st = buf.stats()
        assert st["num_strata"] == 4
        assert st["num_strata_nonempty"] == 4
        assert st["total_samples"] == 4
        assert "per_layer_hard_negative" in st


# ===========================================================================
# make_timestep_bucket tests
# ===========================================================================


class TestTimestepBucket:
    def test_early(self):
        assert make_timestep_bucket(0, 50) == 0
        assert make_timestep_bucket(5, 50) == 0
        assert make_timestep_bucket(16, 50) == 0  # 16 < 50/3

    def test_mid(self):
        assert make_timestep_bucket(17, 50) == 1
        assert make_timestep_bucket(25, 50) == 1
        assert make_timestep_bucket(33, 50) == 1  # 33 < 2*50/3

    def test_late(self):
        assert make_timestep_bucket(34, 50) == 2
        assert make_timestep_bucket(49, 50) == 2

    def test_boundary(self):
        assert make_timestep_bucket(0, 3) == 0
        assert make_timestep_bucket(1, 3) == 1
        assert make_timestep_bucket(2, 3) == 2
