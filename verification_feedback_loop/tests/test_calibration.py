# -*- coding: utf-8 -*-
"""Tests for M2: OnlineCalibrator — EMA threshold, version swap."""

import math
import torch

from verification_feedback_loop.online_calibration import (
    OnlineCalibrator,
    _EMAThreshold,
)
from verification_feedback_loop.verification_hook import (
    VerificationEvent,
    make_timestep_bucket,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_event(layer_id=5, timestep_bucket=1, error_value=0.05,
                decision="reject"):
    """Create a minimal VerificationEvent for calibration testing."""
    return VerificationEvent(
        layer_id=layer_id,
        timestep=500,
        timestep_bucket=timestep_bucket,
        predicted_feature=torch.randn(1, 256, 1152),
        true_feature=torch.randn(1, 256, 1152),
        error_value=error_value,
        decision=decision,
        model="dit",
        base_model_version="v1.0",
        step_idx=10,
    )


# ===========================================================================
# EMA threshold tests
# ===========================================================================


class TestEMAThreshold:
    def test_initial(self):
        ema = _EMAThreshold(window=500, k=2.0)
        assert not ema.is_ready()
        thresh = ema.get_threshold(min_floor=0.001)
        assert thresh >= 0.001

    def test_convergence_to_distribution(self):
        """EMA should estimate mean+std for a known distribution."""
        ema = _EMAThreshold(window=50, k=2.0)  # fast EMA

        # Feed values from N(0.1, 0.05)
        import random
        random.seed(42)
        for _ in range(200):
            val = 0.1 + 0.05 * random.gauss(0, 1)
            ema.update(max(0.0, val))

        assert ema.is_ready()
        # Mean should be close to 0.1
        assert 0.05 < ema.mean < 0.2, f"mean={ema.mean:.4f}"
        # Threshold should be > mean
        thresh = ema.get_threshold()
        assert thresh > ema.mean, f"threshold={thresh:.4f} <= mean={ema.mean:.4f}"

    def test_k_affects_threshold(self):
        """Higher k should produce higher threshold."""
        ema_low = _EMAThreshold(window=100, k=1.0)
        ema_high = _EMAThreshold(window=100, k=3.0)

        import random
        random.seed(42)
        for _ in range(100):
            val = 0.1 + 0.05 * random.gauss(0, 1)
            ema_low.update(max(0.0, val))
            ema_high.update(max(0.0, val))

        assert ema_high.get_threshold() > ema_low.get_threshold()

    def test_window_change(self):
        ema = _EMAThreshold(window=500, k=2.0)
        assert ema.alpha > 0.99  # long window → slow
        ema.set_window(50)
        assert ema.alpha < 0.99  # short window → fast


# ===========================================================================
# OnlineCalibrator integration tests
# ===========================================================================


class TestOnlineCalibrator:
    def test_update_creates_strata(self):
        cal = OnlineCalibrator()
        event = _make_event(layer_id=5, timestep_bucket=1, error_value=0.05)
        cal.update(event)
        assert cal.total_updates == 1
        assert cal.num_strata == 1

    def test_get_threshold_default(self):
        cal = OnlineCalibrator()
        # Before any updates, should return default
        thresh = cal.get_threshold(5, 1, default=0.25)
        assert thresh == 0.25

    def test_get_threshold_after_updates(self):
        cal = OnlineCalibrator(ema_window=50, threshold_k=2.0)

        # Feed events with varying errors
        for i in range(100):
            error = 0.05 + 0.02 * (0.5 - (i % 20) / 20)  # mean ~0.05
            event = _make_event(layer_id=5, timestep_bucket=1, error_value=error)
            cal.update(event)

        thresh = cal.get_threshold(5, 1, default=0.25)
        # Should no longer be the default
        assert thresh != 0.25
        # Should be positive and reasonable
        assert 0.001 <= thresh <= 1.0, f"threshold={thresh}"

    def test_multi_stratum_isolation(self):
        cal = OnlineCalibrator()

        # Stratum A: high error
        for _ in range(50):
            cal.update(_make_event(layer_id=5, timestep_bucket=0, error_value=0.5))

        # Stratum B: low error
        for _ in range(50):
            cal.update(_make_event(layer_id=5, timestep_bucket=1, error_value=0.01))

        # Thresholds should differ across strata
        t_a = cal.get_threshold(5, 0, default=0.25)
        t_b = cal.get_threshold(5, 1, default=0.25)
        assert t_a > t_b, f"High-error stratum should have higher threshold: {t_a} <= {t_b}"

    def test_on_base_model_swap(self):
        cal = OnlineCalibrator(ema_window_long=500, ema_window_short=50)

        # Feed some data first
        for _ in range(30):
            cal.update(_make_event(layer_id=0, timestep_bucket=0, error_value=0.05))

        # Swap should shorten EMA windows
        cal.on_base_model_swap("v2.0")
        for ema in cal._ema.values():
            assert ema.window == 50

    def test_on_converged_restores(self):
        cal = OnlineCalibrator(ema_window_long=500, ema_window_short=50)
        cal.on_base_model_swap("v2.0")
        cal.on_converged()
        for ema in cal._ema.values():
            assert ema.window == 500

    def test_get_stats(self):
        cal = OnlineCalibrator()
        for _ in range(20):
            cal.update(_make_event(layer_id=5, timestep_bucket=1, error_value=0.05))

        stats = cal.get_stats(5, 1)
        assert stats["layer_id"] == 5
        assert stats["timestep_bucket"] == 1
        assert stats["ema_ready"]
        assert "threshold" in stats
        assert stats["threshold"] > 0.0

    def test_empty_stratum(self):
        cal = OnlineCalibrator()
        # Never-seen stratum should return default
        thresh = cal.get_threshold(99, 2, default=0.30)
        assert thresh == 0.30
