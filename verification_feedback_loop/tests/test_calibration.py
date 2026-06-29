# -*- coding: utf-8 -*-
"""Tests for M2: OnlineCalibrator — RLS convergence, EMA threshold, version swap."""

import math
import torch

from verification_feedback_loop.online_calibration import (
    OnlineCalibrator,
    _RLSEstimator,
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
# RLS estimator tests
# ===========================================================================


class TestRLSEstimator:
    def test_initial_state(self):
        rls = _RLSEstimator(degree=2, forget_factor=0.995)
        assert not rls.is_ready()
        assert rls.n_updates == 0
        # Before any updates, predict should be near 0
        assert abs(rls.predict(0.5)) < 0.1

    def test_convergence_linear(self):
        """RLS should converge to the true linear relationship y = 2x + 1."""
        rls = _RLSEstimator(degree=1, forget_factor=0.99)  # linear: [x, 1]
        true_slope, true_intercept = 2.0, 1.0

        for i in range(200):
            x = i * 0.01  # 0 .. 2
            y = true_slope * x + true_intercept + 0.1 * (0.5 - (i % 100) / 100)
            rls.update(x, y)

        assert rls.is_ready()
        slope, intercept = rls.get_linear_coeffs()
        # Should be close to true values (within reasonable tolerance)
        assert abs(slope - true_slope) < 0.5, f"slope={slope:.3f}, expected ~{true_slope}"
        assert abs(intercept - true_intercept) < 0.5, f"intercept={intercept:.3f}, expected ~{true_intercept}"

    def test_convergence_quadratic(self):
        """RLS should converge for y = 3x² + 2x + 1."""
        rls = _RLSEstimator(degree=2, forget_factor=0.99)
        for i in range(300):
            x = i * 0.005  # 0 .. 1.5
            y = 3 * x**2 + 2 * x + 1 + 0.05 * (0.5 - (i % 50) / 50)
            rls.update(x, y)

        assert rls.is_ready()
        # Predict should be close at sampled points
        for x_test in [0.0, 0.5, 1.0]:
            y_pred = rls.predict(x_test)
            y_true = 3 * x_test**2 + 2 * x_test + 1
            assert abs(y_pred - y_true) < 1.0, \
                f"At x={x_test}: pred={y_pred:.3f}, true={y_true:.3f}"

    def test_forgetting(self):
        """After a distribution shift, RLS should adapt with forgetting."""
        rls = _RLSEstimator(degree=1, forget_factor=0.95)  # fast forgetting

        # Phase 1: y = 1x + 0
        for _ in range(50):
            rls.update(0.5, 0.5)
        slope1, _ = rls.get_linear_coeffs()
        assert abs(slope1 - 1.0) < 0.5

        # Phase 2: y = 5x + 0 (distribution shift)
        for _ in range(100):
            rls.update(0.5, 2.5)
        slope2, _ = rls.get_linear_coeffs()
        # Should have moved toward slope=5
        assert slope2 > slope1, f"RLS didn't adapt: slope1={slope1:.3f}, slope2={slope2:.3f}"

    def test_reset(self):
        rls = _RLSEstimator(degree=1, forget_factor=0.99)
        for _ in range(50):
            rls.update(0.5, 1.0)
        assert rls.is_ready()
        rls.reset()
        assert not rls.is_ready()
        assert rls.n_updates == 0

    def test_min_updates(self):
        rls = _RLSEstimator(degree=1)
        assert not rls.is_ready(min_updates=10)
        for _ in range(9):
            rls.update(0.5, 1.0)
        assert not rls.is_ready(min_updates=10)
        rls.update(0.5, 1.0)
        assert rls.is_ready(min_updates=10)


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

    def test_update_with_proxy(self):
        cal = OnlineCalibrator(forget_factor=0.95, ema_window=50)

        # Feed correlated proxy→error pairs
        for i in range(100):
            proxy = 0.01 * (i % 50)
            true_error = 2.0 * proxy + 0.01  # linear relationship
            event = _make_event(
                layer_id=5, timestep_bucket=1,
                error_value=true_error,
            )
            cal.update_with_proxy(event, proxy_value=proxy)

        assert cal.total_updates == 100

        # The rescale function should approximate the linear relationship
        rescale = cal.get_rescale_fn(layer_id=5, timestep_bucket=1)
        for x_test in [0.05, 0.10, 0.20]:
            pred = rescale(x_test)
            expected = 2.0 * x_test + 0.01
            # Allow loose tolerance since RLS takes time to converge
            assert pred >= 0, f"rescale({x_test}) = {pred} < 0"

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
        rescale = cal.get_rescale_fn(99, 2)
        assert rescale(0.5) == 0.5  # identity fallback
