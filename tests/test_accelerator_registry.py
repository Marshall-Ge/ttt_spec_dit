"""Tests for the accelerator adapter registry — COVR's pluggable seam.

These verify the *contract* the registry promises: a new accelerator can
register an ``AcceleratorAdapter`` and become a valid strategy method,
dispatchable through ``apply_strategy``, without editing ``covr_bandit`` or
``strategy_dispatch``.
"""

import pytest

from accelerators.covr_bandit import AccelerationStrategy
from accelerators.registry import (
    AcceleratorAdapter,
    SpecAAdapter,
    TeaCacheAdapter,
    get_adapter,
    is_registered,
    register_adapter,
    registered_methods,
    _REGISTRY,
)
from accelerators.strategy_dispatch import apply_strategy


# ===========================================================================
# Built-in adapters
# ===========================================================================


def test_builtins_registered():
    assert is_registered("speca")
    assert is_registered("teacache")
    assert "speca" in registered_methods()
    assert "teacache" in registered_methods()


def test_get_adapter_returns_right_type():
    assert isinstance(get_adapter("speca"), SpecAAdapter)
    assert isinstance(get_adapter("teacache"), TeaCacheAdapter)


def test_get_adapter_unknown_method_raises():
    with pytest.raises(ValueError, match="unsupported acceleration method"):
        get_adapter("does-not-exist")


def test_adapter_state_keys_match_dispatch():
    assert get_adapter("speca").state_keys == ("cache_dic", "current")
    assert get_adapter("teacache").state_keys == ("teacache_state",)


# ===========================================================================
# The whitelist is now registry-driven
# ===========================================================================


def test_strategy_rejects_unregistered_method():
    """AccelerationStrategy validity now flows through the registry."""
    with pytest.raises(ValueError, match="unsupported acceleration method"):
        AccelerationStrategy(
            strategy_id="x", method="nonexistent",
            params={}, modeled_flops=1.0)


# ===========================================================================
# Pluggability: register a brand-new accelerator, use it end-to-end
# ===========================================================================


class _FakeAdapter(AcceleratorAdapter):
    """A throwaway accelerator that exists only for this test."""

    method = "fakeaccel"
    init_kwargs_key = "fakeaccel_init_kwargs"
    state_keys = ("fake_state",)

    def init_state(self, strategy, base_kwargs=None):
        base = dict(base_kwargs or {})
        return {"fake_state": {"id": strategy.strategy_id, **base}}

    def terminal_reward_active(self, states):
        return states.get("fake_state") is not None

    def add_flops(self, flops_metric, states, *, num_layers):
        flops_metric.calls.append(("fake", num_layers))


@pytest.fixture
def fake_adapter_registered():
    """Register _FakeAdapter for the duration of a test, then clean up."""
    register_adapter(_FakeAdapter())
    try:
        yield
    finally:
        _REGISTRY.pop("fakeaccel", None)


def test_new_accelerator_becomes_valid_strategy(fake_adapter_registered):
    # Before registration this method would raise; now it is accepted with
    # NO change to covr_bandit.py.
    strategy = AccelerationStrategy(
        strategy_id="arm0", method="fakeaccel",
        params={"foo": 1}, modeled_flops=3.0)
    assert strategy.method == "fakeaccel"


def test_apply_strategy_dispatches_new_accelerator(fake_adapter_registered):
    strategy = AccelerationStrategy(
        strategy_id="arm0", method="fakeaccel",
        params={}, modeled_flops=3.0)
    result = apply_strategy(
        strategy, fakeaccel_init_kwargs={"lr": 0.5})
    # apply_strategy forwarded the method's own kwargs bag and returned the
    # adapter's state_keys — all without a method branch in the dispatcher.
    assert result["fake_state"] == {"id": "arm0", "lr": 0.5}


def test_apply_strategy_teacache_threshold_still_works():
    """Regression: the built-in TeaCache threshold path is unchanged."""
    strategy = AccelerationStrategy(
        strategy_id="thresh_0.3", method="teacache",
        params={"rel_l1_thresh": 0.3}, modeled_flops=1.0)
    result = apply_strategy(
        strategy,
        teacache_init_kwargs={
            "num_steps": 5, "coefficients": [0.0, 0.0, 0.0, 1.0, 0.0]})
    assert result["teacache_state"]["rel_l1_thresh"] == 0.3
    assert result["teacache_state"]["refresh_mask"] is None


def test_apply_strategy_rejects_ambiguous_teacache_arm():
    strategy = AccelerationStrategy(
        strategy_id="ambiguous", method="teacache",
        params={
            "rel_l1_thresh": 0.3,
            "refresh_mask": [True, False, True, False, True],
        },
        modeled_flops=3.0,
    )

    with pytest.raises(ValueError, match="cannot combine"):
        apply_strategy(
            strategy,
            teacache_init_kwargs={
                "num_steps": 5,
                "coefficients": [0.0, 0.0, 0.0, 1.0, 0.0],
            },
        )


def test_register_adapter_rejects_empty_method():
    class _Bad(AcceleratorAdapter):
        method = ""
        init_kwargs_key = "x"
        state_keys = ()

        def init_state(self, strategy, base_kwargs=None):
            return {}

        def terminal_reward_active(self, states):
            return False

        def add_flops(self, flops_metric, states, *, num_layers):
            pass

    with pytest.raises(ValueError, match="method must be a non-empty"):
        register_adapter(_Bad())
