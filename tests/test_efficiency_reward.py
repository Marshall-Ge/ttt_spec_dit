"""Efficiency-aware reward: combined = terminal_fidelity + lambda * cost.

The contextual bandit optimizes ``combined_loss`` (terminal fidelity plus a
measured-FLOPs cost) while the analyzer still evaluates quality on terminal
fidelity. These tests pin the reward semantics at all three layers: the
``TemplateFeedback`` schema (bandit_loss preference + v1 compatibility), the
all-None guard (efficiency labels do not satisfy the delayed-sentinel guard),
and the runtime's ``_efficiency_labels`` combination formula.
"""

from types import SimpleNamespace

import pytest

from accelerators.covr_bandit import TemplateFeedback
from accelerators.covr_runtime import COVRRuntime, build_covr_runtime_config


def _runtime(tmp_path, *, efficiency_lambda=1e-3):
    args = SimpleNamespace(
        model="dit", method="teacache", ttt=False, batch_size=1,
        covr_strategy_bandit=True,
        covr_strategy_manifest=str(tmp_path / "m.json"),
        covr_contextual_bandit=True, covr_prefix_steps=2,
        covr_efficiency_lambda=efficiency_lambda, covr_linucb_alpha=1.0,
        covr_bandit_epsilon=0.1, covr_bandit_prior_penalty=0.0,
        covr_safety_sample_rate=0.0, covr_safety_chain_threshold=0,
        covr_sentinel_rate=0.0, covr_sentinel_horizon=0,
        covr_session_id="sess", covr_base_model_version="m",
        covr_profile_stages=False, covr_shadow=False, seed=42, num_steps=4,
    )
    config = build_covr_runtime_config(args, str(tmp_path))
    return COVRRuntime.create(config, scheduler=None, num_steps=4)


def test_bandit_loss_prefers_combined_loss():
    fb = TemplateFeedback(
        trajectory_id=0, template_id="baseline",
        sentinel_propensity=1.0, horizon=4,
        terminal_fidelity_loss=0.2, terminal_efficiency_loss=0.5,
        combined_loss=0.2005)
    assert fb.bandit_loss == pytest.approx(0.2005)


def test_bandit_loss_falls_back_to_fidelity_without_combined():
    fb = TemplateFeedback(
        trajectory_id=0, template_id="baseline",
        sentinel_propensity=1.0, horizon=4,
        terminal_fidelity_loss=0.2, terminal_efficiency_loss=0.5)
    assert fb.bandit_loss == pytest.approx(0.2)


def test_efficiency_loss_not_in_all_none_guard():
    # terminal_efficiency_loss alone does NOT satisfy the delayed-label guard.
    with pytest.raises(ValueError, match="delayed sentinel label"):
        TemplateFeedback(
            trajectory_id=0, template_id="baseline",
            sentinel_propensity=1.0, horizon=4,
            terminal_efficiency_loss=0.5)
    # But it co-exists with a fidelity label without error.
    fb = TemplateFeedback(
        trajectory_id=0, template_id="baseline",
        sentinel_propensity=1.0, horizon=4,
        terminal_fidelity_loss=0.2, terminal_efficiency_loss=0.5)
    assert fb.terminal_efficiency_loss == 0.5


def test_efficiency_loss_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        TemplateFeedback(
            trajectory_id=0, template_id="baseline",
            sentinel_propensity=1.0, horizon=4,
            terminal_fidelity_loss=0.2, terminal_efficiency_loss=-0.1)


def test_v1_feedback_dict_still_constructs():
    # A legacy (schema v1) feedback dict has neither efficiency field.
    legacy = {
        "trajectory_id": 0,
        "template_id": "baseline",
        "sentinel_propensity": 1.0,
        "horizon": 4,
        "terminal_fidelity_loss": 0.2,
    }
    fb = TemplateFeedback(**legacy)
    assert fb.terminal_efficiency_loss is None
    assert fb.combined_loss is None
    assert fb.bandit_loss == pytest.approx(0.2)


def test_efficiency_labels_combine_fidelity_and_cost(tmp_path):
    runtime = _runtime(tmp_path, efficiency_lambda=1e-3)
    cost, combined = runtime._efficiency_labels(0.2, {"measured_cost_ratio": 0.5})
    assert cost == pytest.approx(0.5)
    assert combined == pytest.approx(0.2 + 1e-3 * 0.5)


def test_efficiency_labels_lambda_zero_is_pure_fidelity(tmp_path):
    runtime = _runtime(tmp_path, efficiency_lambda=0.0)
    cost, combined = runtime._efficiency_labels(0.2, {"measured_cost_ratio": 0.5})
    assert cost == pytest.approx(0.5)
    assert combined == pytest.approx(0.2)


def test_efficiency_labels_none_without_cost(tmp_path):
    runtime = _runtime(tmp_path, efficiency_lambda=1e-3)
    cost, combined = runtime._efficiency_labels(0.2, {})
    assert cost is None
    assert combined is None


def test_efficiency_labels_rejects_bad_cost(tmp_path):
    runtime = _runtime(tmp_path, efficiency_lambda=1e-3)
    with pytest.raises(ValueError, match="non-negative"):
        runtime._efficiency_labels(0.2, {"measured_cost_ratio": -0.1})
