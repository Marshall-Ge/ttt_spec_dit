"""Unit tests for the deferred-commit ContextualLinUCBBandit.

Covers: pending begin, commit-time arm selection, distinct contexts choosing
distinct arms, theta updates, schema-v2 persistence round-trip, and the
identity gate that rejects cross-version (v1 <-> v2) resume.
"""

import json

import numpy as np
import pytest

from accelerators.covr_bandit import (
    BANDIT_SCHEMA_VERSION,
    CONTEXTUAL_BANDIT_SCHEMA_VERSION,
    AccelerationStrategy,
    ConservativeTemplateBandit,
    ContextualLinUCBBandit,
    StrategyManifest,
    TemplateFeedback,
)


def _manifest() -> StrategyManifest:
    return StrategyManifest(
        version_key="vk1",
        num_steps=50,
        baseline_strategy_id="baseline",
        strategies=(
            AccelerationStrategy(
                "baseline", "teacache",
                {"rel_l1_thresh": 0.25, "num_steps": 50}, 50.0),
            AccelerationStrategy(
                "thresh_0p15", "teacache",
                {"rel_l1_thresh": 0.15, "num_steps": 50}, 50.0),
            AccelerationStrategy(
                "thresh_0p35", "teacache",
                {"rel_l1_thresh": 0.35, "num_steps": 50}, 50.0),
        ),
    )


def _feedback(trajectory_id: int, arm: str, loss: float = 0.3) -> TemplateFeedback:
    return TemplateFeedback(
        trajectory_id=trajectory_id,
        template_id=arm,
        sentinel_propensity=1.0,
        horizon=50,
        terminal_fidelity_loss=loss,
    )


def test_schema_version_is_two():
    assert CONTEXTUAL_BANDIT_SCHEMA_VERSION == 2
    assert BANDIT_SCHEMA_VERSION == 1
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0, alpha=0.0)
    assert bandit._schema_version == 2
    assert bandit.is_contextual is True


def test_begin_returns_pending_assignment_with_no_arm():
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0)
    pending = bandit.begin_trajectory(0, sample_count=1)
    assert pending.template_id is None
    # The pending assignment is NOT recorded; commit_arm records the real one.
    assert bandit.assignments == []


def test_end_before_commit_raises():
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2)
    bandit.begin_trajectory(0)
    with pytest.raises(RuntimeError, match="before commit_arm"):
        bandit.end_trajectory(0)


def test_commit_wrong_trajectory_raises():
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2)
    bandit.begin_trajectory(0)
    with pytest.raises(RuntimeError, match="active trajectory"):
        bandit.commit_arm(1, [0.0, 0.0])


def test_commit_twice_raises():
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0)
    bandit.begin_trajectory(0)
    bandit.commit_arm(0, [0.0, 0.0])
    with pytest.raises(RuntimeError, match="already been committed"):
        bandit.commit_arm(0, [0.0, 0.0])


def test_commit_bad_context_dim_raises():
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2)
    bandit.begin_trajectory(0)
    with pytest.raises(ValueError, match="context shape"):
        bandit.commit_arm(0, [0.0])  # length 1, expected 2


def test_commit_non_finite_context_raises():
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2)
    bandit.begin_trajectory(0)
    with pytest.raises(ValueError, match="finite"):
        bandit.commit_arm(0, [float("nan"), 0.0])


def _seed(bandit: ContextualLinUCBBandit, arm: str, b_vec) -> None:
    """White-box seed: set A=I, b so theta = b for a deterministic LCB choice."""
    dim = bandit._context_dim
    bandit._linucb[arm]["A"] = np.eye(dim)
    bandit._linucb[arm]["b"] = np.asarray(b_vec, dtype=np.float64)


def test_two_contexts_select_different_arms():
    # alpha=0, epsilon=0 -> deterministic argmin of theta.x (bonus ignored).
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0, alpha=0.0)
    # theta_baseline = [-1,-1]  -> favors context [1,1]   (mean = -2)
    # theta_thresh_0p35 = [-1,1] -> favors context [1,-1]  (mean = -2)
    _seed(bandit, "baseline", [-1.0, -1.0])
    _seed(bandit, "thresh_0p35", [-1.0, 1.0])
    # thresh_0p15 left at the uninformative prior theta = [0,0].

    bandit.begin_trajectory(0)
    a0 = bandit.commit_arm(0, [1.0, 1.0])
    bandit.end_trajectory(0)  # close without feedback; seeds preserved

    bandit.begin_trajectory(1)
    a1 = bandit.commit_arm(1, [1.0, -1.0])
    bandit.end_trajectory(1)

    assert a0.template_id == "baseline"
    assert a1.template_id == "thresh_0p35"
    # The headline contextual property: two contexts pick different arms.
    assert a0.template_id != a1.template_id
    # Context is recorded on the committed assignment.
    assert list(a0.context) == [1.0, 1.0]


def test_update_moves_theta():
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0, alpha=0.0)
    # Fresh prior: theta = A^-1 b = 0.
    assert np.allclose(bandit.theta("baseline"), 0.0)

    bandit.begin_trajectory(0)
    # All priors equal -> epsilon=0 picks manifest-first (baseline).
    committed = bandit.commit_arm(0, [1.0, 0.0])
    assert committed.template_id == "baseline"
    bandit.end_trajectory(0, _feedback(0, "baseline", loss=0.5))

    # A = I + x x^T = [[2,0],[0,1]], b = 0.5 x = [0.5,0] -> theta = [0.25,0].
    theta = bandit.theta("baseline")
    assert not np.allclose(theta, 0.0)
    assert theta == pytest.approx([0.25, 0.0])


def test_epsilon_keeps_propensity_nondegenerate():
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.3, alpha=0.0)
    bandit.begin_trajectory(0)
    assignment = bandit.commit_arm(0, [0.0, 0.0])
    # The headline property: epsilon > 0 keeps the logged propensity strictly
    # inside (0,1) so the deployed policy is IPS-estimable offline.
    assert 0.0 < assignment.propensity < 1.0
    # epsilon=0.3 over 3 arms: greedy gets 0.3/3 + 0.7 = 0.8, others 0.1 each.
    assert (assignment.propensity == pytest.approx(0.8)
            or assignment.propensity == pytest.approx(0.1))


def test_contextual_state_roundtrip(tmp_path):
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0, alpha=0.5,
        run_identity={"dataset": "imagenet", "prefix_steps": 2})
    bandit.begin_trajectory(0)
    bandit.commit_arm(0, [1.0, 0.0])  # picks baseline (manifest-first on ties)
    bandit.end_trajectory(0, _feedback(0, "baseline", loss=0.4))

    path = tmp_path / "ctx_state.json"
    bandit.save_state(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CONTEXTUAL_BANDIT_SCHEMA_VERSION
    assert "linucb" in payload
    assert payload["context_dim"] == 2

    restored = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0, alpha=0.5,
        run_identity={"dataset": "imagenet", "prefix_steps": 2})
    restored.load_state(str(path))
    for arm in ("baseline", "thresh_0p15", "thresh_0p35"):
        assert np.allclose(restored._linucb[arm]["A"], bandit._linucb[arm]["A"])
        assert np.allclose(restored._linucb[arm]["b"], bandit._linucb[arm]["b"])
    summary = restored.summary()
    assert summary["contextual"] is True
    assert summary["context_dim"] == 2
    assert summary["linucb_alpha"] == 0.5
    assert summary["linucb_theta_norm"]["baseline"] > 0.0


def test_v1_state_load_raises(tmp_path):
    # A non-contextual (schema v1) state cannot resume into a contextual bandit.
    v1 = ConservativeTemplateBandit.from_strategies(
        _manifest(), "sess", epsilon=0.0,
        run_identity={"dataset": "imagenet"})
    v1.begin_trajectory(0)
    v1.end_trajectory(0)
    path = tmp_path / "v1_state.json"
    v1.save_state(str(path))
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] \
        == BANDIT_SCHEMA_VERSION

    contextual = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2,
        run_identity={"dataset": "imagenet"})
    with pytest.raises(ValueError, match="identity"):
        contextual.load_state(str(path))


def test_v2_state_load_into_v1_raises(tmp_path):
    # Symmetric: a contextual (v2) state cannot resume into a non-contextual bandit.
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0,
        run_identity={"dataset": "imagenet"})
    bandit.begin_trajectory(0)
    bandit.commit_arm(0, [1.0, 0.0])
    bandit.end_trajectory(0, _feedback(0, "baseline"))
    path = tmp_path / "ctx_state.json"
    bandit.save_state(str(path))

    v1 = ConservativeTemplateBandit.from_strategies(
        _manifest(), "sess", epsilon=0.0,
        run_identity={"dataset": "imagenet"})
    with pytest.raises(ValueError, match="identity"):
        v1.load_state(str(path))


def test_context_dim_mismatch_on_load_raises(tmp_path):
    bandit = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=2, epsilon=0.0)
    bandit.begin_trajectory(0)
    bandit.commit_arm(0, [1.0, 0.0])
    bandit.end_trajectory(0, _feedback(0, "baseline"))
    path = tmp_path / "ctx_state.json"
    bandit.save_state(str(path))

    wrong = ContextualLinUCBBandit.from_strategies(
        _manifest(), "sess", context_dim=3, epsilon=0.0)
    with pytest.raises(ValueError, match="shape"):
        wrong.load_state(str(path))


def _manifest_with_refresh_arm() -> StrategyManifest:
    """Mixed manifest: one threshold arm + one refresh-mask arm (no threshold)."""
    return StrategyManifest(
        version_key="vk1",
        num_steps=50,
        baseline_strategy_id="baseline",
        strategies=(
            AccelerationStrategy(
                "baseline", "teacache",
                {"rel_l1_thresh": 0.25, "num_steps": 50}, 50.0),
            AccelerationStrategy(
                "pattern_uniform_k8", "teacache",
                {"refresh_mask": [True] * 50, "refresh_count": 8,
                 "num_steps": 50}, 8.0),
        ),
    )


def test_mixed_manifest_rejected_at_construction():
    # Deferred-commit cannot commit a refresh-mask arm: commit_arm drops the
    # mask and switches onto the rel_l1_thresh dynamic path, which a mask arm
    # has no value for. The bandit must reject mixed manifests at construction
    # so the failure is loud at startup (not after burning GPU on the first
    # trajectory that happens to select the mask arm).
    with pytest.raises(ValueError, match="threshold-only"):
        ContextualLinUCBBandit.from_strategies(
            _manifest_with_refresh_arm(), "sess", context_dim=2)
