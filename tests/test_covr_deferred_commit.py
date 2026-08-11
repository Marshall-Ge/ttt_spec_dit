"""Runtime-layer tests for the deferred-commit contextual bandit integration.

Verifies that ``COVRRuntime.begin_trajectory`` returns a *pending* trajectory
(template_id=None) backed by the synthetic all-calc prefix strategy, that
``commit_arm`` refreshes the assignment/strategy/template_id to the chosen arm,
and that a non-contextual backend rejects ``commit_arm``.
"""

from types import SimpleNamespace

import pytest

from accelerators.covr_bandit import (
    AccelerationStrategy,
    ConservativeTemplateBandit,
    StrategyManifest,
)
from accelerators.covr_runtime import (
    COVRRuntime,
    ExperimentalBanditBackend,
    build_covr_runtime_config,
    flatten_prefix_features,
)
from accelerators.covr_viability import extract_prefix_features


def _manifest_path(tmp_path):
    manifest = StrategyManifest(
        version_key="vk1",
        num_steps=4,
        baseline_strategy_id="baseline",
        strategies=(
            AccelerationStrategy(
                "baseline", "teacache",
                {"rel_l1_thresh": 0.25, "num_steps": 4}, 4.0),
            AccelerationStrategy(
                "thresh_0p35", "teacache",
                {"rel_l1_thresh": 0.35, "num_steps": 4}, 4.0),
        ),
    )
    path = tmp_path / "manifest.json"
    manifest.save(str(path))
    return path, manifest


def _args(tmp_path, **overrides):
    values = dict(
        model="dit", method="teacache", ttt=False, batch_size=1,
        covr_strategy_bandit=True,
        covr_strategy_manifest=str(tmp_path / "manifest.json"),
        covr_contextual_bandit=True, covr_prefix_steps=2,
        covr_efficiency_lambda=1e-3, covr_linucb_alpha=1.0,
        covr_bandit_epsilon=0.0, covr_bandit_prior_penalty=0.0,
        covr_safety_sample_rate=0.0, covr_safety_chain_threshold=0,
        covr_sentinel_rate=0.0, covr_sentinel_horizon=0,
        covr_session_id="sess", covr_base_model_version="m",
        covr_profile_stages=False, covr_shadow=False, seed=42, num_steps=4,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _contextual_runtime(tmp_path):
    _path, manifest = _manifest_path(tmp_path)
    config = build_covr_runtime_config(_args(tmp_path), str(tmp_path))
    runtime = COVRRuntime.create(config, scheduler=None, num_steps=4)
    runtime.configure_experimental_bandit(
        manifest, epsilon=0.0, run_identity={"prefix_steps": 2},
        state_path=str(tmp_path / "state.json"))
    return runtime, manifest


def test_begin_returns_pending_assignment_with_prefix_strategy(tmp_path):
    runtime, _ = _contextual_runtime(tmp_path)
    trajectory = runtime.begin_trajectory(0, sample_count=1, num_steps=4)

    assert trajectory.bandit_assignment is not None
    assert trajectory.bandit_assignment.template_id is None  # pending
    assert trajectory.template_id is None  # no arm yet
    # Effective strategy during the prefix is the synthetic all-calc mask.
    assert trajectory.strategy is not None
    assert trajectory.strategy.strategy_id == "__contextual_prefix__"
    mask = trajectory.strategy.params["refresh_mask"]
    assert mask == (True, True, True, True)
    assert trajectory.prefix_steps == 0  # set by the runner, not begin


def test_commit_arm_refreshes_assignment_and_strategy(tmp_path):
    runtime, _manifest = _contextual_runtime(tmp_path)
    trajectory = runtime.begin_trajectory(0, sample_count=1, num_steps=4)
    # PREFIX_FEATURE_WIDTH(14) * prefix_steps(2) = 28.
    context = [0.1] * 28

    selection = runtime.commit_arm(trajectory, context)

    assert selection.assignment.template_id in {"baseline", "thresh_0p35"}
    # The trajectory now reflects the committed arm everywhere the loop reads it.
    assert trajectory.template_id == selection.assignment.template_id
    assert trajectory.bandit_assignment.template_id == trajectory.template_id
    assert trajectory.strategy.strategy_id == trajectory.template_id
    # The committed strategy is a real manifest arm, not the prefix placeholder.
    assert trajectory.strategy.strategy_id != "__contextual_prefix__"
    assert "rel_l1_thresh" in trajectory.strategy.params


def test_commit_arm_bad_context_dim_raises(tmp_path):
    runtime, _ = _contextual_runtime(tmp_path)
    trajectory = runtime.begin_trajectory(0, sample_count=1, num_steps=4)
    with pytest.raises(ValueError):
        runtime.commit_arm(trajectory, [0.1] * 5)  # wrong width


def test_commit_arm_without_pending_raises(tmp_path):
    runtime, _ = _contextual_runtime(tmp_path)
    # A non-contextual trajectory has a real template_id at begin time, so its
    # bandit_assignment is committed (not pending); commit_arm rejects it.
    from accelerators.covr_runtime import COVRTrajectoryAssignment
    closed = COVRTrajectoryAssignment(
        trajectory_id=0, sample_count=1, strategy=None,
        bandit_assignment=None, sentinel_selected=False,
        sentinel_start_idx=None, sample_ids=(), session_id="sess",
        sentinel_horizon=0)
    with pytest.raises(RuntimeError, match="pending"):
        runtime.commit_arm(closed, [0.1] * 28)


def test_non_contextual_backend_commit_arm_raises(tmp_path):
    _path, manifest = _manifest_path(tmp_path)
    args = _args(tmp_path, covr_contextual_bandit=False, covr_prefix_steps=0)
    config = build_covr_runtime_config(args, str(tmp_path))
    runtime = COVRRuntime.create(config, scheduler=None, num_steps=4)
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, session_id="sess", epsilon=0.0)
    runtime.backend = ExperimentalBanditBackend(
        bandit, state_path=str(tmp_path / "state.json"))
    trajectory = runtime.begin_trajectory(0, sample_count=1, num_steps=4)
    # Non-contextual bandit selects in begin_trajectory (no pending state).
    assert trajectory.bandit_assignment.template_id is not None
    with pytest.raises(RuntimeError, match="contextual bandit backend"):
        runtime.commit_arm(trajectory, [0.1] * 28)


def test_prefix_strategy_is_cached_and_all_calc(tmp_path):
    runtime, _ = _contextual_runtime(tmp_path)
    first = runtime._contextual_prefix_strategy()
    second = runtime._contextual_prefix_strategy()
    assert first is second  # cached
    assert first.source == "contextual-prefix"
    assert first.params["refresh_mask"] == (True, True, True, True)


def test_flatten_prefix_features_roundtrip():
    import torch
    rows = []
    for _ in range(2):
        latent = torch.randn(1, 4)
        noise = torch.randn(1, 4)
        rows.append(extract_prefix_features(latent, noise, 1))
    flat = flatten_prefix_features(rows)
    assert len(flat) == 28  # PREFIX_FEATURE_WIDTH(14) * 2 prefix steps
    assert all(isinstance(v, float) for v in flat)
