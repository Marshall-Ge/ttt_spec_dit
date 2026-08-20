"""Behavior contracts for the optional COVR runtime boundary."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from config import DIT_REPO
from accelerators.covr_bandit import (
    AccelerationStrategy,
    ConservativeTemplateBandit,
    StrategyManifest,
    TemplateFeedback,
)
from accelerators.covr_runtime import (
    COVRMode,
    COVRRunState,
    COVRRuntime,
    COVRRuntimeConfig,
    COVRTrajectoryAssignment,
    COVRTrajectoryFeedback,
    ExperimentalBanditBackend,
    ForcedStrategyBackend,
    build_covr_runtime_config,
    covr_requested,
    load_strategy_manifest,
)
from accelerators.registry import _REGISTRY, register_adapter, AcceleratorAdapter
from main import parse_args, validate_args


def _runtime_args(**overrides):
    values = {
        "model": "dit",
        "method": "speca",
        "ttt": False,
        "covr_shadow": False,
        "covr_output_dir": None,
        "covr_session_id": None,
        "covr_max_events": None,
        "covr_base_model_version": "test-model",
        "covr_template_bandit": False,
        "covr_template_manifest": None,
        "covr_force_template_id": None,
        "covr_strategy_bandit": False,
        "covr_strategy_manifest": None,
        "covr_force_strategy_id": None,
        "covr_bandit_state": None,
        "covr_bandit_epsilon": 0.1,
        "covr_bandit_prior_penalty": 0.0,
        "batch_size": 1,
        "covr_safety_sample_rate": 0.1,
        "covr_safety_chain_threshold": 0,
        "covr_sentinel_rate": 0.05,
        "covr_sentinel_horizon": 0,
        "covr_profile_stages": False,
        "covr_timestep_feedback": False,
        "covr_timestep_feedback_active": False,
        "covr_timestep_feedback_budget": 8,
        "covr_timestep_feedback_state": None,
        "covr_timestep_feedback_p_min": 0.02,
        "covr_timestep_feedback_beta": 1.0,
        "num_steps": 50,
        "speca_max_taylor_steps": 4,
        "seed": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parse(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["main.py", *extra])
    return parse_args()


def _strategy_manifest(path, *, version_key="runtime-version", method="teacache"):
    manifest = StrategyManifest(
        version_key=version_key,
        num_steps=4,
        baseline_strategy_id="uniform",
        strategies=(
            AccelerationStrategy(
                strategy_id="uniform",
                method=method,
                params={"refresh_mask": [True, False, True, False]},
                modeled_flops=2.0,
            ),
        ),
    )
    manifest.save(str(path))
    return manifest


def test_disabled_runtime_config_is_none(tmp_path):
    config = build_covr_runtime_config(_runtime_args(), str(tmp_path))

    assert config is None
    assert not (tmp_path / "covr").exists()


def test_legacy_manifest_path_alone_does_not_enable_runtime(tmp_path):
    config = build_covr_runtime_config(
        _runtime_args(covr_template_manifest="legacy-manifest.json"),
        str(tmp_path),
    )

    assert config is None


@pytest.mark.parametrize("method", ["baseline", "ddim"])
def test_profile_only_config_does_not_require_adapter(tmp_path, method):
    config = build_covr_runtime_config(
        _runtime_args(method=method, covr_profile_stages=True),
        str(tmp_path),
    )

    assert config is not None
    assert config.mode is COVRMode.OBSERVE
    assert config.capabilities.method == method
    assert config.capabilities.terminal_reward is False


def test_runtime_config_preserves_default_base_model_version(tmp_path):
    config = build_covr_runtime_config(
        _runtime_args(covr_shadow=True, covr_base_model_version=None),
        str(tmp_path),
    )

    assert config is not None
    assert config.base_model_version == os.path.expanduser(DIT_REPO)


def test_default_bandit_state_path_does_not_implicitly_resume(tmp_path):
    config = build_covr_runtime_config(
        _runtime_args(
            covr_strategy_bandit=True,
            covr_strategy_manifest="strategies.json",
        ),
        str(tmp_path),
    )

    assert config is not None
    assert config.bandit_state_path == str(
        tmp_path / "covr" / "template_bandit_state.json")
    assert config.resume_state_path is None


def test_explicit_bandit_state_path_is_the_resume_source(tmp_path):
    state_path = str(tmp_path / "resume.json")
    config = build_covr_runtime_config(
        _runtime_args(
            covr_strategy_bandit=True,
            covr_strategy_manifest="strategies.json",
            covr_bandit_state=state_path,
        ),
        str(tmp_path),
    )

    assert config is not None
    assert config.bandit_state_path == state_path
    assert config.resume_state_path == state_path


def test_profile_stage_accumulation_accepts_plain_dicts():
    from run_dit import _record_profile_stage

    totals = {}
    counts = {}
    _record_profile_stage(totals, counts, "image_save_metrics", 0.25)
    _record_profile_stage(totals, counts, "image_save_metrics", 0.75)

    assert totals == {"image_save_metrics": 1.0}
    assert counts == {"image_save_metrics": 2}


def test_runner_binds_manifest_loader_for_forced_dispatch():
    import run_dit

    assert callable(run_dit.load_strategy_manifest)


def test_observe_runtime_uses_single_trajectory_context(tmp_path):
    config = build_covr_runtime_config(
        _runtime_args(method="baseline", covr_profile_stages=True),
        str(tmp_path),
    )
    runtime = COVRRuntime.create(config, num_steps=4)
    trajectory = runtime.begin_trajectory(
        0, sample_count=2, sample_ids=("0", "1"), num_steps=4)

    assert trajectory.strategy is None
    assert trajectory.sample_ids == ("0", "1")
    assert trajectory.feedback_sink == {}

    outcome = runtime.end_trajectory(
        trajectory,
        generation_profile={"generation_online": 0.25},
        num_steps=4,
        sentinel_rate=0.0,
        sentinel_horizon=0,
    )
    assert outcome["wall_s"] == pytest.approx(0.25)
    assert outcome["feedback"]["terminal_fidelity_loss"] is None


def test_observe_runtime_delegates_existing_accelerator_flops(tmp_path):
    config = build_covr_runtime_config(
        _runtime_args(method="teacache", covr_profile_stages=True),
        str(tmp_path),
    )
    runtime = COVRRuntime.create(config, num_steps=4)
    trajectory = runtime.begin_trajectory(0, sample_count=1, num_steps=4)

    class _Metric:
        generation = None
        vanilla_steps = None

        def add_generation(self, state):
            self.generation = list(state.decisions)

        def add_vanilla_steps(self, steps=None):
            self.vanilla_steps = steps

    metric = _Metric()
    runtime.add_flops(
        metric, trajectory, num_layers=28, method="teacache",
        accelerator_states={
            "teacache_state": {"decisions": ["calc", "skip"]}},
        num_steps=4,
    )

    assert metric.generation == ["calc", "skip"]
    assert metric.vanilla_steps is None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"covr_shadow": True}, COVRMode.OBSERVE),
        ({"covr_profile_stages": True}, COVRMode.OBSERVE),
        (
            {
                "covr_strategy_manifest": "manifest.json",
                "covr_force_strategy_id": "uniform",
            },
            COVRMode.FORCED,
        ),
        (
            {
                "covr_strategy_manifest": "manifest.json",
                "covr_strategy_bandit": True,
            },
            COVRMode.EXPERIMENTAL_BANDIT,
        ),
    ],
)
def test_runtime_config_distinguishes_modes(tmp_path, overrides, expected):
    config = build_covr_runtime_config(
        _runtime_args(**overrides), str(tmp_path))

    assert config is not None
    assert config.mode is expected


def test_runtime_config_threads_bandit_prior_penalty(tmp_path):
    config = build_covr_runtime_config(
        _runtime_args(
            covr_strategy_bandit=True,
            covr_strategy_manifest="manifest.json",
            covr_bandit_prior_penalty=2.5e-5,
        ),
        str(tmp_path),
    )

    assert config is not None
    assert config.bandit_prior_penalty == pytest.approx(2.5e-5)


@pytest.mark.parametrize("batch_size", [0, 2, 32])
def test_strategy_bandit_requires_one_image_per_trajectory(
        tmp_path, batch_size):
    with pytest.raises(ValueError, match="requires --batch_size 1"):
        build_covr_runtime_config(
            _runtime_args(
                covr_strategy_bandit=True,
                covr_strategy_manifest="manifest.json",
                batch_size=batch_size,
            ),
            str(tmp_path),
        )


@pytest.mark.parametrize("penalty", [-1.0, float("inf"), float("nan")])
def test_strategy_bandit_rejects_invalid_prior_penalty(tmp_path, penalty):
    with pytest.raises(ValueError, match="prior-penalty"):
        build_covr_runtime_config(
            _runtime_args(
                covr_strategy_bandit=True,
                covr_strategy_manifest="manifest.json",
                covr_bandit_prior_penalty=penalty,
            ),
            str(tmp_path),
        )


def test_runtime_aggregate_serializes_generic_forced_strategy(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = _strategy_manifest(manifest_path, version_key="runtime-version")
    config = build_covr_runtime_config(
        _runtime_args(
            covr_force_strategy_id="uniform",
            covr_strategy_manifest=str(manifest_path),
        ),
        str(tmp_path),
    )
    assert config is not None
    runtime = COVRRuntime.create(config)

    payload = runtime.aggregate(
        forced_strategy=manifest.strategies[0],
        forced_manifest=manifest,
    )

    forced = payload["covr_forced_template"]
    assert forced["template_id"] == "uniform"
    assert forced["refresh_count"] == 2
    assert forced["modeled_full_block_equivalents"] == 2.0
    assert forced["manifest_hash"] == manifest.manifest_hash


def test_strategy_manifest_version_key_is_checked_for_forced_and_bandit(tmp_path):
    path = tmp_path / "manifest.json"
    _strategy_manifest(path, version_key="manifest-version")

    for mode in (COVRMode.FORCED, COVRMode.EXPERIMENTAL_BANDIT):
        with pytest.raises(ValueError, match="runtime"):
            load_strategy_manifest(
                str(path), expected_version_key="runtime-version", mode=mode)


def test_forced_teacache_strategy_manifest_loads_with_matching_version(tmp_path):
    path = tmp_path / "manifest.json"
    expected = _strategy_manifest(path)

    loaded = load_strategy_manifest(
        str(path),
        expected_version_key="runtime-version",
        mode=COVRMode.FORCED,
    )

    assert loaded.manifest_hash == expected.manifest_hash
    assert loaded.strategy_map["uniform"].refresh_mask == (
        True, False, True, False)


def test_cli_rejects_pixart_covr_runtime_in_phase_one(monkeypatch):
    args = _parse(
        monkeypatch,
        "--model", "pixart",
        "--task", "t2i",
        "--dataset", "coco",
        "--method", "teacache",
        "--covr-strategy-manifest", "manifest.json",
        "--covr-force-strategy-id", "uniform",
    )

    assert validate_args(args) is False


def test_timestep_feedback_requires_shadow(monkeypatch):
    args = _parse(
        monkeypatch,
        "--model", "dit",
        "--task", "c2i",
        "--dataset", "imagenet",
        "--method", "speca",
        "--covr-timestep-feedback",
    )

    assert validate_args(args) is False


def test_active_timestep_feedback_requires_learner(monkeypatch):
    args = _parse(
        monkeypatch,
        "--model", "dit",
        "--task", "c2i",
        "--dataset", "imagenet",
        "--method", "speca",
        "--covr-timestep-feedback-active",
    )

    assert validate_args(args) is False


def test_timestep_feedback_shadow_builds_observe_runtime(tmp_path):
    args = _runtime_args(
        covr_shadow=True,
        covr_timestep_feedback=True,
        covr_output_dir=str(tmp_path / "covr"),
    )

    config = build_covr_runtime_config(args, str(tmp_path))
    assert config is not None
    assert config.shadow is True


def test_cli_rejects_ttt_covr_runtime_in_phase_one(monkeypatch):
    args = _parse(
        monkeypatch,
        "--model", "dit",
        "--task", "c2i",
        "--dataset", "imagenet",
        "--method", "teacache",
        "--ttt",
        "--covr-strategy-manifest", "manifest.json",
        "--covr-force-strategy-id", "uniform",
    )

    assert validate_args(args) is False


def test_cli_accepts_forced_teacache_strategy_in_dit_main_loop(monkeypatch):
    args = _parse(
        monkeypatch,
        "--model", "dit",
        "--task", "c2i",
        "--dataset", "imagenet",
        "--method", "teacache",
        "--covr-strategy-manifest", "manifest.json",
        "--covr-force-strategy-id", "uniform",
    )

    assert validate_args(args) is True


# ===========================================================================
# Runtime construction (config → version → backend)
# ===========================================================================


def _runtime(args, tmp_path, scheduler=None, speca_init_kwargs=None,
             resume=None, backend=None):
    config = build_covr_runtime_config(args, str(tmp_path))
    assert config is not None
    return COVRRuntime.create(
        config, scheduler=scheduler, cfg_scale=0.0,
        speca_init_kwargs=speca_init_kwargs, resume=resume, backend=backend)


class _FakeScheduler:
    num_steps = 20
    config = {"prediction_type": "epsilon", "_use_default_values": False}


def test_create_runtime_builds_version_from_scheduler(tmp_path):
    runtime = _runtime(
        _runtime_args(covr_shadow=True),
        tmp_path,
        scheduler=_FakeScheduler(),
        speca_init_kwargs={"num_steps": 20, "check_layer": 20},
    )

    assert runtime.version is not None
    assert runtime.version.model == "dit"
    assert runtime.version.scheduler == "_FakeScheduler"
    assert runtime.version.num_steps == 20
    assert len(runtime.version.key) == 16
    # speca_config is canonical JSON of the init kwargs.
    assert json.loads(runtime.version.speca_config)["check_layer"] == 20


def test_create_runtime_uses_explicit_inference_step_count(tmp_path):
    config = build_covr_runtime_config(
        _runtime_args(covr_shadow=True), str(tmp_path))
    runtime = COVRRuntime.create(
        config, scheduler=_FakeScheduler(), num_steps=50)

    assert runtime.version.num_steps == 50


def test_runtime_configures_generic_bandit_backend(tmp_path):
    config = build_covr_runtime_config(
        _runtime_args(
            covr_strategy_bandit=True,
            covr_strategy_manifest="manifest.json",
            covr_session_id="session-a",
        ),
        str(tmp_path),
    )
    runtime = COVRRuntime.create(config, num_steps=4)
    manifest = _strategy_manifest(
        tmp_path / "manifest.json", version_key=runtime.version.key)

    backend = runtime.configure_experimental_bandit(
        manifest,
        epsilon=0.1,
        run_identity={"dataset": "imagenet"},
        state_path=str(tmp_path / "bandit-state.json"),
    )

    assert runtime.backend is backend
    assert backend.bandit.session_id == "session-a"
    assert backend.bandit.summary()["completed_trajectories"] == 0


def test_create_runtime_session_id_fallbacks(tmp_path):
    args = _runtime_args(covr_shadow=True, covr_session_id=None)
    config = build_covr_runtime_config(args, str(tmp_path))
    runtime = COVRRuntime.create(config, resume=None)
    # Generated: timestamp-seed
    assert runtime.version is not None
    assert runtime.config.session_id is None
    assert runtime.recorder is not None
    assert runtime.recorder.session_id.endswith(f"-seed{args.seed}")


def test_create_runtime_session_id_from_resume(tmp_path):
    args = _runtime_args(covr_shadow=True, covr_session_id=None)
    config = build_covr_runtime_config(args, str(tmp_path))
    runtime = COVRRuntime.create(
        config, resume={"session_id": "old-session", "processed_samples": 5})

    assert runtime.recorder.session_id == "old-session"


def test_disabled_never_creates_runtime_objects():
    args = _runtime_args()
    assert build_covr_runtime_config(args, "/tmp/out") is None
    assert covr_requested(args) is False


# ===========================================================================
# Bandit backend lifecycle through the runtime
# ===========================================================================


def _bandit_runtime(tmp_path):
    manifest = _strategy_manifest(tmp_path / "manifest.json")
    args = _runtime_args(
        covr_strategy_manifest=str(tmp_path / "manifest.json"),
        covr_strategy_bandit=True,
        covr_bandit_state=str(tmp_path / "bandit_state.json"),
    )
    config = build_covr_runtime_config(args, str(tmp_path))
    runtime = COVRRuntime.create(config, scheduler=None)
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, session_id="sess", epsilon=0.1, seed=42)
    runtime.backend = ExperimentalBanditBackend(
        bandit, state_path=config.bandit_state_path)
    return runtime, bandit, config


def test_runtime_bandit_begin_end_observe_persist(tmp_path):
    runtime, bandit, config = _bandit_runtime(tmp_path)
    trajectory = runtime.begin_trajectory(
        0, sample_count=2, num_steps=4, sentinel_rate=0.0,
        sentinel_horizon=0)

    assert trajectory.strategy is not None
    assert trajectory.bandit_assignment is not None
    assert trajectory.sample_count == 2
    assert not trajectory.sentinel_selected

    runtime.backend.observe_one_step(
        trajectory.trajectory_id, 1, [0.1], [0.5])
    runtime.backend.end_trajectory(
        trajectory.trajectory_id,
        TemplateFeedback(
            trajectory_id=0, template_id=trajectory.bandit_assignment.template_id,
            sentinel_propensity=0.05, horizon=4, terminal_fidelity_loss=0.01,
        ),
    )
    runtime.backend.persist()

    assert bandit.summary()["processed_samples"] == 2
    assert bandit.summary()["completed_trajectories"] == 1
    assert (tmp_path / "bandit_state.json").exists()


def test_runtime_resume_offset_comes_from_state(tmp_path):
    runtime, bandit, config = _bandit_runtime(tmp_path)
    # Simulate a prior session that consumed 7 samples.
    t = runtime.begin_trajectory(0, sample_count=7, num_steps=4,
                                 sentinel_rate=0.0, sentinel_horizon=0)
    runtime.backend.end_trajectory(
        t.trajectory_id,
        TemplateFeedback(
            trajectory_id=0, template_id=t.bandit_assignment.template_id,
            sentinel_propensity=0.05, horizon=4, terminal_fidelity_loss=0.0,
        ),
    )
    runtime.backend.persist()

    # A fresh runtime loading that state must resume with the same offset.
    resume = {"session_id": "sess", "processed_samples": 7}
    assert resume["processed_samples"] == 7


# ===========================================================================
# Forced backend through the runtime
# ===========================================================================


def test_runtime_forced_backend_static_selection(tmp_path):
    manifest = _strategy_manifest(tmp_path / "manifest.json")
    strategy = manifest.strategy_map["uniform"]
    backend = ForcedStrategyBackend(strategy=strategy, manifest=manifest)
    args = _runtime_args(
        covr_strategy_manifest=str(tmp_path / "manifest.json"),
        covr_force_strategy_id="uniform",
    )
    config = build_covr_runtime_config(args, str(tmp_path))
    runtime = COVRRuntime.create(config, backend=backend, scheduler=None)

    trajectory = runtime.begin_trajectory(
        3, sample_count=4, num_steps=4, sentinel_rate=0.0,
        sentinel_horizon=0)
    assert trajectory.strategy is strategy
    assert trajectory.bandit_assignment is None

    # No learning state to persist.
    runtime.backend.persist()


def test_runtime_forced_backend_apply_strategy_masks(tmp_path):
    manifest = _strategy_manifest(tmp_path / "manifest.json")
    strategy = manifest.strategy_map["uniform"]
    runtime = COVRRuntime.create(
        build_covr_runtime_config(
            _runtime_args(
                covr_strategy_manifest=str(tmp_path / "manifest.json"),
                covr_force_strategy_id="uniform",
            ),
            str(tmp_path),
        ),
        backend=ForcedStrategyBackend(strategy, manifest),
        scheduler=None,
    )
    trajectory = runtime.begin_trajectory(
        0, sample_count=1, num_steps=4, sentinel_rate=0.0,
        sentinel_horizon=0)
    states = runtime.apply_acceleration_strategy(
        trajectory,
        teacache_init_kwargs={"num_steps": 4, "coefficients": [0.0, 0.0, 0.0, 1.0, 0.0]},
    )
    assert states["teacache_state"]["refresh_mask"] == (True, False, True, False)
    assert trajectory.accelerator_states["teacache_state"] is states["teacache_state"]


# ===========================================================================
# Feedback object
# ===========================================================================


def test_trajectory_feedback_accumulates():
    feedback = COVRTrajectoryFeedback()
    feedback.record_safety(2, "values")
    feedback.record_terminal_loss(0.1)
    feedback.record_h_step((0.2, 0.3), full_steps=3)

    assert feedback.safety_full_steps == 1
    assert feedback.terminal_full_steps == 4
    assert feedback.pending_safety == [(2, "values")]
    assert feedback.terminal_fidelity_loss_tensor == 0.1
    assert feedback.h_step_components_tensor == (0.2, 0.3)


# ===========================================================================
# End-of-trajectory lifecycle (feedback materialization, bandit close/persist)
# ===========================================================================


def _bandit_config(tmp_path, *, sentinel_rate=0.0, sentinel_horizon=0):
    manifest = _strategy_manifest(tmp_path / "manifest.json")
    args = _runtime_args(
        covr_strategy_manifest=str(tmp_path / "manifest.json"),
        covr_strategy_bandit=True,
        covr_sentinel_rate=sentinel_rate,
        covr_sentinel_horizon=sentinel_horizon,
        covr_bandit_state=str(tmp_path / "bandit_state.json"),
    )
    config = build_covr_runtime_config(args, str(tmp_path))
    runtime = COVRRuntime.create(config, scheduler=None)
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, session_id="sess", epsilon=0.1, seed=42)
    runtime.backend = ExperimentalBanditBackend(
        bandit, state_path=config.bandit_state_path)
    return runtime, bandit, config


def test_terminal_fallback_uses_profile_only_trajectory_context():
    from run_dit import _covr_generate_terminal_fallback

    calls = {}

    class _Generator:
        def generate(self, prompts, seeds, **kwargs):
            calls.update({"prompts": prompts, "seeds": seeds, **kwargs})
            return "latent", "image"

    source = COVRTrajectoryAssignment(
        trajectory_id=7,
        sample_count=2,
        strategy=None,
        bandit_assignment=None,
        sentinel_selected=True,
        sentinel_start_idx=None,
        sample_ids=("sample-0", "sample-1"),
        session_id="session-a",
    )
    profiler = object()

    result = _covr_generate_terminal_fallback(
        _Generator(), [1, 2], [11, 12], 4.5, source, profiler)

    assert result == ("latent", "image")
    assert calls["prompts"] == [1, 2]
    assert calls["seeds"] == [11, 12]
    assert calls["guidance_scale"] == 4.5
    assert calls["method"] == "baseline"
    assert "covr_profiler" not in calls
    context = calls["covr_trajectory"]
    assert context.profiler is profiler
    assert context.trajectory_id == source.trajectory_id
    assert context.sample_count == source.sample_count
    assert context.sentinel_selected is False
    assert context.safety_sample_rate == 0.0
    assert context.terminal_reward_active is False
    assert context.sample_ids == ()
    assert context.feedback_sink == {}


def test_runtime_end_trajectory_terminal_sentinel_with_fallback(tmp_path):
    runtime, bandit, config = _bandit_config(
        tmp_path, sentinel_rate=1.0, sentinel_horizon=0)
    trajectory = runtime.begin_trajectory(
        0, sample_count=2, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=0)
    assert trajectory.sentinel_selected
    assert trajectory.sentinel_start_idx is None

    outcome = runtime.end_trajectory(
        trajectory,
        {"terminal_fidelity_loss_tensor": torch.tensor(0.05),
         "terminal_full_steps": 1},
        generation_profile={
            "generation_online": 1.0,
            "delayed_sentinel_shadow_full": 0.5,
        },
        num_steps=4, sentinel_rate=1.0, sentinel_horizon=0,
        fallback_wall_s=2.0, fallback_full_steps=4,
        fallback_profile={"terminal_fallback_full": 2.0},
    )

    # Bandit closed with the terminal reward and persisted per trajectory.
    assert bandit.summary()["completed_trajectories"] == 1
    assert len(bandit.feedback) == 1
    assert bandit.feedback[0].horizon == 4  # terminal sentinels use num_steps
    assert (tmp_path / "bandit_state.json").exists()

    # Wall time includes generation + fallback + control + bandit persist.
    assert outcome["wall_s"] == pytest.approx(3.0, rel=0.05)
    assert outcome["control_wall_s"] == 0.0
    # Sentinel accounting folds the cheap step plus the fallback rollout.
    assert runtime.state.sentinel_full_steps == 5
    assert runtime.state.sentinel_count == 1
    assert runtime.state.terminal_wall_s == pytest.approx(2.5)
    assert runtime.state.terminal_wall_times == [pytest.approx(2.5)]
    # Fallback stages are folded into the profile buckets too.
    assert runtime.state.profile_stage_totals["terminal_fallback_full"] == \
        pytest.approx(2.0)
    assert runtime.state.profile_stage_counts["terminal_fallback_full"] == 1


def test_runtime_end_trajectory_h_step_sentinel(tmp_path):
    runtime, bandit, config = _bandit_config(
        tmp_path, sentinel_rate=1.0, sentinel_horizon=2)
    trajectory = runtime.begin_trajectory(
        0, sample_count=1, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=2)
    assert trajectory.sentinel_selected
    assert trajectory.sentinel_start_idx is not None

    outcome = runtime.end_trajectory(
        trajectory,
        {"h_step_components_tensor": torch.tensor([0.2, 0.8]),
         "terminal_full_steps": 2},
        generation_profile={"generation_online": 1.0},
        num_steps=4, sentinel_rate=1.0, sentinel_horizon=2,
    )

    assert bandit.feedback[0].h_step_numerator == pytest.approx(0.2)
    assert bandit.feedback[0].h_step_denominator == pytest.approx(0.8)
    assert bandit.feedback[0].horizon == 2  # H-step sentinels use the horizon
    assert runtime.state.sentinel_full_steps == 2
    assert runtime.state.sentinel_count == 1


def test_runtime_end_trajectory_non_sentinel_closes_without_reward(tmp_path):
    runtime, bandit, config = _bandit_config(tmp_path)
    trajectory = runtime.begin_trajectory(
        0, sample_count=1, num_steps=4, sentinel_rate=0.0,
        sentinel_horizon=0)
    assert not trajectory.sentinel_selected

    outcome = runtime.end_trajectory(
        trajectory, {},
        generation_profile={"generation_online": 0.5},
        num_steps=4, sentinel_rate=0.0, sentinel_horizon=0,
    )

    assert bandit.summary()["completed_trajectories"] == 1
    assert len(bandit.feedback) == 0
    assert outcome["wall_s"] == pytest.approx(0.5, rel=0.05)
    assert runtime.state.sentinel_count == 0
    assert runtime.state.sentinel_full_steps == 0


def test_runtime_end_trajectory_observes_pending_safety(tmp_path):
    runtime, bandit, config = _bandit_config(tmp_path)
    trajectory = runtime.begin_trajectory(
        0, sample_count=1, num_steps=4, sentinel_rate=0.0,
        sentinel_horizon=0)
    # Safety labels land as CPU tensors at the batch boundary; the runtime
    # materializes them into the trajectory feedback and observes the bandit
    # at close.
    outcome = runtime.end_trajectory(
        trajectory,
        {"pending_safety_steps": (1,),
         "pending_safety_values": torch.tensor([[0.1, 0.5]]),
         "safety_full_steps": 1},
        generation_profile={"generation_online": 0.5},
        num_steps=4, sentinel_rate=0.0, sentinel_horizon=0,
    )

    assert runtime.state.safety_full_steps == 1
    assert len(bandit._pending_safety) == 0  # flushed through end_trajectory
    assert bandit.summary()["completed_trajectories"] == 1
    assert runtime.state.safety_wall_times == [0.0]


def test_runtime_end_trajectory_terminal_sentinel_missing_reward_raises(tmp_path):
    runtime, bandit, config = _bandit_config(
        tmp_path, sentinel_rate=1.0, sentinel_horizon=0)
    trajectory = runtime.begin_trajectory(
        0, sample_count=1, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=0)

    with pytest.raises(RuntimeError, match="terminal sentinel did not produce"):
        runtime.end_trajectory(
            trajectory, {},
            generation_profile={"generation_online": 0.5},
            num_steps=4, sentinel_rate=1.0, sentinel_horizon=0)


def test_runtime_end_trajectory_forced_telemetry(tmp_path):
    manifest = _strategy_manifest(tmp_path / "manifest.json")
    strategy = manifest.strategy_map["uniform"]
    args = _runtime_args(
        covr_strategy_manifest=str(tmp_path / "manifest.json"),
        covr_force_strategy_id="uniform",
        covr_sentinel_rate=1.0,
        covr_sentinel_horizon=2,
    )
    config = build_covr_runtime_config(args, str(tmp_path))
    runtime = COVRRuntime.create(
        config, backend=ForcedStrategyBackend(strategy, manifest),
        scheduler=None)
    state = runtime.state

    # Terminal sentinel with a cheap reward is recorded.
    trajectory = runtime.begin_trajectory(
        0, sample_count=1, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=0)
    assert trajectory.sentinel_selected
    runtime.end_trajectory(
        trajectory, {"terminal_fidelity_loss_tensor": torch.tensor(0.05)},
        generation_profile={"generation_online": 1.0},
        num_steps=4, sentinel_rate=1.0, sentinel_horizon=0)
    assert state.terminal_losses == [pytest.approx(0.05)]

    # H-step sentinel is recorded.
    trajectory = runtime.begin_trajectory(
        1, sample_count=1, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=2)
    assert trajectory.sentinel_start_idx is not None
    runtime.end_trajectory(
        trajectory, {"h_step_components_tensor": torch.tensor([0.2, 0.8])},
        generation_profile={"generation_online": 1.0},
        num_steps=4, sentinel_rate=1.0, sentinel_horizon=2)
    assert state.h_step_numerators == [pytest.approx(0.2)]
    assert state.h_step_denominators == [pytest.approx(0.8)]

    # Missing reward is counted as skipped, never a fallback.
    trajectory = runtime.begin_trajectory(
        2, sample_count=1, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=0)
    runtime.end_trajectory(
        trajectory, {},
        generation_profile={"generation_online": 1.0},
        num_steps=4, sentinel_rate=1.0, sentinel_horizon=0)
    assert state.sentinel_skipped == 1
    assert len(state.terminal_losses) == 1  # unchanged


def test_runtime_end_trajectory_profile_buckets_and_wall_lists(tmp_path):
    runtime, bandit, config = _bandit_config(tmp_path)
    trajectory = runtime.begin_trajectory(
        0, sample_count=1, num_steps=4, sentinel_rate=0.0,
        sentinel_horizon=0)
    profile = {
        "generation_online": 1.0,
        "strategy_selection": 0.01,
        "accelerator_state_reset": 0.02,
        "speca_state_reset": 0.0,
        "strategy_initialization": 0.03,
        "feedback_materialization": 0.004,
        "vae_decode": 0.5,
    }
    outcome = runtime.end_trajectory(
        trajectory, {},
        generation_profile=profile,
        num_steps=4, sentinel_rate=0.0, sentinel_horizon=0,
    )

    # Control stages are summed into control_wall_s and wall_s.
    assert outcome["control_wall_s"] == pytest.approx(0.064)
    assert outcome["wall_s"] == pytest.approx(
        1.0 + 0.064, rel=0.05)  # + bandit update/persist
    # Non-control stages (vae_decode) are folded into the profile buckets.
    assert runtime.state.profile_stage_totals["vae_decode"] == \
        pytest.approx(0.5)
    assert runtime.state.profile_stage_counts["vae_decode"] == 1
    # Wall lists per trajectory.
    assert runtime.state.safety_wall_times == [0.0]
    assert runtime.state.terminal_wall_times == [0.0]
    assert len(runtime.state.control_wall_times) == 1
    assert runtime.state.control_wall_times[0] == pytest.approx(
        0.064, rel=0.05)
    # Bandit update/persist stages are timed into the same buckets.
    assert runtime.state.profile_stage_counts["bandit_update"] == 1
    assert runtime.state.profile_stage_counts["bandit_state_persist"] == 1


def test_runtime_state_initializes_counter_fields():
    state = COVRRunState()
    assert state.safety_full_steps == 0
    assert state.sentinel_full_steps == 0
    assert state.sentinel_count == 0
    assert state.sentinel_skipped == 0
    assert state.safety_wall_s == 0.0
    assert state.terminal_wall_s == 0.0
    assert state.safety_wall_times == []
    assert state.terminal_wall_times == []
    assert state.control_wall_times == []
    assert state.terminal_losses == []
    assert state.h_step_numerators == []
    assert state.h_step_denominators == []
    assert state.profile_stage_totals == {}
    assert state.profile_stage_counts == {}


# ===========================================================================
# Serializer slice (aggregate / generation_profile / online_accounting /
# config_payload) — golden keys must match the old inline run_dit payloads
# ===========================================================================


def test_online_accounting_matches_old_inline_math():
    # Bandit mode is required for the flops fields to be requested at all.
    runtime = COVRRuntime(
        build_covr_runtime_config(
            _runtime_args(covr_strategy_bandit=True), "/tmp/out"),
        version=None,
        backend=ExperimentalBanditBackend(
            ConservativeTemplateBandit.from_strategies(
                StrategyManifest(
                    version_key="v", num_steps=4,
                    baseline_strategy_id="uniform",
                    strategies=(AccelerationStrategy(
                        strategy_id="uniform", method="speca",
                        params={}, modeled_flops=2.0),),
                ),
                session_id="sess", epsilon=0.1, seed=42,
            ),
            state_path="/tmp/bandit.json",
        ),
        state=COVRRunState(
            safety_full_steps=2,
            sentinel_full_steps=4,
            safety_wall_times=[0.4],
            terminal_wall_times=[0.6],
            control_wall_times=[0.1],
        ),
    )
    wall_times = [1.2]
    n = 16
    cand = 5.0
    vanilla = 10.0
    full = 2.5e12

    result = runtime.online_accounting(
        wall_times=wall_times, n_images=n,
        candidate_flops_T=cand, vanilla_flops_T=vanilla,
        full_step_flops=full)

    # Flops accounting (old helper formula).
    safety_flops = 2.0 / 1 * full / 1e12
    terminal_flops = 4.0 / 1 * full / 1e12
    online_flops = cand + safety_flops + terminal_flops
    assert result["flops_candidate_T"] == pytest.approx(cand)
    assert result["flops_safety_T"] == pytest.approx(safety_flops)
    assert result["flops_terminal_T"] == pytest.approx(terminal_flops)
    assert result["flops_online_T"] == pytest.approx(online_flops)
    assert result["flops_reduction_candidate"] == pytest.approx(
        1.0 - cand / vanilla)
    assert result["flops_reduction_online"] == pytest.approx(
        1.0 - online_flops / vanilla)
    assert result["speedup_flops_candidate"] == pytest.approx(vanilla / cand)
    assert result["speedup_flops_online"] == pytest.approx(
        vanilla / online_flops)
    # Wall-time accounting (old helper formula).
    candidate = max(0.0, 1.2 - 0.4 - 0.6 - 0.1)
    assert result["wall_s_candidate_mean"] == pytest.approx(candidate)
    assert result["wall_s_candidate_total"] == pytest.approx(candidate)
    assert result["wall_s_online_mean"] == pytest.approx(1.2)
    assert result["wall_s_safety_total"] == pytest.approx(0.4)
    assert result["wall_s_terminal_total"] == pytest.approx(0.6)
    assert result["wall_s_control_total"] == pytest.approx(0.1)
    assert result["speed_candidate_img_per_s"] == pytest.approx(n / candidate)
    assert result["speed_online_img_per_s"] == pytest.approx(n / 1.2)
    assert result["safety_full_steps_mean_per_trajectory"] == pytest.approx(2.0)
    assert result["terminal_full_steps_mean_per_trajectory"] == pytest.approx(4.0)


def test_online_accounting_flops_fallbacks():
    runtime = COVRRuntime(
        build_covr_runtime_config(_runtime_args(), "/tmp/out"), version=None)
    runtime.state.safety_wall_times = [0.2]
    runtime.state.terminal_wall_times = [0.1]
    runtime.state.control_wall_times = [0.05]

    # No flops inputs: fall back to the candidate-side values.
    result = runtime.online_accounting(
        wall_times=[0.6], n_images=8,
        candidate_flops_T=4.0, vanilla_flops_T=None, full_step_flops=None)
    assert result["flops_online_T"] == pytest.approx(4.0)
    assert result["speed_online_img_per_s"] == pytest.approx(
        result["speed_candidate_img_per_s"])
    # No flops keys at all.
    assert "flops_candidate_T" not in result
    assert "flops_reduction_online" not in result
    # Empty run returns the empty dict (old helper contract).
    assert runtime.online_accounting(wall_times=[], n_images=0) == {}


def test_online_accounting_rejects_misaligned_wall_times():
    runtime = COVRRuntime(
        build_covr_runtime_config(_runtime_args(), "/tmp/out"), version=None)
    runtime.state.safety_wall_times = [0.2, 0.3]
    runtime.state.terminal_wall_times = [0.1]
    runtime.state.control_wall_times = [0.05]

    with pytest.raises(ValueError, match="must align"):
        runtime.online_accounting(wall_times=[0.6], n_images=8)


def test_generation_profile_golden_schema():
    runtime = COVRRuntime(
        build_covr_runtime_config(_runtime_args(), "/tmp/out"), version=None)
    runtime.state.profile_stage_totals = {
        "generation_online": 2.0,
        "vae_decode": 1.0,
        "cuda_sync_wait": 0.1,
        "cuda_sync_calls": 4.0,
    }
    runtime.state.profile_stage_counts = {
        "generation_online": 2,
        "vae_decode": 2,
        "cuda_sync_wait": 2,
        "cuda_sync_calls": 4,
    }

    payload = runtime.generation_profile(batches=2)

    # cuda_sync_calls is excluded from the stage buckets but kept as a count.
    assert payload["batches"] == 2
    assert payload["cuda_sync_calls"] == 4
    assert payload["stage_total_s"] == {
        "generation_online": 2.0,
        "vae_decode": 1.0,
        "cuda_sync_wait": 0.1,
    }
    assert payload["stage_mean_per_batch_s"] == {
        "generation_online": 1.0,
        "vae_decode": 0.5,
        "cuda_sync_wait": 0.05,
    }
    assert payload["stage_observed_batches"] == {
        "generation_online": 2,
        "vae_decode": 2,
        "cuda_sync_wait": 2,
    }
    # Empty run still yields the full schema with empty buckets.
    empty = COVRRuntime(
        build_covr_runtime_config(_runtime_args(), "/tmp/out"),
        version=None).generation_profile(batches=0)
    assert empty["batches"] == 0
    assert empty["stage_total_s"] == {}


def test_aggregate_bandit_summary_schema(tmp_path):
    runtime, bandit, config = _bandit_config(tmp_path, sentinel_rate=0.5)
    trajectory = runtime.begin_trajectory(
        0, sample_count=2, num_steps=4, sentinel_rate=0.5,
        sentinel_horizon=0)
    assert trajectory.sentinel_selected

    runtime.end_trajectory(
        trajectory,
        {"terminal_fidelity_loss_tensor": torch.tensor(0.25),
         "terminal_full_steps": 1},
        generation_profile={"generation_online": 1.0},
        num_steps=4, sentinel_rate=0.5, sentinel_horizon=0,
    )
    payload = runtime.aggregate(
        state_path=str(tmp_path / "bandit_state.json"),
        dataset_start_index=10, resume_sample_offset=2,
        generation_start_index=10, target_samples=16,
        generated_samples_this_run=2,
    )

    summary = payload["covr_template_bandit"]
    assert summary["completed_trajectories"] == 1
    assert summary["processed_samples"] == 2
    assert summary["state_path"] == str(tmp_path / "bandit_state.json")
    assert summary["dataset_start_index"] == 10
    assert summary["resume_sample_offset"] == 2
    assert summary["generation_start_index"] == 10
    assert summary["target_samples"] == 16
    assert summary["generated_samples_this_run"] == 2
    assert summary["safety_full_steps"] == 0
    assert summary["safety_wall_s"] == 0.0
    assert summary["sentinel_count"] == 1
    assert summary["sentinel_horizon"] == 0
    assert summary["sentinel_full_steps"] == 1
    assert summary["sentinel_wall_s"] == 0.0
    assert summary["terminal_feedback_wall_s"] == 0.0
    # Accounting flags are part of the bandit summary schema.
    assert summary["candidate_flops_exclude_safety"] is True
    assert summary["candidate_flops_exclude_sentinel"] is True
    assert summary["online_flops_include_safety"] is True
    assert summary["online_flops_include_terminal"] is True
    assert summary["online_flops_exclude_sentinel"] is True
    assert summary["online_wall_exclude_sentinel"] is True
    assert "covr_forced_template" not in payload

    # Reward telemetry from the closed trajectory: bandit mode records the
    # counters only — the mean/std keys are forced-mode telemetry (the old
    # code appended to covr_terminal_losses only under covr_forced_strategy).
    reward = payload["covr_reward_telemetry"]
    assert reward["sentinel_count"] == 1
    assert reward["sentinel_full_steps"] == 1
    assert reward["sentinel_wall_s"] == 0.0
    assert reward["sentinel_horizon"] == 0
    assert reward["sentinel_rate"] == 0.5
    assert reward["sentinel_skipped"] == 0
    assert "terminal_fidelity_loss_mean" not in reward


def test_aggregate_reward_telemetry_matches_old_local_stats(tmp_path):
    # Forced single-arm sweep: the old code collected terminal losses,
    # H-step pairs and skipped sentinels into local lists, then aggregated
    # mean/std/n. The runtime state feeds the same payload.
    manifest = _strategy_manifest(tmp_path / "manifest.json")
    strategy = manifest.strategy_map["uniform"]
    runtime = COVRRuntime.create(
        build_covr_runtime_config(
            _runtime_args(
                covr_strategy_manifest=str(tmp_path / "manifest.json"),
                covr_force_strategy_id="uniform",
                covr_sentinel_rate=1.0,
            ),
            str(tmp_path),
        ),
        backend=ForcedStrategyBackend(strategy, manifest),
        scheduler=None,
    )

    # Three trajectories: terminal reward, H-step reward, skipped sentinel.
    t0 = runtime.begin_trajectory(
        0, sample_count=2, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=0)
    assert t0.sentinel_selected
    runtime.end_trajectory(
        t0,
        {"terminal_fidelity_loss_tensor": torch.tensor(0.1),
         "terminal_full_steps": 1},
        generation_profile={"generation_online": 1.0},
        num_steps=4, sentinel_rate=1.0, sentinel_horizon=0,
    )
    h = runtime.begin_trajectory(
        1, sample_count=2, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=2)
    assert h.sentinel_selected and h.sentinel_start_idx is not None
    runtime.end_trajectory(
        h,
        {"h_step_components_tensor": torch.tensor([0.3, 0.6])},
        generation_profile={"generation_online": 1.0},
        num_steps=4, sentinel_rate=1.0, sentinel_horizon=2,
    )
    s = runtime.begin_trajectory(
        2, sample_count=2, num_steps=4, sentinel_rate=1.0,
        sentinel_horizon=0)
    assert s.sentinel_selected
    runtime.end_trajectory(
        s, {},
        generation_profile={"generation_online": 1.0},
        num_steps=4, sentinel_rate=1.0, sentinel_horizon=0,
    )

    reward = runtime.aggregate()["covr_reward_telemetry"]
    # Same statistics the old run_dit local lists produced.
    assert reward["terminal_fidelity_loss_mean"] == pytest.approx(0.1)
    assert reward["terminal_fidelity_loss_n"] == 1
    assert reward["h_step_numerator_mean"] == pytest.approx(0.3)
    assert reward["h_step_numerator_n"] == 1
    assert reward["h_step_denominator_mean"] == pytest.approx(0.6)
    assert reward["h_step_denominator_n"] == 1
    assert reward["sentinel_skipped"] == 1
    # Only the terminal trajectory reported terminal_full_steps>0; the old
    # code counted a sentinel only when terminal_full_steps was nonzero.
    assert reward["sentinel_count"] == 1
    assert reward["sentinel_full_steps"] == 1


def test_aggregate_forced_template_schema(tmp_path):
    manifest = _strategy_manifest(tmp_path / "manifest.json")
    strategy = manifest.strategy_map["uniform"]
    runtime = COVRRuntime.create(
        build_covr_runtime_config(
            _runtime_args(
                covr_strategy_manifest=str(tmp_path / "manifest.json"),
                covr_force_strategy_id="uniform",
            ),
            str(tmp_path),
        ),
        backend=ForcedStrategyBackend(strategy, manifest),
        scheduler=None,
    )
    forced_template = SimpleNamespace(
        template_id="uniform",
        refresh_count=3,
        modeled_full_block_equivalents=20.0,
    )

    payload = runtime.aggregate(
        forced_template=forced_template,
        forced_manifest=manifest,
        speca_probe_full_blocks=7,
    )

    # Bandit mode absent: no covr_template_bandit key.
    assert "covr_template_bandit" not in payload
    forced = payload["covr_forced_template"]
    assert forced["template_id"] == "uniform"
    assert forced["manifest_hash"] == manifest.manifest_hash
    assert forced["refresh_count"] == 3
    assert forced["modeled_full_block_equivalents"] == 20.0
    assert forced["probe_full_blocks"] == 7
    # Reward telemetry present in forced mode too.
    assert payload["covr_reward_telemetry"]["sentinel_rate"] == 0.05
    assert "forced_strategy_id" in payload["covr_reward_telemetry"]
    assert payload["covr_reward_telemetry"]["forced_strategy_id"] == "uniform"


def test_aggregate_disabled_mode_has_no_bandit_key(tmp_path):
    # No backend (observe-only runtime): only reward telemetry is emitted.
    runtime = COVRRuntime.create(
        build_covr_runtime_config(
            _runtime_args(covr_shadow=True), str(tmp_path)),
        scheduler=None,
    )
    payload = runtime.aggregate()
    assert "covr_template_bandit" not in payload
    assert "covr_forced_template" not in payload
    assert payload["covr_reward_telemetry"]["sentinel_rate"] == 0.05
    assert payload["covr_reward_telemetry"]["sentinel_count"] == 0
    # Reward telemetry with no samples carries no mean/std keys.
    assert "terminal_fidelity_loss_mean" not in payload["covr_reward_telemetry"]


def test_config_payload_golden_keys(tmp_path):
    runtime = COVRRuntime.create(
        build_covr_runtime_config(
            _runtime_args(covr_shadow=True, covr_profile_stages=True),
            str(tmp_path)),
        scheduler=None,
    )
    payload = runtime.config_payload(
        covr_forced_template_id="uniform",
        covr_session_id="sess-1",
        covr_version_key="abc123",
    )
    assert payload == {
        "covr_shadow": True,
        "covr_template_bandit": False,
        "covr_force_template_id": "uniform",
        "covr_session_id": "sess-1",
        "covr_version_key": "abc123",
        "covr_profile_stages": True,
        "covr_bandit_prior_penalty": 0.0,
    }

    # Forced runtime flips the bandit flag off, shadow on.
    manifest = _strategy_manifest(tmp_path / "manifest.json")
    strategy = manifest.strategy_map["uniform"]
    forced = COVRRuntime.create(
        build_covr_runtime_config(
            _runtime_args(
                covr_strategy_manifest=str(tmp_path / "manifest.json"),
                covr_force_strategy_id="uniform",
            ),
            str(tmp_path),
        ),
        backend=ForcedStrategyBackend(strategy, manifest),
        scheduler=None,
    )
    assert forced.config_payload(covr_forced_template_id=None)["covr_shadow"] \
        is False
    assert forced.config_payload(
        covr_forced_template_id="uniform")["covr_force_template_id"] == "uniform"


def test_close_returns_recorder_summary_then_none(tmp_path):
    runtime = COVRRuntime.create(
        build_covr_runtime_config(
            _runtime_args(covr_shadow=True), str(tmp_path)),
        scheduler=None,
    )
    assert runtime.recorder is not None
    summary = runtime.close()
    assert summary is not None
    assert "events" in summary and "samples" in summary
    # Second close is a no-op.
    assert runtime.close() is None
