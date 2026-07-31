"""Tests for AccelerationStrategy, StrategyManifest, and method-agnostic bandit."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pytest

from accelerators.covr_bandit import (
    AccelerationStrategy,
    ConservativeTemplateBandit,
    RefreshTemplate,
    StrategyManifest,
    TemplateAssignment,
    TemplateFeedback,
    TemplateManifest,
)


# ===========================================================================
# AccelerationStrategy unit tests
# ===========================================================================


def test_strategy_invalid_method():
    with pytest.raises(ValueError, match="unsupported acceleration method"):
        AccelerationStrategy(
            strategy_id="invalid",
            method="nonexistent",
            params={},
            modeled_flops=1.0,
        )


def test_strategy_empty_id():
    with pytest.raises(ValueError, match="strategy_id must be non-empty"):
        AccelerationStrategy(
            strategy_id="",
            method="speca",
            params={"refresh_mask": []},
            modeled_flops=1.0,
        )


def test_speca_strategy_exposes_refresh_mask():
    strategy = AccelerationStrategy(
        strategy_id="t1",
        method="speca",
        params={"refresh_mask": [True, False, True]},
        modeled_flops=3.0,
    )
    assert strategy.refresh_mask == (True, False, True)
    assert strategy.refresh_count == 2
    assert strategy.template_id == "t1"


def test_teacache_strategy_has_no_refresh_mask():
    strategy = AccelerationStrategy(
        strategy_id="gamma_0.25",
        method="teacache",
        params={"rel_l1_thresh": 0.25, "num_steps": 50},
        modeled_flops=50.0,
    )
    assert strategy.refresh_mask is None
    assert strategy.refresh_count == 0
    assert strategy.params["rel_l1_thresh"] == 0.25


def test_teacache_strategy_template_id_is_strategy_id():
    strategy = AccelerationStrategy(
        strategy_id="thresh_0.1500",
        method="teacache",
        params={"rel_l1_thresh": 0.15},
        modeled_flops=1.0,
    )
    assert strategy.template_id == "thresh_0.1500"


def test_teacache_mask_strategy_exposes_refresh_mask():
    """Forced-schedule TeaCache arm carries a refresh_mask (equal-FLOPs)."""
    strategy = AccelerationStrategy(
        strategy_id="template_01",
        method="teacache",
        params={"refresh_mask": [True, False, True, False], "num_steps": 4},
        modeled_flops=2.0,
    )
    assert strategy.refresh_mask == (True, False, True, False)
    assert strategy.refresh_count == 2
    # to_refresh_template still refuses non-SpecA methods.
    with pytest.raises(RuntimeError, match="cannot convert teacache"):
        strategy.to_refresh_template()


def test_teacache_threshold_strategy_has_no_mask():
    """A threshold-based TeaCache arm (no mask) yields refresh_mask=None."""
    strategy = AccelerationStrategy(
        strategy_id="thresh_0.25",
        method="teacache",
        params={"rel_l1_thresh": 0.25},
        modeled_flops=1.0,
    )
    assert strategy.refresh_mask is None
    assert strategy.refresh_count == 0


# ===========================================================================
# apply_strategy dispatch (method-agnostic)
# ===========================================================================


_TC_COEF = [0.0, 0.0, 0.0, 1.0, 0.0]


def test_apply_strategy_injects_teacache_refresh_mask():
    from accelerators.strategy_dispatch import apply_strategy

    mask = [True, False, True, False, False]
    strategy = AccelerationStrategy(
        strategy_id="template_01",
        method="teacache",
        params={"refresh_mask": mask, "num_steps": 5},
        modeled_flops=2.0,
    )
    result = apply_strategy(
        strategy,
        teacache_init_kwargs={"num_steps": 5, "coefficients": _TC_COEF},
    )
    state = result["teacache_state"]
    assert state["refresh_mask"] == tuple(mask)


def test_apply_strategy_injects_teacache_threshold_when_no_mask():
    from accelerators.strategy_dispatch import apply_strategy

    strategy = AccelerationStrategy(
        strategy_id="thresh_0.35",
        method="teacache",
        params={"rel_l1_thresh": 0.35, "num_steps": 5},
        modeled_flops=1.0,
    )
    result = apply_strategy(
        strategy,
        teacache_init_kwargs={"num_steps": 5, "coefficients": _TC_COEF},
    )
    state = result["teacache_state"]
    assert state["refresh_mask"] is None
    assert state["rel_l1_thresh"] == 0.35


def test_speca_strategy_to_refresh_template():
    strategy = AccelerationStrategy(
        strategy_id="t1",
        method="speca",
        params={"refresh_mask": [True, False, True, False]},
        modeled_flops=4.0,
    )
    rt = strategy.to_refresh_template()
    assert isinstance(rt, RefreshTemplate)
    assert rt.template_id == "t1"
    assert rt.refresh_mask == (True, False, True, False)
    assert rt.modeled_full_block_equivalents == 4


def test_teacache_strategy_to_refresh_template_raises():
    strategy = AccelerationStrategy(
        strategy_id="thresh_0.25",
        method="teacache",
        params={"rel_l1_thresh": 0.25},
        modeled_flops=1.0,
    )
    with pytest.raises(RuntimeError, match="cannot convert teacache"):
        strategy.to_refresh_template()


def test_refresh_template_to_strategy():
    template = RefreshTemplate(
        "my_template", (True, False, True, False), 8)
    strategy = template.to_strategy(num_steps=4)
    assert isinstance(strategy, AccelerationStrategy)
    assert strategy.strategy_id == "my_template"
    assert strategy.method == "speca"
    assert strategy.refresh_mask == (True, False, True, False)
    assert strategy.modeled_flops == 8.0


# ===========================================================================
# StrategyManifest tests
# ===========================================================================


def _teacache_manifest() -> StrategyManifest:
    return StrategyManifest(
        version_key="teacache_v50",
        num_steps=50,
        baseline_strategy_id="baseline",
        strategies=(
            AccelerationStrategy(
                "baseline", "teacache",
                {"rel_l1_thresh": 0.25, "num_steps": 50}, 50.0),
            AccelerationStrategy(
                "thresh_0.15", "teacache",
                {"rel_l1_thresh": 0.15, "num_steps": 50}, 50.0),
            AccelerationStrategy(
                "thresh_0.35", "teacache",
                {"rel_l1_thresh": 0.35, "num_steps": 50}, 50.0),
        ),
    )


def test_strategy_manifest_validates():
    manifest = _teacache_manifest()
    assert manifest.num_steps == 50
    assert len(manifest.strategies) == 3
    assert manifest.baseline_strategy_id == "baseline"
    assert manifest.baseline_template_id == "baseline"


def test_strategy_manifest_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="strategy IDs must be unique"):
        StrategyManifest(
            version_key="v1",
            num_steps=50,
            baseline_strategy_id="a",
            strategies=(
                AccelerationStrategy("a", "teacache", {"rel_l1_thresh": 0.25}, 1.0),
                AccelerationStrategy("a", "teacache", {"rel_l1_thresh": 0.35}, 1.0),
            ),
        )


def test_strategy_manifest_rejects_missing_baseline():
    with pytest.raises(ValueError, match="baseline strategy is missing"):
        StrategyManifest(
            version_key="v1",
            num_steps=50,
            baseline_strategy_id="nonexistent",
            strategies=(
                AccelerationStrategy("a", "teacache", {"rel_l1_thresh": 0.25}, 1.0),
            ),
        )


def test_strategy_manifest_to_dict_roundtrip():
    manifest = _teacache_manifest()
    payload = manifest.to_dict()
    restored = StrategyManifest.from_dict(payload)
    assert restored.version_key == manifest.version_key
    assert restored.num_steps == manifest.num_steps
    assert restored.baseline_strategy_id == manifest.baseline_strategy_id
    assert len(restored.strategies) == len(manifest.strategies)
    for orig, rest in zip(manifest.strategies, restored.strategies):
        assert orig.strategy_id == rest.strategy_id
        assert orig.method == rest.method
        assert orig.params == rest.params
        assert orig.modeled_flops == rest.modeled_flops


def test_strategy_manifest_save_load(tmp_path):
    manifest = _teacache_manifest()
    path = tmp_path / "manifest.json"
    manifest.save(str(path))
    loaded = StrategyManifest.load(str(path))
    assert loaded.manifest_hash == manifest.manifest_hash
    assert loaded.num_steps == manifest.num_steps


def test_strategy_manifest_duck_type_properties():
    manifest = _teacache_manifest()
    # Compat with TimestepSafetyTable / bandit
    assert manifest.baseline_template_id == "baseline"
    assert manifest.templates == manifest.strategies
    assert isinstance(manifest.template_map, dict)
    assert "baseline" in manifest.template_map
    assert manifest.prior_map == {}
    assert isinstance(manifest.safety_numerator_ucb_limit, float)
    assert manifest.safety_denominator_lcb_floor == 0.0
    assert isinstance(manifest.manifest_hash, str)
    assert len(manifest.manifest_hash) > 0


# ===========================================================================
# ConservativeTemplateBandit + AccelerationStrategy integration
# ===========================================================================


def test_bandit_from_strategies_accepts_teacache():
    manifest = _teacache_manifest()
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, "session", epsilon=0.0)
    assert len(bandit._strategies) == 3
    assert bandit.manifest is manifest


def test_bandit_from_strategies_begin_and_end_trajectory():
    manifest = _teacache_manifest()
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, "session", epsilon=0.0)

    assignment = bandit.begin_trajectory(0)
    assert assignment.template_id in ("baseline", "thresh_0.15", "thresh_0.35")

    # active_strategy works
    strategy = bandit.active_strategy
    assert isinstance(strategy, AccelerationStrategy)
    assert strategy.strategy_id == assignment.template_id
    assert strategy.method == "teacache"

    # active_template raises for non-SpecA
    with pytest.raises(RuntimeError, match="cannot convert teacache"):
        bandit.active_template

    # End trajectory with terminal fidelity reward
    feedback = TemplateFeedback(
        trajectory_id=0,
        template_id=assignment.template_id,
        sentinel_propensity=1.0,
        horizon=50,
        terminal_fidelity_loss=0.15,
    )
    bandit.end_trajectory(0, feedback)

    assert bandit.summary()["delayed_feedback"] == 1
    assert bandit.summary()["completed_trajectories"] == 1


def test_bandit_from_strategies_ucb_chooses_lowest_loss():
    """With epsilon=0, the bandit always picks the arm with lowest mean loss."""
    manifest = _teacache_manifest()
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, "session", epsilon=0.0,
        baseline_prior_count=8,
        alternative_prior_penalty=0.25,
    )

    # Baseline has prior count=8, mean=0 (lower)
    # Alternatives have prior count=1, mean=log1p(0.25) ≈ 0.22
    # So baseline should be selected
    assignment = bandit.begin_trajectory(0)
    assert assignment.template_id == "baseline"

    # After collecting feedback with high loss for baseline,
    # the bandit should eventually explore alternatives
    # (but with epsilon=0, it won't — verifying the UCB bound)
    feedback = TemplateFeedback(
        trajectory_id=0,
        template_id="baseline",
        sentinel_propensity=1.0,
        horizon=50,
        terminal_fidelity_loss=1.0,
    )
    bandit.end_trajectory(0, feedback)

    # Baseline should now have a higher mean
    assert bandit._arm_stats["baseline"].mean > 0.0


def test_bandit_from_strategies_state_save_load(tmp_path):
    manifest = _teacache_manifest()
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, "session", epsilon=0.0)
    bandit.begin_trajectory(0)
    feedback = TemplateFeedback(
        trajectory_id=0,
        template_id="baseline",
        sentinel_propensity=1.0,
        horizon=50,
        terminal_fidelity_loss=0.1,
    )
    bandit.end_trajectory(0, feedback)

    path = tmp_path / "state.json"
    bandit.save_state(str(path))

    restored = ConservativeTemplateBandit.from_strategies(
        manifest, "session", epsilon=0.0)
    restored.load_state(str(path))
    assert restored.summary() == bandit.summary()


def test_bandit_from_strategies_state_rejects_wrong_session(tmp_path):
    manifest = _teacache_manifest()
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, "session_a", epsilon=0.0)
    bandit.begin_trajectory(0)
    bandit.end_trajectory(0)
    path = tmp_path / "state.json"
    bandit.save_state(str(path))

    restored = ConservativeTemplateBandit.from_strategies(
        manifest, "session_b", epsilon=0.0)
    with pytest.raises(ValueError, match="identity"):
        restored.load_state(str(path))


def test_bandit_from_strategies_summary_method_agnostic():
    manifest = _teacache_manifest()
    bandit = ConservativeTemplateBandit.from_strategies(
        manifest, "session", epsilon=0.3)
    bandit.begin_trajectory(0, sample_count=2)

    strategy = bandit.active_strategy
    assert strategy.method == "teacache"

    feedback = TemplateFeedback(
        trajectory_id=0,
        template_id=strategy.strategy_id,
        sentinel_propensity=1.0,
        horizon=50,
        terminal_fidelity_loss=0.2,
    )
    bandit.end_trajectory(0, feedback)

    summary = bandit.summary()
    assert summary["assignments"] == 1
    assert summary["processed_samples"] == 2
    assert summary["delayed_feedback"] == 1
    assert "common_refresh_count" not in summary
    assert "common_full_block_equivalents" not in summary
    assert "arm_log1p_loss_mean" in summary
    assert set(summary["arm_log1p_loss_mean"]) == {
        "baseline", "thresh_0.15", "thresh_0.35"}


# ===========================================================================
# Backward compatibility: TemplateManifest bandit unchanged
# ===========================================================================


def _speca_manifest():
    from accelerators.covr_bandit import TimestepSafetyPrior
    priors = tuple(
        TimestepSafetyPrior(
            step_idx=step, log_numerator_mean=-4.0, log_numerator_std=0.1,
            log_denominator_mean=0.0, log_denominator_std=0.1, sample_count=5,
        )
        for step in range(8)
    )
    return TemplateManifest(
        version_key="version",
        num_steps=8, num_layers=2, mandatory_prefix=2, max_taylor_gap=3,
        baseline_template_id="baseline",
        templates=(
            RefreshTemplate("baseline", (True, True, False, True, False, True, False, False), 8),
            RefreshTemplate("alternative", (True, True, True, False, True, False, False, False), 8),
        ),
        timestep_priors=priors,
    )


def test_old_bandit_still_works_with_template_manifest():
    """Verify that the original TemplateManifest codepath is unchanged."""
    manifest = _speca_manifest()
    bandit = ConservativeTemplateBandit(manifest, "session", epsilon=0.0)
    assignment = bandit.begin_trajectory(0)
    assert assignment.template_id == "baseline"

    # active_template returns RefreshTemplate (backward compat)
    template = bandit.active_template
    assert isinstance(template, RefreshTemplate)
    assert template.template_id == "baseline"

    # active_strategy also works
    strategy = bandit.active_strategy
    assert isinstance(strategy, AccelerationStrategy)
    assert strategy.method == "speca"
    assert strategy.refresh_mask == template.refresh_mask

    feedback = TemplateFeedback(
        trajectory_id=0,
        template_id=assignment.template_id,
        sentinel_propensity=1.0,
        horizon=8,
        terminal_fidelity_loss=0.2,
    )
    bandit.end_trajectory(0, feedback)
    assert bandit.summary()["delayed_feedback"] == 1
    assert bandit.summary()["common_refresh_count"] == 4

    # State-save-load still works
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        pass
    try:
        bandit.save_state(f.name)
        restored = ConservativeTemplateBandit(manifest, "session", epsilon=0.0)
        restored.load_state(f.name)
        assert restored.summary()["common_refresh_count"] == 4
    finally:
        import os
        os.unlink(f.name)
