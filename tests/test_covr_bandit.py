import json
import math
import subprocess
import sys

import pytest

from accelerators.covr import (
    ActionAuditContext,
    ActionAuditEvent,
    COVRAction,
    TransitionDefectBatch,
)
from accelerators.covr_bandit import (
    ConservativeTemplateBandit,
    RefreshTemplate,
    TemplateFeedback,
    TemplateManifest,
    TimestepSafetyPrior,
)
from experiments.covr_analysis import build_template_manifest


def _manifest():
    priors = tuple(
        TimestepSafetyPrior(
            step_idx=step,
            log_numerator_mean=-4.0,
            log_numerator_std=0.1,
            log_denominator_mean=0.0,
            log_denominator_std=0.1,
            sample_count=5,
        )
        for step in range(8)
    )
    return TemplateManifest(
        version_key="version",
        num_steps=8,
        num_layers=2,
        mandatory_prefix=2,
        max_taylor_gap=3,
        baseline_template_id="baseline",
        templates=(
            RefreshTemplate(
                "baseline", (True, True, False, True, False, True, False, False), 8),
            RefreshTemplate(
                "alternative", (True, True, True, False, True, False, False, False), 8),
        ),
        timestep_priors=priors,
        safety_numerator_ucb_limit=1.0,
        safety_denominator_lcb_floor=0.1,
    )


def test_manifest_rejects_unequal_refresh_costs():
    with pytest.raises(ValueError, match="equal FLOPs"):
        TemplateManifest(
            version_key="version",
            num_steps=8,
            num_layers=2,
            mandatory_prefix=2,
            max_taylor_gap=3,
            baseline_template_id="baseline",
            templates=(
                RefreshTemplate("baseline", (True,) * 8, 16),
                RefreshTemplate("alternative", (True,) * 7 + (False,), 14),
            ),
        )


def test_bandit_updates_safety_and_reward_only_at_trajectory_boundary():
    bandit = ConservativeTemplateBandit(_manifest(), "session", epsilon=0.0)
    assignment = bandit.begin_trajectory(0)
    assert assignment.template_id == "baseline"
    assert bandit.summary()["delayed_feedback"] == 0

    bandit.observe_one_step(0, 2, (0.1,), (1.0,))
    assert bandit.summary()["delayed_feedback"] == 0
    with pytest.raises(RuntimeError):
        bandit.state_dict()

    feedback = TemplateFeedback(
        trajectory_id=0,
        template_id=assignment.template_id,
        sentinel_propensity=1.0,
        horizon=8,
        terminal_fidelity_loss=0.2,
    )
    bandit.end_trajectory(0, feedback)
    assert bandit.summary()["delayed_feedback"] == 1
    assert bandit.safety._by_template_step[("baseline", 2)].numerator.count == 1


def test_bandit_does_not_accept_cross_trajectory_feedback():
    bandit = ConservativeTemplateBandit(_manifest(), "session", epsilon=0.0)
    assignment = bandit.begin_trajectory(3)
    with pytest.raises(ValueError):
        bandit.end_trajectory(
            3,
            TemplateFeedback(
                trajectory_id=3,
                template_id="alternative",
                sentinel_propensity=1.0,
                horizon=8,
                h_step_numerator=0.1,
                h_step_denominator=1.0,
            ),
        )
    bandit.end_trajectory(3)
    assert assignment.trajectory_id == 3


def test_bandit_state_is_pinned_to_session_and_manifest(tmp_path):
    manifest = _manifest()
    bandit = ConservativeTemplateBandit(manifest, "session", epsilon=0.0)
    bandit.begin_trajectory(0)
    bandit.end_trajectory(0)
    path = tmp_path / "state.json"
    bandit.save_state(str(path))

    restored = ConservativeTemplateBandit(manifest, "session", epsilon=0.0)
    restored.load_state(str(path))
    assert restored.summary() == bandit.summary()

    wrong_session = ConservativeTemplateBandit(manifest, "other", epsilon=0.0)
    with pytest.raises(ValueError, match="identity"):
        wrong_session.load_state(str(path))


def _audit_context(step_idx):
    return ActionAuditContext(
        step_idx=step_idx,
        num_steps=8,
        timestep=float(800 - step_idx),
        log_snr=-1.0 + step_idx * 0.1,
        alpha_t=0.8,
        alpha_prev=0.7,
        latent_coefficient=0.8,
        model_output_coefficient=-0.1,
        distance_since_refresh=step_idx,
    )


def _audit_events():
    masks = (
        (True, True, False, True, False, True, False, False),
        (True, True, True, False, True, False, False, False),
    )
    events = []
    for trajectory_id, mask in enumerate(masks):
        for step_idx, refresh in enumerate(mask):
            if refresh:
                continue
            events.append(ActionAuditEvent(
                session_id="train",
                trajectory_id=trajectory_id,
                sample_ids=(f"sample-{trajectory_id}",),
                class_ids=(trajectory_id,),
                version_key="version",
                context=_audit_context(step_idx),
                committed_action=COVRAction.ACCEPT,
                committed_propensity=1.0,
                audit_action=COVRAction.REFRESH,
                audit_propensity=1.0,
                policy="shadow_static_speca",
                incremental_cost=1.0,
                one_step_transition=TransitionDefectBatch(
                    numerators=(0.1 + step_idx * 0.01,),
                    denominators=(1.0,),
                    ratios=(0.1 + step_idx * 0.01,),
                ),
            ))
    return events


def test_manifest_builder_keeps_trajectory_groups_and_equal_costs():
    manifest = build_template_manifest(
        _audit_events(), num_layers=2, template_count=3,
        mandatory_prefix=2, max_taylor_gap=3,
    )
    assert manifest.common_refresh_count == 4
    assert len(manifest.templates) == 3
    assert {template.modeled_full_block_equivalents for template in manifest.templates} == {8}
    assert manifest.source_groups == ("train:0", "train:1")
    assert manifest.baseline_template_id == "timestep_prior"
    assert all(math.isfinite(prior.log_numerator_mean)
               for prior in manifest.timestep_priors)


def test_manifest_builder_repairs_missing_mandatory_prefix():
    manifest = build_template_manifest(
        _audit_events(), num_layers=2, template_count=3,
        mandatory_prefix=3, max_taylor_gap=3,
    )

    assert manifest.common_refresh_count == 4
    assert all(all(template.refresh_mask[:3])
               for template in manifest.templates)
    assert all(sum(template.refresh_mask) == 4
               for template in manifest.templates)


def test_state_tracks_samples_and_pins_run_identity(tmp_path):
    manifest = _manifest()
    identity = {"dataset": "imagenet", "seed": 42, "batch_size": 2}
    bandit = ConservativeTemplateBandit(
        manifest, "session", epsilon=0.0, run_identity=identity)
    bandit.begin_trajectory(0, sample_count=2)
    bandit.end_trajectory(0)
    path = tmp_path / "state.json"
    bandit.save_state(str(path))

    restored = ConservativeTemplateBandit(
        manifest, "session", epsilon=0.0, run_identity=identity)
    restored.load_state(str(path))
    assert restored.summary()["processed_samples"] == 2
    assert restored.assignments[0].sample_count == 2

    wrong_identity = ConservativeTemplateBandit(
        manifest, "session", epsilon=0.0,
        run_identity={"dataset": "imagenet", "seed": 7, "batch_size": 2})
    with pytest.raises(ValueError, match="run identity"):
        wrong_identity.load_state(str(path))


def test_manifest_cli_writes_loadable_manifest(tmp_path):
    audits = tmp_path / "audits.jsonl"
    with audits.open("w", encoding="utf-8") as handle:
        for event in _audit_events():
            handle.write(json.dumps(event.to_dict()) + "\n")
    output = tmp_path / "manifest.json"

    completed = subprocess.run(
        [
            sys.executable, "scripts/build_covr_manifest.py", str(audits),
            "--output", str(output), "--num-layers", "2",
            "--template-count", "3", "--mandatory-prefix", "2",
            "--max-taylor-gap", "3",
        ],
        check=True, capture_output=True, text=True,
    )
    manifest = TemplateManifest.load(str(output))
    summary = json.loads(completed.stdout)
    assert summary["manifest_hash"] == manifest.manifest_hash
    assert summary["refresh_count"] == 4
    assert len(manifest.templates) == 3
