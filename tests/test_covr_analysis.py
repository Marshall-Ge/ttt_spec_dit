from dataclasses import replace

import numpy as np
import pytest

from accelerators.covr import (
    ActionAuditContext,
    ActionAuditEvent,
    COVRAction,
    COVRContext,
    CounterfactualEvent,
    TransitionDefectBatch,
)
from experiments.covr_analysis import (
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_action_aligned_timestep,
    evaluate_action_audit_integrity,
    evaluate_denominator_tail,
    evaluate_gate_c,
    run_all_gates,
    run_phase0_analysis,
)


def _events(count=350, seed=4):
    rng = np.random.default_rng(seed)
    events = []
    for index in range(count):
        signal = float(rng.uniform(0.0, 1.0))
        target = 1.5 * signal + float(rng.normal(0.0, 0.03))
        local = float(rng.uniform(0.0, 1.0))
        context = COVRContext(
            step_idx=index % 50,
            num_steps=50,
            timestep=999 - index % 50,
            distance_since_refresh=index % 5,
            taylor_term_norms=(signal, signal ** 2, 0.0, 0.0),
            previous_defect=float(rng.uniform(0.0, 0.1)),
        )
        events.append(CounterfactualEvent(
            session_id="session",
            trajectory_id=index // 5,
            sample_id=str(index),
            version_key="version",
            context=context,
            action=COVRAction.REFRESH,
            propensity=1.0,
            policy="shadow",
            incremental_cost=1.0,
            one_step_defect=max(target, 0.0),
            local_probe_error=local,
            h_step_defect=max(target + float(rng.normal(0.0, 0.02)), 0.0),
            terminal_intervention_gain=max(
                target + float(rng.normal(0.0, 0.02)), 0.0),
            metadata={"class_id": index % 100},
        ))
    return events


def test_gate_a_accepts_valid_whole_step_signal():
    result = evaluate_gate_a(_events(), bootstrap_samples=100)
    assert result.status == "pass"
    assert result.metrics["selected_metrics"]["spearman"] > 0.9


def test_gate_b_finds_oracle_headroom():
    result = evaluate_gate_b(_events(), random_trials=30)
    assert result.status == "pass"
    assert result.metrics["oracle_wins"] >= result.metrics["required_wins"]


def test_gate_c_requires_context_beyond_timestep():
    result = evaluate_gate_c(_events())
    assert result.status == "pass"
    aggregate = result.metrics["aggregate"]
    assert aggregate["full_context"]["mse"] < aggregate["timestep_only"]["mse"]


def test_gate_order_stops_at_missing_terminal_labels():
    events = [
        CounterfactualEvent(
            session_id="session",
            trajectory_id=0,
            sample_id=str(index),
            version_key="version",
            context=COVRContext(index % 50, 50, 999 - index % 50),
            action=COVRAction.REFRESH,
            propensity=1.0,
            policy="shadow",
            incremental_cost=1.0,
            one_step_defect=0.1,
        )
        for index in range(220)
    ]
    result = run_all_gates(events)
    assert result["decision"] == "stop"
    assert result["stopped_at"] == "A"
    assert result["gates"][0]["status"] == "insufficient_data"


def _audit_context(step_idx):
    return ActionAuditContext(
        step_idx=step_idx,
        num_steps=2,
        timestep=2 - step_idx,
        log_snr=2.0 if step_idx == 0 else -2.0,
        alpha_t=0.8,
        alpha_prev=0.6,
        latent_coefficient=0.866,
        model_output_coefficient=-0.1,
        distance_since_refresh=step_idx,
    )


def _audit_events(final_numerator=1.0, final_denominator=0.25,
                  session_count=2):
    events = []
    for trajectory_id in range(2):
        session_id = f"session-{trajectory_id % session_count}"
        sample_ids = (f"sample-{trajectory_id}-0", f"sample-{trajectory_id}-1")
        class_ids = (trajectory_id, trajectory_id + 10)
        for step_idx, numerator, denominator in (
            (0, 1.0, 1.0),
            (1, final_numerator, final_denominator),
        ):
            transition = TransitionDefectBatch(
                numerators=(numerator, numerator),
                denominators=(denominator, denominator),
                ratios=(numerator / denominator, numerator / denominator),
            )
            events.append(ActionAuditEvent(
                session_id=session_id,
                trajectory_id=trajectory_id,
                sample_ids=sample_ids,
                class_ids=class_ids,
                version_key="version",
                context=_audit_context(step_idx),
                committed_action=COVRAction.ACCEPT,
                committed_propensity=1.0,
                audit_action=COVRAction.REFRESH,
                audit_propensity=1.0,
                policy="shadow_static_speca",
                incremental_cost=1.0,
                one_step_transition=transition,
            ))
    return events


def test_integrity_rejects_duplicate_batch_context():
    events = _audit_events()
    result = evaluate_action_audit_integrity(events + [events[0]])
    assert result.status == "stop"
    assert result.metrics["duplicate_contexts"] == 1


def test_integrity_accepts_selective_timestep_feedback_propensity():
    events = _audit_events()
    events[0] = replace(
        events[0],
        policy="timestep_feedback_shadow",
        audit_propensity=0.02,
    )
    result = evaluate_action_audit_integrity(events)
    assert result.status == "pass"
    assert result.metrics["propensity_violations"] == 0


def test_denominator_tail_distinguishes_numerator_spike_and_collapse():
    denominator = evaluate_denominator_tail(_audit_events(1.0, 0.25))
    assert denominator.status == "stop"
    assert denominator.metrics["delta_log_numerator"] == pytest.approx(0.0)
    assert denominator.metrics["denominator_share"] == pytest.approx(1.0)
    numerator = evaluate_denominator_tail(_audit_events(4.0, 1.0))
    assert numerator.status == "pass"
    assert numerator.metrics["delta_log_ratio"] == pytest.approx(np.log(4.0))
    assert numerator.metrics["denominator_share"] == pytest.approx(0.0)


def test_action_aligned_timestep_uses_session_held_out_folds_and_trajectory_curves():
    result = evaluate_action_aligned_timestep(_audit_events())
    assert result.status == "pass"
    assert result.metrics["fold_unit"] == "session"
    assert len(result.metrics["folds"]) == 2
    curve = result.metrics["allocation"]["curve"]
    assert curve[0]["oracle_residual_defect"] <= curve[0]["one_hot_residual_defect"]


def test_action_aligned_timestep_uses_trajectory_folds_without_session_fallback():
    result = evaluate_action_aligned_timestep(_audit_events(session_count=1))
    assert result.status == "pass"
    assert result.metrics["fold_unit"] == "trajectory"
    assert result.metrics["groups"] == 2


def test_phase0_stops_on_denominator_diagnostic():
    result = run_phase0_analysis(_audit_events())
    assert result["schema_version"] == 2
    assert result["decision"] == "stop"
    assert result["stopped_at"] == "denominator_tail"
