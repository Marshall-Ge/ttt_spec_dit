import json

import numpy as np
import pytest
import torch

from accelerators.covr import (
    FEATURE_NAMES,
    ActionAuditContext,
    ActionAuditEvent,
    ActionAuditRecorder,
    BudgetLedger,
    COVRAction,
    COVRContext,
    COVRDecision,
    COVRPolicy,
    COVRVersion,
    CounterfactualEvent,
    OnlineRidgeUCB,
    PrimalDualBudget,
    ShadowAuditRecorder,
    TransitionDefectBatch,
    load_policy_state,
    normalized_transition_defect,
    read_action_audits,
    save_policy_state,
    transition_defect_batch,
    transition_defects,
)


def _version(cfg_scale=4.5):
    return COVRVersion(
        model="dit",
        base_model_version="test",
        scheduler="FakeScheduler",
        scheduler_config="{}",
        num_steps=50,
        cfg_scale=cfg_scale,
        speca_config="{}",
    )


def _context():
    return COVRContext(
        step_idx=10,
        num_steps=50,
        timestep=800,
        log_snr=float("nan"),
        distance_since_refresh=3,
        taylor_term_norms=(1.0, 2.0, 3.0, 4.0),
        remaining_budget=0.4,
    )


def test_context_has_fixed_finite_feature_schema():
    features = _context().features()
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert features[0] == 1.0


def test_transition_defect_is_per_sample_and_normalized():
    current = torch.zeros(2, 1, 1, 2)
    full = torch.tensor([[[[2.0, 0.0]]], [[[4.0, 0.0]]]])
    approx = torch.tensor([[[[3.0, 0.0]]], [[[6.0, 0.0]]]])
    defects = transition_defects(approx, full, current)
    assert defects == pytest.approx((0.5, 0.5))
    assert normalized_transition_defect(approx, full, current) == pytest.approx(0.5)


def _audit_context(step_idx):
    return ActionAuditContext(
        step_idx=step_idx,
        num_steps=3,
        timestep=3 - step_idx,
        log_snr=float(step_idx),
        alpha_t=0.8,
        alpha_prev=0.6,
        latent_coefficient=0.8660254,
        model_output_coefficient=-0.1,
        distance_since_refresh=step_idx,
        previous_defect_mean=0.25,
    )


def _audit_event(recorder, step_idx, ratio):
    transition = TransitionDefectBatch(
        numerators=(ratio, ratio * 2.0),
        denominators=(1.0, 2.0),
        ratios=(ratio, ratio),
    )
    return ActionAuditEvent(
        session_id=recorder.session_id,
        trajectory_id=0,
        sample_ids=("sample-0", "sample-1"),
        class_ids=(3, 4),
        version_key=recorder.version.key,
        context=_audit_context(step_idx),
        committed_action=COVRAction.ACCEPT,
        committed_propensity=1.0,
        audit_action=COVRAction.REFRESH,
        audit_propensity=1.0,
        policy="shadow_static_speca",
        incremental_cost=1.0,
        one_step_transition=transition,
    )


def test_transition_defect_batch_records_rms_components_and_validates_shapes():
    current = torch.zeros(2, 1, 1, 2)
    full = torch.tensor([[[[2.0, 0.0]]], [[[4.0, 0.0]]]])
    approx = torch.tensor([[[[3.0, 0.0]]], [[[6.0, 0.0]]]])
    transition = transition_defect_batch(approx, full, current)
    assert transition.numerators == pytest.approx((2 ** -0.5, 2 ** 0.5))
    assert transition.denominators == pytest.approx((2 ** 0.5, 2 ** 1.5))
    assert transition.ratios == pytest.approx((0.5, 0.5))
    with pytest.raises(ValueError, match="identical shapes"):
        transition_defect_batch(approx[:1], full, current)


def test_action_audit_round_trip_and_batch_mean_history(tmp_path):
    recorder = ActionAuditRecorder(str(tmp_path), "session", _version(), max_events=2)
    first = _audit_event(recorder, 0, 0.25)
    second = _audit_event(recorder, 1, 0.5)
    assert recorder.record(first)
    assert recorder.previous_defect == pytest.approx(0.25)
    assert recorder.record(second)
    summary = recorder.close()
    assert summary["schema_version"] == 2
    assert summary["event_type"] == "batch_step_audit"
    assert summary["events"] == 2
    assert summary["samples"] == 4
    assert summary["mean_one_step_defect"] == pytest.approx(0.375)
    restored = read_action_audits([str(recorder.event_path)])
    assert len(restored) == 2
    assert restored[0].sample_ids == ("sample-0", "sample-1")
    assert restored[1].context.previous_defect_mean == pytest.approx(0.25)


def test_policy_records_propensity_and_conserves_budget():
    model = OnlineRidgeUCB(len(FEATURE_NAMES), beta=0.0)
    budget = BudgetLedger(1.0)
    dual = PrimalDualBudget(target_cost=0.2, initial_price=1.0)
    policy = COVRPolicy(model, budget, dual, p_min=0.2, seed=0)

    accepted = policy.decide(_context(), incremental_cost=0.6)
    assert accepted.action == COVRAction.ACCEPT
    assert accepted.propensity == pytest.approx(0.8)
    policy.commit(accepted)
    assert budget.spent == 0.0

    refreshed = policy.decide(_context(), incremental_cost=0.6, force_refresh=True)
    assert refreshed.action == COVRAction.REFRESH
    assert refreshed.propensity == 1.0
    policy.commit(refreshed)
    assert budget.spent == pytest.approx(0.6)

    blocked = policy.decide(_context(), incremental_cost=0.6, force_refresh=True)
    assert blocked.action == COVRAction.ACCEPT
    assert blocked.reason == "budget_exhausted"
    with pytest.raises(RuntimeError):
        budget.spend(0.5, "overflow")


def test_policy_uses_clipped_stabilized_ipw():
    model = OnlineRidgeUCB(len(FEATURE_NAMES), beta=0.0)
    policy = COVRPolicy(
        model,
        BudgetLedger(10.0),
        PrimalDualBudget(target_cost=0.1, initial_price=1.0),
        p_min=0.2,
        ipw_clip=10.0,
        stabilized_numerator=0.2,
    )
    decision = COVRDecision(
        COVRAction.REFRESH, propensity=0.01, refresh_probability=0.01,
        reason="exploration", risk_hat=0.0, risk_ucb=0.0,
        incremental_cost=1.0,
    )
    weight = policy.observe(_context(), decision, defect=0.4)
    assert weight == 10.0
    assert model.observations == 1


def test_recorder_rejects_tensor_payload_and_round_trips(tmp_path):
    recorder = ShadowAuditRecorder(
        str(tmp_path), "session", _version(), max_events=1)
    bad = CounterfactualEvent(
        session_id="session",
        trajectory_id=0,
        sample_id="sample",
        version_key=recorder.version.key,
        context=_context(),
        action=COVRAction.REFRESH,
        propensity=1.0,
        policy="shadow",
        incremental_cost=1.0,
        one_step_defect=0.2,
        metadata={"tensor": torch.ones(1)},
    )
    with pytest.raises(TypeError):
        recorder.record(bad)

    good = CounterfactualEvent(
        session_id="session",
        trajectory_id=0,
        sample_id="sample",
        version_key=recorder.version.key,
        context=_context(),
        action=COVRAction.REFRESH,
        propensity=1.0,
        policy="shadow",
        incremental_cost=1.0,
        one_step_defect=0.2,
        metadata={"class_id": 3},
    )
    assert recorder.record(good)
    assert not recorder.enabled
    summary = recorder.close()
    assert summary["events"] == 1
    payload = json.loads(recorder.event_path.read_text().strip())
    assert payload["metadata"] == {"class_id": 3}


def test_policy_state_isolated_by_inference_version(tmp_path):
    policy = COVRPolicy(
        OnlineRidgeUCB(len(FEATURE_NAMES)),
        BudgetLedger(5.0),
        PrimalDualBudget(target_cost=0.2),
    )
    path = tmp_path / "policy.json"
    save_policy_state(str(path), _version(), policy, "session")
    restored = load_policy_state(str(path), _version())
    assert restored.risk_model.feature_dim == len(FEATURE_NAMES)
    with pytest.raises(ValueError, match="version"):
        load_policy_state(str(path), _version(cfg_scale=7.5))
