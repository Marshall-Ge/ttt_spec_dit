import pytest

from accelerators.covr import (
    ActionAuditContext,
    ActionAuditEvent,
    COVRAction,
    TransitionDefectBatch,
)
from accelerators.timestep_feedback import TimestepFeedbackController
from scripts.analyze.analyze_timestep_feedback import evaluate_prequential


def _controller(**kwargs):
    defaults = dict(
        num_steps=6,
        budget_refreshes=2,
        version_key="runtime-v1",
        p_min=0.1,
        ucb_beta=1.0,
        prior_std=0.0,
        initial_price=0.5,
        seed=7,
    )
    defaults.update(kwargs)
    return TimestepFeedbackController(**defaults)


def test_risk_statistics_use_clipped_ipw():
    controller = _controller(max_ipw_weight=3.0)

    assert controller.observe(2, 0.2, 0.5) == pytest.approx(2.0)
    assert controller.observe(2, 0.8, 0.01) == pytest.approx(3.0)

    summary = controller.summary()["timesteps"][2]
    assert summary["observations"] == 2
    assert summary["mean"] == pytest.approx((0.2 * 2.0 + 0.8 * 3.0) / 5.0)
    assert summary["effective_count"] == pytest.approx(25.0 / 13.0)


def test_high_defect_timestep_becomes_refresh_candidate():
    controller = _controller(prior_std=0.0, initial_price=0.3)
    controller.observe(4, 0.9, 1.0)

    decision = controller.decide(4, remaining_budget=1.0)
    assert decision.refresh
    assert decision.reason == "risk_ucb"
    assert decision.risk_ucb == pytest.approx(0.9)


def test_exploration_has_positive_propensity_for_low_risk_steps():
    controller = _controller(
        p_min=1.0, prior_std=0.0, initial_price=1.0, seed=3)
    decision = controller.decide(1, remaining_budget=1.0)

    assert decision.refresh
    assert decision.propensity == pytest.approx(1.0)
    assert decision.reason == "explore"


def test_hard_budget_is_never_exceeded():
    controller = _controller(
        p_min=1.0, prior_std=0.0, initial_price=0.0)
    controller.begin_trajectory()
    remaining = 2.0
    decisions = []
    for step in range(6):
        decision = controller.decide(step, remaining_budget=remaining)
        decisions.append(decision)
        if decision.refresh:
            remaining -= 1.0

    assert sum(decision.refresh for decision in decisions) == 2
    assert controller.trajectory_refreshes == 2
    assert all(decision.reason == "hard_budget"
               for decision in decisions[2:])


def test_mandatory_steps_are_validated_and_forced():
    controller = _controller(mandatory_steps=(0, 5), budget_refreshes=2)
    controller.begin_trajectory()
    first = controller.decide(0, remaining_budget=2.0)
    last = controller.decide(5, remaining_budget=1.0)

    assert first.refresh and first.reason == "mandatory"
    assert last.refresh and last.reason == "mandatory"


def test_refresh_mask_falls_back_to_even_schedule_then_uses_risk():
    controller = _controller(num_steps=10, budget_refreshes=4)
    fresh_mask = controller.recommended_refresh_mask()
    assert sum(fresh_mask) == 4
    assert fresh_mask[0]
    assert fresh_mask[3] and fresh_mask[6] and fresh_mask[9]

    controller.observe(4, 2.0, 1.0)
    controller.observe(7, 1.5, 1.0)
    learned_mask = controller.recommended_refresh_mask()
    assert sum(learned_mask) == 4
    assert learned_mask[0] and learned_mask[4] and learned_mask[7]


def test_refresh_mask_rejects_unsafe_budget_and_repairs_gaps():
    controller = _controller(num_steps=10, budget_refreshes=1)
    with pytest.raises(ValueError, match="need at least 2 refreshes"):
        controller.recommended_refresh_mask(max_taylor_gap=4)

    controller = _controller(num_steps=10, budget_refreshes=3)
    mask = controller.recommended_refresh_mask(max_taylor_gap=4)
    assert sum(mask) == 3
    assert mask[0]
    longest_gap = 0
    current_gap = 0
    for refresh in mask:
        current_gap = 0 if refresh else current_gap + 1
        longest_gap = max(longest_gap, current_gap)
    assert longest_gap <= 4


def test_refresh_mask_repairs_gap_before_risk_spends_budget():
    controller = _controller(num_steps=50, budget_refreshes=10)
    for step in (1, 3, 4, 5, 6, 8, 9, 10):
        controller.observe(step, 1.5, 1.0)

    mask = controller.recommended_refresh_mask(max_taylor_gap=4)
    assert sum(mask) == 10
    assert mask[0]
    longest_gap = 0
    current_gap = 0
    for refresh in mask:
        current_gap = 0 if refresh else current_gap + 1
        longest_gap = max(longest_gap, current_gap)
    assert longest_gap <= 4


def test_price_updates_toward_target_refresh_rate():
    controller = _controller(
        budget_refreshes=3, initial_price=0.5, price_learning_rate=0.2)
    controller.begin_trajectory()
    controller._trajectory_steps = 6
    controller._trajectory_refreshes = 0

    assert controller.end_trajectory() == pytest.approx(0.4)
    assert controller.trajectories == 1


def test_state_roundtrip_and_version_gate():
    controller = _controller(mandatory_steps=(0,))
    controller.observe(3, 0.4, 0.5)
    controller.begin_trajectory()
    controller.decide(0, remaining_budget=2.0)
    controller.end_trajectory(observed_cost=1.0 / 6.0)

    restored = TimestepFeedbackController.from_state_dict(
        controller.state_dict(), expected_version_key="runtime-v1")
    assert restored.summary() == controller.summary()

    with pytest.raises(ValueError, match="does not match runtime"):
        TimestepFeedbackController.from_state_dict(
            controller.state_dict(), expected_version_key="runtime-v2")


def test_prequential_feedback_learns_a_high_risk_timestep():
    controller = _controller(
        num_steps=4,
        budget_refreshes=1,
        p_min=0.2,
        prior_std=0.05,
        initial_price=0.5,
        seed=11,
    )
    defects = (0.05, 0.05, 0.8, 0.05)

    for _ in range(100):
        controller.begin_trajectory()
        remaining = 1.0
        for step, defect in enumerate(defects):
            decision = controller.decide(step, remaining_budget=remaining)
            if decision.refresh:
                controller.observe(step, defect, decision.propensity)
                remaining -= 1.0
        controller.end_trajectory()

    summary = controller.summary()["timesteps"]
    assert summary[2]["observations"] > summary[0]["observations"]
    assert summary[2]["mean"] == pytest.approx(0.8)


def test_invalid_observations_are_rejected():
    controller = _controller()
    with pytest.raises(ValueError):
        controller.observe(0, -0.1, 1.0)
    with pytest.raises(ValueError):
        controller.observe(0, 0.1, 0.0)
    with pytest.raises(ValueError):
        controller.decide(0, remaining_budget=-1.0)


def test_session_held_out_analysis_updates_after_prediction():
    events = []
    for session_id in ("session-a", "session-b"):
        for step_idx, defect in ((0, 0.05), (1, 0.8), (2, 0.05)):
            transition = TransitionDefectBatch(
                numerators=(defect,), denominators=(1.0,), ratios=(defect,))
            events.append(ActionAuditEvent(
                session_id=session_id,
                trajectory_id=0,
                sample_ids=(f"{session_id}-sample",),
                class_ids=(0,),
                version_key="runtime-v1",
                context=ActionAuditContext(
                    step_idx=step_idx,
                    num_steps=3,
                    timestep=3 - step_idx,
                    log_snr=0.0,
                    alpha_t=0.8,
                    alpha_prev=0.7,
                    latent_coefficient=1.0,
                    model_output_coefficient=0.0,
                    distance_since_refresh=step_idx,
                ),
                committed_action=COVRAction.ACCEPT,
                committed_propensity=1.0,
                audit_action=COVRAction.REFRESH,
                audit_propensity=1.0,
                policy="shadow_static_speca",
                incremental_cost=1.0,
                one_step_transition=transition,
            ))

    result = evaluate_prequential(
        events,
        num_steps=3,
        budget_refreshes=1,
        version_key="runtime-v1",
        controller_kwargs={"prior_std": 0.0, "initial_price": 1.0},
    )
    assert result["status"] == "pass"
    assert result["session_rows"][1]["mse"] < result["session_rows"][0]["mse"]
    assert "baseline_prequential_mse" in result
    assert "baseline_mse" in result["session_rows"][1]
    assert result["final_controller"]["trajectories"] == 2
    assert result["target"] == "one_step_transition_defect_only"
