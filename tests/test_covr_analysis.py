import numpy as np

from accelerators.covr import (
    COVRAction,
    COVRContext,
    CounterfactualEvent,
)
from experiments.covr_analysis import (
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    run_all_gates,
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
