# -*- coding: utf-8 -*-
"""Offline falsification gates for COVR-Spec event streams."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from accelerators.covr import ActionAuditEvent, COVRAction, CounterfactualEvent


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: str
    reason: str
    metrics: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_gate_a(
    events: Sequence[CounterfactualEvent],
    min_samples: int = 100,
    min_metric_gain: float = 0.02,
    bootstrap_samples: int = 300,
    seed: int = 0,
) -> GateResult:
    paired = [
        event for event in events
        if event.one_step_defect is not None
        and event.local_probe_error is not None
        and event.terminal_intervention_gain is not None
    ]
    if len(paired) < min_samples:
        return GateResult(
            "A", "insufficient_data",
            "Gate A requires paired local, one-step, and terminal intervention labels",
            {"paired_samples": len(paired), "required_samples": min_samples},
        )

    target = np.asarray(
        [event.terminal_intervention_gain for event in paired], dtype=np.float64)
    local = np.asarray(
        [event.local_probe_error for event in paired], dtype=np.float64)
    one_step = np.asarray(
        [event.one_step_defect for event in paired], dtype=np.float64)
    h_mask = np.asarray(
        [event.h_step_defect is not None for event in paired], dtype=bool)

    local_metrics = _ranking_metrics(local, target)
    one_step_metrics = _ranking_metrics(one_step, target)
    candidate_name = "one_step"
    candidate_metrics = one_step_metrics
    if int(h_mask.sum()) >= min_samples:
        h_step = np.asarray(
            [event.h_step_defect for event in paired if event.h_step_defect is not None],
            dtype=np.float64,
        )
        h_target = target[h_mask]
        h_metrics = _ranking_metrics(h_step, h_target)
        if h_metrics["spearman"] > candidate_metrics["spearman"]:
            candidate_name = "h_step"
            candidate_metrics = h_metrics
    else:
        h_metrics = None

    ci_low, ci_high = _bootstrap_spearman_delta(
        local, one_step, target, bootstrap_samples, seed)
    gains = {
        key: candidate_metrics[key] - local_metrics[key]
        for key in ("spearman", "auroc", "top_decile_lift")
    }
    passed = (
        ci_low > 0.0
        and gains["auroc"] >= min_metric_gain
        and gains["top_decile_lift"] >= min_metric_gain
    )
    metrics: Dict[str, object] = {
        "paired_samples": len(paired),
        "local_probe": local_metrics,
        "one_step": one_step_metrics,
        "selected_whole_step_signal": candidate_name,
        "selected_metrics": candidate_metrics,
        "metric_gains_over_local": gains,
        "one_step_spearman_delta_bootstrap_ci": [ci_low, ci_high],
    }
    if h_metrics is not None:
        metrics["h_step"] = h_metrics
    return GateResult(
        "A",
        "pass" if passed else "stop",
        (
            "whole-step feedback ranks terminal intervention gain better than the local probe"
            if passed
            else "whole-step feedback did not clear the predeclared signal-validity margin"
        ),
        metrics,
    )


def evaluate_gate_b(
    events: Sequence[CounterfactualEvent],
    budget_fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4),
    min_samples: int = 100,
    min_headroom: float = 0.02,
    random_trials: int = 200,
    seed: int = 0,
) -> GateResult:
    labeled = [event for event in events if event.one_step_defect is not None]
    if len(labeled) < min_samples:
        return GateResult(
            "B", "insufficient_data",
            "Gate B requires one-step counterfactual labels",
            {"labeled_samples": len(labeled), "required_samples": min_samples},
        )

    defects = np.asarray(
        [max(float(event.one_step_defect), 0.0) for event in labeled],
        dtype=np.float64,
    )
    probe_scores = np.asarray([
        float(event.local_probe_error)
        if event.local_probe_error is not None else -math.inf
        for event in labeled
    ], dtype=np.float64)
    timestep_scores = np.asarray([
        event.context.step_idx / max(event.context.num_steps - 1, 1)
        for event in labeled
    ], dtype=np.float64)
    calibrator_scores = _online_calibrator_scores(labeled)
    ordered_indices = np.asarray(sorted(
        range(len(labeled)),
        key=lambda index: (
            labeled[index].trajectory_id,
            labeled[index].context.step_idx,
            labeled[index].sample_id,
        ),
    ))
    rng = np.random.default_rng(seed)
    curves: List[Dict[str, float]] = []
    oracle_wins = 0

    for fraction in budget_fractions:
        if not 0 < fraction < 1:
            raise ValueError("budget fractions must be in (0, 1)")
        refreshes = max(1, int(round(len(labeled) * fraction)))
        oracle = _residual_defect(defects, np.argsort(defects)[-refreshes:])
        local = _residual_defect(defects, np.argsort(probe_scores)[-refreshes:])
        timestep = _residual_defect(defects, np.argsort(timestep_scores)[-refreshes:])
        calibrator = _residual_defect(
            defects, np.argsort(calibrator_scores)[-refreshes:])
        periodic_positions = np.linspace(
            0, len(ordered_indices) - 1, refreshes, dtype=int)
        periodic = _residual_defect(defects, ordered_indices[periodic_positions])
        random_values = [
            _residual_defect(
                defects, rng.choice(len(labeled), size=refreshes, replace=False))
            for _ in range(random_trials)
        ]
        random_mean = float(np.mean(random_values))
        strongest_baseline = min(
            local, timestep, calibrator, periodic, random_mean)
        relative_headroom = (
            (strongest_baseline - oracle) / max(strongest_baseline, 1e-12))
        if relative_headroom >= min_headroom:
            oracle_wins += 1
        curves.append({
            "budget_fraction": float(fraction),
            "refreshes": float(refreshes),
            "oracle": oracle,
            "static_speca": local,
            "timestep_only": timestep,
            "online_calibrator": calibrator,
            "periodic": periodic,
            "random": random_mean,
            "relative_oracle_headroom": relative_headroom,
        })

    required_wins = math.ceil(0.8 * len(curves))
    passed = oracle_wins >= required_wins
    return GateResult(
        "B",
        "pass" if passed else "stop",
        (
            "full-information oracle has budget-matched Pareto headroom"
            if passed
            else "oracle did not consistently improve over budget-matched baselines"
        ),
        {
            "labeled_samples": len(labeled),
            "curve": curves,
            "oracle_wins": oracle_wins,
            "required_wins": required_wins,
        },
    )


def evaluate_gate_c(
    events: Sequence[CounterfactualEvent],
    min_samples: int = 200,
    folds: int = 5,
    ridge_alpha: float = 1.0,
    min_relative_mse_gain: float = 0.02,
    min_recall_gain: float = 0.02,
) -> GateResult:
    labeled = [event for event in events if event.one_step_defect is not None]
    if len(labeled) < min_samples:
        return GateResult(
            "C", "insufficient_data",
            "Gate C requires enough labeled class/seed groups for held-out folds",
            {"labeled_samples": len(labeled), "required_samples": min_samples},
        )

    targets = np.asarray(
        [float(event.one_step_defect) for event in labeled], dtype=np.float64)
    full_features = np.stack([event.context.features() for event in labeled])
    progress = full_features[:, 1]
    distance = full_features[:, 4]
    previous = full_features[:, 13]
    timestep_features = np.column_stack([np.ones(len(labeled)), progress])
    distance_features = np.column_stack([
        np.ones(len(labeled)), progress, distance, previous])
    probe = np.asarray([
        float(event.local_probe_error)
        if event.local_probe_error is not None else 0.0
        for event in labeled
    ], dtype=np.float64)
    static_features = np.column_stack([np.ones(len(labeled)), probe])
    calibrator_features = np.column_stack([
        np.ones(len(labeled)), _online_calibrator_scores(labeled)])
    fold_ids = np.asarray([
        _stable_fold(event, folds) for event in labeled
    ], dtype=np.int64)

    fold_metrics: List[Dict[str, Dict[str, float]]] = []
    full_mse_wins = 0
    valid_folds = 0
    for fold in range(folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        if int(test_mask.sum()) < 5 or int(train_mask.sum()) < 5:
            continue
        valid_folds += 1
        per_model: Dict[str, Dict[str, float]] = {}
        for name, features in (
            ("static_threshold", static_features),
            ("online_calibrator", calibrator_features),
            ("timestep_only", timestep_features),
            ("cache_distance", distance_features),
            ("full_context", full_features),
        ):
            predictions = _ridge_predict(
                features[train_mask], targets[train_mask],
                features[test_mask], ridge_alpha)
            per_model[name] = _prediction_metrics(
                predictions, targets[test_mask])
        if per_model["full_context"]["mse"] < per_model["timestep_only"]["mse"]:
            full_mse_wins += 1
        fold_metrics.append(per_model)

    if valid_folds < max(3, folds - 1):
        return GateResult(
            "C", "insufficient_data",
            "held-out class/seed hashing left too few populated folds",
            {"valid_folds": valid_folds, "requested_folds": folds},
        )

    aggregate = {
        name: {
            metric: float(np.mean([
                fold[name][metric] for fold in fold_metrics
            ]))
            for metric in ("mse", "top_risk_recall")
        }
        for name in (
            "static_threshold", "online_calibrator", "timestep_only",
            "cache_distance", "full_context")
    }
    timestep = aggregate["timestep_only"]
    full = aggregate["full_context"]
    relative_mse_gain = (
        (timestep["mse"] - full["mse"]) / max(timestep["mse"], 1e-12))
    recall_gain = full["top_risk_recall"] - timestep["top_risk_recall"]
    required_fold_wins = math.ceil(0.8 * valid_folds)
    passed = (
        full_mse_wins >= required_fold_wins
        and relative_mse_gain >= min_relative_mse_gain
        and recall_gain >= min_recall_gain
    )
    return GateResult(
        "C",
        "pass" if passed else "stop",
        (
            "full context generalizes beyond the timestep-only policy"
            if passed
            else "full context did not stably beat timestep-only held-out performance"
        ),
        {
            "labeled_samples": len(labeled),
            "valid_folds": valid_folds,
            "full_context_mse_wins": full_mse_wins,
            "required_fold_wins": required_fold_wins,
            "aggregate": aggregate,
            "relative_mse_gain": relative_mse_gain,
            "top_risk_recall_gain": recall_gain,
        },
    )


def run_all_gates(events: Sequence[CounterfactualEvent]) -> Dict[str, object]:
    results = [
        evaluate_gate_a(events),
        evaluate_gate_b(events),
        evaluate_gate_c(events),
    ]
    first_failure: Optional[GateResult] = next(
        (result for result in results if result.status != "pass"), None)
    return {
        "decision": "proceed" if first_failure is None else "stop",
        "stopped_at": None if first_failure is None else first_failure.gate,
        "gates": [result.to_dict() for result in results],
    }


def evaluate_action_audit_integrity(
    events: Sequence[ActionAuditEvent],
    ratio_epsilon: float = 1e-8,
) -> GateResult:
    if not events:
        return GateResult(
            "integrity",
            "insufficient_data",
            "integrity analysis requires at least one action audit",
            {"events": 0},
        )

    contexts = set()
    duplicate_contexts = 0
    action_violations = 0
    propensity_violations = 0
    policy_violations = 0
    budget_violations = 0
    cardinality_violations = 0
    finite_violations = 0
    ratio_violations = 0
    trajectory_label_violations = 0
    trajectory_shape_violations = 0
    sample_labels = 0
    finite_values = 0
    total_metric_values = 0
    trajectory_labels = {}
    trajectory_steps = {}
    trajectory_costs = {}

    for event in events:
        context_key = (
            event.session_id, event.trajectory_id, event.context.step_idx)
        if context_key in contexts:
            duplicate_contexts += 1
        contexts.add(context_key)

        trajectory_key = (event.session_id, event.trajectory_id)
        labels = (event.sample_ids, event.class_ids)
        previous_labels = trajectory_labels.setdefault(trajectory_key, labels)
        if previous_labels != labels:
            trajectory_label_violations += 1
        previous_steps = trajectory_steps.setdefault(
            trajectory_key, event.context.num_steps)
        if previous_steps != event.context.num_steps:
            trajectory_shape_violations += 1
        trajectory_costs[trajectory_key] = (
            trajectory_costs.get(trajectory_key, 0.0) + event.incremental_cost)

        transition = event.one_step_transition
        count = len(event.sample_ids)
        sample_labels += count
        if (
            count == 0
            or len(set(event.sample_ids)) != count
            or len(event.class_ids) != count
            or transition.sample_count != count
        ):
            cardinality_violations += 1

        if (
            event.committed_action is not COVRAction.ACCEPT
            or event.audit_action is not COVRAction.REFRESH
            or event.committed_action is event.audit_action
        ):
            action_violations += 1
        if (
            not 0 < event.committed_propensity <= 1
            or not 0 < event.audit_propensity <= 1
            or not math.isclose(event.committed_propensity, 1.0)
            or not math.isclose(event.audit_propensity, 1.0)
        ):
            propensity_violations += 1
        if event.policy != "shadow_static_speca":
            policy_violations += 1
        if (
            not math.isfinite(event.incremental_cost)
            or event.incremental_cost <= 0
            or event.incremental_cost > 1.0
        ):
            budget_violations += 1

        for numerator, denominator, ratio in zip(
            transition.numerators,
            transition.denominators,
            transition.ratios,
        ):
            values = (
                float(numerator), float(denominator), float(ratio))
            total_metric_values += len(values)
            finite_values += sum(
                math.isfinite(value) and value >= 0 for value in values)
            if not all(
                    math.isfinite(value) and value >= 0 for value in values):
                finite_violations += 1
                continue
            expected = numerator / max(denominator, ratio_epsilon)
            if not math.isclose(
                    ratio, expected, rel_tol=1e-5, abs_tol=1e-8):
                ratio_violations += 1

    for trajectory_key, cost in trajectory_costs.items():
        if cost > trajectory_steps[trajectory_key] + 1e-8:
            budget_violations += 1

    failures = sum((
        duplicate_contexts,
        action_violations,
        propensity_violations,
        policy_violations,
        budget_violations,
        cardinality_violations,
        finite_violations,
        ratio_violations,
        trajectory_label_violations,
        trajectory_shape_violations,
    ))
    return GateResult(
        "integrity",
        "pass" if failures == 0 else "stop",
        (
            "action audit contexts and sample labels are internally consistent"
            if failures == 0
            else "action audit integrity violations were detected"
        ),
        {
            "events": len(events),
            "unique_contexts": len(contexts),
            "trajectories": len(trajectory_labels),
            "sample_labels": sample_labels,
            "finite_rate": finite_values / max(total_metric_values, 1),
            "duplicate_contexts": duplicate_contexts,
            "action_violations": action_violations,
            "propensity_violations": propensity_violations,
            "policy_violations": policy_violations,
            "budget_violations": budget_violations,
            "cardinality_violations": cardinality_violations,
            "finite_violations": finite_violations,
            "ratio_reconstruction_violations": ratio_violations,
            "trajectory_label_violations": trajectory_label_violations,
            "trajectory_shape_violations": trajectory_shape_violations,
        },
    )


def evaluate_denominator_tail(
    events: Sequence[ActionAuditEvent],
    bootstrap_samples: int = 1000,
    seed: int = 0,
    log_epsilon: float = 1e-12,
) -> GateResult:
    trajectories = {}
    for event in events:
        key = (event.session_id, event.trajectory_id)
        trajectories.setdefault(key, []).append(event)

    decomposition = []
    skipped_nonterminal = 0
    sample_set_mismatches = 0
    paired_sample_labels = 0
    terminal_steps = set()
    previous_steps = set()
    for trajectory_events in trajectories.values():
        ordered = sorted(
            trajectory_events, key=lambda event: event.context.step_idx)
        if len(ordered) < 2:
            continue
        previous, terminal = ordered[-2], ordered[-1]
        if terminal.context.step_idx != terminal.context.num_steps - 1:
            skipped_nonterminal += 1
            continue
        if previous.sample_ids != terminal.sample_ids:
            sample_set_mismatches += 1
            continue

        previous_transition = previous.one_step_transition
        terminal_transition = terminal.one_step_transition
        delta_log_numerator = []
        delta_log_denominator = []
        for index in range(len(terminal.sample_ids)):
            delta_log_numerator.append(
                math.log(terminal_transition.numerators[index] + log_epsilon)
                - math.log(previous_transition.numerators[index] + log_epsilon))
            delta_log_denominator.append(
                math.log(terminal_transition.denominators[index] + log_epsilon)
                - math.log(previous_transition.denominators[index] + log_epsilon))
        mean_delta_numerator = float(np.mean(delta_log_numerator))
        mean_delta_denominator = float(np.mean(delta_log_denominator))
        decomposition.append((mean_delta_numerator, mean_delta_denominator))
        paired_sample_labels += len(terminal.sample_ids)
        terminal_steps.add(terminal.context.step_idx)
        previous_steps.add(previous.context.step_idx)

    if len(decomposition) < 2:
        return GateResult(
            "denominator_tail",
            "insufficient_data",
            "terminal decomposition requires at least two paired trajectories",
            {
                "paired_trajectories": len(decomposition),
                "required_trajectories": 2,
                "skipped_nonterminal": skipped_nonterminal,
                "sample_set_mismatches": sample_set_mismatches,
            },
        )

    values = np.asarray(decomposition, dtype=np.float64)
    mean_delta_numerator = float(values[:, 0].mean())
    mean_delta_denominator = float(values[:, 1].mean())
    mean_delta_ratio = mean_delta_numerator - mean_delta_denominator
    denominator_contribution = -mean_delta_denominator
    denominator_share = (
        denominator_contribution / mean_delta_ratio
        if mean_delta_ratio > 0 else None)

    rng = np.random.default_rng(seed)
    bootstrap = np.empty((bootstrap_samples, 4), dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled = values[rng.integers(0, len(values), len(values))]
        delta_numerator = float(sampled[:, 0].mean())
        delta_denominator = float(sampled[:, 1].mean())
        delta_ratio = delta_numerator - delta_denominator
        bootstrap[index] = (
            delta_numerator,
            delta_denominator,
            delta_ratio,
            -delta_denominator / delta_ratio if delta_ratio > 0 else np.nan,
        )

    confidence_intervals = {}
    names = (
        "delta_log_numerator",
        "delta_log_denominator",
        "delta_log_ratio",
        "denominator_share",
    )
    for column, name in enumerate(names):
        finite = bootstrap[:, column][np.isfinite(bootstrap[:, column])]
        confidence_intervals[name] = (
            [float(np.quantile(finite, 0.025)),
             float(np.quantile(finite, 0.975))]
            if len(finite) else None)

    normalized_defect_prohibited = (
        denominator_share is not None and denominator_share >= 0.5)
    return GateResult(
        "denominator_tail",
        "stop" if normalized_defect_prohibited else "pass",
        (
            "denominator shrinkage explains at least half of the terminal ratio increase"
            if normalized_defect_prohibited
            else "denominator shrinkage does not dominate the terminal ratio change"
        ),
        {
            "paired_trajectories": len(decomposition),
            "paired_sample_labels": paired_sample_labels,
            "terminal_step_indices": sorted(terminal_steps),
            "previous_step_indices": sorted(previous_steps),
            "skipped_nonterminal": skipped_nonterminal,
            "sample_set_mismatches": sample_set_mismatches,
            "delta_log_numerator": mean_delta_numerator,
            "delta_log_denominator": mean_delta_denominator,
            "delta_log_ratio": mean_delta_ratio,
            "denominator_contribution": denominator_contribution,
            "denominator_share": denominator_share,
            "normalized_defect_primary_target_prohibited": (
                normalized_defect_prohibited),
            "trajectory_bootstrap_95_ci": confidence_intervals,
            "log_epsilon": log_epsilon,
        },
    )


def evaluate_action_aligned_timestep(
    events: Sequence[ActionAuditEvent],
    budget_fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4),
    ridge_alpha: float = 1e-6,
) -> GateResult:
    if len(events) < 4:
        return GateResult(
            "action_aligned_timestep",
            "insufficient_data",
            "timestep analysis requires at least four batch-step contexts",
            {"contexts": len(events), "required_contexts": 4},
        )

    sessions = sorted({event.session_id for event in events})
    if len(sessions) >= 2:
        fold_unit = "session"
        group_keys = [event.session_id for event in events]
    else:
        fold_unit = "trajectory"
        group_keys = [
            (event.session_id, event.trajectory_id) for event in events]
    groups = list(dict.fromkeys(group_keys))
    if len(groups) < 2:
        return GateResult(
            "action_aligned_timestep",
            "insufficient_data",
            "held-out analysis requires at least two isolated groups",
            {
                "contexts": len(events),
                "fold_unit": fold_unit,
                "groups": len(groups),
                "required_groups": 2,
            },
        )

    targets = np.asarray([
        event.one_step_transition.mean_ratio for event in events
    ], dtype=np.float64)
    steps = np.asarray([
        event.context.step_idx for event in events
    ], dtype=np.int64)
    log_snr = np.asarray([
        event.context.log_snr for event in events
    ], dtype=np.float64)
    one_hot_predictions = np.full(len(events), np.nan, dtype=np.float64)
    spline_predictions = np.full(len(events), np.nan, dtype=np.float64)
    fold_sizes = []

    group_array = np.asarray(group_keys, dtype=object)
    for group in groups:
        if fold_unit == "session":
            test_mask = group_array == group
        else:
            test_mask = np.asarray([
                key == group for key in group_keys], dtype=bool)
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            continue

        train_targets = targets[train_mask]
        global_mean = float(train_targets.mean())
        step_means = {
            int(step): float(train_targets[steps[train_mask] == step].mean())
            for step in np.unique(steps[train_mask])
        }
        one_hot_predictions[test_mask] = [
            step_means.get(int(step), global_mean) for step in steps[test_mask]
        ]

        train_log_snr = log_snr[train_mask]
        center = float(train_log_snr.mean())
        scale = float(train_log_snr.std())
        if scale < 1e-12:
            scale = 1.0
        train_scaled = (train_log_snr - center) / scale
        test_scaled = (log_snr[test_mask] - center) / scale
        knots = np.quantile(train_scaled, (0.25, 0.5, 0.75))
        train_design = _cubic_spline_design(train_scaled, knots)
        test_design = _cubic_spline_design(test_scaled, knots)
        precision = (
            train_design.T @ train_design
            + ridge_alpha * np.eye(train_design.shape[1]))
        coefficients = np.linalg.solve(
            precision, train_design.T @ train_targets)
        spline_predictions[test_mask] = test_design @ coefficients
        fold_sizes.append({
            "group": str(group),
            "train_contexts": int(train_mask.sum()),
            "test_contexts": int(test_mask.sum()),
        })

    valid = np.isfinite(one_hot_predictions) & np.isfinite(spline_predictions)
    if not valid.all():
        return GateResult(
            "action_aligned_timestep",
            "insufficient_data",
            "one or more held-out groups could not be evaluated",
            {
                "contexts": len(events),
                "valid_contexts": int(valid.sum()),
                "fold_unit": fold_unit,
                "groups": len(groups),
            },
        )

    one_hot_metrics = _prediction_metrics(one_hot_predictions, targets)
    spline_metrics = _prediction_metrics(spline_predictions, targets)
    one_hot_metrics["spearman"] = _spearman(one_hot_predictions, targets)
    spline_metrics["spearman"] = _spearman(spline_predictions, targets)
    allocation = _trajectory_allocation_curves(
        events,
        targets,
        one_hot_predictions,
        spline_predictions,
        budget_fractions,
    )
    return GateResult(
        "action_aligned_timestep",
        "pass",
        "held-out timestep baselines were evaluated on isolated action contexts",
        {
            "contexts": len(events),
            "fold_unit": fold_unit,
            "groups": len(groups),
            "folds": fold_sizes,
            "one_hot_timestep": one_hot_metrics,
            "cubic_truncated_power_spline_log_snr": spline_metrics,
            "allocation": {
                "oracle_kind": "noncausal_static_path_one_step",
                "selection_unit": "batch_step_context_within_trajectory",
                "curve": allocation,
            },
        },
    )


def run_phase0_analysis(
    events: Sequence[ActionAuditEvent],
) -> Dict[str, object]:
    reports = [
        evaluate_action_audit_integrity(events),
        evaluate_denominator_tail(events),
        evaluate_action_aligned_timestep(events),
    ]
    first_failure: Optional[GateResult] = next(
        (report for report in reports if report.status != "pass"), None)
    return {
        "schema_version": 2,
        "decision": "proceed" if first_failure is None else "stop",
        "stopped_at": None if first_failure is None else first_failure.gate,
        "reports": [report.to_dict() for report in reports],
    }


def _cubic_spline_design(
    values: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    columns = [
        np.ones(len(values), dtype=np.float64),
        values,
        values ** 2,
        values ** 3,
    ]
    columns.extend(np.maximum(values - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(columns)


def _trajectory_allocation_curves(
    events: Sequence[ActionAuditEvent],
    targets: np.ndarray,
    one_hot_predictions: np.ndarray,
    spline_predictions: np.ndarray,
    budget_fractions: Sequence[float],
) -> List[Dict[str, float]]:
    trajectories = {}
    for index, event in enumerate(events):
        key = (event.session_id, event.trajectory_id)
        trajectories.setdefault(key, []).append(index)

    curves = []
    for fraction in budget_fractions:
        if not 0 < fraction <= 1:
            raise ValueError("budget fractions must be in (0, 1]")
        residuals = {
            "oracle": [],
            "one_hot_timestep": [],
            "spline_log_snr": [],
        }
        for indices in trajectories.values():
            trajectory_indices = np.asarray(indices, dtype=np.int64)
            count = max(1, int(round(len(trajectory_indices) * fraction)))
            trajectory_targets = targets[trajectory_indices]
            scores = {
                "oracle": trajectory_targets,
                "one_hot_timestep": one_hot_predictions[trajectory_indices],
                "spline_log_snr": spline_predictions[trajectory_indices],
            }
            for name, values in scores.items():
                selected_local = np.argsort(
                    values, kind="stable")[-count:]
                selected = trajectory_indices[selected_local]
                residual = (
                    float(targets[trajectory_indices].sum())
                    - float(targets[selected].sum())
                ) / len(trajectory_indices)
                residuals[name].append(max(residual, 0.0))
        curves.append({
            "budget_fraction": float(fraction),
            "trajectories": len(trajectories),
            "oracle_residual_defect": float(np.mean(residuals["oracle"])),
            "one_hot_residual_defect": float(
                np.mean(residuals["one_hot_timestep"])),
            "spline_residual_defect": float(
                np.mean(residuals["spline_log_snr"])),
        })
    return curves


def _ranking_metrics(scores: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    threshold = float(np.quantile(target, 0.9))
    labels = target >= threshold
    top_count = max(1, int(math.ceil(0.1 * len(scores))))
    top_indices = np.argsort(scores)[-top_count:]
    target_std = max(float(np.std(target)), 1e-12)
    lift = (float(np.mean(target[top_indices])) - float(np.mean(target))) / target_std
    return {
        "spearman": _spearman(scores, target),
        "auroc": _auroc(scores, labels),
        "top_decile_lift": lift,
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = _average_ranks(scores) + 1.0
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _bootstrap_spearman_delta(
    local: np.ndarray,
    whole: np.ndarray,
    target: np.ndarray,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        indices = rng.integers(0, len(target), size=len(target))
        deltas.append(
            _spearman(whole[indices], target[indices])
            - _spearman(local[indices], target[indices]))
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _residual_defect(defects: np.ndarray, refreshed: Iterable[int]) -> float:
    accepted = np.ones(len(defects), dtype=bool)
    accepted[np.asarray(list(refreshed), dtype=np.int64)] = False
    return float(defects[accepted].mean()) if accepted.any() else 0.0


def _online_calibrator_scores(
    events: Sequence[CounterfactualEvent],
) -> np.ndarray:
    states: Dict[int, Tuple[int, float, float]] = {}
    scores = []
    for event in events:
        bucket = min(
            int(event.context.step_idx * 3 / max(event.context.num_steps, 1)),
            2,
        )
        count, mean, m2 = states.get(bucket, (0, 0.0, 0.0))
        variance = m2 / max(count - 1, 1) if count > 1 else 0.0
        threshold = mean + 3.0 * math.sqrt(max(variance, 0.0))
        error = (
            float(event.local_probe_error)
            if event.local_probe_error is not None else 0.0)
        scores.append(error - threshold)
        count += 1
        delta = error - mean
        mean += delta / count
        m2 += delta * (error - mean)
        states[bucket] = (count, mean, m2)
    return np.asarray(scores, dtype=np.float64)


def _stable_fold(event: CounterfactualEvent, folds: int) -> int:
    class_id = event.metadata.get("class_id", "unknown")
    value = f"{class_id}:{event.sample_id}".encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:8], 16) % folds


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray,
                   test_x: np.ndarray, alpha: float) -> np.ndarray:
    precision = train_x.T @ train_x + alpha * np.eye(train_x.shape[1])
    coefficients = np.linalg.solve(precision, train_x.T @ train_y)
    return test_x @ coefficients


def _prediction_metrics(predictions: np.ndarray,
                        targets: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((predictions - targets) ** 2))
    count = max(1, int(math.ceil(0.1 * len(targets))))
    predicted_top = set(np.argsort(predictions)[-count:].tolist())
    actual_top = set(np.argsort(targets)[-count:].tolist())
    recall = len(predicted_top & actual_top) / len(actual_top)
    return {"mse": mse, "top_risk_recall": float(recall)}
