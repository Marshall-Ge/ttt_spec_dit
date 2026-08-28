#!/usr/bin/env python3
"""Session-held-out analysis for the timestep feedback learner.

This is an offline analysis of action-audit JSONL files. It predicts each
session before ingesting that session's refresh labels, then reports one-step
defect metrics. It does not claim terminal quality or simulate a per-image
contextual policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from accelerators.covr import COVRAction, read_action_audits  # noqa: E402
from accelerators.timestep_feedback import (  # noqa: E402
    TimestepFeedbackController,
)


def _top_decile_recall(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if not predictions or len(predictions) != len(targets):
        return 0.0
    count = max(1, int(math.ceil(len(targets) * 0.1)))
    predicted = set(sorted(range(len(predictions)),
                           key=lambda index: predictions[index], reverse=True)[:count])
    actual = set(sorted(range(len(targets)),
                       key=lambda index: targets[index], reverse=True)[:count])
    return len(predicted & actual) / len(actual)


def evaluate_prequential(
        events: Iterable[Any],
        *,
        num_steps: int,
        budget_refreshes: int,
        version_key: str,
        min_sessions: int = 2,
        controller_kwargs: Mapping[str, Any] | None = None,
        ) -> Dict[str, Any]:
    """Evaluate sessions in order, updating only after each session's labels."""
    events = list(events)
    if not events:
        raise ValueError("timestep feedback analysis requires at least one event")
    if min_sessions < 2:
        raise ValueError("min_sessions must be at least 2")
    versions = {str(event.version_key) for event in events}
    if versions != {version_key}:
        raise ValueError("events do not match the requested runtime version")

    by_session: Dict[str, List[Any]] = defaultdict(list)
    for event in events:
        by_session[str(event.session_id)].append(event)
    sessions = sorted(by_session)
    if len(sessions) < min_sessions:
        return {
            "status": "insufficient_data",
            "sessions": len(sessions),
            "required_sessions": min_sessions,
        }

    controller = TimestepFeedbackController(
        num_steps=num_steps,
        budget_refreshes=budget_refreshes,
        version_key=version_key,
        **dict(controller_kwargs or {}),
    )
    baseline_sum = [0.0] * num_steps
    baseline_count = [0] * num_steps
    session_rows = []
    for session_id in sessions:
        session_events = sorted(
            by_session[session_id],
            key=lambda event: (int(event.trajectory_id), int(event.context.step_idx)),
        )
        predictions = []
        targets = []
        baseline_predictions = []
        for event in session_events:
            step_idx = int(event.context.step_idx)
            mean, _ = controller.risk(step_idx)
            predictions.append(mean)
            target = float(event.one_step_transition.mean_ratio)
            targets.append(target)
            baseline_predictions.append(
                baseline_sum[step_idx] / baseline_count[step_idx]
                if baseline_count[step_idx] else 0.0)

        squared = [
            (prediction - target) ** 2
            for prediction, target in zip(predictions, targets)
        ]
        baseline_squared = [
            (prediction - target) ** 2
            for prediction, target in zip(baseline_predictions, targets)
        ]
        session_rows.append({
            "session_id": session_id,
            "events": len(session_events),
            "mse": sum(squared) / len(squared),
            "mae": sum(abs(prediction - target)
                       for prediction, target in zip(predictions, targets)) / len(targets),
            "top_decile_recall": _top_decile_recall(predictions, targets),
            "baseline_mse": sum(baseline_squared) / len(baseline_squared),
            "baseline_mae": sum(abs(prediction - target)
                                 for prediction, target in zip(
                                     baseline_predictions, targets)) / len(targets),
            "baseline_top_decile_recall": _top_decile_recall(
                baseline_predictions, targets),
            "observations_before_update": sum(
                item["observations"] for item in controller.summary()["timesteps"]),
        })

        # Session labels are ingested only after the held-out predictions above.
        for event in session_events:
            if event.audit_action is COVRAction.REFRESH:
                step_idx = int(event.context.step_idx)
                target = float(event.one_step_transition.mean_ratio)
                controller.observe(
                    step_idx,
                    target,
                    float(event.audit_propensity),
                )
                baseline_sum[step_idx] += target
                baseline_count[step_idx] += 1
        controller.begin_trajectory()
        observed_refreshes = sum(
            event.audit_action is COVRAction.REFRESH
            for event in session_events)
        controller.end_trajectory(
            observed_cost=observed_refreshes / num_steps)

    total_events = sum(row["events"] for row in session_rows)
    return {
        "status": "pass",
        "version_key": version_key,
        "num_steps": num_steps,
        "budget_refreshes": budget_refreshes,
        "sessions": len(session_rows),
        "events": total_events,
        "prequential_mse": sum(row["mse"] for row in session_rows) / len(session_rows),
        "prequential_mae": sum(row["mae"] for row in session_rows) / len(session_rows),
        "prequential_top_decile_recall": (
            sum(row["top_decile_recall"] for row in session_rows)
            / len(session_rows)),
        "baseline_prequential_mse": (
            sum(row["baseline_mse"] for row in session_rows)
            / len(session_rows)),
        "baseline_prequential_mae": (
            sum(row["baseline_mae"] for row in session_rows)
            / len(session_rows)),
        "baseline_prequential_top_decile_recall": (
            sum(row["baseline_top_decile_recall"] for row in session_rows)
            / len(session_rows)),
        "session_rows": session_rows,
        "final_controller": controller.summary(),
        "target": "one_step_transition_defect_only",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", nargs="+", help="action-audit JSONL files")
    parser.add_argument("--version-key", required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--budget-refreshes", type=int, required=True)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    result = evaluate_prequential(
        read_action_audits(args.events),
        num_steps=args.num_steps,
        budget_refreshes=args.budget_refreshes,
        version_key=args.version_key,
        min_sessions=args.min_sessions,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
