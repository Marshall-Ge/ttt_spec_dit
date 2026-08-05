#!/usr/bin/env python3
"""Validate a two-leg COVR experimental-bandit resume smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def _find_manifest(root: Path) -> Path:
    candidates = sorted(root.glob("teacache_*.json"))
    if not candidates:
        raise SystemExit(f"no teacache manifest found under {root}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise SystemExit(
            f"multiple manifests found under {root}: {names}; "
            "pass --manifest explicitly")
    return candidates[0]


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else []


def _expect_equal(
        errors: List[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label} is {actual!r}, expected {expected!r}")


def _assignment_sample_count(assignments: Sequence[Any]) -> int:
    total = 0
    for assignment in assignments:
        if isinstance(assignment, Mapping):
            try:
                total += int(assignment.get("sample_count", 0))
            except (TypeError, ValueError):
                continue
    return total


def validate_artifacts(
        *, manifest: Mapping[str, Any], first_result: Mapping[str, Any],
        resumed_result: Mapping[str, Any], first_state: Mapping[str, Any],
        final_state: Mapping[str, Any], samples_per_leg: int,
        dataset_start_index: int, session_id: str,
        state_path: str) -> List[str]:
    """Return all persistence/resume contract violations."""
    errors: List[str] = []
    total_samples = 2 * samples_per_leg
    version_key = manifest.get("version_key")
    manifest_hash = _manifest_hash(manifest)

    if not version_key:
        errors.append("manifest has no version_key")
    strategies = _sequence(manifest.get("strategies"))
    strategy_ids = {
        item.get("strategy_id")
        for item in strategies if isinstance(item, Mapping)
    }
    baseline_id = manifest.get("baseline_strategy_id")
    if not strategy_ids:
        errors.append("manifest has no strategies")
    elif baseline_id not in strategy_ids:
        errors.append("manifest baseline_strategy_id is absent from strategies")

    for label, result in (
            ("first results", first_result),
            ("resumed results", resumed_result)):
        config = _mapping(result.get("config"))
        _expect_equal(
            errors, f"{label} covr_version_key",
            config.get("covr_version_key"), version_key)
        _expect_equal(
            errors, f"{label} covr_session_id",
            config.get("covr_session_id"), session_id)
        _expect_equal(
            errors, f"{label} covr_template_bandit",
            config.get("covr_template_bandit"), True)
        _expect_equal(
            errors, f"{label} covr_force_template_id",
            config.get("covr_force_template_id"), None)
        _expect_equal(
            errors, f"{label} covr_profile_stages",
            config.get("covr_profile_stages"), True)

        aggregate = _mapping(result.get("aggregate"))
        for key in (
                "covr_template_bandit", "covr_reward_telemetry",
                "generation_profile"):
            if not isinstance(aggregate.get(key), Mapping):
                errors.append(f"{label} aggregate is missing {key}")

    for label, state in (
            ("first state", first_state), ("final state", final_state)):
        for key in (
                "schema_version", "session_id", "version_key",
                "manifest_hash", "run_identity", "assignments",
                "completed_trajectories", "feedback"):
            if key not in state:
                errors.append(f"{label} is missing {key}")
        _expect_equal(
            errors, f"{label} session_id", state.get("session_id"), session_id)
        _expect_equal(
            errors, f"{label} version_key", state.get("version_key"),
            version_key)
        _expect_equal(
            errors, f"{label} manifest_hash", state.get("manifest_hash"),
            manifest_hash)
        if not isinstance(state.get("run_identity"), Mapping):
            errors.append(f"{label} run_identity is not an object")
        if not isinstance(state.get("assignments"), list):
            errors.append(f"{label} assignments is not a list")
        if not isinstance(state.get("completed_trajectories"), list):
            errors.append(f"{label} completed_trajectories is not a list")
        if not isinstance(state.get("feedback"), list):
            errors.append(f"{label} feedback is not a list")

    _expect_equal(
        errors, "resumed run_identity", final_state.get("run_identity"),
        first_state.get("run_identity"))
    run_identity = _mapping(first_state.get("run_identity"))
    _expect_equal(
        errors, "run_identity dataset_start_index",
        run_identity.get("dataset_start_index"), dataset_start_index)
    _expect_equal(
        errors, "run_identity batch_size", run_identity.get("batch_size"), 1)

    first_assignments = _sequence(first_state.get("assignments"))
    final_assignments = _sequence(final_state.get("assignments"))
    _expect_equal(
        errors, "first assignment count", len(first_assignments),
        samples_per_leg)
    _expect_equal(
        errors, "final assignment count", len(final_assignments), total_samples)
    _expect_equal(
        errors, "first assignment sample total",
        _assignment_sample_count(first_assignments), samples_per_leg)
    _expect_equal(
        errors, "final assignment sample total",
        _assignment_sample_count(final_assignments), total_samples)

    for label, assignments in (
            ("first", first_assignments), ("final", final_assignments)):
        sample_counts = [
            item.get("sample_count") if isinstance(item, Mapping) else None
            for item in assignments
        ]
        if any(value != 1 for value in sample_counts):
            errors.append(
                f"{label} assignment sample_count values are not all 1: "
                f"{sample_counts!r}")
        trajectory_ids = [
            item.get("trajectory_id") if isinstance(item, Mapping) else None
            for item in assignments
        ]
        expected_ids = list(range(len(assignments)))
        if trajectory_ids != expected_ids:
            errors.append(
                f"{label} trajectory IDs are {trajectory_ids!r}, "
                f"expected {expected_ids!r}")

    if list(final_assignments[:len(first_assignments)]) != list(first_assignments):
        errors.append("final assignments do not preserve the first-state prefix")

    first_completed = _sequence(first_state.get("completed_trajectories"))
    final_completed = _sequence(final_state.get("completed_trajectories"))
    _expect_equal(
        errors, "first completed trajectories", list(first_completed),
        list(range(samples_per_leg)))
    _expect_equal(
        errors, "final completed trajectories", list(final_completed),
        list(range(total_samples)))

    expected_summaries = (
        (
            "first", _mapping(
                _mapping(first_result.get("aggregate")).get(
                    "covr_template_bandit")),
            0, dataset_start_index, samples_per_leg, samples_per_leg,
        ),
        (
            "resumed", _mapping(
                _mapping(resumed_result.get("aggregate")).get(
                    "covr_template_bandit")),
            samples_per_leg, dataset_start_index + samples_per_leg,
            total_samples, samples_per_leg,
        ),
    )
    for label, summary, resume_offset, generation_start, target, generated in (
            expected_summaries):
        _expect_equal(
            errors, f"{label} summary session_id", summary.get("session_id"),
            session_id)
        _expect_equal(
            errors, f"{label} summary manifest_hash",
            summary.get("manifest_hash"), manifest_hash)
        _expect_equal(
            errors, f"{label} summary state_path", summary.get("state_path"),
            state_path)
        _expect_equal(
            errors, f"{label} summary dataset_start_index",
            summary.get("dataset_start_index"), dataset_start_index)
        _expect_equal(
            errors, f"{label} summary resume_sample_offset",
            summary.get("resume_sample_offset"), resume_offset)
        _expect_equal(
            errors, f"{label} summary generation_start_index",
            summary.get("generation_start_index"), generation_start)
        _expect_equal(
            errors, f"{label} summary target_samples",
            summary.get("target_samples"), target)
        _expect_equal(
            errors, f"{label} summary generated_samples_this_run",
            summary.get("generated_samples_this_run"), generated)
        _expect_equal(
            errors, f"{label} summary processed_samples",
            summary.get("processed_samples"), target)
        _expect_equal(
            errors, f"{label} summary assignments",
            summary.get("assignments"), target)
        _expect_equal(
            errors, f"{label} summary completed_trajectories",
            summary.get("completed_trajectories"), target)

    return errors


def _print_summary(
        first_result: Mapping[str, Any], resumed_result: Mapping[str, Any],
        first_state: Mapping[str, Any], final_state: Mapping[str, Any]) -> None:
    first_summary = _mapping(
        _mapping(first_result.get("aggregate")).get("covr_template_bandit"))
    resumed_summary = _mapping(
        _mapping(resumed_result.get("aggregate")).get("covr_template_bandit"))
    print("Bandit state progression:")
    print(
        "  first:   "
        f"assignments={len(_sequence(first_state.get('assignments')))}, "
        f"processed_samples={first_summary.get('processed_samples')}, "
        f"generation_start_index={first_summary.get('generation_start_index')}")
    print(
        "  resumed: "
        f"assignments={len(_sequence(final_state.get('assignments')))}, "
        f"processed_samples={resumed_summary.get('processed_samples')}, "
        f"resume_sample_offset={resumed_summary.get('resume_sample_offset')}, "
        f"generation_start_index={resumed_summary.get('generation_start_index')}")
    print(f"  session_id={final_state.get('session_id')}")
    print(f"  version_key={final_state.get('version_key')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a completed COVR bandit persistence/resume smoke")
    parser.add_argument(
        "root", type=Path,
        help="run root created by run_covr_bandit_resume_smoke.sh")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--first-state", type=Path, default=None)
    parser.add_argument("--samples-per-leg", type=int, default=2)
    parser.add_argument("--dataset-start-index", type=int, default=0)
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    if args.samples_per_leg <= 0:
        parser.error("--samples-per-leg must be positive")
    if args.dataset_start_index < 0:
        parser.error("--dataset-start-index must be non-negative")

    root = args.root
    manifest_path = args.manifest or _find_manifest(root)
    state_path = args.state or (root / "template_bandit_state.json")
    first_state_path = args.first_state or (root / "state_after_first.json")
    first_results_path = root / "first_run" / "results.json"
    resumed_results_path = root / "resumed_run" / "results.json"

    manifest = _load(manifest_path)
    first_result = _load(first_results_path)
    resumed_result = _load(resumed_results_path)
    first_state = _load(first_state_path)
    final_state = _load(state_path)
    session_id = args.session_id or str(first_state.get("session_id", ""))

    print(f"root:          {root}")
    print(f"manifest:      {manifest_path}")
    print(f"first results: {first_results_path}")
    print(f"resume result: {resumed_results_path}")
    print(f"state:         {state_path}")
    _print_summary(first_result, resumed_result, first_state, final_state)

    errors = validate_artifacts(
        manifest=manifest,
        first_result=first_result,
        resumed_result=resumed_result,
        first_state=first_state,
        final_state=final_state,
        samples_per_leg=args.samples_per_leg,
        dataset_start_index=args.dataset_start_index,
        session_id=session_id,
        state_path=str(state_path),
    )
    if errors:
        print("CHECK: FAIL")
        for error in errors:
            print(f"  [FAIL] {error}")
        return 1

    print("CHECK: PASS")
    print("  bandit state identity, assignment continuity, and resume window are valid")
    print("  no adaptive-effectiveness claim is implied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
