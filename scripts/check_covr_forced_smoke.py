#!/usr/bin/env python3
"""Validate one run produced by ``run_covr_forced_smoke.sh``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


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
        names = ", ".join(str(path.name) for path in candidates)
        raise SystemExit(
            f"multiple manifests found under {root}: {names}; "
            "pass --manifest explicitly")
    return candidates[0]


def _print_covr_payload(result: Dict[str, Any]) -> None:
    config = result.get("config", {})
    aggregate = result.get("aggregate", {})
    print("COVR config:")
    for key in sorted(key for key in config if key.startswith("covr_")):
        print(f"  {key}: {config[key]}")

    print("COVR aggregate keys:")
    for key in sorted(key for key in aggregate if key.startswith("covr_")):
        value = aggregate[key]
        if isinstance(value, dict):
            print(f"  {key}: {', '.join(sorted(value))}")
        else:
            print(f"  {key}: {value}")

    profile = aggregate.get("generation_profile")
    if isinstance(profile, dict):
        stages = profile.get("stage_total_s", {})
        print("Profiler stages:")
        if isinstance(stages, dict):
            for name, value in sorted(stages.items()):
                print(f"  {name}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a completed COVR forced-strategy smoke run")
    parser.add_argument(
        "root", nargs="?", default="/tmp/covr_forced_smoke",
        help="run root created by run_covr_forced_smoke.sh")
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="manifest path (default: the only teacache_*.json under root)")
    parser.add_argument(
        "--strategy-id", default=None,
        help="expected forced strategy ID (default: read from config)")
    args = parser.parse_args()

    root = Path(args.root)
    results_path = root / "forced_run" / "results.json"
    manifest_path = args.manifest or _find_manifest(root)
    result = _load(results_path)
    manifest = _load(manifest_path)
    config = result.get("config", {})
    aggregate = result.get("aggregate", {})

    errors = []
    version_key = config.get("covr_version_key")
    if not version_key:
        errors.append("results config has no covr_version_key")
    if manifest.get("version_key") != version_key:
        errors.append(
            "manifest version_key does not match results covr_version_key")

    actual_id = config.get("covr_force_template_id")
    expected_id = args.strategy_id or actual_id
    ids = [item.get("strategy_id") for item in manifest.get("strategies", [])]
    if not expected_id:
        errors.append("forced strategy ID is missing from results config")
    elif expected_id not in ids:
        errors.append(f"strategy ID {expected_id!r} is absent from manifest")
    elif actual_id != expected_id:
        errors.append(
            f"results forced strategy is {actual_id!r}, expected {expected_id!r}")

    if not config.get("covr_profile_stages"):
        errors.append("COVR profiler stages were not enabled")
    if "covr_forced_template" not in aggregate:
        errors.append("aggregate is missing covr_forced_template")
    if "generation_profile" not in aggregate:
        errors.append("aggregate is missing generation_profile")

    bandit_state = root / "forced_run" / "covr" / "template_bandit_state.json"
    if bandit_state.exists():
        errors.append(f"forced mode created bandit state: {bandit_state}")

    print(f"results:  {results_path}")
    print(f"manifest: {manifest_path}")
    print(f"strategy: {expected_id}")
    _print_covr_payload(result)

    if errors:
        print("CHECK: FAIL")
        for error in errors:
            print(f"  [FAIL] {error}")
        return 1

    print("CHECK: PASS")
    print("  forced manifest identity, telemetry, profiler, and state policy are valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
