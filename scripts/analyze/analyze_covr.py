#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run COVR-Spec falsification gates over scalar event files."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerators.covr import read_action_audits, read_events
from experiments.covr_analysis import run_all_gates, run_phase0_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze legacy COVR events or schema-v2 action audits")
    parser.add_argument("events", nargs="+", help="COVR events_*.jsonl files")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional machine-readable JSON result path")
    return parser.parse_args()


def _detect_schema(paths) -> int:
    versions = set()
    for path in paths:
        found = False
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                versions.add(int(payload.get("schema_version", 1)))
                found = True
                break
        if not found:
            raise ValueError(f"COVR event file is empty: {path}")
    if len(versions) != 1:
        raise ValueError("COVR analysis does not accept mixed schema versions")
    version = versions.pop()
    if version not in (1, 2):
        raise ValueError(f"Unsupported COVR schema version: {version}")
    return version


def main() -> int:
    args = parse_args()
    schema_version = _detect_schema(args.events)
    if schema_version == 2:
        events = read_action_audits(args.events)
        result = run_phase0_analysis(events)
    else:
        events = read_events(args.events)
        result = run_all_gates(events)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0 if result["decision"] == "proceed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
