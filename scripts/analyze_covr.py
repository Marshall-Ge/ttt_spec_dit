#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run COVR-Spec falsification gates over scalar event files."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerators.covr import read_events
from experiments.covr_analysis import run_all_gates


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate COVR-Spec Gate A/B/C in falsification order")
    parser.add_argument("events", nargs="+", help="COVR events_*.jsonl files")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional machine-readable JSON result path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
