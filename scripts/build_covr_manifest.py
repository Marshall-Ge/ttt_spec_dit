#!/usr/bin/env python3
"""Build an equal-FLOPs SpecA template manifest from action-audit JSONL files."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerators.covr import read_action_audits
from experiments.covr_analysis import build_template_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a COVR equal-FLOPs template manifest")
    parser.add_argument("audits", nargs="+",
                        help="schema-v2 batch-step audit JSONL files")
    parser.add_argument("--output", required=True,
                        help="Manifest JSON output path")
    parser.add_argument("--num-layers", type=int, default=28,
                        help="Number of DiT blocks used for modeled FLOPs")
    parser.add_argument("--template-count", type=int, default=4,
                        help="Total number of templates including the baseline")
    parser.add_argument("--mandatory-prefix", type=int, default=3,
                        help="Number of initial denoising steps forced to refresh")
    parser.add_argument("--max-taylor-gap", type=int, default=5,
                        help="Maximum consecutive Taylor steps")
    parser.add_argument("--refresh-count", type=int, default=None,
                        help="Required refresh count; defaults to the source trajectories")
    parser.add_argument("--training-session", action="append", default=None,
                        help="Session ID to use for manifest construction; repeatable")
    parser.add_argument("--version-key", default=None,
                        help="Expected COVR version key; defaults to the source version")
    parser.add_argument("--safety-numerator-ucb-limit", type=float, default=1.0,
                        help="Numerator UCB safety limit")
    parser.add_argument("--safety-denominator-lcb-floor", type=float, default=1e-8,
                        help="Denominator LCB safety floor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = read_action_audits(args.audits)
    manifest = build_template_manifest(
        events,
        num_layers=args.num_layers,
        template_count=args.template_count,
        mandatory_prefix=args.mandatory_prefix,
        max_taylor_gap=args.max_taylor_gap,
        refresh_count=args.refresh_count,
        version_key=args.version_key,
        training_sessions=args.training_session,
        safety_numerator_ucb_limit=args.safety_numerator_ucb_limit,
        safety_denominator_lcb_floor=args.safety_denominator_lcb_floor,
    )
    manifest.save(args.output)
    print(json.dumps({
        "manifest": os.path.abspath(args.output),
        "manifest_hash": manifest.manifest_hash,
        "version_key": manifest.version_key,
        "num_steps": manifest.num_steps,
        "num_layers": manifest.num_layers,
        "templates": len(manifest.templates),
        "baseline_template_id": manifest.baseline_template_id,
        "refresh_count": manifest.common_refresh_count,
        "modeled_full_block_equivalents": (
            manifest.common_full_block_equivalents),
        "source_groups": list(manifest.source_groups),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
