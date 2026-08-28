#!/usr/bin/env python3
"""Build accelerator strategy manifests from action-audit JSONL files.

Supports:
  - SpecA (default): builds an equal-FLOPs RefreshTemplate manifest
    from action-audit counterfactual events.
  - TeaCache (--method teacache): builds a StrategyManifest with one
    AccelerationStrategy per threshold value.
  - TeaCache equal-FLOPs masks (--method teacache-mask): reuses the SpecA
    equal-FLOPs template search over action-audit events, then emits each
    refresh mask as a ``method="teacache"`` AccelerationStrategy. Every arm
    shares the same refresh count, so arms are equal-FLOPs and the COVR
    bandit compares them apple-to-apple.

The SpecA path uses the existing ``build_template_manifest`` pipeline;
the TeaCache path creates strategies directly from the CLI thresholds;
the TeaCache-mask path bridges the two (SpecA search → TeaCache arms).
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from accelerators.covr import read_action_audits
from accelerators.covr_bandit import (
    AccelerationStrategy,
    StrategyManifest,
    TemplateManifest,
)
from scripts.analyze.covr_analysis import build_template_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a COVR equal-FLOPs template manifest")
    parser.add_argument("audits", nargs="*", default=[],
                        help="schema-v2 batch-step audit JSONL files "
                             "(required for SpecA; optional for TeaCache)")

    # Shared
    parser.add_argument("--output", required=True,
                        help="Manifest JSON output path")
    parser.add_argument("--method", type=str, default="speca",
                        choices=["speca", "teacache", "teacache-mask"],
                        help="Acceleration method (default: speca). "
                             "'teacache-mask' reuses the SpecA equal-FLOPs "
                             "search to build equal-FLOPs TeaCache arms.")

    # SpecA-specific
    parser.add_argument("--num-layers", type=int, default=28,
                        help="Number of DiT blocks used for modeled FLOPs (SpecA)")
    parser.add_argument("--template-count", type=int, default=4,
                        help="Total number of templates including the baseline (SpecA)")
    parser.add_argument("--mandatory-prefix", type=int, default=3,
                        help="Number of initial denoising steps forced to refresh (SpecA)")
    parser.add_argument("--max-taylor-gap", type=int, default=5,
                        help="Maximum consecutive Taylor steps (SpecA)")
    parser.add_argument("--refresh-count", type=int, default=None,
                        help="Required refresh count; defaults to the source trajectories (SpecA)")
    parser.add_argument("--training-session", action="append", default=None,
                        help="Session ID to use for manifest construction; repeatable (SpecA)")
    parser.add_argument("--safety-numerator-ucb-limit", type=float, default=1.0,
                        help="Numerator UCB safety limit (SpecA)")
    parser.add_argument("--safety-denominator-lcb-floor", type=float, default=1e-8,
                        help="Denominator LCB safety floor (SpecA)")

    # TeaCache-specific
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated TeaCache rel_l1_thresh values "
                             "(e.g. '0.15,0.25,0.35,0.45')")
    parser.add_argument("--baseline-threshold", type=float, default=0.25,
                        help="Baseline TeaCache threshold (default: 0.25)")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Denoising steps (default: 50, TeaCache)")
    parser.add_argument("--modeled-flops-per-step", type=float, default=1.0,
                        help="Modeled FLOPs per denoising step (default: 1.0, "
                             "arbitrary unit; only the *ratio* between arms matters)")

    # Common
    parser.add_argument("--version-key", default=None,
                        help="Expected COVR version key")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.method == "teacache":
        return _build_teacache_manifest(args)
    elif args.method == "teacache-mask":
        return _build_teacache_mask_manifest(args)
    else:
        return _build_speca_manifest(args)


def _build_speca_manifest(args) -> int:
    if not args.audits:
        print("[ERROR] SpecA manifest requires action-audit JSONL files.", file=sys.stderr)
        return 1
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


def _build_teacache_manifest(args) -> int:
    if not args.thresholds:
        print("[ERROR] --thresholds is required for --method teacache "
              "(e.g. '0.15,0.25,0.35,0.45').", file=sys.stderr)
        return 1

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    if not thresholds:
        print("[ERROR] --thresholds must contain at least one value.", file=sys.stderr)
        return 1
    if args.baseline_threshold not in thresholds:
        print(f"[ERROR] --baseline-threshold {args.baseline_threshold} must be one "
              f"of the --thresholds values.", file=sys.stderr)
        return 1

    version_key = args.version_key or f"teacache_v{args.num_steps}"
    modeled_flops = args.modeled_flops_per_step * args.num_steps

    strategies = []
    for t in thresholds:
        is_baseline = math.isclose(t, args.baseline_threshold)
        strategy_id = "baseline" if is_baseline else f"thresh_{t:.4f}"
        strategies.append(AccelerationStrategy(
            strategy_id=strategy_id,
            method="teacache",
            params={"rel_l1_thresh": t, "num_steps": args.num_steps},
            modeled_flops=modeled_flops,
            source="teacache_threshold" if is_baseline else "teacache_alternative",
        ))

    manifest = StrategyManifest(
        version_key=version_key,
        num_steps=args.num_steps,
        baseline_strategy_id="baseline",
        strategies=tuple(strategies),
    )
    manifest.save(args.output)
    print(json.dumps({
        "manifest": os.path.abspath(args.output),
        "manifest_hash": manifest.manifest_hash,
        "version_key": manifest.version_key,
        "num_steps": manifest.num_steps,
        "strategies": len(manifest.strategies),
        "baseline_strategy_id": manifest.baseline_strategy_id,
        "thresholds": thresholds,
        "source_groups": [],
    }, indent=2, sort_keys=True))
    return 0


def _build_teacache_mask_manifest(args) -> int:
    """Reuse the SpecA equal-FLOPs template search to build TeaCache arms.

    Each ``RefreshTemplate`` produced by ``build_template_manifest`` becomes a
    ``method="teacache"`` ``AccelerationStrategy`` carrying the same refresh
    mask. Because all masks share one refresh count, the arms are equal-FLOPs
    and the COVR bandit compares them apple-to-apple (unlike threshold arms).
    """
    if not args.audits:
        print("[ERROR] --method teacache-mask requires action-audit JSONL files "
              "(the SpecA equal-FLOPs search runs over them).", file=sys.stderr)
        return 1

    events = read_action_audits(args.audits)
    speca_manifest = build_template_manifest(
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

    num_steps = speca_manifest.num_steps
    version_key = args.version_key or f"teacache_mask_{speca_manifest.version_key}"
    refresh_count = speca_manifest.common_refresh_count

    strategies = []
    for template in speca_manifest.templates:
        strategies.append(AccelerationStrategy(
            strategy_id=template.template_id,
            method="teacache",
            params={
                "refresh_mask": list(template.refresh_mask),
                "num_steps": num_steps,
            },
            modeled_flops=float(refresh_count),
            source=template.source,
        ))

    manifest = StrategyManifest(
        version_key=version_key,
        num_steps=num_steps,
        baseline_strategy_id=speca_manifest.baseline_template_id,
        strategies=tuple(strategies),
        source_groups=speca_manifest.source_groups,
    )
    manifest.save(args.output)
    print(json.dumps({
        "manifest": os.path.abspath(args.output),
        "manifest_hash": manifest.manifest_hash,
        "version_key": manifest.version_key,
        "num_steps": manifest.num_steps,
        "strategies": len(manifest.strategies),
        "baseline_strategy_id": manifest.baseline_strategy_id,
        "refresh_count": refresh_count,
        "source_groups": list(manifest.source_groups),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
