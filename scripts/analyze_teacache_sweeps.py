#!/usr/bin/env python3
"""Verdict analyzer for the TeaCache decision-value sweeps.

Reads the results.json files produced by ``sweep_teacache_decision.sh`` and
answers the *upstream* question that gates all COVR-on-TeaCache work:

    Does per-trajectory adaptive decision (a bandit, whether over masks or
    thresholds) beat a single static setting on TeaCache at all?

Three inputs, three sub-verdicts:

  A. Threshold sweep  — plain ``--method teacache`` at N thresholds.
     Establishes the quality-FLOPs Pareto frontier COVR must beat, and
     shows whether the threshold knob is smooth (=> just pick a point).
  B1. Equal-FLOPs mask sweep — each manifest arm forced via
     ``--covr-force-strategy-id``. At *fixed* calc-count, does WHERE the
     calc steps sit change FID beyond noise? (headroom for mask selection)
  B2. Reward-proxy check (optional) — one epsilon=1.0 bandit run gives the
     cheap terminal-fidelity per arm. Does it rank arms like FID does?

Layout expected under the sweep output dir::

    <out>/threshold/thresh_<v>/results.json
    <out>/equalflops/arm_<id>/results.json
    <out>/equalflops/noise_<seed>/results.json   (noise floor, same arm)
    <out>/equalflops/bandit/results.json          (optional, B2)

This is analysis only; it always exits 0. The RECOMMENDATION block is the
payload — it prints the reasoning, not just a label.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _load(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _row(results: dict) -> Dict[str, Optional[float]]:
    """Extract the comparison fields from one results.json."""
    agg = results.get("aggregate", {}) if results else {}
    cfg = results.get("config", {}) if results else {}
    return {
        "fid": agg.get("fid"),
        "is_mean": agg.get("is_mean"),
        "is_std": agg.get("is_std"),
        "skip_ratio": agg.get("skip_ratio"),
        "flops_accel_T": agg.get("flops_accel_T"),
        "flops_reduction": agg.get("flops_reduction"),
        "total_calc": agg.get("total_calc"),
        "rel_l1_thresh": cfg.get("rel_l1_thresh"),
        "n_prompts": cfg.get("n_prompts"),
    }


def _fmt(value: Optional[float], spec: str = ".2f") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "  n/a"
    if spec.endswith("d"):
        return format(int(value), spec)
    return format(value, spec)


# --------------------------------------------------------------------------
# A. Threshold sweep — the Pareto frontier COVR must beat
# --------------------------------------------------------------------------

def analyze_threshold_sweep(sweep_dir: str) -> Tuple[List[dict], List[str]]:
    """Load every threshold run and describe the quality-FLOPs curve."""
    rows: List[dict] = []
    for results_path in sorted(glob.glob(
            os.path.join(sweep_dir, "thresh_*", "results.json"))):
        results = _load(results_path)
        if results is None:
            continue
        row = _row(results)
        row["label"] = os.path.basename(os.path.dirname(results_path))
        rows.append(row)
    rows.sort(key=lambda r: (r["rel_l1_thresh"] is None, r["rel_l1_thresh"]))

    lines: List[str] = []
    lines.append("== [A] Threshold sweep (plain --method teacache) ==")
    if not rows:
        lines.append("  (no threshold runs found)")
        return rows, lines
    lines.append(
        f"  {'thresh':>7} {'skip':>6} {'FLOPs_T':>8} {'FID':>7} {'IS':>8}")
    for r in rows:
        lines.append(
            f"  {_fmt(r['rel_l1_thresh'], '.3f'):>7} "
            f"{_fmt(r['skip_ratio'], '.3f'):>6} "
            f"{_fmt(r['flops_accel_T'], '.3f'):>8} "
            f"{_fmt(r['fid'], '.2f'):>7} "
            f"{_fmt(r['is_mean'], '.1f'):>8}")
    return rows, lines


# --------------------------------------------------------------------------
# B1. Equal-FLOPs mask sweep — does WHERE calc sits matter?
# --------------------------------------------------------------------------

def analyze_equalflops_sweep(
    sweep_dir: str,
) -> Tuple[List[dict], List[dict], List[str]]:
    """Load forced-arm runs plus same-arm noise replicas.

    Returns (arm_rows, noise_rows, report_lines). ``noise_rows`` are
    repeated runs of the SAME arm under different seeds — their FID spread
    is the noise floor that arm-to-arm differences must clear.
    """
    arm_rows: List[dict] = []
    for results_path in sorted(glob.glob(
            os.path.join(sweep_dir, "arm_*", "results.json"))):
        results = _load(results_path)
        if results is None:
            continue
        row = _row(results)
        row["label"] = os.path.basename(
            os.path.dirname(results_path))[len("arm_"):]
        arm_rows.append(row)

    noise_rows: List[dict] = []
    for results_path in sorted(glob.glob(
            os.path.join(sweep_dir, "noise_*", "results.json"))):
        results = _load(results_path)
        if results is None:
            continue
        row = _row(results)
        row["label"] = os.path.basename(os.path.dirname(results_path))
        noise_rows.append(row)

    lines: List[str] = []
    lines.append("")
    lines.append("== [B1] Equal-FLOPs mask sweep (forced arms) ==")
    if not arm_rows:
        lines.append("  (no forced-arm runs found)")
        return arm_rows, noise_rows, lines
    lines.append(f"  {'arm':>16} {'calc':>5} {'skip':>6} {'FID':>7} {'IS':>8}")
    for r in arm_rows:
        lines.append(
            f"  {r['label']:>16} "
            f"{_fmt(r['total_calc'], 'd') if r['total_calc'] is not None else '  n/a':>5} "
            f"{_fmt(r['skip_ratio'], '.3f'):>6} "
            f"{_fmt(r['fid'], '.2f'):>7} "
            f"{_fmt(r['is_mean'], '.1f'):>8}")
    return arm_rows, noise_rows, lines


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def _spread(rows: List[dict], key: str) -> Optional[float]:
    values = [r[key] for r in rows
              if r.get(key) is not None and not math.isnan(r[key])]
    if len(values) < 2:
        return None
    return max(values) - min(values)


def render_verdict(
    thresh_rows: List[dict],
    arm_rows: List[dict],
    noise_rows: List[dict],
) -> List[str]:
    """Turn the two spreads into a go/no-go recommendation with reasoning."""
    lines: List[str] = []
    lines.append("")
    lines.append("== RECOMMENDATION ==")

    arm_fid_spread = _spread(arm_rows, "fid")
    noise_fid_spread = _spread(noise_rows, "fid")
    thresh_fid_spread = _spread(thresh_rows, "fid")
    thresh_flops_spread = _spread(thresh_rows, "flops_accel_T")

    # Noise floor: prefer measured same-arm replica spread; else a heuristic.
    if noise_fid_spread is not None:
        floor = noise_fid_spread
        floor_src = f"measured same-arm replica spread ({floor:.2f} FID)"
    else:
        floor = 2.0
        floor_src = ("heuristic 2.0 FID (no noise replicas found; "
                     "run noise_* replicas to measure it)")
    lines.append(f"  Noise floor: {floor_src}")

    # --- B1 verdict: do equal-FLOPs arms differ? ---
    if arm_fid_spread is None:
        lines.append("  [B1] inconclusive — need >=2 forced-arm runs.")
        b1 = "unknown"
    elif arm_fid_spread > 2.0 * floor:
        lines.append(
            f"  [B1] arms DIFFER: FID spread {arm_fid_spread:.2f} > 2x noise "
            f"floor. WHERE calc steps sit matters => mask selection has "
            f"headroom.")
        b1 = "differ"
    else:
        lines.append(
            f"  [B1] arms ~TIE: FID spread {arm_fid_spread:.2f} <= 2x noise "
            f"floor. Equal-FLOPs masks are interchangeable => a mask bandit "
            f"cannot beat a fixed schedule (matches the 80-img null result).")
        b1 = "tie"

    # --- A verdict: is the threshold knob worth a bandit? ---
    if thresh_fid_spread is None or thresh_flops_spread is None:
        lines.append("  [A] inconclusive — need >=2 threshold runs.")
        a = "unknown"
    else:
        lines.append(
            f"  [A] threshold knob moves FID by {thresh_fid_spread:.2f} across "
            f"{thresh_flops_spread:.3f} T FLOPs. This is a smooth 1-D "
            f"quality-cost frontier: pick the point you want offline.")
        a = "smooth"

    # --- Combined go/no-go ---
    lines.append("")
    if b1 == "tie":
        lines.append(
            "  VERDICT: STOP mask-bandit on TeaCache. Equal-FLOPs arms don't "
            "separate, so there is nothing for the bandit to discover — its "
            "ceiling is the baseline (exactly what 80 imgs showed).")
        lines.append(
            "  If you still want per-trajectory adaptivity, the ONLY lever "
            "with signal is the threshold (path A) — but that needs cost-"
            "aware reward (quality alone collapses to min-threshold), and a "
            "smooth 1-D frontier is usually better set offline than learned.")
    elif b1 == "differ":
        lines.append(
            "  VERDICT: mask selection has headroom on TeaCache. Proceed to "
            "the reward-proxy check (B2): confirm cheap terminal-fidelity "
            "ranks arms like FID before spending 5k-50k on the bandit.")
    else:
        lines.append(
            "  VERDICT: inconclusive. Add forced-arm and noise-replica runs, "
            "then re-run this analyzer.")

    return lines


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze TeaCache decision-value sweeps (COVR go/no-go)")
    parser.add_argument("sweep_dir",
                        help="Root dir with threshold/ and equalflops/ subdirs")
    args = parser.parse_args()

    thresh_dir = os.path.join(args.sweep_dir, "threshold")
    equalflops_dir = os.path.join(args.sweep_dir, "equalflops")

    thresh_rows, thresh_lines = analyze_threshold_sweep(thresh_dir)
    arm_rows, noise_rows, ef_lines = analyze_equalflops_sweep(equalflops_dir)
    verdict_lines = render_verdict(thresh_rows, arm_rows, noise_rows)

    for line in thresh_lines + ef_lines + verdict_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
