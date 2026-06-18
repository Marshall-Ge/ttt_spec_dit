# -*- coding: utf-8 -*-
"""
Task 2.3: Bottleneck Profiling & Rejection Tracking

Tracks and logs the exact timestamps, step indices, and DiT layers where
the feature residual explodes beyond threshold gamma, forcing a full
re-computation (rejection).

Exports a stage-vs-rejection-frequency heatmap as JSON/CSV for the
Phase 3 TTT paper motivation chart.
"""

import json
import os
import csv
import time
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Stage classifier
# ---------------------------------------------------------------------------

def denoising_stage(step_idx: int, num_steps: int) -> str:
    """Classify a denoising step into Early/Middle/Late."""
    third = num_steps // 3
    if step_idx < third:
        return "Early (Noise)"
    elif step_idx < 2 * third:
        return "Middle (Structure)"
    else:
        return "Late (Details)"


# ---------------------------------------------------------------------------
# Step-level tracker
# ---------------------------------------------------------------------------

class StepTracker:
    """Tracks per-step telemetry: residual, skip/rejection decision, latency."""

    def __init__(self):
        self.records: List[Dict] = []
        self._step_start: float = 0.0

    def start_step(self):
        self._step_start = time.time()

    def record(self, step_idx: int, timestep: int, num_steps: int,
               decision: str,  # "full_forward", "skip", "rejection"
               residual: Optional[float] = None,
               skip_count: int = 0,
               extra: Optional[Dict] = None):
        elapsed = time.time() - self._step_start
        rec = {
            "step": step_idx,
            "timestep": timestep,
            "stage": denoising_stage(step_idx, num_steps),
            "decision": decision,
            "latency_ms": round(elapsed * 1000, 2),
            "residual": round(residual, 8) if residual is not None else None,
            "consecutive_skips": skip_count,
        }
        if extra:
            rec.update(extra)
        self.records.append(rec)

    def to_dataframe(self):
        """Return records as a list of dicts (JSON-serializable)."""
        return self.records


# ---------------------------------------------------------------------------
# Stage-level aggregation for bottleneck heatmap
# ---------------------------------------------------------------------------

class BottleneckAnalyzer:
    """Aggregates per-step tracking into a stage-vs-rejection-frequency matrix.

    Produces the core motivation data for the Phase 3 TTT paper:
      - Which denoising stages suffer from high rejection rates
      - What is the average residual per stage
      - How often does skip_count hit max_skip before re-calibration
    """

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.stage_buckets = defaultdict(lambda: {
            "total_steps": 0,
            "full_forwards": 0,
            "skips": 0,
            "rejections": 0,
            "residuals": [],     # list of float residuals at rejection
            "max_skip_rejects": 0,  # rejections where skip_count == max_skip
            "threshold_rejects": 0,  # rejections due to residual >= gamma
        })

    def ingest(self, tracker: StepTracker):
        """Ingest all records from a StepTracker.

        Counts total steps, full forwards (including rejections), and skips.
        Rejection classification is handled separately via ingest_rejections().
        """
        for rec in tracker.records:
            stage = rec["stage"]
            bucket = self.stage_buckets[stage]
            bucket["total_steps"] += 1

            if rec["decision"] == "skip":
                bucket["skips"] += 1
            else:
                # full_forward or rejection — both run the full forward
                bucket["full_forwards"] += 1
            # Rejection details (count, classification, residuals) come from
            # ingest_rejections() below.

    def ingest_rejections(self, rejections: List[Dict], max_skip: int):
        """Ingest raw rejection records from TeaCacheController.

        Classifies each rejection using the 'reason' field from the controller.
        If that field is absent, falls back to the consecutive-skips heuristic.
        """
        for r in rejections:
            stage = denoising_stage(r["step"], self.num_steps)
            bucket = self.stage_buckets[stage]
            bucket["rejections"] += 1
            bucket["residuals"].append(r["residual"])
            reason = r.get("reason", "")
            if reason == "residual_threshold":
                bucket["threshold_rejects"] += 1
            elif reason == "max_skip":
                bucket["max_skip_rejects"] += 1
            else:
                # Fallback heuristic
                if r.get("consecutive_skips_before_reject", 0) >= max_skip:
                    bucket["max_skip_rejects"] += 1
                else:
                    bucket["threshold_rejects"] += 1

    def summary(self) -> Dict:
        """Return aggregated stage summary."""
        out = {}
        stage_order = ["Early (Noise)", "Middle (Structure)", "Late (Details)"]
        for stage in stage_order:
            bucket = self.stage_buckets.get(stage, {})
            n = bucket.get("total_steps", 0)
            rej = bucket.get("rejections", 0)
            residuals = bucket.get("residuals", [])
            out[stage] = {
                "total_steps": n,
                "full_forwards": bucket.get("full_forwards", 0),
                "skips": bucket.get("skips", 0),
                "rejections": rej,
                "rejection_rate": round(rej / n, 3) if n > 0 else 0.0,
                "mean_residual_at_rejection": round(float(np.mean(residuals)), 6) if residuals else None,
                "max_residual_at_rejection": round(float(np.max(residuals)), 6) if residuals else None,
                "max_skip_rejects": bucket.get("max_skip_rejects", 0),
                "threshold_rejects": bucket.get("threshold_rejects", 0),
            }
        return out

    def export_heatmap_json(self, path: str) -> None:
        """Export stage-vs-rejection-frequency data as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "num_steps": self.num_steps,
            "stages": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_heatmap_csv(self, path: str) -> None:
        """Export stage-vs-rejection-frequency data as CSV."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        summary = self.summary()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "stage", "total_steps", "full_forwards", "skips", "rejections",
                "rejection_rate", "mean_residual_at_rejection",
                "max_residual_at_rejection", "max_skip_rejects", "threshold_rejects"
            ])
            for stage, s in summary.items():
                writer.writerow([
                    stage, s["total_steps"], s["full_forwards"], s["skips"],
                    s["rejections"], s["rejection_rate"],
                    s["mean_residual_at_rejection"], s["max_residual_at_rejection"],
                    s["max_skip_rejects"], s["threshold_rejects"],
                ])


# ---------------------------------------------------------------------------
# Deep Water Reporter — formatted Markdown output
# ---------------------------------------------------------------------------

def print_deep_water_report(analyzer: BottleneckAnalyzer,
                            teacache_stats: Dict,
                            eval_agg: Optional[Dict] = None) -> str:
    """Generate a Markdown-formatted "Deep Water" bottleneck report.

    This is the core deliverable: a clear summary showing which denoising
    stages still suffer from high rejection/re-calibration rates despite
    using the SOTA adaptive cache mechanism.
    """
    summary = analyzer.summary()
    lines = []
    lines.append("## 🌊 \"Deep Water\" Bottleneck Report")
    lines.append("")
    lines.append("> **What this shows:** Even with a SOTA-inspired adaptive cache")
    lines.append("> (TeaCache-style relative L1 residual thresholding), certain")
    lines.append("> denoising stages force frequent full re-computation — the")
    lines.append("> \"Deep Water\" where features change too fast for caching to help.")
    lines.append("")
    lines.append(f"- **Cache config:** γ = {teacache_stats.get('gamma', 'N/A')}, "
                 f"max_skip = {teacache_stats.get('max_skip', 'N/A')}")
    lines.append(f"- **Total full forwards:** {teacache_stats.get('total_full_forwards', 'N/A')}")
    lines.append(f"- **Total skips:** {teacache_stats.get('total_skips', 'N/A')}")
    lines.append(f"- **Skip ratio:** {teacache_stats.get('skip_ratio', 0.0):.2%}")
    lines.append(f"- **Total rejections:** {teacache_stats.get('total_rejections', 'N/A')}")
    lines.append("")

    # Stage table
    lines.append("### Rejection Frequency by Denoising Stage")
    lines.append("")
    lines.append("| Stage | Steps | Rejections | Rejection Rate | Mean Residual @ Reject | Max Residual @ Reject | Max-Skip Rejects | Threshold Rejects |")
    lines.append("|-------|-------|------------|----------------|------------------------|------------------------|------------------|-------------------|")
    for stage in ["Early (Noise)", "Middle (Structure)", "Late (Details)"]:
        s = summary.get(stage, {})
        if s.get("total_steps", 0) == 0:
            continue
        lines.append(
            f"| {stage} | {s['total_steps']} | {s['rejections']} | "
            f"{s['rejection_rate']:.1%} | "
            f"{s['mean_residual_at_rejection'] or 'N/A'} | "
            f"{s['max_residual_at_rejection'] or 'N/A'} | "
            f"{s['max_skip_rejects']} | {s['threshold_rejects']} |"
        )
    lines.append("")

    # Identify the bottleneck stage
    worst_stage = max(summary.items(), key=lambda x: x[1]["rejection_rate"])
    lines.append(f"### 🔴 Primary Bottleneck: **{worst_stage[0]}**")
    lines.append(f"")
    lines.append(f"The **{worst_stage[0]}** stage exhibits a rejection rate of "
                 f"**{worst_stage[1]['rejection_rate']:.1%}**, meaning that even "
                 f"with adaptive thresholding, {worst_stage[1]['rejection_rate']:.0%} "
                 f"of steps in this stage cannot be cached and require full re-computation.")
    lines.append("")
    if worst_stage[1]["threshold_rejects"] > worst_stage[1]["max_skip_rejects"]:
        lines.append(f"Most rejections ({worst_stage[1]['threshold_rejects']}/{worst_stage[1]['rejections']}) "
                     f"are **residual-threshold** rejections — the features change too much between steps "
                     f"for the L1 residual check to pass at γ={teacache_stats.get('gamma', 0.1)}.")
    else:
        lines.append(f"Most rejections ({worst_stage[1]['max_skip_rejects']}/{worst_stage[1]['rejections']}) "
                     f"are **max-skip** rejections — the cache drifts too far after "
                     f"{teacache_stats.get('max_skip', 3)} consecutive skips and must re-calibrate.")
    lines.append("")
    lines.append("**Implication for Phase 3 TTT:** This stage is where a learned, online ")
    lines.append("predictor (Test-Time Training) can potentially outperform the static ")
    lines.append("residual-threshold cache by learning to predict feature evolution ")
    lines.append("adaptively, reducing the rejection rate.")
    lines.append("")

    # If evaluation aggregate is available, include it
    if eval_agg:
        lines.append("---")
        lines.append("")
        lines.append("## 📊 SOTA Baseline Performance Table")
        lines.append("")
        lines.append("| Metric | Vanilla DPMSolver++ | Our SOTA Caching Baseline |")
        lines.append("|--------|---------------------|---------------------------|")
        lines.append(f"| **Latency (s)** | {eval_agg.get('latency_vanilla_s_mean', 0):.2f} ± {eval_agg.get('latency_vanilla_s_std', 0):.2f} | "
                     f"{eval_agg.get('latency_accel_s_mean', 0):.2f} ± {eval_agg.get('latency_accel_s_std', 0):.2f} |")
        lines.append(f"| **Speedup Ratio** | 1.00× | **{eval_agg.get('speedup_ratio_mean', 1.0):.2f}×** |")
        lines.append(f"| **CLIP Score** | {eval_agg.get('clip_score_vanilla_mean', 0):.2f} ± {eval_agg.get('clip_score_vanilla_std', 0):.2f} | "
                     f"{eval_agg.get('clip_score_accel_mean', 0):.2f} ± {eval_agg.get('clip_score_accel_std', 0):.2f} |")
        lines.append(f"| **CLIP Δ** | — | {eval_agg.get('clip_delta_mean', 0):.2f} |")
        lines.append(f"| **Pixel MSE** | — | {eval_agg.get('pixel_mse_mean', 0):.6f} |")
        lines.append(f"| **Latent MSE** | — | {eval_agg.get('latent_mse_mean', 0):.6f} |")
        lines.append("")
        lines.append(f"*Evaluated on {eval_agg.get('n_prompts', 0)} prompts.*")

    return "\n".join(lines)
