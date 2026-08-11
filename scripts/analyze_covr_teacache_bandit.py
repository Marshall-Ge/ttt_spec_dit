#!/usr/bin/env python3
"""Analyze a persisted COVR TeaCache strategy-bandit state.

The state contains randomized assignments and delayed sentinel feedback, but it
contains no paired counterfactual outcome for unselected arms. The report
therefore labels pairwise comparisons as unpaired observed probabilities and
uses propensity-weighted estimates for static-arm comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#60646c"
GRID = "#d9d9e0"
BLUE = "#256abf"
BLUE_DARK = "#1c5cab"
RED = "#d03b3b"
NEUTRAL = "#f0efec"


def _load_object(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]):
    # extrasaction='ignore' so each CSV's fieldnames is a curated projection of
    # the row dicts (e.g. reward_observations omits the raw context vector that
    # only context_observations.csv expands). restval writes an empty cell for
    # keys a row does not carry (e.g. shorter context vectors).
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _feedback_loss(feedback: Mapping[str, Any]) -> tuple[Optional[float], str]:
    quality = _finite_float(feedback.get("terminal_quality_loss"))
    if quality is not None:
        return quality, "terminal_quality_loss"
    fidelity = _finite_float(feedback.get("terminal_fidelity_loss"))
    if fidelity is not None:
        return fidelity, "terminal_fidelity_loss"
    return None, "missing"


def _effective_sample_size(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    denominator = float(np.square(weights).sum())
    if denominator <= 0.0:
        return 0.0
    return float(weights.sum() ** 2 / denominator)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    if values.size == 0 or float(weights.sum()) <= 0.0:
        return None
    return float(np.average(values, weights=weights))


def _probability_superiority(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if left.size == 0 or right.size == 0:
        return None
    wins = 0.0
    for value in left:
        wins += float(np.count_nonzero(value < right))
        wins += 0.5 * float(np.count_nonzero(value == right))
    return wins / float(left.size * right.size)


def _dimension_stats(
        reward_rows: List[Mapping[str, Any]],
        loss_key: str,
        arm_ids: List[str]) -> Dict[str, Any]:
    """Per-arm SNIPS + contextual-policy IPS value for one loss dimension.

    Mirrors the fidelity SNIPS/policy/best-static logic but reads ``loss_key``
    from each reward row, skipping rows where that dimension is absent. Used
    for the efficiency-aware ``combined`` dimension (terminal_fidelity +
    lambda*measured_cost) so the analyzer can report whether the bandit's
    *actual* objective yields a contextual gain even when pure terminal
    fidelity shows no per-image crossover. Returns ``available=False`` when no
    row carries the key (a pure-fidelity probe with no measured cost).
    """
    rewards_by_arm: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in reward_rows:
        if row.get(loss_key) is None:
            continue
        rewards_by_arm[row["arm_id"]].append(row)
    arm_snips: Dict[str, Optional[float]] = {}
    for arm_id in arm_ids:
        rows = rewards_by_arm.get(arm_id, [])
        losses = np.asarray([row[loss_key] for row in rows], dtype=np.float64)
        weights = np.asarray(
            [row["arm_snips_weight"] for row in rows], dtype=np.float64)
        arm_snips[arm_id] = _weighted_mean(losses, weights)
    valid = {key: value for key, value in arm_snips.items()
             if value is not None}
    best_static_id = min(valid, key=valid.get) if valid else None
    best_static_loss = valid.get(best_static_id) if best_static_id else None
    policy_rows = [row for row in reward_rows if row.get(loss_key) is not None]
    policy_values = np.asarray(
        [row[loss_key] for row in policy_rows], dtype=np.float64)
    policy_weights = np.asarray(
        [row["policy_weight"] for row in policy_rows], dtype=np.float64)
    return {
        "available": bool(policy_rows),
        "arm_snips": arm_snips,
        "best_static_id": best_static_id,
        "best_static_loss": best_static_loss,
        "policy_loss": _weighted_mean(policy_values, policy_weights),
        "policy_ess": _effective_sample_size(policy_weights),
    }


def _manifest_metadata(manifest: Optional[Mapping[str, Any]]) -> Dict[str, Dict]:
    if manifest is None:
        return {}
    result = {}
    for strategy in manifest.get("strategies", []):
        strategy_id = str(strategy["strategy_id"])
        result[strategy_id] = {
            "method": strategy.get("method"),
            "modeled_flops": strategy.get("modeled_flops"),
            "params": strategy.get("params", {}),
            "source": strategy.get("source", ""),
        }
    return result


def analyze_state(
        state: Mapping[str, Any], manifest: Optional[Mapping[str, Any]],
        window_size: int) -> tuple[Dict[str, Any], Dict[str, List[Dict]]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    assignments = sorted(
        state.get("assignments", []), key=lambda row: int(row["prequential_index"]))
    feedback = state.get("feedback", [])
    if not assignments:
        raise ValueError("bandit state has no assignments")

    assignment_by_trajectory = {}
    for row in assignments:
        trajectory_id = int(row["trajectory_id"])
        if trajectory_id in assignment_by_trajectory:
            raise ValueError(f"duplicate assignment for trajectory {trajectory_id}")
        assignment_by_trajectory[trajectory_id] = row

    arm_metadata = _manifest_metadata(manifest)
    arm_ids = list(arm_metadata)
    for row in assignments:
        arm_id = str(row["template_id"])
        if arm_id not in arm_ids:
            arm_ids.append(arm_id)

    assignment_counts = Counter(str(row["template_id"]) for row in assignments)
    reward_rows = []
    invalid_feedback = 0
    for row in feedback:
        trajectory_id = int(row["trajectory_id"])
        assignment = assignment_by_trajectory.get(trajectory_id)
        if assignment is None:
            invalid_feedback += 1
            continue
        arm_id = str(row["template_id"])
        if arm_id != str(assignment["template_id"]):
            invalid_feedback += 1
            continue
        loss, reward_kind = _feedback_loss(row)
        if loss is None or loss < 0.0:
            invalid_feedback += 1
            continue
        assignment_propensity = float(assignment["propensity"])
        sentinel_propensity = float(row["sentinel_propensity"])
        joint_propensity = assignment_propensity * sentinel_propensity
        if joint_propensity <= 0.0:
            invalid_feedback += 1
            continue
        # Contextual telemetry (None on non-contextual states). The context
        # vector and the efficiency cost are carried alongside the reward so
        # context_observations.csv can correlate per-trajectory features with
        # the chosen arm, its loss, and its measured FLOPs ratio.
        context = assignment.get("context")
        terminal_efficiency_loss = _finite_float(
            row.get("terminal_efficiency_loss"))
        combined_loss = _finite_float(row.get("combined_loss"))
        reward_rows.append({
            "trajectory_id": trajectory_id,
            "prequential_index": int(assignment["prequential_index"]),
            "arm_id": arm_id,
            "loss": loss,
            "reward_kind": reward_kind,
            "assignment_propensity": assignment_propensity,
            "sentinel_propensity": sentinel_propensity,
            "joint_propensity": joint_propensity,
            "arm_snips_weight": 1.0 / joint_propensity,
            "policy_weight": 1.0 / sentinel_propensity,
            "context": list(context) if context is not None else None,
            "terminal_efficiency_loss": terminal_efficiency_loss,
            "combined_loss": combined_loss,
        })

    reward_rows.sort(key=lambda row: row["prequential_index"])
    rewards_by_arm = defaultdict(list)
    for row in reward_rows:
        rewards_by_arm[row["arm_id"]].append(row)

    arm_rows = []
    arm_snips = {}
    for arm_id in arm_ids:
        rows = rewards_by_arm[arm_id]
        losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
        weights = np.asarray(
            [row["arm_snips_weight"] for row in rows], dtype=np.float64)
        count = assignment_counts[arm_id]
        snips = _weighted_mean(losses, weights)
        arm_snips[arm_id] = snips
        midpoint = losses.size // 2
        first = losses[:midpoint]
        second = losses[midpoint:]
        mean = float(losses.mean()) if losses.size else None
        std = float(losses.std(ddof=1)) if losses.size > 1 else None
        arm_rows.append({
            "arm_id": arm_id,
            "assignments": count,
            "assignment_share": count / len(assignments),
            "feedback_count": int(losses.size),
            "feedback_coverage": (losses.size / count) if count else 0.0,
            "loss_mean": mean,
            "loss_median": float(np.median(losses)) if losses.size else None,
            "loss_std": std,
            "loss_cv": (
                std / mean if std is not None and mean is not None and mean > 0.0
                else None),
            "loss_p95": (
                float(np.quantile(losses, 0.95)) if losses.size else None),
            "loss_zero_fraction": (
                float(np.mean(losses <= 1e-12)) if losses.size else None),
            "loss_unique_count": int(np.unique(losses).size),
            "early_late_drift": (
                float(second.mean() - first.mean())
                if first.size and second.size else None),
            "snips_loss": snips,
            "snips_effective_sample_size": _effective_sample_size(weights),
            **arm_metadata.get(arm_id, {}),
        })

    timeline_rows = []
    num_windows = math.ceil(len(assignments) / window_size)
    for window_index in range(num_windows):
        start = window_index * window_size
        stop = min(start + window_size, len(assignments))
        counts = Counter(
            str(row["template_id"]) for row in assignments[start:stop])
        for arm_id in arm_ids:
            timeline_rows.append({
                "window_index": window_index,
                "start_prequential_index": start,
                "end_prequential_index": stop - 1,
                "arm_id": arm_id,
                "assignments": counts[arm_id],
                "share": counts[arm_id] / (stop - start),
            })

    pairwise_rows = []
    pairwise_matrix = {}
    for left_id in arm_ids:
        left = np.asarray(
            [row["loss"] for row in rewards_by_arm[left_id]], dtype=np.float64)
        pairwise_matrix[left_id] = {}
        for right_id in arm_ids:
            right = np.asarray(
                [row["loss"] for row in rewards_by_arm[right_id]],
                dtype=np.float64)
            probability = _probability_superiority(left, right)
            pairwise_matrix[left_id][right_id] = probability
            pairwise_rows.append({
                "left_arm": left_id,
                "right_arm": right_id,
                "probability_left_loss_is_lower": probability,
                "left_feedback_count": int(left.size),
                "right_feedback_count": int(right.size),
            })

    policy_values = np.asarray(
        [row["loss"] for row in reward_rows], dtype=np.float64)
    policy_weights = np.asarray(
        [row["policy_weight"] for row in reward_rows], dtype=np.float64)
    policy_loss = _weighted_mean(policy_values, policy_weights)
    baseline_id = str(
        state.get("run_identity", {}).get("baseline_strategy_id")
        or (manifest or {}).get("baseline_strategy_id")
        or arm_ids[0])
    baseline_loss = arm_snips.get(baseline_id)
    valid_static = {key: value for key, value in arm_snips.items()
                    if value is not None}
    best_static_id = min(valid_static, key=valid_static.get) if valid_static else None
    best_static_loss = valid_static.get(best_static_id) if best_static_id else None

    # Efficiency-aware combined dimension (the bandit's actual objective):
    # SNIPS/best-static/policy-gain recomputed on combined_loss so the Pareto
    # thesis can be checked independently of pure terminal fidelity.
    combined_dim = _dimension_stats(reward_rows, "combined_loss", arm_ids)

    # ---- Contextual (deferred-commit LinUCB) telemetry ---------------------
    # Detected by schema version >= 2 (the contextual bandit) or by any
    # assignment carrying a context vector. The IPS policy-value estimate is
    # computed the same way as the non-contextual path (per-trajectory
    # propensity is logged at commit time), so the existing policy_loss /
    # best_static_loss ARE the contextual-policy-vs-best-static comparison;
    # we surface them under contextual names and add the LinUCB diagnostics.
    is_contextual = (
        int(state.get("schema_version", 1)) >= 2
        or any(row.get("context") is not None for row in reward_rows))
    context_dim = _finite_float(state.get("context_dim"))
    linucb_alpha = _finite_float(state.get("linucb_alpha"))
    run_identity = state.get("run_identity", {}) or {}
    prefix_steps = _finite_float(run_identity.get("prefix_steps"))
    efficiency_lambda = _finite_float(run_identity.get("efficiency_lambda"))
    linucb_theta_norm: Dict[str, Optional[float]] = {}
    linucb_state = state.get("linucb") or {}
    for arm_id in arm_ids:
        entry = linucb_state.get(arm_id)
        if not entry or "A" not in entry or "b" not in entry:
            linucb_theta_norm[arm_id] = None
            continue
        try:
            a_mat = np.asarray(entry["A"], dtype=np.float64)
            b_vec = np.asarray(entry["b"], dtype=np.float64)
            theta = np.linalg.solve(a_mat, b_vec)
            linucb_theta_norm[arm_id] = float(np.linalg.norm(theta))
        except (np.linalg.LinAlgError, ValueError):
            linucb_theta_norm[arm_id] = None
    # context_observations: one row per rewarded trajectory with its context
    # vector expanded into named columns (c0..cN-1). Only emitted when the run
    # actually produced contexts; downstream tooling treats absence as
    # "non-contextual state".
    context_rows: List[Dict[str, Any]] = []
    context_width = 0
    if is_contextual:
        for row in reward_rows:
            ctx = row.get("context")
            if ctx is None:
                continue
            context_width = max(context_width, len(ctx))
        for row in reward_rows:
            ctx = row.get("context") or []
            base = {
                "trajectory_id": row["trajectory_id"],
                "prequential_index": row["prequential_index"],
                "arm_id": row["arm_id"],
                "loss": row["loss"],
                "terminal_efficiency_loss": row.get("terminal_efficiency_loss"),
                "combined_loss": row.get("combined_loss"),
                "assignment_propensity": row["assignment_propensity"],
            }
            for idx in range(context_width):
                base[f"c{idx}"] = float(ctx[idx]) if idx < len(ctx) else None
            context_rows.append(base)
    distinct_theta_norms = {
        value for value in linucb_theta_norm.values()
        if value is not None}

    first_window = [row for row in timeline_rows if row["window_index"] == 0]
    last_window = [row for row in timeline_rows
                   if row["window_index"] == num_windows - 1]
    global_losses = np.asarray(
        [row["loss"] for row in reward_rows], dtype=np.float64)
    reward_midpoint = global_losses.size // 2
    first_rewards = global_losses[:reward_midpoint]
    last_rewards = global_losses[reward_midpoint:]
    uniform_share = 1.0 / len(arm_ids)
    last_shares = np.asarray(
        [row["share"] for row in last_window], dtype=np.float64)

    def normalized_entropy(rows):
        shares = np.asarray([row["share"] for row in rows], dtype=np.float64)
        positive = shares[shares > 0.0]
        if len(arm_ids) <= 1 or positive.size == 0:
            return 0.0
        return float(-(positive * np.log(positive)).sum() / math.log(len(arm_ids)))

    summary = {
        "schema_version": int(state.get("schema_version", 1)),
        "session_id": state.get("session_id"),
        "manifest_hash": state.get("manifest_hash"),
        "baseline_arm_id": baseline_id,
        "assignment_count": len(assignments),
        "feedback_count": len(reward_rows),
        "invalid_feedback_count": invalid_feedback,
        "terminal_feedback_coverage": len(reward_rows) / len(assignments),
        "arm_count": len(arm_ids),
        "window_size": window_size,
        "first_window_normalized_share_entropy": normalized_entropy(first_window),
        "last_window_normalized_share_entropy": normalized_entropy(last_window),
        "uniform_arm_share": uniform_share,
        "last_window_max_arm_share": float(last_shares.max()),
        "last_window_l1_distance_from_uniform": float(
            np.abs(last_shares - uniform_share).sum()),
        "terminal_reward_zero_fraction": (
            float(np.mean(global_losses <= 1e-12))
            if global_losses.size else None),
        "terminal_reward_unique_count": int(np.unique(global_losses).size),
        "terminal_reward_early_late_drift": (
            float(last_rewards.mean() - first_rewards.mean())
            if first_rewards.size and last_rewards.size else None),
        "terminal_reward_degenerate": bool(
            not global_losses.size or np.all(global_losses <= 1e-12)),
        "policy_terminal_loss_ipw_mean": policy_loss,
        "policy_terminal_loss_effective_sample_size": _effective_sample_size(
            policy_weights),
        "baseline_terminal_loss_snips": baseline_loss,
        "estimated_online_gain_vs_baseline": (
            baseline_loss - policy_loss
            if baseline_loss is not None and policy_loss is not None else None),
        "best_static_arm_id_in_sample": best_static_id,
        "best_static_terminal_loss_snips_in_sample": best_static_loss,
        "estimated_online_gain_vs_best_static_in_sample": (
            best_static_loss - policy_loss
            if best_static_loss is not None and policy_loss is not None else None),
        "counterfactual_warning": (
            "The bandit state has outcomes only for selected arms. The pairwise "
            "matrix is an unpaired observed probability, not a per-image "
            "crossover matrix. SNIPS estimates can have high variance; inspect "
            "effective sample sizes before claiming online gain."),
        "per_image_crossover_identifiable": False,
        "pairwise_metric": "observed_unpaired_probability_left_loss_is_lower",
        "policy_scope": (
            "contextual LinUCB (deferred-commit)" if is_contextual
            else "per-image assignment, non-contextual epsilon-greedy"),
        "policy_scope_warning": (
            "The contextual policy selects the arm from a causal-prefix context "
            "after a forced-calc prefix; the IPS policy value below compares it "
            "to the best static arm. Per-image counterfactuals for unselected "
            "arms remain unidentified, so the gain is an IPS policy-value "
            "comparison, not an oracle." if is_contextual else
            "The current bandit learns a global arm mean. Batch size 1 makes "
            "the assignment and reward per-image, but it does not condition "
            "selection on image features."),
        "contextual": bool(is_contextual),
        "context_dim": (int(context_dim) if context_dim is not None else None),
        "prefix_steps": (int(prefix_steps) if prefix_steps is not None else None),
        "efficiency_lambda": efficiency_lambda,
        "linucb_alpha": linucb_alpha,
        "linucb_theta_norm": linucb_theta_norm,
        "linucb_theta_norms_distinct": (
            len(distinct_theta_norms) > 1 if distinct_theta_norms else False),
        "context_observations": len(context_rows),
        # Contextual gain vs best static arm (IPS policy value vs best static
        # SNIPS). Same estimator as the non-contextual path; surfaced under a
        # contextual name so reports name what they measure.
        "contextual_policy_loss_ips": policy_loss if is_contextual else None,
        "estimated_contextual_gain_vs_best_static": (
            (best_static_loss - policy_loss) if is_contextual
            and best_static_loss is not None and policy_loss is not None
            else None),
        # Efficiency-aware combined dimension: does the contextual policy beat
        # the best static arm on the objective the bandit actually minimizes
        # (combined = terminal_fidelity + lambda*measured_cost)? A positive
        # gain here is the Pareto-thesis signal; None when the state carries no
        # measured cost (a pure-fidelity probe, or FLOPs not profiled).
        "combined_loss_available": combined_dim["available"],
        "contextual_policy_combined_loss_ips": (
            combined_dim["policy_loss"] if is_contextual else None),
        "policy_combined_loss_effective_sample_size": (
            combined_dim["policy_ess"] if is_contextual else None),
        "best_static_arm_id_in_sample_combined": (
            combined_dim["best_static_id"] if is_contextual else None),
        "best_static_combined_loss_snips_in_sample": (
            combined_dim["best_static_loss"] if is_contextual else None),
        "estimated_contextual_gain_vs_best_static_combined": (
            (combined_dim["best_static_loss"] - combined_dim["policy_loss"])
            if is_contextual and combined_dim["available"]
            and combined_dim["best_static_loss"] is not None
            and combined_dim["policy_loss"] is not None else None),
        "combined_dimension_note": (
            "combined_loss = terminal_fidelity + efficiency_lambda * "
            "measured_cost_ratio (the bandit's actual objective). A positive "
            "gain means the contextual policy beats the best static arm on the "
            "efficiency-aware objective, not on pure fidelity; SNIPS variance "
            "is high, so inspect policy_combined_loss_effective_sample_size "
            "before claiming gain."
            if combined_dim["available"]
            else "combined_loss unavailable: the state carries no measured "
            "cost (terminal_efficiency_loss/combined_loss are null). Run with "
            "efficiency_lambda > 0 and a FLOPs metric to populate it."),
        "arms": arm_rows,
        "pairwise": pairwise_matrix,
    }
    tables = {
        "arm_summary": arm_rows,
        "arm_share_timeline": timeline_rows,
        "reward_observations": reward_rows,
        "pairwise_probability": pairwise_rows,
        "context_observations": context_rows,
    }
    return summary, tables


def _plot_reports(output_dir: Path, summary: Mapping[str, Any],
                  tables: Mapping[str, List[Dict]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except ImportError:
        return []

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
    })
    generated = []
    arms = [row["arm_id"] for row in tables["arm_summary"]]

    cols = 2 if len(arms) > 1 else 1
    rows = math.ceil(len(arms) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, max(2.4 * rows, 3.0)),
                             squeeze=False)
    timeline = tables["arm_share_timeline"]
    for index, arm_id in enumerate(arms):
        axis = axes[index // cols][index % cols]
        values = [row for row in timeline if row["arm_id"] == arm_id]
        axis.plot([row["end_prequential_index"] for row in values],
                  [row["share"] for row in values], color=BLUE, linewidth=2)
        axis.scatter([row["end_prequential_index"] for row in values],
                     [row["share"] for row in values], color=BLUE_DARK, s=18,
                     zorder=3)
        axis.set_title(arm_id, fontsize=10)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.set_ylabel("Assignment share")
        axis.set_xlabel("Prequential index")
    for index in range(len(arms), rows * cols):
        axes[index // cols][index % cols].set_visible(False)
    fig.suptitle("COVR TeaCache arm shares by window", fontsize=14,
                 fontweight="semibold", x=0.07, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    share_path = output_dir / "arm_share_timeline.png"
    fig.savefig(share_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated.append(share_path.name)

    reward_rows = tables["reward_observations"]
    fig, axis = plt.subplots(figsize=(11, max(3.0, 0.5 * len(arms) + 1.5)))
    plotted = False
    for y, arm_id in enumerate(arms):
        values = np.asarray(
            [row["loss"] for row in reward_rows if row["arm_id"] == arm_id],
            dtype=np.float64)
        if values.size == 0:
            continue
        plotted = True
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axis.plot([q1, q3], [y, y], color=BLUE, linewidth=4,
                  solid_capstyle="round")
        axis.scatter([median], [y], color=BLUE_DARK, s=45, zorder=3,
                     edgecolor=SURFACE, linewidth=1.5)
    axis.set_yticks(range(len(arms)), arms)
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.set_xlabel("Terminal loss (median and interquartile range)")
    axis.set_title("Observed terminal reward distributions")
    if plotted and all(row["loss"] > 0.0 for row in reward_rows):
        axis.set_xscale("log")
    fig.tight_layout()
    reward_path = output_dir / "terminal_reward_distributions.png"
    fig.savefig(reward_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated.append(reward_path.name)

    matrix = np.full((len(arms), len(arms)), np.nan, dtype=np.float64)
    for row in tables["pairwise_probability"]:
        if row["probability_left_loss_is_lower"] is not None:
            i = arms.index(row["left_arm"])
            j = arms.index(row["right_arm"])
            matrix[i, j] = row["probability_left_loss_is_lower"]
    cmap = LinearSegmentedColormap.from_list(
        "covr_probability", [RED, NEUTRAL, BLUE_DARK])
    fig_size = max(6.0, 0.65 * len(arms) + 2.5)
    fig, axis = plt.subplots(figsize=(fig_size, fig_size))
    image = axis.imshow(
        matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0))
    axis.set_xticks(range(len(arms)), arms, rotation=45, ha="right")
    axis.set_yticks(range(len(arms)), arms)
    axis.set_xlabel("Right arm")
    axis.set_ylabel("Left arm")
    axis.set_title("Observed unpaired P(left loss < right loss)")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04,
                 label="Probability (0.5 = no observed advantage)")
    fig.tight_layout()
    matrix_path = output_dir / "observed_pairwise_probability.png"
    fig.savefig(matrix_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated.append(matrix_path.name)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path, help="persisted bandit state JSON")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    state = _load_object(args.state)
    manifest = _load_object(args.manifest) if args.manifest else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, tables = analyze_state(state, manifest, args.window_size)

    _write_csv(
        args.output_dir / "arm_summary.csv",
        [
            "arm_id", "assignments", "assignment_share", "feedback_count",
            "feedback_coverage", "loss_mean", "loss_median", "loss_std",
            "loss_cv", "loss_p95", "loss_zero_fraction",
            "loss_unique_count", "early_late_drift", "snips_loss",
            "snips_effective_sample_size", "method", "modeled_flops",
            "params", "source",
        ],
        tables["arm_summary"],
    )
    _write_csv(
        args.output_dir / "arm_share_timeline.csv",
        [
            "window_index", "start_prequential_index",
            "end_prequential_index", "arm_id", "assignments", "share",
        ],
        tables["arm_share_timeline"],
    )
    _write_csv(
        args.output_dir / "reward_observations.csv",
        [
            "trajectory_id", "prequential_index", "arm_id", "loss",
            "reward_kind", "assignment_propensity", "sentinel_propensity",
            "joint_propensity", "arm_snips_weight", "policy_weight",
            "terminal_efficiency_loss", "combined_loss",
        ],
        tables["reward_observations"],
    )
    _write_csv(
        args.output_dir / "pairwise_probability.csv",
        [
            "left_arm", "right_arm", "probability_left_loss_is_lower",
            "left_feedback_count", "right_feedback_count",
        ],
        tables["pairwise_probability"],
    )
    context_rows = tables.get("context_observations", [])
    if context_rows:
        # Context columns are c0..cN-1; width is the max across observed rows.
        context_width = max(
            (1 + max(
                (int(key[1:]) for key in row if key.startswith("c")
                 and key[1:].isdigit()),
                default=-1))
            for row in context_rows)
        context_fields = (
            ["trajectory_id", "prequential_index", "arm_id", "loss",
             "terminal_efficiency_loss", "combined_loss",
             "assignment_propensity"]
            + [f"c{idx}" for idx in range(context_width)])
        _write_csv(
            args.output_dir / "context_observations.csv",
            context_fields,
            context_rows,
        )

    plots = [] if args.no_plots else _plot_reports(args.output_dir, summary, tables)
    summary["plots_generated"] = plots
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"assignments: {summary['assignment_count']}")
    print(f"terminal feedback: {summary['feedback_count']} "
          f"({summary['terminal_feedback_coverage']:.1%})")
    print(f"terminal reward degenerate: {summary['terminal_reward_degenerate']}")
    print(f"last-window max arm share: "
          f"{summary['last_window_max_arm_share']:.1%}")
    print(f"baseline arm: {summary['baseline_arm_id']}")
    print("estimated gain vs baseline: "
          f"{summary['estimated_online_gain_vs_baseline']}")
    if summary.get("contextual"):
        print(f"policy scope: {summary['policy_scope']}")
        print(f"context_dim: {summary['context_dim']} "
              f"prefix_steps: {summary['prefix_steps']} "
              f"efficiency_lambda: {summary['efficiency_lambda']}")
        print(f"linucb theta norms distinct: "
              f"{summary['linucb_theta_norms_distinct']} "
              f"({summary['linucb_theta_norm']})")
        print("estimated contextual gain vs best static: "
              f"{summary['estimated_contextual_gain_vs_best_static']}")
    print(f"wrote report -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
