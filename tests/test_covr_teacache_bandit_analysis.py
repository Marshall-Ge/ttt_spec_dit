import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze.analyze_covr_teacache_bandit import analyze_state


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_covr_teacache_bandit.py"


def _fixtures():
    assignments = []
    feedback = []
    losses = {
        "arm_a": [1.0, 2.0, 1.0, 2.0],
        "arm_b": [3.0, 4.0, 3.0, 4.0],
    }
    offsets = {"arm_a": 0, "arm_b": 0}
    for index in range(8):
        arm_id = "arm_a" if index % 2 == 0 else "arm_b"
        offset = offsets[arm_id]
        offsets[arm_id] += 1
        assignments.append({
            "session_id": "session",
            "trajectory_id": index,
            "prequential_index": index,
            "template_id": arm_id,
            "propensity": 0.5,
            "manifest_hash": "hash",
            "sample_count": 1,
        })
        feedback.append({
            "trajectory_id": index,
            "template_id": arm_id,
            "sentinel_propensity": 1.0,
            "horizon": 1,
            "terminal_fidelity_loss": losses[arm_id][offset],
            "terminal_quality_loss": None,
        })
    state = {
        "session_id": "session",
        "manifest_hash": "hash",
        "assignments": assignments,
        "feedback": feedback,
        "run_identity": {},
    }
    manifest = {
        "baseline_strategy_id": "arm_a",
        "strategies": [
            {
                "strategy_id": "arm_a",
                "method": "teacache",
                "params": {"rel_l1_thresh": 0.25},
                "modeled_flops": 10.0,
            },
            {
                "strategy_id": "arm_b",
                "method": "teacache",
                "params": {"refresh_mask": [True, False]},
                "modeled_flops": 1.0,
            },
        ],
    }
    return state, manifest


def test_analysis_reports_unpaired_comparisons_and_snips_gain():
    state, manifest = _fixtures()

    summary, tables = analyze_state(state, manifest, window_size=4)

    assert summary["assignment_count"] == 8
    assert summary["feedback_count"] == 8
    assert summary["per_image_crossover_identifiable"] is False
    assert summary["terminal_reward_degenerate"] is False
    assert summary["last_window_max_arm_share"] == pytest.approx(0.5)
    assert summary["pairwise"]["arm_a"]["arm_b"] == 1.0
    assert summary["pairwise"]["arm_b"]["arm_a"] == 0.0
    assert summary["baseline_terminal_loss_snips"] == pytest.approx(1.5)
    assert summary["policy_terminal_loss_ipw_mean"] == pytest.approx(2.5)
    assert summary["estimated_online_gain_vs_baseline"] == pytest.approx(-1.0)
    assert len(tables["arm_share_timeline"]) == 4


def test_analysis_flags_all_zero_terminal_rewards():
    state, manifest = _fixtures()
    for feedback in state["feedback"]:
        feedback["terminal_fidelity_loss"] = 0.0

    summary, _ = analyze_state(state, manifest, window_size=4)

    assert summary["terminal_reward_degenerate"] is True
    assert summary["terminal_reward_zero_fraction"] == 1.0


def test_analysis_cli_writes_machine_readable_report(tmp_path):
    state, manifest = _fixtures()
    state_path = tmp_path / "state.json"
    manifest_path = tmp_path / "manifest.json"
    report_dir = tmp_path / "report"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(state_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(report_dir),
            "--window-size",
            "4",
            "--no-plots",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["plots_generated"] == []
    assert (report_dir / "arm_summary.csv").is_file()
    assert (report_dir / "arm_share_timeline.csv").is_file()
    assert (report_dir / "reward_observations.csv").is_file()
    assert (report_dir / "pairwise_probability.csv").is_file()


# ===========================================================================
# Contextual (schema v2) state analysis
# ===========================================================================


def _contextual_fixtures():
    # Three trajectories where the contextual policy picks baseline on
    # trajectories 0/1 (low fidelity loss) and thresh_0p35 on trajectory 2.
    # The point is to exercise the contextual diagnostics end-to-end; the sign
    # of estimated_contextual_gain_vs_best_static is NOT asserted (synthetic
    # data cannot beat the selection-biased best-static floor by construction).
    picks = [
        ("baseline", 0.10, 0.40),
        ("baseline", 0.12, 0.40),
        ("thresh_0p35", 0.20, 0.65),
    ]
    assignments = []
    feedback = []
    for index, (arm, fid, cost) in enumerate(picks):
        assignments.append({
            "session_id": "session",
            "trajectory_id": index,
            "prequential_index": index,
            "template_id": arm,
            "propensity": 0.5,
            "manifest_hash": "hash",
            "sample_count": 1,
            "context": [
                float(index),
                float(index) * 0.5,
                1.0 - float(index) * 0.25,
            ],
        })
        feedback.append({
            "trajectory_id": index,
            "template_id": arm,
            "sentinel_propensity": 1.0,
            "horizon": 1,
            "terminal_fidelity_loss": fid,
            "terminal_quality_loss": None,
            "terminal_efficiency_loss": cost,
            "combined_loss": fid + 1e-3 * cost,
        })
    dim = 3
    state = {
        "schema_version": 2,
        "session_id": "session",
        "manifest_hash": "hash",
        "context_dim": dim,
        "linucb_alpha": 1.0,
        "linucb": {
            "baseline": {"A": np.eye(dim).tolist(), "b": [0.1, 0.0, 0.0]},
            "thresh_0p35": {"A": np.eye(dim).tolist(), "b": [0.0, 0.2, 0.0]},
        },
        "assignments": assignments,
        "feedback": feedback,
        "run_identity": {"prefix_steps": 3, "efficiency_lambda": 1e-3},
    }
    manifest = {
        "baseline_strategy_id": "baseline",
        "strategies": [
            {
                "strategy_id": "baseline",
                "method": "teacache",
                "params": {"rel_l1_thresh": 0.25},
                "modeled_flops": 10.0,
            },
            {
                "strategy_id": "thresh_0p35",
                "method": "teacache",
                "params": {"rel_l1_thresh": 0.35},
                "modeled_flops": 6.0,
            },
        ],
    }
    return state, manifest


def test_contextual_analysis_reports_combined_dimension():
    # _contextual_fixtures populates combined_loss = fid + 1e-3*cost, so the
    # combined dimension (the bandit's actual objective) is analyzable and
    # should mirror the fidelity SNIPS/best-static/gain structure.
    state, manifest = _contextual_fixtures()
    summary, _ = analyze_state(state, manifest, window_size=3)
    assert summary["combined_loss_available"] is True
    # baseline combined SNIPS = weighted_mean([0.1004, 0.1204]) = 0.1104 beats
    # thresh_0p35 (0.20065), so best static on combined is baseline too.
    assert summary["best_static_arm_id_in_sample_combined"] == "baseline"
    assert summary["best_static_combined_loss_snips_in_sample"] == \
        pytest.approx(0.1104, rel=1e-4)
    assert summary["contextual_policy_combined_loss_ips"] is not None
    assert summary["policy_combined_loss_effective_sample_size"] > 0.0
    assert summary["estimated_contextual_gain_vs_best_static_combined"] is not None


def test_combined_dimension_unavailable_when_no_measured_cost():
    # A pure-fidelity probe (lambda=0 or FLOPs not profiled) leaves combined_loss
    # null; the analyzer must report the dimension unavailable rather than
    # fabricate a gain.
    state, manifest = _contextual_fixtures()
    for fb in state["feedback"]:
        fb["combined_loss"] = None
        fb["terminal_efficiency_loss"] = None
    summary, _ = analyze_state(state, manifest, window_size=3)
    assert summary["combined_loss_available"] is False
    assert summary["contextual_policy_combined_loss_ips"] is None
    assert summary["best_static_arm_id_in_sample_combined"] is None
    assert summary["estimated_contextual_gain_vs_best_static_combined"] is None


def test_contextual_analysis_reports_contextual_diagnostics():
    state, manifest = _contextual_fixtures()

    summary, tables = analyze_state(state, manifest, window_size=3)

    assert summary["contextual"] is True
    assert summary["schema_version"] == 2
    assert summary["context_dim"] == 3
    assert summary["prefix_steps"] == 3
    assert summary["efficiency_lambda"] == pytest.approx(1e-3)
    assert summary["linucb_alpha"] == pytest.approx(1.0)
    assert summary["policy_scope"] == "contextual LinUCB (deferred-commit)"
    # Two arms with distinct theta norms -> contexts can prefer different arms.
    assert summary["linucb_theta_norms_distinct"] is True
    assert set(summary["linucb_theta_norm"]) == {"baseline", "thresh_0p35"}
    # One context row per rewarded trajectory.
    assert summary["context_observations"] == 3
    assert summary["contextual_policy_loss_ips"] is not None
    assert summary["estimated_contextual_gain_vs_best_static"] is not None
    # context_observations table carries the expanded context + efficiency cols.
    ctx_table = tables["context_observations"]
    assert len(ctx_table) == 3
    assert "c0" in ctx_table[0]
    assert "c2" in ctx_table[0]
    assert "terminal_efficiency_loss" in ctx_table[0]
    assert "combined_loss" in ctx_table[0]


def test_contextual_cli_writes_context_observations_csv(tmp_path):
    state, manifest = _contextual_fixtures()
    state_path = tmp_path / "ctx_state.json"
    manifest_path = tmp_path / "manifest.json"
    report_dir = tmp_path / "report"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(state_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(report_dir),
            "--window-size",
            "3",
            "--no-plots",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["contextual"] is True
    assert (report_dir / "context_observations.csv").is_file()
    # reward_observations.csv gains the efficiency columns.
    reward_header = (report_dir / "reward_observations.csv").read_text().splitlines()[0]
    assert "terminal_efficiency_loss" in reward_header
    assert "combined_loss" in reward_header
