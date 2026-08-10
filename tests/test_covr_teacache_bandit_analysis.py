import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analyze_covr_teacache_bandit import analyze_state


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
