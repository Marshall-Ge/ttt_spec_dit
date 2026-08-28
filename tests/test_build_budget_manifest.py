import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_budget_manifest.py"


def _run_builder(tmp_path, *args):
    output = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--version-key",
            "test-version",
            *args,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8")), completed


def test_legacy_single_budget_keeps_original_arm_ids(tmp_path):
    manifest, _ = _run_builder(
        tmp_path,
        "--num-steps",
        "50",
        "--refresh-count",
        "8",
    )

    ids = [arm["strategy_id"] for arm in manifest["strategies"]]
    assert ids == ["front_loaded", "uniform", "back_loaded", "geometric"]
    assert manifest["baseline_strategy_id"] == "uniform"
    assert all(arm["modeled_flops"] == 8.0 for arm in manifest["strategies"])


def test_mixed_manifest_emits_disjoint_threshold_and_pattern_arms(tmp_path):
    manifest, _ = _run_builder(
        tmp_path,
        "--num-steps",
        "50",
        "--threshold-arm",
        "0.15:0.30",
        "--threshold-arm",
        "0.25:0.48",
        "--refresh-counts",
        "8",
        "10",
        "--baseline-arm",
        "threshold_0p25",
    )

    arms = {arm["strategy_id"]: arm for arm in manifest["strategies"]}
    assert len(arms) == 10
    assert manifest["baseline_strategy_id"] == "threshold_0p25"
    assert arms["threshold_0p15"]["params"] == {
        "rel_l1_thresh": 0.15,
        "expected_skip_rate": 0.3,
        "num_steps": 50,
    }
    assert arms["threshold_0p15"]["modeled_flops"] == 35.0
    assert "refresh_mask" not in arms["threshold_0p15"]["params"]

    pattern = arms["pattern_uniform_k8"]
    assert pattern["params"]["refresh_count"] == 8
    assert sum(pattern["params"]["refresh_mask"]) == 8
    assert "rel_l1_thresh" not in pattern["params"]
    assert "pattern_geometric_k10" in arms


def test_random_null_arms_are_reproducible_and_budget_matched(tmp_path):
    first, _ = _run_builder(
        tmp_path / "first",
        "--num-steps", "12",
        "--refresh-count", "5",
        "--random-count", "3",
        "--random-seed", "17",
    )
    second, _ = _run_builder(
        tmp_path / "second",
        "--num-steps", "12",
        "--refresh-count", "5",
        "--random-count", "3",
        "--random-seed", "17",
    )

    first_random = [
        arm for arm in first["strategies"]
        if arm["source"] == "budget_random_null_k5"
    ]
    second_random = [
        arm for arm in second["strategies"]
        if arm["source"] == "budget_random_null_k5"
    ]
    assert [arm["params"]["refresh_mask"] for arm in first_random] == [
        arm["params"]["refresh_mask"] for arm in second_random
    ]
    assert len(first_random) == 3
    assert len({tuple(arm["params"]["refresh_mask"]) for arm in first_random}) == 3
    for arm in first_random:
        mask = arm["params"]["refresh_mask"]
        assert len(mask) == 12
        assert sum(mask) == 5
        assert mask[:3] == [True, True, True]
        assert arm["modeled_flops"] == 5.0
    assert first["baseline_strategy_id"] == "pattern_uniform_k5"
