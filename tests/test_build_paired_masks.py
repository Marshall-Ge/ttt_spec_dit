import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze" / "build_paired_masks.py"


def _run(tmp_path, *args):
    output = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output", str(output),
            "--version-key", "runtime-version",
            *args,
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(output.read_text(encoding="utf-8")), completed


def _error(tmp_path, *args):
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output", str(tmp_path / "manifest.json"),
            "--version-key", "runtime-version",
            *args,
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    return completed.stderr


def test_two_arm_manifest_roundtrip(tmp_path):
    manifest, _ = _run(
        tmp_path,
        "--num-steps", "50",
        "--refresh-count", "12",
        "--max-taylor-gap", "4",
        "--uniform-steps", "0,2,5,10,15,20,25,30,35,40,45,47",
        "--learned-steps", "0,4,5,6,10,15,20,25,30,35,40,45",
    )

    arms = {arm["strategy_id"]: arm for arm in manifest["strategies"]}
    assert set(arms) == {"uniform_static", "risk_learned"}
    assert manifest["baseline_strategy_id"] == "uniform_static"
    for arm in arms.values():
        assert arm["method"] == "speca"
        assert sum(arm["params"]["refresh_mask"]) == 12
        assert arm["modeled_flops"] == 12.0
    uniform_steps = [
        step for step, flag in enumerate(arms["uniform_static"]["params"]["refresh_mask"])
        if flag
    ]
    learned_steps = [
        step for step, flag in enumerate(arms["risk_learned"]["params"]["refresh_mask"])
        if flag
    ]
    assert uniform_steps == [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 47]
    assert learned_steps == [0, 4, 5, 6, 10, 15, 20, 25, 30, 35, 40, 45]


def test_rejects_missing_first_step(tmp_path):
    message = _error(
        tmp_path,
        "--num-steps", "15",
        "--refresh-count", "3",
        "--max-taylor-gap", "4",
        "--uniform-steps", "0,5,10",
        "--learned-steps", "1,6,11",
    )
    assert "first step must refresh" in message


def test_rejects_unequal_refresh_counts(tmp_path):
    message = _error(
        tmp_path,
        "--num-steps", "15",
        "--refresh-count", "4",
        "--max-taylor-gap", "4",
        "--uniform-steps", "0,4,8,12",
        "--learned-steps", "0,5,10",
    )
    assert "refresh count" in message


def test_rejects_unsafe_gap(tmp_path):
    message = _error(
        tmp_path,
        "--num-steps", "15",
        "--refresh-count", "3",
        "--max-taylor-gap", "4",
        "--uniform-steps", "0,5,10",
        "--learned-steps", "0,3,14",
    )
    assert "Taylor gap" in message


def test_rejects_step_out_of_range(tmp_path):
    message = _error(
        tmp_path,
        "--num-steps", "15",
        "--refresh-count", "3",
        "--max-taylor-gap", "4",
        "--uniform-steps", "0,5,10",
        "--learned-steps", "0,5,15",
    )
    assert "exceeds num_steps" in message
