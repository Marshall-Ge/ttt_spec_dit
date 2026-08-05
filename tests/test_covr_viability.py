# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import main as cli
from accelerators.covr_viability import (
    COVRViabilityRecorder,
    extract_prefix_features,
)
from scripts.analyze_covr_v2_viability import analyze


def _write_image(path: Path, value: int) -> None:
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def test_recorder_writes_verified_prefix_scalars(tmp_path):
    output = tmp_path / "viability.jsonl"
    recorder = COVRViabilityRecorder(
        str(output), prefix_steps=2, run_identity={"model": "dit"})
    recorder.begin_trajectory(
        global_idx=7,
        latent_seed=99,
        strategy_id="uniform",
        manifest_hash="manifest",
        refresh_mask=(True, True, False),
        batch_size=1,
    )
    for step in range(2):
        recorder.record_step(
            step_idx=step,
            timestep=49 - step,
            latent_input=torch.ones(2, 4, 2, 2) * (step + 1),
            noise_pred=torch.zeros(2, 4, 2, 2),
        )
    recorder.end_trajectory()
    recorder.close()

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["feature_source"] == "causal_prefix"
    assert rows[0]["prefix_verified"] is True
    assert rows[0]["strategy_id"] == "uniform"
    assert rows[0]["features"]["latent_mean"] == 1.0
    assert "noise_l2" in rows[0]["features"]


def test_recorder_rejects_unverified_prefix_and_incomplete_close(tmp_path):
    recorder = COVRViabilityRecorder(str(tmp_path / "bad.jsonl"), prefix_steps=3)
    with pytest.raises(ValueError, match="shared prefix"):
        recorder.begin_trajectory(
            global_idx=0,
            latent_seed=0,
            strategy_id="arm",
            manifest_hash="m",
            refresh_mask=(True, False, True),
        )

    recorder.begin_trajectory(
        global_idx=0,
        latent_seed=0,
        strategy_id="arm",
        manifest_hash="m",
        refresh_mask=(True, True, True),
    )
    with pytest.raises(RuntimeError, match="active trajectory"):
        recorder.close()
    recorder._file.close()


def test_extract_features_requires_single_image_batch():
    with pytest.raises(ValueError, match="batch_size=1"):
        extract_prefix_features(torch.zeros(2, 3), torch.zeros(2, 3), batch_size=2)


def _make_probe_fixture(tmp_path: Path, *, feature_source="causal_prefix"):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    for index in range(32):
        _write_image(reference_dir / f"{index:06d}.png", 128)
    arm_specs = []
    for arm_idx, arm in enumerate(("front", "uniform", "back"), start=0):
        arm_dir = tmp_path / arm
        generated = arm_dir / "generated"
        generated.mkdir(parents=True)
        jsonl = arm_dir / "viability.jsonl"
        with jsonl.open("w", encoding="utf-8") as handle:
            for index in range(32):
                best = index % 3
                value = 128 if best == arm_idx else 230
                _write_image(generated / f"{index:06d}.png", value)
                features = {
                    f"arm_signal_{j}": float(j == best) for j in range(3)
                }
                for step in range(2):
                    handle.write(json.dumps({
                        "schema_version": 1,
                        "feature_source": feature_source,
                        "model": "dit",
                        "method": "teacache",
                        "num_steps": 50,
                        "manifest_hash": "manifest",
                        "prefix_steps": 2,
                        "prefix_verified": True,
                        "batch_size": 1,
                        "latent_seed_offset": 0,
                        "global_idx": index,
                        "latent_seed": index,
                        "strategy_id": arm,
                        "step_idx": step,
                        "timestep": 49 - step,
                        "features": features,
                    }) + "\n")
        arm_specs.append(f"{arm}={jsonl}")
    return reference_dir, arm_specs


def test_analyzer_reports_oos_viability(tmp_path):
    reference_dir, arm_specs = _make_probe_fixture(tmp_path)
    report = analyze(
        reference_dir=str(reference_dir),
        arm_specs=arm_specs,
        output_path=str(tmp_path / "report.json"),
        permutations=50,
    )
    assert report["recommendation"] == "PASS"
    assert report["feature_source"] == "causal_prefix"
    assert report["n_train"] == 16
    assert report["n_test"] == 16
    assert report["capturability"] > 0.5


def test_analyzer_rejects_reference_derived_features(tmp_path):
    reference_dir, arm_specs = _make_probe_fixture(
        tmp_path, feature_source="reference_image")
    with pytest.raises(ValueError, match="causal_prefix"):
        analyze(
            reference_dir=str(reference_dir),
            arm_specs=arm_specs,
            output_path=str(tmp_path / "report.json"),
            permutations=0,
        )


def test_cli_probe_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--model", "dit", "--task", "c2i",
        "--dataset", "imagenet", "--n_prompts", "1",
    ])
    args = cli.parse_args()
    assert args.covr_viability_output is None
    assert cli.validate_args(args) is True


def test_cli_probe_requires_batch_one_and_forced_arm(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--model", "dit", "--task", "c2i",
        "--dataset", "imagenet", "--n_prompts", "1",
        "--batch_size", "2", "--covr-viability-output", "/tmp/probe.jsonl",
    ])
    args = cli.parse_args()
    assert cli.validate_args(args) is False
