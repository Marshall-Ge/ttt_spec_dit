"""Tests for paired static-mask and plain-TeaCache comparison analysis."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import analyze_static_masks as asm  # noqa: E402
from analyze_teacache_sweeps import analyze_equalflops_sweep  # noqa: E402


def _write_result(path, *, offset=0, fid=100.0, is_mean=30.0,
                  total_calc=8, seed=42, n_images=500, batch_size=32,
                  num_steps=50):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "seed": seed,
            "latent_seed_offset": offset,
            "n_prompts": n_images,
            "total_images": n_images,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "dataset_start_index": 0,
            "generation_start_index": 0,
            "rel_l1_thresh": 0.5,
        },
        "aggregate": {
            "n_images": n_images,
            "fid": fid,
            "is_mean": is_mean,
            "is_std": 0.1,
            "total_calc": total_calc,
            "skip_ratio": 0.84,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_arm(root, arm, offset, **kwargs):
    arm_dir = root / "k8" / "equalflops" / f"arm_{arm}"
    result = (arm_dir / "results.json" if offset == 0 else
              arm_dir / f"rep_{offset}" / "results.json")
    _write_result(result, offset=offset, **kwargs)


def _write_threshold(root, label, offset, **kwargs):
    threshold_dir = root / "threshold" / f"thresh_{label}"
    result = (threshold_dir / "results.json" if offset == 0 else
              threshold_dir / f"rep_{offset}" / "results.json")
    _write_result(result, offset=offset, **kwargs)


def test_discovers_root_and_nested_replicas_and_pairs_by_offset(tmp_path):
    for offset in range(3):
        _write_arm(tmp_path, "uniform", offset,
                   fid=100.0 + offset, is_mean=31.0 + offset)
        _write_arm(tmp_path, "geometric", offset,
                   fid=101.0 + offset, is_mean=29.0 + offset)

    arms, warnings = asm.discover_arm_runs(str(tmp_path / "k8"))

    assert warnings == []
    assert sorted(arms) == ["geometric", "uniform"]
    assert sorted(arms["uniform"]) == [0, 1, 2]
    fid_deltas, pair_warnings = asm.paired_deltas(
        arms["uniform"], arms["geometric"], "fid")
    is_deltas, _ = asm.paired_deltas(
        arms["uniform"], arms["geometric"], "is_mean")
    assert pair_warnings == []
    assert fid_deltas == [(0, -1.0), (1, -1.0), (2, -1.0)]
    assert is_deltas == [(0, 2.0), (1, 2.0), (2, 2.0)]


def test_mismatched_config_and_missing_offsets_are_not_paired(tmp_path):
    _write_arm(tmp_path, "uniform", 0)
    _write_arm(tmp_path, "uniform", 1)
    _write_arm(tmp_path, "geometric", 0, seed=99)
    _write_arm(tmp_path, "geometric", 2)
    arms, _ = asm.discover_arm_runs(str(tmp_path / "k8"))

    deltas, warnings = asm.paired_deltas(
        arms["uniform"], arms["geometric"], "fid")

    assert deltas == []
    assert any("left-only latent offsets: [1]" in item for item in warnings)
    assert any("right-only latent offsets: [2]" in item for item in warnings)
    assert any("config.seed differs" in item for item in warnings)


def test_directory_offset_must_match_persisted_offset(tmp_path):
    path = (tmp_path / "k8" / "equalflops" / "arm_uniform" /
            "rep_2" / "results.json")
    _write_result(path, offset=3)

    arms, warnings = asm.discover_arm_runs(str(tmp_path / "k8"))

    assert arms == {}
    assert any("directory offset 2 != config offset 3" in item
               for item in warnings)


def test_paired_stats_and_exact_sign_gates():
    deltas = [(index, value) for index, value in enumerate(
        [-0.2, -0.1, -0.3, -0.15, -0.25])]

    stats = asm.paired_stats(deltas)

    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(-0.2)
    assert stats["negative"] == 5
    assert asm.exact_sign_p([value for _, value in deltas], "less") == 1 / 32
    assert asm.exact_sign_p([1, 2, 3, 4, 5], "greater") == 1 / 32
    assert asm.exact_sign_p([-1, -2, 3, 4, 5], "greater") == pytest.approx(0.5)
    assert asm.exact_sign_p([0, 0], "greater") is None


def test_calc_per_trajectory_and_nearest_threshold(tmp_path):
    _write_threshold(tmp_path, "0.50", 0, total_calc=8)
    _write_threshold(tmp_path, "0.75", 0, total_calc=12)
    thresholds, warnings = asm.discover_threshold_runs(str(tmp_path))

    label, mean_calc, distance = asm.nearest_threshold(thresholds, 8)

    assert warnings == []
    assert asm.calc_per_trajectory(thresholds["0.50"][0]) == pytest.approx(8.0)
    assert label == "0.50"
    assert mean_calc == pytest.approx(8.0)
    assert distance == pytest.approx(0.0)


def test_legacy_equalflops_analyzer_ignores_nested_replicas(tmp_path):
    _write_arm(tmp_path, "uniform", 0, fid=100.0)
    _write_arm(tmp_path, "uniform", 1, fid=999.0)
    _write_arm(tmp_path, "geometric", 0, fid=101.0)
    _write_arm(tmp_path, "geometric", 1, fid=998.0)

    rows, _, _ = analyze_equalflops_sweep(
        str(tmp_path / "k8" / "equalflops"))

    assert len(rows) == 2
    assert sorted(row["fid"] for row in rows) == [100.0, 101.0]


def test_cli_passes_five_pair_gate_and_matches_threshold(tmp_path, capsys):
    for offset in range(5):
        _write_arm(tmp_path, "uniform", offset,
                   fid=100.0 + offset, is_mean=35.0 + offset)
        _write_arm(tmp_path, "geometric", offset,
                   fid=100.5 + offset, is_mean=30.0 + offset)
    _write_threshold(tmp_path, "0.50", 0, fid=101.0,
                     is_mean=29.0, total_calc=8)

    assert asm.main([str(tmp_path), "--fid-margin", "1.86"]) == 0
    output = capsys.readouterr().out

    assert "FID non-inferiority: margin=1.860, p=0.03125 -> PASS" in output
    assert "IS superiority: margin=0.000, p=0.03125 -> PASS" in output
    assert "nearest plain TeaCache: thresh=0.50 calc/trajectory=8.00" in output
    assert "descriptive only" in output
    assert "MASK WINNER: uniform" in output
    assert "comparator still needs paired runs" in output


def test_cli_paired_threshold_gate_produces_static_winner(tmp_path, capsys):
    for offset in range(5):
        _write_arm(tmp_path, "uniform", offset,
                   fid=100.0 + offset, is_mean=35.0 + offset)
        _write_arm(tmp_path, "geometric", offset,
                   fid=100.5 + offset, is_mean=30.0 + offset)
        _write_threshold(tmp_path, "0.50", offset,
                         fid=100.4 + offset, is_mean=31.0 + offset,
                         total_calc=8)

    assert asm.main([str(tmp_path), "--fid-margin", "1.86"]) == 0
    output = capsys.readouterr().out

    assert "comparison: uniform - threshold_0.50" in output
    assert "STATIC WINNER: uniform" in output


def test_cli_requires_five_complete_pairs(tmp_path, capsys):
    for offset in range(4):
        _write_arm(tmp_path, "uniform", offset, fid=100.0, is_mean=35.0)
        _write_arm(tmp_path, "geometric", offset, fid=101.0, is_mean=30.0)

    assert asm.main([str(tmp_path)]) == 0
    output = capsys.readouterr().out

    assert "INSUFFICIENT EVIDENCE (need >= 5" in output
    assert "INSUFFICIENT EVIDENCE: need >= 5 paired latent offsets" in output


def test_cli_compares_all_discovered_arms(tmp_path, capsys):
    for offset in range(5):
        _write_arm(tmp_path, "uniform", offset,
                   fid=100.0 + offset, is_mean=35.0 + offset)
        _write_arm(tmp_path, "geometric", offset,
                   fid=101.0 + offset, is_mean=30.0 + offset)
        _write_arm(tmp_path, "random_00", offset,
                   fid=102.0 + offset, is_mean=29.0 + offset)

    assert asm.main([
        str(tmp_path), "--left", "uniform", "--right", "all",
    ]) == 0
    output = capsys.readouterr().out
    assert "comparison: uniform - geometric" in output
    assert "comparison: uniform - random_00" in output
