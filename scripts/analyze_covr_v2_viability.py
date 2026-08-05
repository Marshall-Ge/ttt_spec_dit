#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze the COVR-v2 causal-feature viability probe.

This is an exploratory gate.  Its outcome is a recommendation about whether
an online policy has measurable headroom, not a FID/IS effectiveness claim.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image


_INDEX_RE = re.compile(r"^(\d+)(?:_|\.)")


def _read_jsonl(path: str) -> List[Mapping[str, object]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_no} is not an object")
            rows.append(row)
    return rows


def _image_map(directory: str) -> Dict[int, str]:
    result: Dict[int, str] = {}
    paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(directory, pattern)))
    for path in sorted(paths):
        match = _INDEX_RE.match(os.path.basename(path))
        if match is None:
            continue
        index = int(match.group(1))
        if index in result:
            raise ValueError(f"duplicate image index {index} in {directory}")
        result[index] = path
    return result


def _mse(path: str, reference: str) -> float:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    target = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32) / 255.0
    if image.shape != target.shape:
        raise ValueError(
            f"image shape mismatch: {path}={image.shape}, reference={target.shape}")
    return float(np.mean((image - target) ** 2))


def _feature_rows(rows: Sequence[Mapping[str, object]]) -> Tuple[int, Dict[str, float], Mapping[str, object]]:
    if not rows:
        raise ValueError("viability JSONL is empty")
    ordered = sorted(rows, key=lambda row: int(row["step_idx"]))
    expected = list(range(len(ordered)))
    actual = [int(row["step_idx"]) for row in ordered]
    if actual != expected:
        raise ValueError(f"prefix steps are not contiguous: {actual}")
    if not all(bool(row.get("prefix_verified")) for row in ordered):
        raise ValueError("viability record has an unverified shared prefix")
    if any(row.get("feature_source") != "causal_prefix" for row in ordered):
        raise ValueError("viability policy features must come from causal_prefix")
    values: Dict[str, float] = {}
    for row in ordered:
        features = row.get("features")
        if not isinstance(features, dict) or not features:
            raise ValueError("viability record has no features")
        step = int(row["step_idx"])
        for name, value in features.items():
            if not isinstance(name, str) or not isinstance(value, (int, float)):
                raise ValueError("viability features must be finite numeric scalars")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("viability features must be finite")
            values[f"step{step}_{name}"] = value
    return int(ordered[0]["global_idx"]), values, ordered[0]


def _load_arm(path: str) -> Dict[int, Tuple[Dict[str, float], Mapping[str, object]]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in _read_jsonl(path):
        index = int(row.get("global_idx", -1))
        grouped.setdefault(index, []).append(row)
    if not grouped or -1 in grouped:
        raise ValueError(f"invalid global_idx values in {path}")
    result = {}
    for index, rows in grouped.items():
        parsed_index, features, identity = _feature_rows(rows)
        if parsed_index != index:
            raise ValueError("inconsistent global_idx in trajectory records")
        result[index] = (features, identity)
    return result


def _identity(arm_rows: Mapping[int, Tuple[Dict[str, float], Mapping[str, object]]]) -> Dict[str, object]:
    first = next(iter(arm_rows.values()))[1]
    fields = ("schema_version", "model", "method", "num_steps", "manifest_hash",
              "prefix_steps", "prefix_verified", "batch_size", "latent_seed_offset")
    return {field: first.get(field) for field in fields}


def _validate_identity(
    arms: Mapping[str, Mapping[int, Tuple[Dict[str, float], Mapping[str, object]]]],
) -> Dict[str, object]:
    names = list(arms)
    if len(names) < 3:
        raise ValueError("viability gate requires at least three strategy arms")
    reference = _identity(arms[names[0]])
    if reference["prefix_verified"] is not True or reference["batch_size"] != 1:
        raise ValueError("viability probe must be verified batch-size-one data")
    for name in names[1:]:
        current = _identity(arms[name])
        for field, expected in reference.items():
            if current.get(field) != expected:
                raise ValueError(f"identity mismatch for {name}: {field}")
    return reference


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    train = np.concatenate([np.ones((len(train), 1)), train], axis=1)
    test = np.concatenate([np.ones((len(test), 1)), test], axis=1)
    reg = np.eye(train.shape[1], dtype=np.float64) * float(ridge)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(train.T @ train + reg, train.T @ y_train)
    return test @ coef


def analyze(
    *,
    reference_dir: str,
    arm_specs: Sequence[str],
    output_path: str,
    test_parity: int = 1,
    ridge: float = 1e-3,
    min_capturability: float = 0.10,
    min_policy_gain: float = 1e-7,
    permutations: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    arms: Dict[str, Dict[int, Tuple[Dict[str, float], Mapping[str, object]]]] = {}
    for spec in arm_specs:
        if "=" not in spec:
            raise ValueError(f"arm must use id=JSONL syntax: {spec}")
        name, path = spec.split("=", 1)
        if not name or not path:
            raise ValueError(f"invalid arm specification: {spec}")
        arms[name] = _load_arm(path)
    identity = _validate_identity(arms)
    names = list(arms)
    indices = sorted(next(iter(arms.values())))
    for name in names:
        if sorted(arms[name]) != indices:
            raise ValueError(f"global_idx set mismatch for {name}")

    feature_names = sorted(arms[names[0]][indices[0]][0])
    for name in names[1:]:
        for index in indices:
            current = arms[name][index][0]
            if sorted(current) != feature_names:
                raise ValueError(f"shared-prefix feature schema mismatch for {name}")
            if any(abs(float(current[key]) - float(arms[names[0]][index][0][key])) > 1e-6
                   for key in feature_names):
                raise ValueError(f"shared-prefix features differ across arms at {index}")
    x = np.asarray([[arms[names[0]][i][0][key] for key in feature_names] for i in indices], dtype=np.float64)
    image_maps = {name: _image_map(os.path.join(os.path.dirname(path), "generated"))
                  for name, path in ((spec.split("=", 1)[0], spec.split("=", 1)[1]) for spec in arm_specs)}
    reference_images = _image_map(reference_dir)
    missing = [i for i in indices if i not in reference_images]
    if missing:
        raise ValueError(f"reference is missing {len(missing)} global indices")
    losses = np.asarray([
        [_mse(image_maps[name][i], reference_images[i]) for name in names]
        for i in indices
    ], dtype=np.float64)
    if not np.isfinite(losses).all():
        raise ValueError("non-finite arm outcome")

    train_mask = np.asarray([(i % 2) == int(test_parity) for i in indices], dtype=bool)
    test_mask = ~train_mask
    if train_mask.sum() < 16 or test_mask.sum() < 16:
        return _write_report(output_path, {
            "schema_version": 1, "identity": identity, "arms": names,
            "n_images": len(indices), "recommendation": "INSUFFICIENT_DATA",
            "reason": "parity split needs at least 16 train and test images",
        })

    predictions = np.column_stack([
        _fit_ridge(x[train_mask], losses[train_mask, arm], x[test_mask], ridge)
        for arm in range(len(names))
    ])
    test_losses = losses[test_mask]
    policy_choice = predictions.argmin(axis=1)
    policy_loss = float(test_losses[np.arange(len(policy_choice)), policy_choice].mean())
    train_fixed = losses[train_mask].mean(axis=0)
    fixed_arm = int(train_fixed.argmin())
    fixed_loss = float(test_losses[:, fixed_arm].mean())
    oracle_loss = float(test_losses.min(axis=1).mean())
    oracle_gap = max(0.0, fixed_loss - oracle_loss)
    policy_gain = fixed_loss - policy_loss
    capturability = policy_gain / oracle_gap if oracle_gap > 1e-12 else 0.0

    rng = np.random.RandomState(seed)
    null_gains = []
    for _ in range(max(0, int(permutations))):
        shuffled = np.array(policy_choice, copy=True)
        rng.shuffle(shuffled)
        null_gains.append(fixed_loss - float(test_losses[np.arange(len(shuffled)), shuffled].mean()))
    p_value = ((1 + sum(gain >= policy_gain for gain in null_gains)) /
               (1 + len(null_gains))) if null_gains else None

    recommendation = "PASS"
    reason = "OOS causal-prefix policy recovers measurable fixed-arm headroom"
    if oracle_gap <= 1e-12 or policy_gain < min_policy_gain or capturability < min_capturability:
        recommendation = "STOP"
        reason = "OOS policy does not clear the oracle-gap viability floor"
    elif p_value is not None and p_value > 0.10:
        recommendation = "STOP"
        reason = "OOS policy gain is not separated from the permutation null"

    report = {
        "schema_version": 1,
        "identity": identity,
        "arms": names,
        "feature_source": "causal_prefix",
        "feature_names": feature_names,
        "n_images": len(indices),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_split": "global_idx_parity",
        "test_parity": int(test_parity),
        "fixed_arm": names[fixed_arm],
        "fixed_test_loss": fixed_loss,
        "policy_test_loss": policy_loss,
        "oracle_test_loss": oracle_loss,
        "oracle_gap": oracle_gap,
        "policy_gain": policy_gain,
        "capturability": capturability,
        "permutation_p_value": p_value,
        "permutations": int(permutations),
        "outcome": "pixel_mse_to_full_reference",
        "recommendation": recommendation,
        "reason": reason,
    }
    return _write_report(output_path, report)


def _write_report(path: str, report: Dict[str, object]) -> Dict[str, object]:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--arm", action="append", required=True,
                        help="strategy_id=path/to/viability.jsonl; generated/ is beside it")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-parity", type=int, choices=[0, 1], default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-capturability", type=float, default=0.10)
    parser.add_argument("--min-policy-gain", type=float, default=1e-7)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = analyze(
            reference_dir=args.reference_dir,
            arm_specs=args.arm,
            output_path=args.output,
            test_parity=args.test_parity,
            ridge=args.ridge,
            min_capturability=args.min_capturability,
            min_policy_gain=args.min_policy_gain,
            permutations=args.permutations,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"VIABILITY ERROR: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("recommendation") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
