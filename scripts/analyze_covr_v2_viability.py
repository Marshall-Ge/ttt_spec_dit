#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze the COVR-v2 causal-feature viability probe.

This is an exploratory gate.  Its outcome is a recommendation about whether
an online policy has measurable headroom, not a FID/IS effectiveness claim.

Schema evolution
----------------
- v1 rows (schema_version 1): ``features`` only (legacy).
- v2 rows (schema_version 2): ``features`` + optional ``boundary`` dict added
  by ``--covr-viability-boundary-telemetry``.

This module:
  1. loads all arms,
  2. runs artifact diagnostics (index/format/outlier/crossover/per-arm stats),
  3. evaluates a frozen repeated-OOS policy gate (fixed 4x4 fold design,
     ridge lambda=1.0 primary, tree/legacy/ablations = diagnostics only).

The gating recommendation is PASS / STOP / INSUFFICIENT_DATA. Exit codes:
  0 PASS, 1 STOP, 2 VIABILITY ERROR, 3 artifacts missing.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import math
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


_INDEX_RE = re.compile(r"^(\d+)(?:_|\.)")

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


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


_LPIPS_MODEL = None


def _get_lpips() -> object:
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        try:
            import lpips  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "LPIPS outcome requires 'lpips' package.  Install with: "
                "pip install lpips")
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False)
    return _LPIPS_MODEL


def _lpips(path: str, reference: str, lpips_model: object) -> float:
    import torch
    image = (np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5) - 1.0
    target = (np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32) / 127.5) - 1.0
    t_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    t_ref = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        result = float(lpips_model(t_img, t_ref).item())
    return result


# ---------------------------------------------------------------------------
# Feature row parsing (handles v1 + v2 boundary rows)
# ---------------------------------------------------------------------------


def _feature_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    collect_boundary: bool = True,
) -> Tuple[int, Dict[str, float], Mapping[str, object]]:
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
    # Collect boundary arrays for slope/curvature derivations
    boundary_series: List[Dict[str, float]] = []
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
        # Collect boundary scalars (v2 only)
        if collect_boundary:
            boundary = row.get("boundary")
            if isinstance(boundary, dict):
                bvals: Dict[str, float] = {}
                for bname, bval in boundary.items():
                    if not isinstance(bname, str) or not isinstance(bval, (int, float)):
                        raise ValueError("boundary scalars must be numeric")
                    bvals[str(bname)] = float(bval) if np.isfinite(float(bval)) else float("nan")
                bvals["_has_boundary"] = 1.0
                boundary_series.append(bvals)
            elif boundary is not None:
                raise ValueError("boundary must be a dict or absent")

    # Derive slopes / curvatures for latent/noise/delta scalar families and
    # for boundary series when present.
    _derive_step_slopes(values, ordered)
    if boundary_series:
        # Flatten per-step boundary into step-prefixed names
        for si, bvals in enumerate(boundary_series):
            s = int(ordered[si]["step_idx"])
            for bname, bval in bvals.items():
                if bname.startswith("_"):
                    continue
                values[f"step{s}_b_{bname}"] = bval
        # Slopes of selected boundary scalars
        _derive_boundary_slopes(values, boundary_series, ordered)

    return int(ordered[0]["global_idx"]), values, ordered[0]


def _derive_step_slopes(
    values: Dict[str, float],
    ordered: Sequence[Mapping[str, object]],
) -> None:
    """Add step-to-step differences (slopes) for per-step scalar groups."""
    scalar_groups = [
        "latent_mean", "latent_std", "latent_l1", "latent_l2",
        "noise_mean", "noise_std", "noise_l1", "noise_l2",
        "delta_mean", "delta_std", "delta_l1", "delta_l2",
        "latent_noise_cosine", "delta_relative_l1",
    ]
    for key in scalar_groups:
        series = []
        for step in range(len(ordered)):
            v = values.get(f"step{step}_{key}")
            series.append(v if v is not None and np.isfinite(v) else float("nan"))
        for step in range(1, len(ordered)):
            if np.isfinite(series[step]) and np.isfinite(series[step - 1]):
                values[f"step{step}_{key}_slope"] = series[step] - series[step - 1]
        # Curvature (second difference) at step 2
        for step in range(2, len(ordered)):
            s0 = series[step - 2]
            s1 = series[step - 1]
            s2 = series[step]
            if all(np.isfinite(v) for v in (s0, s1, s2)):
                values[f"step{step}_{key}_curv"] = s2 - 2*s1 + s0


def _derive_boundary_slopes(
    values: Dict[str, float],
    boundary_series: List[Dict[str, float]],
    ordered: Sequence[Mapping[str, object]],
) -> None:
    """Add slopes/curvatures for key boundary scalars."""
    target_keys = ["raw_diff", "rescaled", "shadow_accum_before",
                   "prev_residual_l1", "prev_residual_l2",
                   "residual_modulated_ratio", "prev_mod_l1", "prev_mod_l2"]
    for key in target_keys:
        series = []
        for bvals in boundary_series:
            v = bvals.get(key)
            series.append(v if v is not None and np.isfinite(v) else float("nan"))
        n = len(ordered)
        for step in range(n):
            if np.isfinite(series[step]):
                values[f"step{step}_b_{key}_level"] = series[step]
        for step in range(1, n):
            if np.isfinite(series[step]) and np.isfinite(series[step - 1]):
                values[f"step{step}_b_{key}_slope"] = series[step] - series[step - 1]


def _load_arm(
    path: str,
    *,
    collect_boundary: bool = True,
) -> Dict[int, Tuple[Dict[str, float], Mapping[str, object]]]:
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for row in _read_jsonl(path):
        index = int(row.get("global_idx", -1))
        grouped.setdefault(index, []).append(row)
    if not grouped or -1 in grouped:
        raise ValueError(f"invalid global_idx values in {path}")
    result = {}
    for index, rows in grouped.items():
        parsed_index, features, identity = _feature_rows(
            rows, collect_boundary=collect_boundary)
        if parsed_index != index:
            raise ValueError("inconsistent global_idx in trajectory records")
        result[index] = (features, identity)
    return result


def _identity(
    arm_rows: Mapping[int, Tuple[Dict[str, float], Mapping[str, object]]],
) -> Dict[str, object]:
    first = next(iter(arm_rows.values()))[1]
    fields = ("schema_version", "model", "method", "num_steps", "manifest_hash",
              "prefix_steps", "prefix_verified", "batch_size", "latent_seed_offset")
    result: Dict[str, object] = {field: first.get(field) for field in fields}
    # v2 fields (present only when boundary telemetry was recorded)
    for extra in ("image_format", "boundary_telemetry"):
        val = first.get(extra)
        if val is not None:
            result[extra] = val
    return result


def _validate_identity(
    arms: Mapping[str, Mapping[int, Tuple[Dict[str, float], Mapping[str, object]]]],
) -> Dict[str, object]:
    names = list(arms)
    if len(names) < 2:
        raise ValueError("viability gate requires at least two strategy arms")
    reference = _identity(arms[names[0]])
    if reference["prefix_verified"] is not True or reference["batch_size"] != 1:
        raise ValueError("viability probe must be verified batch-size-one data")
    for name in names[1:]:
        current = _identity(arms[name])
        for field, expected in reference.items():
            if current.get(field) != expected:
                raise ValueError(f"identity mismatch for {name}: {field}")
    return reference


# ---------------------------------------------------------------------------
# Ridge regression (legacy + primary)
# ---------------------------------------------------------------------------


def _fit_ridge(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, ridge: float,
) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Deterministic balanced folds (4x4 = 16 evaluations)
# ---------------------------------------------------------------------------

def _build_folds(indices: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    """Deterministic balanced folds: permute indices with fixed seed, then
    assign to folds round-robin. Returns list of test-mask arrays (one per fold)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(indices))
    fold_id = np.zeros(len(indices), dtype=int)
    for i, pi in enumerate(perm):
        fold_id[pi] = i % n_folds
    masks = []
    for f in range(n_folds):
        masks.append(fold_id == f)
    return masks


_FOLD_N_REPEATS = 4
_FOLD_N_SPLITS = 4
_FOLD_SEED = 42
_RIDGE_LAMBDA = 1.0
_MIN_LEAF = 8


# ---------------------------------------------------------------------------
# Low-capacity tree baseline (diagnostic only, does not gate)
# ---------------------------------------------------------------------------

def _fit_tree(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray,
    min_leaf: int = _MIN_LEAF,
    depth: int = 2,
) -> np.ndarray:
    """Pure-numpy depth-2 regression tree returning per-arm predictions."""
    n_train = len(x_train)
    n_arms = y_train.shape[1]
    n_test = len(x_test)
    predictions = np.zeros((n_test, n_arms), dtype=np.float64)

    for arm in range(n_arms):
        y = y_train[:, arm].astype(np.float64)
        # Initialize: predict train means
        leaf_train = np.zeros(n_train, dtype=int)
        leaf_test = np.zeros(n_test, dtype=int)
        for d in range(depth):
            # Split each current leaf on the feature with best variance reduction
            max_leaf = int(leaf_train.max()) + 1
            new_train = leaf_train.copy()
            new_test = leaf_test.copy()
            next_leaf_id = max_leaf
            for lid in range(max_leaf):
                in_leaf = leaf_train == lid
                if in_leaf.sum() < 2 * min_leaf:
                    continue
                y_leaf = y[in_leaf]
                best_gain = -1.0
                best_feat = -1
                best_thresh = 0.0
                for feat in range(x_train.shape[1]):
                    vals = x_train[in_leaf, feat]
                    if np.std(vals) < 1e-12:
                        continue
                    thresh_candidates = np.quantile(vals, [0.25, 0.5, 0.75])
                    for th in thresh_candidates:
                        left = vals <= th
                        right = ~left
                        nl, nr = left.sum(), right.sum()
                        if nl < min_leaf or nr < min_leaf:
                            continue
                        var_left = y_leaf[left].var() if nl > 0 else 0.0
                        var_right = y_leaf[right].var() if nr > 0 else 0.0
                        gain = y_leaf.var() - (nl * var_left + nr * var_right) / len(y_leaf)
                        if gain > best_gain:
                            best_gain = gain
                            best_feat = feat
                            best_thresh = th
                if best_feat >= 0:
                    # Apply split
                    in_leaf_mask = np.where(in_leaf)[0]
                    vals = x_train[in_leaf_mask, best_feat]
                    new_train[in_leaf_mask[vals <= best_thresh]] = lid
                    new_train[in_leaf_mask[vals > best_thresh]] = next_leaf_id
                    # Test split
                    test_in_leaf = np.where(leaf_test == lid)[0]
                    if len(test_in_leaf) > 0:
                        test_vals = x_test[test_in_leaf, best_feat]
                        new_test[test_in_leaf[test_vals <= best_thresh]] = lid
                        new_test[test_in_leaf[test_vals > best_thresh]] = next_leaf_id
                    next_leaf_id += 1
            leaf_train = new_train
            leaf_test = new_test

        # Predict: train-leaf means
        for lid in range(int(leaf_train.max()) + 1):
            in_leaf = leaf_train == lid
            pred_val = y[in_leaf].mean() if in_leaf.sum() > 0 else y.mean()
            predictions[leaf_test == lid, arm] = pred_val

    return predictions


# ---------------------------------------------------------------------------
# Bootstrap CI for paired gains
# ---------------------------------------------------------------------------

def _paired_bootstrap(
    policy_losses: np.ndarray,
    fixed_losses: np.ndarray,
    oracle_losses: np.ndarray,
    n_boot: int = 2000,
    seed: int = 1,
) -> Dict[str, object]:
    rng = np.random.RandomState(seed)
    n = len(policy_losses)
    gains = fixed_losses - policy_losses
    oracle_gaps = fixed_losses - oracle_losses
    cap = np.zeros(n_boot, dtype=np.float64)
    gain_boot = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        g = gains[idx].mean()
        o = oracle_gaps[idx]
        o_mean = o.mean()
        gain_boot[b] = g
        cap[b] = g / o_mean if o_mean > 1e-12 else 0.0
    return {
        "gain_mean": float(gains.mean()),
        "gain_ci_low": float(np.percentile(gain_boot, 2.5)),
        "gain_ci_high": float(np.percentile(gain_boot, 97.5)),
        "capturability_mean": float(cap.mean()),
        "capturability_ci_low": float(np.percentile(cap, 2.5)),
        "capturability_ci_high": float(np.percentile(cap, 97.5)),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Permutation null (feature-outcome, preserves arm-loss vector per image)
# ---------------------------------------------------------------------------

def _permutation_null(
    train_predictor: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    losses: np.ndarray,
    fixed_arm_idx: int,
    n_perm: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.RandomState(seed)
    n_arms = losses.shape[1]
    n = len(losses)
    # Observed gain
    obs_pred = train_predictor(x, losses, x)
    obs_choice = obs_pred.argmin(axis=1)
    obs_loss = losses[np.arange(n), obs_choice].mean()
    fixed_loss = losses[:, fixed_arm_idx].mean()
    obs_gain = fixed_loss - obs_loss

    null_gains = np.zeros(n_perm, dtype=np.float64)
    for p in range(n_perm):
        # Shuffle rows of x, keeping losses fixed
        perm = rng.permutation(n)
        x_perm = x[perm]
        # Retrain on permuted features
        pred = train_predictor(x_perm, losses, x_perm)
        choice = pred.argmin(axis=1)
        null_gains[p] = fixed_loss - losses[np.arange(n), choice].mean()

    p_value = (1.0 + (null_gains >= obs_gain).sum()) / (1.0 + n_perm)
    null_zero = float((null_gains <= 1e-9).mean())
    return {
        "permutation_p": float(p_value),
        "null_zero_fraction": null_zero,
        "permutations": n_perm,
        "observed_gain": float(obs_gain),
    }


# ---------------------------------------------------------------------------
# Per-arm statistics diagnostics
# ---------------------------------------------------------------------------

def _arm_diagnostics(
    losses: np.ndarray,
    arm_names: List[str],
    indices: np.ndarray,
    fixed_arm_idx: int,
    seed: int = 1,
) -> Dict[str, object]:
    n_arms = len(arm_names)
    rng = np.random.RandomState(seed)
    per_arm = {}
    for a, name in enumerate(arm_names):
        l_a = losses[:, a]
        win_mask = np.argmin(losses, axis=1) == a
        ties = (losses == losses.min(axis=1, keepdims=True)).sum(axis=1)
        # Fractional win rate: each image contributes 1/N_winners to its argmin arms
        win_rate = float(np.sum(np.where(win_mask, 1.0 / ties, 0.0)) / len(losses))
        # Bootstrap CI for mean loss
        n = len(losses)
        means = np.zeros(2000, dtype=np.float64)
        for b in range(2000):
            idx = rng.randint(0, n, size=n)
            means[b] = l_a[idx].mean()
        per_arm[name] = {
            "mean": float(l_a.mean()),
            "median": float(np.median(l_a)),
            "std": float(l_a.std()),
            "mean_ci_low": float(np.percentile(means, 2.5)),
            "mean_ci_high": float(np.percentile(means, 97.5)),
            "win_rate": win_rate,
            "train_mean": float(l_a.mean()),
        }
    # Paired delta bootstrap vs fixed arm
    l_fixed = losses[:, fixed_arm_idx]
    deltas = {}
    for a, name in enumerate(arm_names):
        if a == fixed_arm_idx:
            continue
        delta = l_fixed - losses[:, a]
        n = len(delta)
        d_boot = np.zeros(2000, dtype=np.float64)
        for b in range(2000):
            idx = rng.randint(0, n, size=n)
            d_boot[b] = delta[idx].mean()
        deltas[name] = {
            "delta_mean": float(delta.mean()),
            "delta_ci_low": float(np.percentile(d_boot, 2.5)),
            "delta_ci_high": float(np.percentile(d_boot, 97.5)),
        }
    return {"per_arm": per_arm, "paired_deltas_vs_fixed": deltas}


# ---------------------------------------------------------------------------
# Outlier diagnostics
# ---------------------------------------------------------------------------

def _outlier_diagnostics(
    losses: np.ndarray,
    arm_names: List[str],
    n_sigma: float = 4.0,
) -> Dict[str, object]:
    n_arms = len(arm_names)
    # Per-arm robust z-scores (MAD-based)
    robust_z: Dict[str, list] = {}
    for a, name in enumerate(arm_names):
        med = np.median(losses[:, a])
        mad = np.median(np.abs(losses[:, a] - med))
        if mad < 1e-12:
            mad = 1e-12
        robust_z[name] = (losses[:, a] - med) / (1.4826 * mad)

    uniformly_hard = []
    arm_specific = []
    for i in range(len(losses)):
        exceed = [name for name in arm_names if abs(robust_z[name][i]) > n_sigma]
        if len(exceed) >= n_arms:
            uniformly_hard.append(int(i))
        elif len(exceed) == 1:
            arm_specific.append({"index": int(i), "arm": exceed[0]})

    return {
        "uniformly_hard_count": len(uniformly_hard),
        "uniformly_hard_indices": uniformly_hard[:20],
        "arm_specific_count": len(arm_specific),
        "arm_specific_examples": arm_specific[:20],
        "n_sigma": n_sigma,
    }


# ---------------------------------------------------------------------------
# Oracle robustness
# ---------------------------------------------------------------------------

def _oracle_robustness(
    losses: np.ndarray,
    arm_names: List[str],
    fixed_arm_idx: int,
) -> Dict[str, object]:
    n = len(losses)
    fixed_loss = losses[:, fixed_arm_idx]
    oracle_loss = losses.min(axis=1)
    per_image_gains = fixed_loss - oracle_loss
    sorted_gains = np.sort(per_image_gains)[::-1]

    # Trimmed means
    trim_05 = sorted_gains[:int(n * 0.95)].mean() if n >= 20 else per_image_gains.mean()

    # Leave-one-out range
    loo_gains = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        loo_gains[i] = fixed_loss[mask].mean() - oracle_loss[mask].mean()

    # Top-1/Top-5% contribution share
    total_gain = per_image_gains.sum()
    top1 = sorted_gains[0] / total_gain if total_gain > 1e-12 else 0.0
    k = max(1, int(n * 0.05))
    top5 = sorted_gains[:k].sum() / total_gain if total_gain > 1e-12 else 0.0

    return {
        "oracle_gap": float(per_image_gains.mean()),
        "oracle_gap_median": float(np.median(per_image_gains)),
        "oracle_gap_trimmed_05": float(trim_05),
        "leave_one_out_lo": float(loo_gains.min()),
        "leave_one_out_hi": float(loo_gains.max()),
        "top1_contribution": float(top1),
        "top5pct_contribution": float(top5),
        "n_zero_or_neg_gain": int((per_image_gains <= 1e-12).sum()),
    }


# ---------------------------------------------------------------------------
# Codec sensitivity
# ---------------------------------------------------------------------------

def _codec_sensitivity(
    arm_names: List[str],
    image_maps: Dict[str, Dict[int, str]],
    reference_images: Dict[int, str],
    indices: np.ndarray,
    image_format: str,
) -> Dict[str, object]:
    report: Dict[str, object] = {"image_format": image_format}
    if image_format != "jpeg":
        report["jpeg_roundtrip_noise_floor"] = None
        report["note"] = "noise-floor analysis not applicable (non-JPEG artifacts)"
        return report

    # JPEG re-encode noise floor: re-save reference images at quality 85
    # and measure pixel MSE vs original.
    ref_mse_values = []
    for i in indices:
        ref_path = reference_images[i]
        import io
        img = Image.open(ref_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        re_encoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        original = np.asarray(img, dtype=np.float32) / 255.0
        ref_mse_values.append(float(np.mean((re_encoded - original) ** 2)))
    jpeg_floor = float(np.mean(ref_mse_values))
    report["jpeg_roundtrip_noise_floor"] = jpeg_floor
    report["jpeg_floor_2x"] = 2.0 * jpeg_floor

    return report


# ---------------------------------------------------------------------------
# Crossover concentration
# ---------------------------------------------------------------------------

def _crossover_diagnostics(
    losses: np.ndarray,
    arm_names: List[str],
) -> Dict[str, object]:
    n_arms = len(arm_names)
    # Pairwise crossover rates
    pairs = {}
    for a in range(n_arms):
        for b in range(a + 1, n_arms):
            rate = float((losses[:, a] < losses[:, b]).mean())
            pairs[f"{arm_names[a]}_vs_{arm_names[b]}"] = {
                f"{arm_names[a]}_wins": rate,
                f"{arm_names[b]}_wins": 1.0 - rate,
            }
    # Layout win counts
    layout_wins = {}
    for a, name in enumerate(arm_names):
        layout_wins[name] = int((losses.argmin(axis=1) == a).sum())
    layout_win_rates = {name: float(cnt) / len(losses)
                        for name, cnt in layout_wins.items()}
    return {
        "pairwise_crossover": pairs,
        "layout_win_counts": layout_wins,
        "layout_win_rates": layout_win_rates,
        "n_images": len(losses),
    }


# ---------------------------------------------------------------------------
# Feature ablation helpers
# ---------------------------------------------------------------------------

def _select_boundary_features(feature_names: List[str]) -> List[str]:
    """Return features with boundary prefix (step*_b_*)."""
    return [n for n in feature_names if "_b_" in n]


def _select_prefix_features(feature_names: List[str]) -> List[str]:
    """Return legacy step-features (excluding boundary and derived slopes)."""
    return [n for n in feature_names
            if "_b_" not in n
            and not n.endswith("_slope")
            and not n.endswith("_curv")]


# ---------------------------------------------------------------------------
# Primary repeated-OOS gate
# ---------------------------------------------------------------------------


def analyze(
    *,
    reference_dir: str,
    arm_specs: Sequence[str],
    output_path: str,
    # Legacy v1 params kept for backward compatibility
    test_parity: int = 1,
    ridge: float = 1e-3,
    min_capturability: float = 0.10,
    min_policy_gain: float = 1e-7,
    permutations: int = 200,
    seed: int = 0,
    outcome: str = "mse",
    lpips_gpu: bool = False,
) -> Dict[str, object]:
    # ---- Load arms ----
    arms: Dict[str, Dict[int, Tuple[Dict[str, float], Mapping[str, object]]]] = {}
    for spec in arm_specs:
        if "=" not in spec:
            raise ValueError(f"arm must use id=JSONL syntax: {spec}")
        name, path = spec.split("=", 1)
        if not name or not path:
            raise ValueError(f"invalid arm specification: {spec}")
        arms[name] = _load_arm(path, collect_boundary=True)
    identity = _validate_identity(arms)
    names = list(arms)
    indices_list = sorted(next(iter(arms.values())))
    indices = np.array(indices_list, dtype=int)
    for name in names:
        if sorted(arms[name]) != indices_list:
            raise ValueError(f"global_idx set mismatch for {name}")

    # Has boundary telemetry?
    has_boundary = bool(identity.get("boundary_telemetry", False))

    feature_names = sorted(arms[names[0]][indices_list[0]][0])
    for name in names[1:]:
        for index in indices_list:
            current = arms[name][index][0]
            if sorted(current) != feature_names:
                raise ValueError(f"shared-prefix feature schema mismatch for {name}")
            if any(abs(float(current[key]) - float(arms[names[0]][index][0][key])) > 1e-6
                   for key in feature_names):
                raise ValueError(f"shared-prefix features differ across arms at {index}")

    # ---- Image maps & losses ----
    image_maps = {name: _image_map(os.path.join(os.path.dirname(
        spec.split("=", 1)[1]), "generated"))
        for name, spec in zip(names, arm_specs)}
    reference_images = _image_map(reference_dir)
    missing = [i for i in indices_list if i not in reference_images]
    if missing:
        raise ValueError(f"reference is missing {len(missing)} global indices")

    # ---- Per-image loss metric ----
    _loss_fn: Callable[[str, str], float]
    _outcome_label: str
    if outcome == "lpips":
        import torch
        _lpips_model = _get_lpips()
        if lpips_gpu and torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()
        _loss_fn = functools.partial(_lpips, lpips_model=_lpips_model)
        _outcome_label = "lpips_to_full_reference"
    else:
        _loss_fn = _mse
        _outcome_label = "pixel_mse_to_full_reference"

    losses = np.asarray([
        [_loss_fn(image_maps[name][i], reference_images[i]) for name in names]
        for i in indices_list
    ], dtype=np.float64)
    if not np.isfinite(losses).all():
        raise ValueError("non-finite arm outcome")

    n = len(indices)
    n_arms = len(names)

    # ---- Build feature matrix ----
    x_all = np.asarray(
        [[float(arms[names[0]][int(i)][0].get(key, float("nan")))
          for key in feature_names]
         for i in indices_list],
        dtype=np.float64,
    )
    # Drop columns that are constant or all-NaN
    col_std = np.nanstd(x_all, axis=0)
    keep_cols = col_std > 1e-12
    x_all = x_all[:, keep_cols]
    kept_features = [f for f, k in zip(feature_names, keep_cols) if k]
    # Fill NaN with column mean (or 0 if entire column NaN)
    col_mean = np.nanmean(x_all, axis=0)
    col_mean[np.isnan(col_mean)] = 0.0
    nan_mask = np.isnan(x_all)
    for c in range(x_all.shape[1]):
        if nan_mask[:, c].any():
            x_all[nan_mask[:, c], c] = col_mean[c]

    # ---- Train-selected fixed arm on all data (used for diagnostics) ----
    train_fixed_all = losses.mean(axis=0)
    fixed_arm_idx = int(train_fixed_all.argmin())
    fixed_arm = names[fixed_arm_idx]

    # ---- Build deterministic folds ----
    fold_masks = _build_folds(indices, _FOLD_N_SPLITS, _FOLD_SEED)

    # ---- Feature sets for ablation ----
    prefix_feats = _select_prefix_features(kept_features)
    boundary_feats = _select_boundary_features(kept_features)
    # Primary: boundary + prefix slopes/curvatures.  Fall back to all
    # features when boundary is absent (fold results are computed for
    # diagnostic completeness, but the gate still requires boundary).
    primary_feats = [
        f for f in kept_features
        if f.startswith("step") and (
            "_b_" in f
            or f.endswith("_slope")
            or f.endswith("_curv")
        )
    ]
    if not primary_feats:
        primary_feats = list(kept_features)
    combined_feats = list(kept_features)

    feature_sets = {
        "primary": primary_feats,
        "prefix_legacy": prefix_feats,
        "boundary_level": boundary_feats,
        "combined": combined_feats,
    }

    # ---- Repeat x Fold cross-fitting ----
    all_fold_results: List[Dict] = []
    pool_policy_losses: List[float] = []
    pool_fixed_losses: List[float] = []
    pool_oracle_losses: List[float] = []

    for rep in range(_FOLD_N_REPEATS):
        # Fresh permutation for each repeat
        rng_rep = np.random.RandomState(_FOLD_SEED + rep * 1000)
        perm = rng_rep.permutation(n)
        for fold in range(_FOLD_N_SPLITS):
            test_mask = fold_masks[fold]
            train_mask = ~test_mask
            if train_mask.sum() < 16 or test_mask.sum() < 16:
                continue

            # Select fixed arm from train only
            train_fixed = losses[train_mask].mean(axis=0).argmin()
            n_test = test_mask.sum()

            fold_result: Dict[str, Any] = {
                "repeat": rep, "fold": fold,
                "n_train": int(train_mask.sum()), "n_test": n_test,
                "train_fixed_arm": names[train_fixed],
                "models": {},
            }

            # Evaluate primary model (ridge on primary features)
            x_primary = x_all[:, [kept_features.index(f) for f in primary_feats if f in kept_features]]
            if x_primary.shape[1] > 0:
                x_train = x_primary[train_mask]
                x_test = x_primary[test_mask]
                y_train = losses[train_mask]
                pred = np.column_stack([
                    _fit_ridge(x_train, y_train[:, a], x_test, _RIDGE_LAMBDA)
                    for a in range(n_arms)
                ])
                choice = pred.argmin(axis=1)
                test_losses = losses[test_mask]
                policy_loss = float(test_losses[np.arange(n_test), choice].mean())
                fixed_loss = float(test_losses[:, train_fixed].mean())
                oracle_loss = float(test_losses.min(axis=1).mean())
                oracle_gap = max(0.0, fixed_loss - oracle_loss)
                policy_gain = fixed_loss - policy_loss
                cap = policy_gain / oracle_gap if oracle_gap > 1e-12 else 0.0

                fold_result["primary_gain"] = policy_gain
                fold_result["primary_capturability"] = cap
                fold_result["primary_oracle_gap"] = oracle_gap
                fold_result["primary_fixed_loss"] = fixed_loss
                fold_result["primary_policy_loss"] = policy_loss

                pool_policy_losses.extend(test_losses[np.arange(n_test), choice].tolist())
                pool_fixed_losses.extend(test_losses[:, train_fixed].tolist())
                pool_oracle_losses.extend(test_losses.min(axis=1).tolist())

            all_fold_results.append(fold_result)

    if not all_fold_results:
        # Edge case: no folds passed the minimum-size check.  Still
        # produce diagnostics; mark INSUFFICIENT_DATA.
        pool_policy = np.array([0.0])
        pool_fixed = np.array([0.0])
        pool_oracle = np.array([0.0])
        mean_policy_gain = 0.0
        capturability = 0.0
        boot = _paired_bootstrap(pool_policy, pool_fixed, pool_oracle, n_boot=2000, seed=1)
        x_p = x_all[:, [kept_features.index(f) for f in primary_feats if f in kept_features]]
        perm_null = {"permutation_p": 1.0, "null_zero_fraction": 0.0,
                     "permutations": 0, "observed_gain": 0.0}
        fold_direction_ok = False
        folds_positive = 0
        folds_total = 0
    else:
        # ---- Aggregate pooled gains ----
        pool_policy = np.array(pool_policy_losses or [0.0], dtype=np.float64)
        pool_fixed = np.array(pool_fixed_losses or [0.0], dtype=np.float64)
        pool_oracle = np.array(pool_oracle_losses or [0.0], dtype=np.float64)

        mean_policy_gain = float(pool_fixed.mean() - pool_policy.mean()) if len(pool_fixed) > 0 else 0.0
        mean_oracle_gap = max(0.0, float(pool_fixed.mean() - pool_oracle.mean())) if len(pool_fixed) > 0 else 0.0
        capturability = mean_policy_gain / mean_oracle_gap if mean_oracle_gap > 1e-12 else 0.0

        # ---- Bootstrap CI ----
        boot = _paired_bootstrap(pool_policy, pool_fixed, pool_oracle, n_boot=2000, seed=1)

        # ---- Permutation null (feature-outcome) ----
        x_p = x_all[:, [kept_features.index(f) for f in primary_feats if f in kept_features]]
        perm_null = _permutation_null(
            lambda x_tr, y_tr, x_te: np.column_stack([
                _fit_ridge(x_tr, y_tr[:, a], x_te, _RIDGE_LAMBDA)
                for a in range(n_arms)
            ]),
            x_p if x_p.shape[1] > 0 else x_all,
            losses, fixed_arm_idx, n_perm=500, seed=1,
        )

        # ---- Fold direction consistency ----
        fold_gains = [r.get("primary_gain") for r in all_fold_results if "primary_gain" in r]
        fold_gains_valid = [g for g in fold_gains if isinstance(g, (int, float))]
        folds_positive = sum(1 for g in fold_gains_valid if g > 0)
        folds_total = len(fold_gains_valid) if fold_gains_valid else 0
        fold_direction_ok = folds_total > 0 and folds_positive >= 0.75 * folds_total

    # ---- Diagnostics (always computed) ----
    arm_diag = _arm_diagnostics(losses, names, indices, fixed_arm_idx, seed=1)
    outlier_diag = _outlier_diagnostics(losses, names)
    oracle_diag = _oracle_robustness(losses, names, fixed_arm_idx)
    crossover_diag = _crossover_diagnostics(losses, names)
    image_format = str(identity.get("image_format", "png"))
    codec_diag = _codec_sensitivity(names, image_maps, reference_images, indices, image_format)

    # ---- Gating ----
    gate_checks = {
        "policy_gain_positive": mean_policy_gain > 0,
        "capturability_floor": capturability >= 0.10,
        "bootstrap_ci_low_ok": boot["gain_ci_low"] >= -1e-4,
        "permutation_p_ok": perm_null["permutation_p"] <= 0.10,
        "fold_direction_ok": fold_direction_ok,
    }

    if not has_boundary:
        recommendation = "INSUFFICIENT_DATA"
        reason = "boundary telemetry not present; rerun with --covr-viability-boundary-telemetry"
    elif all(gate_checks.values()):
        recommendation = "PASS"
        reason = "primary boundary-ridge policy clears all frozen gates"
    else:
        failed = [k for k, v in gate_checks.items() if not v]
        recommendation = "STOP"
        reason = f"gates failed: {failed}"

    # ---- Legacy v1-style fields for compatibility ----
    parity_train = np.array([(int(i) % 2) == int(test_parity) for i in indices_list], dtype=bool)
    parity_test = ~parity_train
    legacy_fixed_loss = float(losses[parity_test][:, fixed_arm_idx].mean())
    legacy_oracle_loss = float(losses[parity_test].min(axis=1).mean())
    legacy_oracle_gap = max(0.0, legacy_fixed_loss - legacy_oracle_loss)

    report: Dict[str, object] = {
        "schema_version": 2,
        "identity": identity,
        "arms": names,
        "feature_source": "causal_prefix",
        "feature_names": kept_features,
        "n_images": n,
        "n_train": int(parity_train.sum()),
        "n_test": int(parity_test.sum()),
        "train_split": "global_idx_parity",
        "test_parity": int(test_parity),
        "fixed_arm": fixed_arm,
        "fixed_test_loss": legacy_fixed_loss,
        "policy_test_loss": None,
        "oracle_test_loss": legacy_oracle_loss,
        "oracle_gap": legacy_oracle_gap,
        "policy_gain": None,
        "capturability": None,
        "permutation_p_value": None,
        "permutations": 0,
        "outcome": _outcome_label,
        "recommendation": recommendation,
        "reason": reason,
        # --- v2 extensions ---
        "primary": {
            "policy_gain": mean_policy_gain,
            "capturability": capturability,
            "oracle_gap": max(0.0, float(pool_fixed.mean() - pool_oracle.mean())) if len(pool_fixed) > 0 else 0.0,
            "folds_positive": folds_positive,
            "folds_total": folds_total,
            "fold_direction_ok": fold_direction_ok,
            "fold_results": all_fold_results,
        },
        "bootstrap": boot,
        "null": perm_null,
        "gate_checks": gate_checks,
        "diagnostics": {
            "arm_statistics": arm_diag,
            "outliers": outlier_diag,
            "oracle_robustness": oracle_diag,
            "crossover": crossover_diag,
            "codec_sensitivity": codec_diag,
            "artifact_integrity": {
                "indices_per_arm": {name: sorted(arms[name])
                                   for name in names},
                "reference_indices": sorted(reference_images),
                "missing_reference": missing,
                "image_format": image_format,
                "has_boundary_telemetry": has_boundary,
            },
        },
    }
    return _write_report(output_path, report)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: object) -> object:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _write_report(path: str, report: Dict[str, object]) -> Dict[str, object]:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, cls=_NumpyEncoder)
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
    parser.add_argument("--outcome", choices=["mse", "lpips"], default="mse",
                        help="Per-image loss metric (default: mse)")
    parser.add_argument("--lpips-gpu", action="store_true", default=False,
                        help="Use GPU for LPIPS computation")
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
            outcome=args.outcome,
            lpips_gpu=args.lpips_gpu,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"VIABILITY ERROR: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, cls=_NumpyEncoder))
    return 0 if report.get("recommendation") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
