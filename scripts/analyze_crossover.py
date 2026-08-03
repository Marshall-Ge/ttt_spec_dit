#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-image crossover verdict for the COVR aggressive-budget probe.

The budget probe's global FID cannot tell "one arm is best for every image"
(pick it offline, bandit unnecessary) from "the best arm flips per image"
(the only case a per-trajectory bandit has something to discover). This script
settles it from the saved 256x256 PNGs, which ``sweep_budget_probe.sh`` now
saves in full (--img_save_limit N_PROMPTS) and anchors to a shared full-compute
reference (--method baseline, same seed/window).

Why per-image and not per-class:
  ``ImageNetDataset`` shuffles all 50k and takes the first n, so n=500 gives
  ~500 *different* classes at ~1 image each. FID is a distribution metric; a
  per-class FID on one image is meaningless. The pairing anchor that makes
  per-image comparison valid is global_idx: the same global_idx across arms
  (and vs the reference) is the same dataset sample, the same generation seed
  (``run_dit.py`` seeds ``100000 + absolute_idx`` and saves
  ``{global_idx:06d}_{cls}.png``), so every arm-vs-arm and arm-vs-reference
  comparison is properly paired.

Four sections per budget; only [d] is a verdict gate. Lettering below matches
the printed output exactly.

  [a] validity   per-arm MEAN per-image loss must rank the arms the same way
                FID does. If not, the proxy measures something other than what
                FID measures and the crossover conclusion is void.
                ``--metric`` picks the loss, and WHICH loss fails [a] changes
                what the failure means:
                  ``inception`` (default when torch-fidelity loads) — squared
                    distance in FID's OWN InceptionV3 pool3 ("2048") features.
                    Strictly the strongest reference-based per-image proxy for
                    FID available here, so an [a] failure under it is evidence
                    about the fidelity-to-full-compute REWARD FAMILY (terminal
                    and its H-step sentinel variant alike), not about metric
                    choice — it closes the family without spending the
                    sentinel's +125% forwards.
                  ``inception_conf`` — REFERENCE-FREE predictive entropy of the
                    same network's ``logits_unbiased``, i.e. the per-image half
                    of Inception Score (IS's confidence term is -H[p(y|x)]; the
                    diversity term is a distribution property and cannot exist
                    per image). Its purpose is to remove the one escape route
                    ``inception`` leaves open — that the reference is not the FID
                    optimum, so distance-TO-reference is the wrong anchor
                    regardless of feature space. Known bias, printed in the
                    output rather than hidden: entropy rewards
                    class-prototypicality, so an arm collapsing toward confident
                    generic exemplars scores well here while FID punishes it.
                    FID therefore still gates; a second Spearman against IS is
                    printed to separate "unfaithful to its own aggregate" from
                    "faithful to IS, and IS disagrees with FID".
                  ``lpips`` / ``pixel_mse`` — neither is in FID's feature space,
                    so their [a] failures do NOT close anything; they may just
                    mean the proxy is weak. The RECOMMENDATION block says so.
  [b] floor      PNGs are 8-bit. If the arm-to-arm signal sits at the uint8
                rounding floor, the measured crossover is rounding noise and the
                budget is excluded from the verdict.
  [c] crossover  per-image oracle (min over arms) mean loss vs best-single-arm
                mean loss, reported as DESCRIPTIVE only — never as a gate.
                ``oracle - best_single`` is not a test of structure: the
                per-image min converts iid measurement noise into positive
                "benefit" (more arm-to-arm variance => bigger apparent gain), so
                it is positive under pure noise. Measured on constructed trees
                (scenarios reproducible from this file's docstring): pure
                independent noise produced an in-sample gap of 24% of the best
                arm's loss, while a genuinely difficulty-driven crossover
                produced 18% — the NOISE case looked bigger. No permutation null
                repairs this. Three nulls are printed to make the failure
                legible rather than hidden:
                  null A — shuffle arm labels WITHIN each image. Structurally
                    degenerate: the per-image min is INVARIANT under a row
                    shuffle (``sort(D, axis=1)`` is unchanged), so the oracle
                    term does not move at all. Printed because the probe design
                    asked for it.
                  null B — permute each arm's column independently. Preserves
                    each arm's marginal but destroys the cross-arm coupling, so
                    the null mean moves relative to the observed statistic by
                    the SIGN of that coupling, not by whether crossover is
                    exploitable: positively coupled arms (shared image
                    difficulty) push the null mean far ABOVE the observation
                    (measured obs +0.098 vs null mean +0.30, p=1.00 on real
                    interaction data), anti-correlated arms push it BELOW and
                    yield p~0. Both are artifacts of the coupling's sign.
                  null R — permute the residual after removing BOTH main effects
                    (image difficulty and arm offset). Correctly centred, but
                    unreliable in both directions: measured p=0.47 on one
                    real-interaction construction (a miss) and p=0.065 on pure
                    independent noise (a near-false-positive).
  [d] out-of-sample — THE VERDICT GATE. Split images by global_idx parity, learn
                on one half, settle on the other. In-sample oracle is always
                >= 0 (it is a per-image min) and cannot distinguish noise from
                structure; out-of-sample transfer can, because a rule fit on
                noise does not transfer. The transfer feature MUST be exogenous
                — gradient energy / contrast / luminance of the REFERENCE image,
                never anything derived from the distance matrix. A feature taken
                from D (the per-image distance under the train best arm) leaks
                and declared PURE IID NOISE "utilizable" 25% of the time at
                n=40, 61% at n=100, 93% at n=250 and 99% at n=500 — at this
                sweep's own N it was a near-certain false positive.
                Three rules are printed; only the middle one gates.
                  constant majority — feature-free ceiling, cannot beat the test
                    best single arm, so <= 0 by construction.
                  COST REGRESSION (the gate) — least squares of each arm's cost
                    on the features, argmin of the predictions. Measured over
                    300 trials at n=500/250, gate rate = 0.00/0.00 on all four
                    no-crossover constructions (pure-noise features, difficulty
                    main effect only, per-image hetero variance, arm-specific
                    noise scales), 0.95/0.83 on a strong real interaction and
                    0.19/0.13 on a weak one. With uninformative features every
                    arm's fit collapses to its own mean and the rule degenerates
                    to the best single arm (frac exactly 0).
                  1NN LABEL TRANSFER (diagnostic only) — the former gate. It
                    transfers a neighbour's argmin LABEL, which forces a
                    commitment to a per-image winner that is mostly noise when
                    arms sit within a noise width of each other: on a
                    construction with a GENUINE crossover at this sweep's noise
                    level it recovered a negative share in 200/200 trials while
                    reaching p<=0.05 in 95% of them. Read its p as "the features
                    predict which arm wins" and its negative frac as "label
                    transfer cannot monetize it". That combination arises 2.3%
                    of the time when no crossover exists and 96.7% of the time
                    when one does, so it is worth reporting separately from
                    "pure noise" — but it is not the gate.
                What a positive result does NOT establish: reference features are
                ORACLE-side (a deployed policy does not have the full-compute
                output). It shows per-image structure exists and is predictable
                from image content; whether an ONLINE-available feature sees it
                is a separate experiment.

Layout expected (written by scripts/sweep_budget_probe.sh)::

    <probe>/reference/results.json + <probe>/reference/generated/*.png
    <probe>/k<K>/equalflops/arm_<id>/results.json + .../generated/*.png
    <probe>/k<K>/manifest.json

Analysis only; always exits 0. The RECOMMENDATION block carries the reasoning.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_teacache_sweeps import _load          # noqa: E402
from analyze_budget_probe import _budget_dirs, _f  # noqa: E402
from analyze_reward_proxy import _average_ranks, _spearman  # noqa: E402

# uint8 rounding floor, per-pixel MSE in [0,1] units.
# Rounding to nearest on an 8-bit grid has error uniform in +-0.5/255, variance
# (1/255)^2/12. Two independently rounded images double that.
FLOOR_ONE_ROUNDED = (1.0 / 255.0) ** 2 / 12.0          # image vs continuous ref
FLOOR_TWO_ROUNDED = 2.0 * FLOOR_ONE_ROUNDED             # two rounded images
_FLOOR = FLOOR_TWO_ROUNDED


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _load_images(gen_dir: str) -> Dict[int, np.ndarray]:
    """``<gen_dir>/{global_idx:06d}_*.png`` -> {global_idx: uint8 [H,W,3]}."""
    out: Dict[int, np.ndarray] = {}
    if not os.path.isdir(gen_dir):
        return out
    for path in glob.glob(os.path.join(gen_dir, "*.png")):
        head = os.path.basename(path).split("_", 1)[0]
        if not head.isdigit():
            continue
        try:
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        if arr.ndim == 3 and arr.shape[2] == 3:
            out[int(head)] = arr
    return out


def _load_reference(ref_dir: str
                    ) -> Tuple[Optional[float], dict, Dict[int, np.ndarray]]:
    """Reference FID, config, and generated images from a reference run."""
    results = _load(os.path.join(ref_dir, "results.json"))
    fid = None
    config: dict = {}
    if results:
        fid = (results.get("aggregate") or {}).get("fid")
        config = results.get("config") or {}
    return fid, config, _load_images(os.path.join(ref_dir, "generated"))


# --------------------------------------------------------------------------
# Distance metric
# --------------------------------------------------------------------------

def _pixel_mse_vectorized(ref_al: np.ndarray, arm_al: np.ndarray) -> np.ndarray:
    """Per-image pixel MSE, (n,H,W,3) uint8 -> (n,) in [0,1] units. Same math as
    ``eval/mse.py compute_pixel_mse`` (mean of squared [0,1] differences)."""
    d = ref_al.astype(np.float32) - arm_al.astype(np.float32)
    d /= 255.0
    return (d * d).mean(axis=(1, 2, 3))


def _metric_desc(metric: str, device: str) -> str:
    """One-line description of a metric NAME, printed in the header and per
    budget. Kept separate from the metric implementations so the header and the
    per-budget rows cannot disagree (they did: ``_analyze_budget`` used to
    re-resolve the metric and always landed on pixel MSE)."""
    if metric == "lpips":
        return f"lpips (eval/lpips.py LPIPSScorer, device={device})"
    if metric == "inception":
        return ("inception (mean squared difference of torch-fidelity "
                "InceptionV3 pool3 '2048' features vs the reference; "
                f"smaller = closer to full compute, device={device})")
    if metric == "inception_conf":
        return ("inception_conf (REFERENCE-FREE: predictive entropy of "
                "torch-fidelity InceptionV3 'logits_unbiased' in nats; "
                f"smaller = more confident, device={device})")
    return "pixel_mse (eval/mse.py compute_pixel_mse)"


# --- InceptionV3 feature metrics -------------------------------------------
# FID is a Frechet distance between Gaussians fitted to torch-fidelity's
# InceptionV3 pool3 ("2048") activations, and IS comes from the same network's
# "logits_unbiased". Both metrics below therefore live in the SAME feature space
# as the aggregate numbers [a] validates against, which is the entire reason to
# prefer them over pixel MSE or LPIPS: if a per-image distance in FID's own
# feature space still cannot rank the arms like FID, no reference-based per-image
# metric can, and the whole fidelity-to-full-compute reward family (terminal AND
# the H-step sentinel) is closed rather than merely badly measured.

_INCEPTION_CACHE: Dict[Tuple[str, str], object] = {}


def _inception_extractor(feature: str, device: str):
    """Cached torch-fidelity InceptionV3 extractor for one features_list entry.

    ``forward`` asserts a uint8 4-D BxCxHxW tensor and performs the 299x299
    resize and the (x-128)/128 scaling internally, TF-compatibly — so the input
    must NOT be pre-normalized to [0,1] the way the LPIPS and pixel-MSE paths
    do. Passing floats trips the extractor's own dtype assert.
    """
    key = (feature, device)
    if key in _INCEPTION_CACHE:
        return _INCEPTION_CACHE[key]
    try:
        from torch_fidelity.registry import FEATURE_EXTRACTORS_REGISTRY
        cls = FEATURE_EXTRACTORS_REGISTRY["inception-v3-compat"]
    except Exception:
        from torch_fidelity.feature_extractor_inceptionv3 import (
            FeatureExtractorInceptionV3 as cls)
    ext = cls("inception-v3-compat", [feature]).to(device).eval()
    for p in ext.parameters():
        p.requires_grad_(False)
    _INCEPTION_CACHE[key] = ext
    return ext


def _inception_features(imgs: List[np.ndarray], feature: str, device: str,
                        batch: int = 32) -> np.ndarray:
    """uint8 [H,W,3] list -> (n, d) float64 activations."""
    import torch
    ext = _inception_extractor(feature, device)
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), batch):
            chunk = np.stack(imgs[s:s + batch]).transpose(0, 3, 1, 2)
            t = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
            out.append(ext(t)[0].float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def _entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    """(n, 1008) logits -> (n,) predictive entropy in nats; smaller = more
    confident. The single-image analogue of Inception Score: IS is the mean KL
    from p(y|x) to the marginal p(y), whose per-image term is exactly
    -H[p(y|x)] plus a distribution-level constant. Using entropy directly keeps
    this script's "smaller = better" convention, at the cost of dropping IS's
    diversity term — which is a DISTRIBUTION property and cannot exist per
    image, and is also the half of IS that a per-image reward would need in
    order not to collapse onto class prototypes."""
    m = logits.max(axis=1, keepdims=True)
    z = logits - m
    logsum = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - logsum
    p = np.exp(logp)
    return -(p * logp).sum(axis=1)


def _distance_matrix(metric: str, device: str, ref_al: np.ndarray,
                     arm_stacks: List[np.ndarray]) -> np.ndarray:
    """(n, n_arm) loss matrix, smaller = better, for the BATCHED metrics."""
    n_arm = len(arm_stacks)
    D = np.empty((ref_al.shape[0], n_arm))
    if metric == "inception":
        ref_f = _inception_features(list(ref_al), "2048", device)
        for j, arm_al in enumerate(arm_stacks):
            d = ref_f - _inception_features(list(arm_al), "2048", device)
            D[:, j] = (d * d).mean(axis=1)
        return D
    if metric == "inception_conf":
        # REFERENCE-FREE: ref_al is deliberately never touched here.
        for j, arm_al in enumerate(arm_stacks):
            lg = _inception_features(list(arm_al), "logits_unbiased", device)
            D[:, j] = _entropy_from_logits(lg)
        return D
    raise ValueError(f"not a batched metric: {metric}")


def _make_distance(metric: str, device: str):
    """Return (fn(ref_arr, arm_arr)->float, description) for the PER-PAIR
    metrics. Both args are uint8 [H,W,3] arrays. Batched metrics
    (``inception``, ``inception_conf``) do not go through this — see
    ``_distance_matrix``."""
    import torch

    if metric == "lpips":
        from eval.lpips import LPIPSScorer
        scorer = LPIPSScorer(device=device)

        def fn(ref, arm):
            rt = torch.from_numpy(
                ref.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
            at = torch.from_numpy(
                arm.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
            return float(scorer.score(rt, at))

        return fn, _metric_desc("lpips", device)

    from eval.mse import compute_pixel_mse

    def fn(ref, arm):
        rt = torch.from_numpy(ref.transpose(2, 0, 1).astype(np.float32) / 255.0)
        at = torch.from_numpy(arm.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return float(compute_pixel_mse(at, rt))

    return fn, _metric_desc("pixel_mse", device)


def _lpips_usable(device: str) -> bool:
    """Instantiate LPIPSScorer and score one pair; NaN/exception -> unusable."""
    try:
        import torch
        from eval.lpips import LPIPSScorer
        s = LPIPSScorer(device=device)
        a = torch.zeros(1, 3, 256, 256)
        v = s.score(a, a)
        return isinstance(v, float) and v == v
    except Exception:
        return False


def _inception_usable(device: str) -> Tuple[bool, str]:
    """Build the extractor and score one 8x8 image; (ok, reason)."""
    try:
        _inception_features([np.zeros((8, 8, 3), dtype=np.uint8)],
                            "2048", device)
        return True, ""
    except Exception as exc:  # torch_fidelity missing, no weights, no net
        return False, f"{type(exc).__name__}: {exc}"


# Metrics that need the batched path and a reference-free [a] gate.
_BATCHED = ("inception", "inception_conf")
_REFERENCE_FREE = ("inception_conf",)


def _resolve_metric(choice: str, device: str) -> Tuple[str, str]:
    """Resolve ``--metric`` to a concrete (name, description).

    Returns the NAME, not a callable: ``inception``/``inception_conf`` are
    batched over the whole image set, so the per-pair callable would be the
    wrong interface for them. ``_analyze_budget`` dispatches on the name.
    """
    if choice == "auto":
        # Prefer FID's own feature space when it is available, since that is the
        # only per-image metric whose failure at [a] is informative about the
        # reward family rather than about the metric.
        ok, _ = _inception_usable(device)
        if ok:
            choice = "inception"
        elif _lpips_usable(device):
            choice = "lpips"
        else:
            choice = "pixel_mse"
    elif choice == "lpips" and not _lpips_usable(device):
        print("  [WARN] --metric lpips requested but LPIPS is unusable "
              "(torchmetrics / VGG weights unavailable); falling back to "
              "pixel_mse")
        choice = "pixel_mse"
    elif choice in _BATCHED:
        ok, why = _inception_usable(device)
        if not ok:
            print(f"  [WARN] --metric {choice} requested but the "
                  f"torch-fidelity InceptionV3 extractor is unusable ({why}); "
                  "falling back to pixel_mse. Note this changes what the "
                  "verdict means: pixel MSE is NOT in FID's feature space, so "
                  "an [a] failure under it does not close the reward family.")
            choice = "pixel_mse"
    return choice, _metric_desc(choice, device)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _benefit(D: np.ndarray) -> Tuple[float, float, float]:
    """(benefit, best_single_mean, oracle_mean). Benefit = best single arm's
    mean loss minus per-image oracle mean loss; > 0 means per-image selection
    beats the best global arm (loss units, smaller = better)."""
    best_single = float(D.mean(axis=0).min())
    oracle = float(D.min(axis=1).mean())
    return best_single - oracle, best_single, oracle


def _null_within_image(D: np.ndarray, n_perm: int, rng: np.random.RandomState
                       ) -> Tuple[np.ndarray, float]:
    """Design's null: shuffle arm labels WITHIN each image.

    Structurally degenerate for the benefit statistic: a row shuffle leaves
    every row's min untouched, so the oracle term is INVARIANT and only the
    best-single term moves. Printed for completeness; never a gate.
    """
    benefit, _, _ = _benefit(D)
    n_img, n_arm = D.shape
    nulls = np.empty(n_perm)
    for p in range(n_perm):
        Dp = D.copy()
        for i in range(n_img):
            Dp[i] = Dp[i][rng.permutation(n_arm)]
        nulls[p], _, _ = _benefit(Dp)
    return nulls, float((nulls >= benefit - 1e-12).mean())


def _null_column_perm(D: np.ndarray, n_perm: int, rng: np.random.RandomState
                      ) -> Tuple[np.ndarray, float]:
    """Independent per-arm column permutation.

    Preserves each arm's marginal distribution but destroys the cross-arm
    coupling, so its p-value tracks the SIGN of that coupling rather than
    whether crossover is exploitable: with arms positively coupled through
    shared image difficulty the null mean lands far above the observation
    (p ~ 1 even when crossover is real), with anti-correlated arms it lands
    below (p ~ 0 for the same reason). Printed for completeness; never a gate.
    """
    benefit, _, _ = _benefit(D)
    n_img, n_arm = D.shape
    nulls = np.empty(n_perm)
    for p in range(n_perm):
        Dp = D.copy()
        for a in range(n_arm):
            Dp[:, a] = Dp[rng.permutation(n_img), a]
        nulls[p], _, _ = _benefit(Dp)
    return nulls, float((nulls >= benefit - 1e-12).mean())


def _null_residual_perm(D: np.ndarray, n_perm: int,
                        rng: np.random.RandomState
                        ) -> Tuple[np.ndarray, float]:
    """Residual permutation keeping BOTH main effects.

    Decompose ``D = mu + row_effect + col_effect + residual`` (image difficulty
    and arm offset), shuffle the residual within each row, and reassemble. This
    is the only one of the three nulls centred on the observed statistic, but it
    is unreliable in both directions (measured p=0.47 on a real-interaction
    construction, p=0.065 on pure independent noise) because the statistic it
    tests cannot separate the two. Reported as a diagnostic; the verdict rests
    on out-of-sample transfer instead.
    """
    benefit, _, _ = _benefit(D)
    mu = float(D.mean())
    row = D.mean(axis=1, keepdims=True) - mu
    col = D.mean(axis=0, keepdims=True) - mu
    base = mu + row + col
    resid = D - base
    n_img, n_arm = D.shape
    nulls = np.empty(n_perm)
    for p in range(n_perm):
        Rp = resid.copy()
        for i in range(n_img):
            Rp[i] = Rp[i][rng.permutation(n_arm)]
        nulls[p], _, _ = _benefit(base + Rp)
    return nulls, float((nulls >= benefit - 1e-12).mean())


def _image_features(imgs: Dict[int, np.ndarray], keys: List[int]) -> np.ndarray:
    """Per-image features from the REFERENCE image ONLY -> (n, 3).

    Exogenous by construction: nothing here touches any arm's output or the
    distance matrix ``D``, which is what makes the out-of-sample transfer test
    valid (see ``_out_of_sample``). Columns: gradient energy (how much
    high-frequency detail a cache approximation has to reproduce), contrast,
    mean luminance.
    """
    feats = []
    for k in keys:
        g = imgs[k].astype(np.float32).mean(axis=2) / 255.0
        dx = np.diff(g, axis=1)
        dy = np.diff(g, axis=0)
        feats.append([float((dx * dx).mean() + (dy * dy).mean()),
                      float(g.std()), float(g.mean())])
    return np.asarray(feats, dtype=np.float64)


def _cost_regression(Dtr: np.ndarray, Ftr: np.ndarray, Fte: np.ndarray,
                     Dte: np.ndarray) -> float:
    """Per-arm least squares of COST on features; pick argmin of predictions.

    Returns the test-half mean cost of the resulting rule. Unlike transferring
    a neighbour's argmin LABEL, this never has to commit to a noisy per-image
    winner: with uninformative features every arm's fit collapses to its own
    mean, the argmin becomes the globally best arm, and the rule degenerates to
    "always pick the best single arm" (benefit exactly 0) instead of going
    deeply negative.
    """
    rows = np.arange(Dte.shape[0])
    X = np.concatenate([Ftr, np.ones((Ftr.shape[0], 1))], axis=1)
    Xe = np.concatenate([Fte, np.ones((Fte.shape[0], 1))], axis=1)
    beta = np.linalg.lstsq(X, Dtr, rcond=None)[0]
    return float(Dte[rows, (Xe @ beta).argmin(axis=1)].mean())


def _out_of_sample(D: np.ndarray, idxs: List[int], feat: np.ndarray,
                   n_perm: int, rng: np.random.RandomState
                   ) -> Dict[str, object]:
    """Split by global_idx parity; learn on even, settle on odd. THE GATE.

    The transfer feature MUST be exogenous — computed from the reference image
    alone (``_image_features``), never from ``D``. An earlier version used
    ``D[:, train_best]`` (the per-image distance under the train best arm) as
    the "difficulty" feature; that leaks, because the same noise realization
    that moves an image's feature also moves which arm wins it and what the
    test-half cost is. Measured false-positive rate of the leaky version on
    pure iid noise: 25% at n=40, 61% at n=100, 93% at n=250, **99% at n=500** —
    i.e. at this sweep's own N it declared noise "utilizable" almost always.

    THE GATE IS THE COST REGRESSION, not the 1NN label transfer. The label rule
    was the original gate and is kept only as a diagnostic, because it fails in
    a way that is easy to misread as "no crossover": on a construction with a
    genuine difficulty-driven crossover at this sweep's own noise level it
    recovered a NEGATIVE share of the oracle gain in 200/200 trials while
    reaching p<=0.05 in 95% of them — significant and unusable at the same
    time. Transferring an argmin LABEL forces a commitment to a per-image
    winner that is mostly noise when the arms sit within a noise width of each
    other; the cost regression predicts each arm's cost instead and only
    departs from the best single arm where the features actually say so.

    Measured through ``_cost_regression``, 300 trials per construction, at
    n=500 and n=250 (gate = frac > 0 AND p <= 0.05):
      pure-noise features           0.00 / 0.00
      difficulty main effect only   0.00 / 0.00
      per-image hetero variance     0.00 / 0.00
      arm-specific noise scales     0.00 / 0.00
      real interaction (strong)     0.95 / 0.83   <- power
      real interaction (weak)       0.19 / 0.13   <- underpowered, honest
    The same six constructions through the label rule: 0.00 gate everywhere
    except the strong low-noise one.

    Caveat this test cannot settle: reference-derived features are ORACLE-side
    (a deployed policy does not have the full-compute output). A positive result
    establishes that per-image structure exists and is predictable from image
    content; whether an ONLINE-available feature can see it is a separate
    question.

    Returns the test-half oracle benefit and the fraction each transfer rule
    recovers. The constant majority rule is the feature-free ceiling (<= 0 by
    construction, since it cannot beat the test best single arm).
    """
    even = np.array([i % 2 == 0 for i in idxs], dtype=bool)
    tr_idx = np.where(even)[0]
    te_idx = np.where(~even)[0]
    if len(tr_idx) < 2 or len(te_idx) < 2:
        return {"skip": "parity split too small"}
    Dtr, Dte = D[tr_idx], D[te_idx]

    best_single_test = float(Dte.mean(axis=0).min())
    oracle_test = float(Dte.min(axis=1).mean())
    oracle_benefit = best_single_test - oracle_test

    train_win = Dtr.argmin(axis=1)  # per-image oracle winners on train

    # constant majority rule (feature-free ceiling)
    counts = np.bincount(train_win, minlength=D.shape[1])
    cands = np.where(counts == counts.max())[0]
    rule_const = int(cands[np.argmin(Dtr.mean(axis=0)[cands])])
    const_cost = float(Dte[:, rule_const].mean())
    const_benefit = best_single_test - const_cost

    # exogenous-feature transfer; z-scored so the three columns are comparable
    scale = feat.std(axis=0)
    if not np.all(scale > 1e-12):
        return {"skip": "reference-image features are degenerate (zero "
                        "variance) — 1NN transfer is undefined"}
    F = (feat - feat.mean(axis=0)) / scale
    Ftr, Fte = F[tr_idx], F[te_idx]

    # --- THE GATE: per-arm cost regression, permuted by shuffling train rows ---
    reg_benefit = best_single_test - _cost_regression(Dtr, Ftr, Fte, Dte)
    reg_nulls = np.empty(n_perm)
    for p in range(n_perm):
        reg_nulls[p] = best_single_test - _cost_regression(
            Dtr[rng.permutation(Dtr.shape[0])], Ftr, Fte, Dte)
    reg_p = float((reg_nulls >= reg_benefit - 1e-12).mean())

    # --- diagnostic: 1NN label transfer (the former gate) ---
    nn_map = np.array([int(np.argmin(((Ftr - f) ** 2).sum(axis=-1)))
                       for f in Fte])

    def _nn_cost(win_labels: np.ndarray) -> float:
        rule = win_labels[nn_map]
        return float(Dte[np.arange(len(te_idx)), rule].mean())

    nn_benefit = best_single_test - _nn_cost(train_win)

    nulls = np.empty(n_perm)
    for p in range(n_perm):
        nulls[p] = best_single_test - _nn_cost(
            train_win[rng.permutation(len(train_win))])
    nn_p = float((nulls >= nn_benefit - 1e-12).mean())

    def _frac(benefit: float) -> float:
        return (benefit / oracle_benefit if oracle_benefit > 1e-12 else float("nan"))

    return {
        "skip": None,
        "train_n": int(len(tr_idx)),
        "test_n": int(len(te_idx)),
        "best_single_test": best_single_test,
        "oracle_test": oracle_test,
        "oracle_benefit_test": oracle_benefit,
        "rule_const": rule_const,
        "rule_nn_seed_arms": sorted(int(a) for a in set(train_win.tolist())),
        "const_frac": _frac(const_benefit),
        "reg_frac": _frac(reg_benefit),
        "reg_benefit": reg_benefit,
        "reg_p": reg_p,
        "reg_null_mean": float(reg_nulls.mean()),
        "reg_null_std": float(reg_nulls.std()),
        "nn_frac": _frac(nn_benefit),
        "nn_benefit": nn_benefit,
        "nn_p": nn_p,
        "nn_null_mean": float(nulls.mean()),
        "nn_null_std": float(nulls.std()),
    }


# --------------------------------------------------------------------------
# Per-budget analysis
# --------------------------------------------------------------------------

def _analyze_budget(budget_dir: str, ref_imgs: Dict[int, np.ndarray],
                    metric: str, device: str, args
                    ) -> Dict[str, object]:
    budget = int(os.path.basename(budget_dir)[1:])
    manifest = _load(os.path.join(budget_dir, "manifest.json"))
    num_steps = int((manifest or {}).get("num_steps", 0))

    arm_dirs = sorted(glob.glob(os.path.join(budget_dir, "equalflops", "arm_*")))
    labels: List[str] = []
    results: Dict[str, dict] = {}
    arm_imgs: Dict[str, Dict[int, np.ndarray]] = {}
    for adir in arm_dirs:
        label = os.path.basename(adir)[len("arm_"):]
        labels.append(label)
        results[label] = _load(os.path.join(adir, "results.json")) or {}
        arm_imgs[label] = _load_images(os.path.join(adir, "generated"))

    row: Dict[str, object] = {
        "budget": budget,
        "num_steps": num_steps,
        "labels": labels,
        "valid": None,
        "skip": None,
    }
    if not labels:
        row["skip"] = "no forced-arm runs"
        return row

    common = set(ref_imgs)
    for label in labels:
        common &= set(arm_imgs[label])
    common = sorted(common)
    if len(common) < 2:
        row["skip"] = f"only {len(common)} images shared with reference (need >=2)"
        return row

    metric_desc = _metric_desc(metric, device)
    ref_al = np.stack([ref_imgs[k] for k in common])
    n = len(common)
    n_arm = len(labels)

    # Per-arm uint8 stacks, aligned to `common`, shared by every metric below.
    arm_stacks = [np.stack([arm_imgs[label][k] for k in common])
                  for label in labels]

    # loss matrix in the primary metric's units (smaller = better everywhere)
    row["pixel_mse_crosscheck"] = None
    if metric in _BATCHED:
        try:
            D = _distance_matrix(metric, device, ref_al, arm_stacks)
        except Exception as exc:
            row["skip"] = (f"{metric} extractor failed at analysis time "
                           f"({type(exc).__name__}: {exc})")
            return row
        if not np.isfinite(D).all():
            row["skip"] = f"{metric} produced non-finite values"
            return row
    elif metric == "pixel_mse":
        D = np.empty((n, n_arm))
        for j, arm_al in enumerate(arm_stacks):
            D[:, j] = _pixel_mse_vectorized(ref_al, arm_al)
        # cross-check the vectorized path against eval/mse.py's compute_pixel_mse
        try:
            import torch
            from eval.mse import compute_pixel_mse
            rt = torch.from_numpy(ref_al[0].transpose(2, 0, 1).astype(np.float32) / 255.0)
            at = torch.from_numpy(
                arm_stacks[0][0].transpose(2, 0, 1).astype(np.float32) / 255.0)
            chk = float(compute_pixel_mse(at, rt))
            row["pixel_mse_crosscheck"] = abs(chk - D[0, 0])
        except Exception:
            row["pixel_mse_crosscheck"] = None
    else:
        fn, _ = _make_distance(metric, device)
        D = np.empty((n, n_arm))
        for j, label in enumerate(labels):
            for i, k in enumerate(common):
                D[i, j] = fn(ref_imgs[k], arm_imgs[label][k])
        if np.isnan(D).any():
            row["skip"] = "LPIPS produced NaN (model load failed)"
            return row

    # [b]'s quantization floor is ALWAYS in pixel-MSE units: it asks whether the
    # arm IMAGES differ by more than 8-bit rounding, which is a property of the
    # PNGs and not of whatever metric ranks them. For a reference-free metric it
    # therefore answers "do the images differ above rounding", not "does the
    # score differ above rounding" — stated explicitly in the rendering.
    if metric == "pixel_mse":
        D_pmse = D
    else:
        D_pmse = np.empty((n, n_arm))
        for j, arm_al in enumerate(arm_stacks):
            D_pmse[:, j] = _pixel_mse_vectorized(ref_al, arm_al)

    fids: Dict[str, Optional[float]] = {
        label: (results[label].get("aggregate") or {}).get("fid")
        for label in labels}
    is_means: Dict[str, Optional[float]] = {
        label: (results[label].get("aggregate") or {}).get("is_mean")
        for label in labels}
    row["fids"] = fids
    row["is_means"] = is_means
    row["metric"] = metric
    row["reference_free"] = metric in _REFERENCE_FREE
    row["metric_desc"] = metric_desc
    row["n_images"] = n
    row["mean_dist"] = {label: float(D[:, j].mean()) for j, label in enumerate(labels)}
    row["argmin_counts"] = {label: int((D.argmin(axis=1) == j).sum())
                            for j, label in enumerate(labels)}

    # --- (a) validity: mean per-image loss must rank arms like FID ---
    dist_vals = [float(D[:, j].mean()) for j in range(n_arm)]
    fid_vals = [fids[label] for label in labels]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in fid_vals):
        # Leave ``valid`` at None so the renderer prints a SKIPPED line instead
        # of trying to build the [a] table: it would call float() on the missing
        # FID and read row["spearman"], neither of which exists here.
        row["skip"] = "some arm results.json missing FID"
        return row
    # Rank consistency ignoring ties: every pair of arms whose FIDs differ must
    # be ranked the same way by the per-image mean loss. Tied FIDs impose
    # no constraint (near-tied losses must not flip the gate).
    consistent = True
    for a in range(n_arm):
        for b in range(a + 1, n_arm):
            df = float(fid_vals[a]) - float(fid_vals[b])  # type: ignore[arg-type]
            if abs(df) < 1e-6:
                continue
            dd = dist_vals[a] - dist_vals[b]
            if dd * df < 0:
                consistent = False
    rho = _spearman(dist_vals, [float(v) for v in fid_vals])  # type: ignore[arg-type]
    row["valid"] = consistent
    row["spearman"] = rho
    # A reference-free confidence score is structurally the per-image half of IS,
    # not of FID (see _entropy_from_logits), so ALSO rank it against IS with the
    # sign flipped (higher IS is better, lower entropy is better). This separates
    # two very different failures when [a] fails: the proxy is unfaithful to its
    # own aggregate, versus the proxy is faithful to IS and IS disagrees with FID.
    row["spearman_is"] = None
    if metric in _REFERENCE_FREE and not any(
            v is None or (isinstance(v, float) and math.isnan(v))
            for v in is_means.values()):
        row["spearman_is"] = _spearman(
            dist_vals, [-float(is_means[label]) for label in labels])  # type: ignore[arg-type]
    if not row["valid"]:
        row["skip"] = "mean per-image loss does not rank arms like FID"
        return row

    # --- (b) crossover ---
    rng = np.random.RandomState(args.perm_seed)
    benefit, best_single, oracle = _benefit(D)
    null_a, p_a = _null_within_image(D, args.max_perm, rng)
    null_b, p_b = _null_column_perm(D, args.max_perm, rng)
    null_r, p_r = _null_residual_perm(D, args.max_perm, rng)
    row["benefit"] = benefit
    row["best_single_mean"] = best_single
    row["oracle_mean"] = oracle
    row["null_a_mean"] = float(null_a.mean())
    row["null_a_std"] = float(null_a.std())
    row["null_a_p"] = p_a
    row["null_b_mean"] = float(null_b.mean())
    row["null_b_std"] = float(null_b.std())
    row["null_b_p"] = p_b
    row["null_r_mean"] = float(null_r.mean())
    row["null_r_std"] = float(null_r.std())
    row["null_r_p"] = p_r
    row["oos"] = _out_of_sample(D, common, _image_features(ref_imgs, common),
                                args.oos_perm, rng)

    # --- (c) quantization floor (always in pixel-MSE units) ---
    spread = D_pmse.max(axis=1) - D_pmse.min(axis=1)
    mean_spread = float(spread.mean())
    # direct arm-vs-arm pixel MSE (6 pairs for 4 arms)
    pair_mses = []
    for a in range(n_arm):
        for b in range(a + 1, n_arm):
            pair_mses.append(float(_pixel_mse_vectorized(
                arm_stacks[a], arm_stacks[b]).mean()))
    row["floor_ratio"] = mean_spread / _FLOOR
    row["mean_spread_pmse"] = mean_spread
    row["mean_arm_arm_pmse"] = float(np.mean(pair_mses)) if pair_mses else None
    row["near_floor_frac"] = float((spread <= 5.0 * _FLOOR).mean())
    return row


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt(value: Optional[float], spec: str = ".4g") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def _render_budget(row: Dict[str, object], floor_thresh: float) -> List[str]:
    out: List[str] = []
    budget = row["budget"]
    ns = row["num_steps"]
    out.append("")
    out.append(f"== budget k{budget}/{ns} ==")
    labels = row["labels"]
    if row.get("valid") is None:
        out.append(f"  SKIPPED: {row.get('skip') or 'no arms'}")
        return out
    out.append(f"  arms: {' '.join(labels)}   images paired to reference: "
               f"{row['n_images']}   metric: {row['metric_desc']}")
    if row.get("pixel_mse_crosscheck") is not None:
        out.append(f"  [cross-check] vectorized pixel MSE vs eval/mse.py "
                   f"compute_pixel_mse on one pair: abs diff "
                   f"{row['pixel_mse_crosscheck']:.3e}")
    mean_dist = row["mean_dist"]
    fids = row["fids"]
    ref_free = bool(row.get("reference_free"))
    loss_name = "mean_conf" if ref_free else "mean_dist"
    if ref_free:
        out.append("  [a] validity — mean per-image Inception entropy vs FID "
                   "(REFERENCE-FREE metric: lower entropy = more confident = "
                   "asserted better, so it must still rank arms like FID)")
    else:
        out.append("  [a] validity — mean per-image distance vs FID")
    dist_ranks = _average_ranks([mean_dist[l] for l in labels])
    fid_ranks = _average_ranks([float(fids[l]) for l in labels])  # type: ignore[arg-type]
    is_means = row.get("is_means") or {}
    out.append(f"    {'arm':<14} {loss_name:>12} {'FID':>8} {'IS':>7} "
               f"{'d_rank':>6} {'fid_rank':>8}")
    for j, label in enumerate(labels):
        out.append(f"    {label:<14} {_fmt(mean_dist[label]):>12} "
                   f"{_fmt(fids[label], '.2f'):>8} "
                   f"{_fmt(is_means.get(label), '.1f'):>7} "
                   f"{dist_ranks[j]:>6.1f} {fid_ranks[j]:>8.1f}")
    rho = row["spearman"]
    out.append(f"    Spearman({loss_name}, FID) = {rho:.3f}; rank consistency "
               "(pairs with tied FIDs impose no constraint) -> "
               + ("PROXY VALID" if row["valid"]
                  else "度量不可比 — crossover 结论不成立 (this budget skipped)"))
    if ref_free:
        rho_is = row.get("spearman_is")
        out.append("    Spearman(mean_conf, -IS) = "
                   + (f"{rho_is:.3f}" if rho_is is not None else "n/a")
                   + " — a reference-free confidence score is the per-image half"
                     " of IS, not of FID, so this second correlation separates "
                     "two different [a] failures: proxy unfaithful to its OWN "
                     "aggregate (both low), versus proxy faithful to IS while IS "
                     "disagrees with FID (this high, the FID one low). Only the "
                     "FID column gates.")
        out.append("    [KNOWN BIAS, not a bug] entropy rewards "
                   "class-prototypicality, so an arm that collapses toward "
                   "confident, generic exemplars scores WELL here while FID "
                   "punishes it. That is exactly why FID remains the gate.")
    if not row["valid"]:
        return out

    floor_ratio = row["floor_ratio"]
    out.append("  [b] quantization floor (8-bit PNG; per-pixel MSE units)")
    out.append(f"    uint8 rounding floor, two rounded images: {_FLOOR:.3e}; "
               f"one rounded: {FLOOR_ONE_ROUNDED:.3e}")
    if row.get("metric") not in (None, "pixel_mse"):
        out.append("    (the floor is deliberately in pixel-MSE units even "
                   "though [a]/[c] are not: it asks whether the arm IMAGES "
                   "differ by more than 8-bit rounding, which is a property of "
                   "the saved PNGs, not of the ranking metric)")
    out.append(f"    mean per-image arm-distance spread: "
               f"{_fmt(row['mean_spread_pmse']):>10} = {floor_ratio:.0f}x floor")
    arm_arm = row["mean_arm_arm_pmse"]
    if arm_arm is not None:
        out.append(f"    mean arm-vs-arm pixel MSE: {arm_arm:.3e} "
                   f"(= {arm_arm / _FLOOR:.0f}x floor)")
    out.append(f"    images whose arm spread <= 5x floor: "
               f"{100.0 * row['near_floor_frac']:.1f}%")
    if floor_ratio < floor_thresh:
        out.append("    => SIGNAL AT THE QUANTIZATION FLOOR — measured crossover "
                   "is rounding noise; need latent or fp16 images, not PNG")
        out.append("    (crossover numbers below are informational only — this "
                   "budget is excluded from the verdict)")
    else:
        out.append(f"    => signal above the floor ({floor_ratio:.0f}x), "
                   "rounding noise does not dominate")

    out.append("  [c] crossover, DESCRIPTIVE ONLY (loss units; smaller = better)")
    counts = row["argmin_counts"]
    count_s = ", ".join(f"{l}={counts[l]}" for l in labels)
    out.append(f"    per-image winner counts: {count_s}")
    best_arm = labels[int(np.argmin([float(row['mean_dist'][l]) for l in labels]))]
    out.append(f"    best single arm: {best_arm}  mean loss {row['best_single_mean']:.4g}")
    out.append(f"    per-image oracle: mean loss {row['oracle_mean']:.4g}")
    out.append(f"    in-sample benefit: {row['benefit']:.4g} "
               "(>0 = per-image selection beats the best global arm)")
    out.append("      [WHY NOT A GATE] a per-image min turns iid measurement "
               "noise into positive benefit — more arm-to-arm variance means "
               "bigger apparent gain — so this is positive under pure noise "
               "too, and its MAGNITUDE does not rank the two apart. Measured on "
               "constructed trees: pure independent noise gave a gap worth 24% "
               "of the best arm's loss, real difficulty-driven crossover gave "
               "18% — the noise case looked BIGGER. The verdict therefore rests "
               "on [d] out-of-sample, not on this number or on any of the three "
               "nulls below.")
    out.append(f"    null A (labels shuffled within each image): "
               f"mean {row['null_a_mean']:.4g} +/- {row['null_a_std']:.4g} "
               f"p={row['null_a_p']:.4g}")
    out.append("      [DEGENERATE] the per-image min is invariant under a "
               "within-image label shuffle (sort(D, axis=1) is unchanged), so "
               "the oracle term does not move at all; only the best-single term "
               "does, which is not the quantity of interest.")
    out.append(f"    null B (each arm's column permuted independently): "
               f"mean {row['null_b_mean']:.4g} +/- {row['null_b_std']:.4g} "
               f"p={row['null_b_p']:.4g}")
    out.append("      [TRACKS COUPLING SIGN, NOT CROSSOVER] permuting columns "
               "destroys the cross-arm coupling, so p reflects that coupling's "
               "sign: positively coupled arms (shared image difficulty) put the "
               "null mean far ABOVE the observation (measured obs +0.098 vs "
               "null mean +0.30, p=1.00 with crossover REAL), anti-correlated "
               "arms put it below and give p~0. Either way it is not evidence "
               "about exploitability.")
    out.append(f"    null R (residual permuted, both main effects kept): "
               f"mean {row['null_r_mean']:.4g} +/- {row['null_r_std']:.4g} "
               f"p={row['null_r_p']:.4g}")
    out.append("      [CENTRED BUT UNRELIABLE] the only null of the three "
               "centred on the observed statistic, yet it errs in both "
               "directions — measured p=0.47 on a real-interaction construction "
               "(missed it) and p=0.065 on pure independent noise (nearly a "
               "false positive) — because the statistic cannot separate them.")

    oos = row["oos"]
    if oos.get("skip"):
        out.append(f"  [d] out-of-sample: {oos['skip']}")
    elif oos["oracle_benefit_test"] <= 1e-12:
        out.append(f"  [d] out-of-sample (global_idx parity split: "
                   f"{oos['train_n']} train / {oos['test_n']} test)")
        out.append(f"    test-half oracle benefit: 0 — best single arm already "
                   "equals the per-image oracle there, so there is no oracle "
                   "gain to transfer (out-of-sample is undefined)")
    else:
        out.append(f"  [d] out-of-sample (global_idx parity split: "
                   f"{oos['train_n']} train / {oos['test_n']} test)")
        out.append(f"    test-half oracle benefit: {oos['oracle_benefit_test']:.4g} "
                   "(=100%)")
        const_s = _fmt(oos["const_frac"] * 100.0, ".1f")
        reg_s = _fmt(oos["reg_frac"] * 100.0, ".1f")
        nn_s = _fmt(oos["nn_frac"] * 100.0, ".1f")
        out.append(f"    constant majority rule: {const_s}% "
                   "(feature-free ceiling, <=0 by construction)")
        out.append(f"    reference-feature COST REGRESSION (the gate): {reg_s}% "
                   f"(perm p={oos['reg_p']:.4g}, null mean "
                   f"{oos['reg_null_mean']:.4g} +/- {oos['reg_null_std']:.4g})")
        out.append("      features are EXOGENOUS: gradient energy / contrast / "
                   "luminance of the REFERENCE image only, never D. A leaky "
                   "D-derived feature declared pure noise utilizable 99% of the "
                   "time at n=500. This rule fits each arm's COST on those "
                   "features and takes the argmin of the predictions, so with "
                   "uninformative features it degenerates to the best single "
                   "arm (frac exactly 0) rather than going negative. Measured "
                   "gate rate over 300 trials at n=500/250: 0.00/0.00 on all "
                   "four no-crossover constructions (pure-noise features, "
                   "difficulty main effect, per-image hetero variance, "
                   "arm-specific noise scales), 0.95/0.83 on a real strong "
                   "interaction, 0.19/0.13 on a weak one (underpowered).")
        out.append(f"    [diagnostic, NOT the gate] 1NN label transfer: {nn_s}% "
                   f"(perm p={oos['nn_p']:.4g}, null mean "
                   f"{oos['nn_null_mean']:.4g} +/- {oos['nn_null_std']:.4g})")
        out.append("      This was the former gate and it fails misleadingly: "
                   "transferring a neighbour's argmin LABEL forces a commitment "
                   "to a per-image winner that is mostly noise when arms sit "
                   "within a noise width of each other. On a construction with "
                   "a GENUINE crossover at this sweep's noise level it recovered "
                   "a negative share in 200/200 trials while reaching p<=0.05 in "
                   "95% of them. So read its p as 'the features do carry "
                   "information about which arm wins' and its negative frac as "
                   "'label transfer cannot monetize it' — the two are not in "
                   "conflict, and only the cost regression above gates.")
        out.append("    => VERDICT GATE: crossover " +
                   ("UTILIZABLE out-of-sample" if oos["reg_frac"] > 0.0
                    and oos["reg_p"] <= 0.05
                    else "NOT utilizable — the cost regression recovers no "
                         "positive share of the oracle gain at p<=0.05, which "
                         "is what pure measurement noise looks like"))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-image crossover verdict for the COVR budget probe")
    ap.add_argument("probe_dir", help="root containing k<K>/ and reference/")
    ap.add_argument("--metric",
                    choices=["auto", "inception", "inception_conf", "lpips",
                             "pixel_mse"],
                    default="auto",
                    help="per-image loss (auto: inception if torch-fidelity's "
                         "InceptionV3 loads, else LPIPS, else pixel MSE). "
                         "inception = squared difference of FID's own pool3 "
                         "'2048' features vs the reference; inception_conf = "
                         "REFERENCE-FREE Inception predictive entropy (the "
                         "per-image half of IS). The resolved choice is always "
                         "printed in the output.")
    ap.add_argument("--device", default="auto",
                    help="device for the loss network (auto: cuda when "
                         "available, else cpu). Only matters for inception* and "
                         "lpips; on the GPU box auto keeps the 299x299 "
                         "Inception forwards off the CPU.")
    ap.add_argument("--max-perm", type=int, default=20000,
                    help="permutations for the crossover nulls")
    ap.add_argument("--oos-perm", type=int, default=2000,
                    help="permutations for the out-of-sample NN rule")
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--floor-ratio", type=float, default=5.0,
                    help="arm spread / floor below this = quantization-limited")
    args = ap.parse_args()

    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    probe = args.probe_dir
    ref_dir = os.path.join(probe, "reference")
    ref_fid, ref_config, ref_imgs = _load_reference(ref_dir)

    print("== COVR per-image crossover verdict ==")
    print(f"  probe dir: {probe}")
    print(f"  reference dir: {ref_dir}")

    if not ref_imgs:
        print("  REFERENCE MISSING: no generated PNGs under "
              f"{ref_dir}/generated/. Re-run the sweep with REFERENCE=1 (or "
              "point REFERENCE_DIR at an existing full-compute reference). "
              "Per-image crossover cannot be established without it.")
    metric, metric_desc = _resolve_metric(args.metric, args.device)
    print(f"  metric: {metric_desc}")
    if metric in _REFERENCE_FREE:
        print("  note: reference-free applies to the LOSS only. The reference "
              "run is still required — [b]'s quantization floor and [d]'s "
              "exogenous 1NN features both come from the reference image, and "
              "arms are still paired to it by global_idx.")

    if ref_fid is not None:
        print(f"  reference FID: {_fmt(ref_fid, '.2f')}"
              f"  (config: seed={ref_config.get('seed')} "
              f"num_steps={ref_config.get('num_steps')} "
              f"n_prompts={ref_config.get('n_prompts')})")

    budgets = _budget_dirs(probe)
    if not budgets:
        print(f"  no k<K>/ budget dirs under {probe}")
        print("")
        print("== RECOMMENDATION ==")
        print("  INSUFFICIENT EVIDENCE: no budget dirs. Run "
              "scripts/sweep_budget_probe.sh first.")
        return 0

    rows: List[Dict[str, object]] = []
    for _, bdir in budgets:
        row = _analyze_budget(bdir, ref_imgs, metric, args.device, args)
        rows.append(row)
        for line in _render_budget(row, args.floor_ratio):
            print(line)

    # ---- cross-budget: does the harshest budget actually bind? ----
    print("")
    print("== budget binding vs the full-compute reference ==")
    if ref_fid is not None:
        print(f"    {'calc':>4} {'best-arm FID':>13} {'worst-arm FID':>13} "
              f"{'best - ref':>11}")
        best_by_budget: Dict[int, float] = {}
        for row in rows:
            if not row.get("skip") and row.get("valid"):
                fids = {l: float(v) for l, v in row["fids"].items()
                        if v is not None}
                if len(fids) >= 2:
                    b = min(fids.values())
                    w = max(fids.values())
                    best_by_budget[row["budget"]] = b
                    print(f"    k{row['budget']:>3} {b:>13.2f} {w:>13.2f} "
                          f"{b - ref_fid:>+11.2f}")
        if len(best_by_budget) >= 2:
            ks = sorted(best_by_budget)
            loosest = max(ks)      # most calc steps = least aggressive
            harshest = min(ks)     # fewest calc steps = most aggressive
            delta = best_by_budget[harshest] - best_by_budget[loosest]
            print(f"    harshest (k{harshest}) vs loosest (k{loosest}) "
                  f"best-arm FID: {delta:+.2f}  -> budget "
                  + ("BINDS (harsher budget costs real quality)" if delta > 0
                     else "does NOT bind (aggressive budgets cost nothing) — "
                          "the probe is not testing what it claims"))
        else:
            print("    (need >=2 valid budgets to check binding)")
    else:
        print("    (reference FID unavailable — set REFERENCE=1)")

    # ---- per-class crossover: what it would take ----
    print("")
    print("== per-class crossover (not measured) ==")
    print("  The dataset window is 50k images shuffled then first-n taken, so "
          "n=500 gives ~500 DIFFERENT classes at ~1 image each. Per-class FID "
          "(a distribution metric) on 1 image is meaningless, so per-class "
          "crossover is unmeasurable at this N.")
    print("  To measure it the sweep's SAMPLING must change (an owner "
          "decision — this script does not touch the sweep): sample a FEW "
          "classes x MANY images, e.g. 10 classes x 50 images = 500 (same N, "
          "same GPU cost) or 20 classes x 100 = 2000 (4x current). Per-class "
          "crossover then needs per-class best-arm-vs-oracle on ~50+ images "
          "and a per-class FID, which needs >= ~50 images/class to be "
          "meaningful — that is the real driver of the cost change, not the "
          "grouping itself.")

    # ---- recommendation ----
    print("")
    print("== RECOMMENDATION ==")
    usable = [r for r in rows if not r.get("skip") and r.get("valid")]
    if not rows:
        print("  INSUFFICIENT EVIDENCE: no budget data.")
    elif not ref_imgs:
        print("  REFERENCE MISSING: re-run the sweep with REFERENCE=1 (or set "
              "REFERENCE_DIR) so every arm's PNG can be paired against the "
              "full-compute output for the same (seed, global_idx). Without it "
              "there is no per-image quality anchor and no crossover verdict.")
    elif not usable:
        if all(r.get("skip") and "rank arms like FID" in str(r.get("skip"))
               for r in rows):
            print("  度量不可比，crossover 结论不成立: in every budget, the "
                  "per-image loss's arm ordering disagrees with the FID "
                  "ordering. The proxy is not measuring what FID measures, so "
                  "no crossover conclusion can be drawn from it.")
            if metric == "inception":
                print("  WHAT THIS METRIC'S FAILURE ADDITIONALLY BUYS: the loss "
                      "here is a squared distance in FID's OWN InceptionV3 "
                      "pool3 feature space — the strongest reference-based "
                      "per-image proxy available, strictly stronger than pixel "
                      "MSE or LPIPS for predicting FID. Its failing [a] is "
                      "therefore evidence about the REWARD FAMILY, not about "
                      "metric choice: 'per-image fidelity to the full-compute "
                      "output' does not order arms the way FID does at this N. "
                      "That closes the terminal-fidelity reward AND its H-step "
                      "sentinel variant (same family, intermediate latents) "
                      "without spending the sentinel's +125% forwards. The "
                      "remaining escape is that the reference is not the FID "
                      "optimum (measured: reference 128.61 vs k8 uniform 119.29 "
                      "under identical pairing), i.e. distance-to-reference is "
                      "the wrong anchor no matter how good the feature space — "
                      "test that with --metric inception_conf, which needs no "
                      "reference at all.")
            elif metric == "inception_conf":
                print("  WHAT THIS METRIC'S FAILURE ADDITIONALLY BUYS: this "
                      "loss is reference-FREE, so its failure cannot be blamed "
                      "on the reference not being the FID optimum. Check the "
                      "printed Spearman(mean_conf, -IS) per budget: high there "
                      "and low against FID means the proxy is faithful to IS "
                      "and IS is the thing that disagrees with FID (a known "
                      "property of prototypicality-rewarding scores, not a "
                      "measurement defect); low against BOTH means the "
                      "per-image entropy is simply not tracking either "
                      "aggregate at n=500.")
            else:
                print(f"  CAVEAT ON THIS RUN'S METRIC ({metric}): it is not in "
                      "FID's feature space, so its failure does NOT close the "
                      "reward family — it may only mean the proxy is weak. "
                      "Re-run with --metric inception (torch-fidelity's "
                      "InceptionV3 pool3, i.e. FID's own features) before "
                      "concluding anything about the reward design.")
        else:
            print(f"  INSUFFICIENT EVIDENCE: no budget passed the validity gate "
                  f"(skips: {sorted({str(r.get('skip')) for r in rows})}).")
    else:
        quant_limited = [r for r in usable
                         if r["floor_ratio"] < args.floor_ratio]
        clean = [r for r in usable if r not in quant_limited]
        # VERDICT GATE = out-of-sample transfer with EXOGENOUS (reference-image)
        # features ONLY. The in-sample oracle gap and all three permutation
        # nulls are descriptive: pure iid noise produces an in-sample gap as
        # large as a genuinely crossing one (24% vs 18% of the best arm's loss
        # on constructed trees — the NOISE case larger), and no null separates
        # them (A degenerate, B tracks the coupling's sign, R errs both ways).
        # The transfer feature must not come from D: a D-derived one declared
        # pure noise utilizable 99% of the time at n=500. See _out_of_sample.
        gap = [r for r in clean if r["benefit"] > 1e-12]
        util = [r for r in clean
                if r["oos"].get("reg_frac", 0.0) > 0.0
                and r["oos"].get("reg_p", 1.0) <= 0.05]
        # Significant p with a NEGATIVE frac on the label-transfer diagnostic:
        # the features do carry information about which arm wins, but label
        # transfer cannot monetize it. Worth naming, because it is not the same
        # thing as "pure noise" and a reader would otherwise lump them together.
        informative = [r for r in clean
                       if r not in util
                       and r["oos"].get("nn_p", 1.0) <= 0.05]
        if quant_limited:
            print(f"  QUANTIZATION-LIMITED at "
                  + ", ".join(f"k{r['budget']}" for r in quant_limited)
                  + f": arm-distance spread is within {args.floor_ratio}x of the "
                    "uint8 rounding floor — the measured differences there are "
                    "rounding noise, NOT real arm quality. No crossover "
                    "conclusion is drawn for those budgets; re-run with latent "
                    "or fp16 image saving instead of 8-bit PNGs.")
        if not clean:
            print("  VERDICT: 差异淹没在量化噪声里 — every usable budget sits at "
                  "the quantization floor, so neither crossover nor its absence "
                  "can be established from PNGs. Save latent or fp16 images and "
                  "re-run the analysis.")
        elif util:
            print(f"  CROSSOVER REAL AND UTILIZABLE at budget k{util[0]['budget']}"
                  + (f" (also k{util[1]['budget']})" if len(util) > 1 else "")
                  + ": a per-arm COST REGRESSION on REFERENCE-IMAGE features "
                    "(gradient energy / contrast / luminance — nothing derived "
                    "from the distance matrix), fit on one parity half and "
                    "applied to the other, recovers a positive share of the "
                    "oracle benefit at p<=0.05. Out-of-sample transfer on "
                    "exogenous features is the only evidence here that iid noise "
                    "cannot fake. This is the case the bandit is FOR. NEXT GATE, "
                    "not skippable: the features used are ORACLE-side (they need "
                    "the full-compute image). Before spending bandit GPU budget, "
                    "confirm an ONLINE-available signal — early-step latent "
                    "statistics, class embedding, TeaCache raw_diff at the first "
                    "calc step — reproduces this transfer. Structure being "
                    "predictable from image content does not mean a deployable "
                    "policy can see it.")
        elif gap:
            print("  NO UTILIZABLE CROSSOVER at "
                  + ", ".join(f"k{r['budget']}" for r in gap)
                  + ": the per-image oracle does beat the best single arm "
                    "in-sample, but the out-of-sample cost regression recovers "
                    "no positive share of that gain (reg_frac<=0 or p>0.05). An "
                    "in-sample gap of this size is exactly what independent "
                    "per-image measurement noise produces — taking the min over "
                    "4 noisy columns always looks better than any one column. "
                    "On this evidence the gap is NOT established as real "
                    "structure, so do not commit bandit GPU budget to it. Two "
                    "ways forward: stronger per-image features, or more images "
                    "per arm to shrink the per-image noise the oracle is "
                    "harvesting.")
            if informative:
                print("  BUT NOT PURE NOISE EITHER at "
                      + ", ".join(f"k{r['budget']}" for r in informative)
                      + ": the label-transfer diagnostic reached p<=0.05 there "
                        "while recovering a negative share. Its permutation null "
                        "shuffles the winner labels, so a significant p means "
                        "the reference features DO predict which arm wins — what "
                        "fails is acting on that prediction by committing to a "
                        "single per-image winner. Pure iid noise produces this "
                        "pattern 2.3% of the time; a real crossover at this "
                        "noise level produces it 96.7% of the time. So the "
                        "structure is likely there and too weak to pay for at "
                        "n=500 with these 3 features. The cheap next move is "
                        "more images per arm (shrinks the per-image noise the "
                        "oracle harvests) or richer exogenous features, NOT a "
                        "bandit run.")
        else:
            print("  NO CROSSOVER FOUND at any budget with a measurable signal: "
                  "the per-image oracle does not even beat the best single arm "
                  "in-sample. One arm is effectively best everywhere -> pick it "
                  "OFFLINE; a per-trajectory bandit has nothing to discover "
                  "here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
