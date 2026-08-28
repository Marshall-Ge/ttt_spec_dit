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

Five sections per budget; [d] is the verdict gate and [e] can veto it.
Lettering below matches the printed output exactly.

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
                    THE GATE IS NOT p<=0.05 ALONE. Permuting the train rows
                    destroys feature->cost, so every arm's fit collapses to its
                    own mean and the permuted benefit is EXACTLY 0 for 0.56-0.60
                    of the null draws; p<=0.05 is therefore nearly the same
                    statement as frac>0 and carries almost no effect-size
                    information. Two magnitude floors are required on top:
                      RELATIVE — the recovered share must exceed 5% of the oracle
                        gain. No-crossover worlds that still reach p<=0.05 report
                        a median 0.67% (noise features) to 1.58% (high noise); a
                        real crossover reports ~21%. The floor keeps 0.963 power
                        on the real interaction and cuts the worst false-positive
                        rate from 0.048 to 0.003.
                      ABSOLUTE — the recovered benefit, converted to FID with the
                        budget's own (mean loss, FID) arm pairs, must exceed 2x
                        the same-arm replica FID noise SD from ``noise_*``
                        (unbiased: ``sd / c4(n)``, NOT the range — see
                        ``_noise_fid_stats`` for why range would tighten the
                        floor as replicas are added). This is the same
                        2x-noise-floor rule the project applies to every other
                        arm-spread claim. A per-image gain smaller than
                        resampling noise cannot be spent even if it is real.
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
  [e] granularity — WHAT THE BANDIT CAN ACTUALLY CAPTURE. [d] measures
                  per-IMAGE crossover, but the bandit does not decide per image:
                  ``run_dit.py`` assigns each contiguous block of global_idx to
                  ONE trajectory (``batch_start // bs``) and calls
                  ``apply_strategy`` once per batch, so a single refresh mask
                  serves the whole batch (``accelerators/teacache.py``'s forced
                  mask is indexed by step only). The per-batch oracle gain is
                  therefore the ceiling a per-batch bandit could reach: the
                  best single arm's mean cost minus, per batch, that batch's
                  own best-mean-cost arm (rows grouped by
                  ``(global_idx - generation_start_index) // batch_size``,
                  never by row position, since ``common`` may have holes).
                  Converted to FID with the same max loss->FID slope [d] uses,
                  it must clear the same absolute floor — and because the whole
                  point of a bandit is to make per-batch choices, a per-image
                  crossover that the batch granularity cannot monetize makes
                  the verdict NOT utilizable even if checks 1-3 all pass.
                  Simulated at bs=32 on the real k8 arm table the omniscient
                  ceiling is two orders of magnitude below what [d] reports.
                  At n=500 / bs=32 there are only ~16 bandit decisions, i.e.
                  ~16 reward samples, not 500.

Layout expected (written by scripts/sweep_budget_probe.sh)::

    <probe>/reference/results.json + <probe>/reference/generated/*.png
    <probe>/k<K>/equalflops/arm_<id>/results.json + .../generated/*.png
    <probe>/k<K>/manifest.json

Analysis only; always exits 0. The RECOMMENDATION block carries the reasoning.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analyze.analyze_teacache_sweeps import _load          # noqa: E402
from scripts.analyze.analyze_budget_probe import _budget_dirs, _f  # noqa: E402
from scripts.analyze.analyze_reward_proxy import _average_ranks, _spearman  # noqa: E402

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

# Magnitude floors for the [d] gate. See _gate for the calibration behind both.
# _REL_FLOOR: the share of the oracle gain the transfer rule must recover.
# Constructed no-crossover worlds that slip past p<=0.05 report a median 0.67%
# (noise features) to 1.58% (high noise); a real crossover reports ~21%. 5%
# keeps 0.963 power on the real one and cuts the worst false-positive rate from
# 0.048 to 0.003.
_REL_FLOOR = 0.05
# Used only when the sweep produced no noise_* replicas; matches the heuristic
# in analyze_teacache_sweeps.py so the two scripts cannot disagree.
_FALLBACK_FID_FLOOR = 2.0


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


def _loss_to_fid_slopes(dist_vals: List[float],
                        fid_vals: List[float]) -> Optional[Tuple[float, float]]:
    """(min, max) plausible FID change per unit of the per-image loss.

    The gate's recovered benefit is in loss units, which are not comparable to
    anything. The arms themselves supply the only available conversion: four
    (mean loss, FID) pairs. Adjacent-pair slopes plus the OLS slope bracket it.
    The MAX is what the magnitude test uses — deliberately the most generous
    conversion, so a signal cannot be dismissed by a stingy slope.
    """
    order = np.argsort(np.asarray(dist_vals, dtype=np.float64))
    loss = np.asarray(dist_vals, dtype=np.float64)[order]
    fid = np.asarray(fid_vals, dtype=np.float64)[order]
    slopes = [(fid[i + 1] - fid[i]) / (loss[i + 1] - loss[i])
              for i in range(len(loss) - 1) if loss[i + 1] - loss[i] > 1e-12]
    if len(loss) >= 2 and np.ptp(loss) > 1e-12:
        slopes.append(float(np.polyfit(loss, fid, 1)[0]))
    slopes = [s for s in slopes if np.isfinite(s) and s > 0.0]
    if not slopes:
        return None
    return (min(slopes), max(slopes))


def _c4(n: int) -> float:
    """``E[s] / sigma`` for n iid normal draws — the Bessel-corrected sample sd
    is still biased LOW, badly so at small n (0.798 at n=2, 0.886 at n=3), so
    dividing by this is what makes ``sigma_hat`` unbiased."""
    return math.sqrt(2.0 / (n - 1)) * math.exp(
        math.lgamma(n / 2.0) - math.lgamma((n - 1) / 2.0))


# 5-95% central span of ``sigma_hat / sigma`` at n replicas. Exact: sigma_hat
# ~ sigma * chi_{n-1} / (c4 * sqrt(n-1)), so these are chi quantiles, not
# simulation estimates (cross-checked against 3e6 draws, agreeing to 3dp; the
# n=2 pair is verifiable by hand -- sigma_hat = |Z| / c4(2), so the span is
# [z_0.525, z_0.975] / 0.79788 = [0.0627, 1.9600] / 0.79788).
_SIGMA_HAT_SPAN = {
    2: (0.079, 2.456), 3: (0.256, 1.953), 4: (0.372, 1.752),
    5: (0.448, 1.638), 6: (0.503, 1.564), 7: (0.544, 1.510),
    8: (0.577, 1.469),
}


def _noise_fid_stats(budget_dir: str) -> Optional[Dict[str, object]]:
    """Same-arm replica FID noise, as an UNBIASED SD rather than a range.

    ``sweep_budget_probe.sh`` reruns the baseline arm at SEED+1/+2 into
    ``equalflops/noise_<seed>/`` precisely so arm-to-arm differences can be
    compared against resampling noise. The crossover gate needs it because a
    recovered per-image gain smaller than the floor cannot be spent.

    DELIBERATE DIVERGENCE from ``analyze_teacache_sweeps.py:168`` ``_spread``,
    which uses ``max - min``. That is not a bug there and it is not copied
    here: the expected range GROWS with the replica count
    (``E[range] = d2(n) * sigma``, d2 = 1.13 / 1.69 / 2.06 / 2.33 at
    n = 2 / 3 / 4 / 5), so a floor built from it silently TIGHTENS as replicas
    are added — adding three replicas roughly doubles the floor at unchanged
    true noise, and a gain that cleared the floor at n=2 can fail at n=5 purely
    because the estimator changed. ``sigma_hat`` is n-invariant, so the floor
    means the same thing at every replica count.

    THIS DOES MOVE THE FLOOR, in the LOOSENING direction. The two SD estimators
    ``range / d2(n)`` and ``s / c4(n)`` coincide exactly at n=2, so the choice
    between them is immaterial there — but the previous code used the raw RANGE
    as if it were an sd, and the range overstates sigma by d2(2) = 1.128. The
    real two-replica case (FID spread 2.4) therefore goes 2*2.4 = 4.80 FID ->
    2*2.127 = 4.25 FID: the floor drops by 11.4% of its old value (that is
    1 - 1/d2(2), fixed at n=2 regardless of the spread; stated the other way
    round the old floor was 12.8% higher). No observed verdict flips, but
    for a magnitude reason rather than an algebraic one: the largest gain any
    real budget recovered is 0.284 FID (k8) and 0.171 FID (k6), 15-28x below
    either floor. A future gain landing in [4.25, 4.80] would pass now and
    would have failed before.

    The estimator's own uncertainty is reported by the renderer, because at n=2
    it is enormous and a reader must not treat the floor as a precise number.
    """
    fids = []
    for path in sorted(glob.glob(os.path.join(
            budget_dir, "equalflops", "noise_*", "results.json"))):
        agg = (_load(path) or {}).get("aggregate") or {}
        fid = agg.get("fid")
        if fid is not None and not math.isnan(float(fid)):
            fids.append(float(fid))
    return _noise_stats_from_fids(fids)


def _noise_stats_from_fids(fids: List[float]) -> Optional[Dict[str, object]]:
    """The statistics half of ``_noise_fid_stats``, split out so the estimator
    is testable without a directory tree on disk."""
    if len(fids) < 2:
        return None
    n = len(fids)
    mean = sum(fids) / n
    sd = math.sqrt(sum((f - mean) ** 2 for f in fids) / (n - 1))
    c4 = _c4(n)
    lo, hi = _SIGMA_HAT_SPAN.get(n, (0.60, 1.40))
    sigma_hat = sd / c4
    return {"n": n, "fids": list(fids), "range": max(fids) - min(fids),
            "sd": sd, "c4": c4, "sigma_hat": sigma_hat,
            "rel_sd": math.sqrt(1.0 / c4 ** 2 - 1.0),
            # Inverted to a band on the TRUE sigma: sigma_hat/sigma in [lo,hi]
            # means sigma in sigma_hat/[hi,lo]. That is the decision-relevant
            # direction — how much the floor itself could be off.
            "true_lo": sigma_hat / hi, "true_hi": sigma_hat / lo,
            "span_exact": n in _SIGMA_HAT_SPAN}


def _noise_floor_fid(noise: Optional[Dict[str, object]]) -> Tuple[float, bool]:
    """(floor, measured) — ``2 * sigma_hat`` when replicas exist, else the
    ``2 * _FALLBACK_FID_FLOOR`` heuristic. One place so gate and renderer can
    never disagree about which floor was applied."""
    if noise is None:
        return 2.0 * _FALLBACK_FID_FLOOR, False
    return 2.0 * float(noise["sigma_hat"]), True


def _batch_oracle_gain(D: np.ndarray, idxs: List[int],
                       batch_size: int, start_index: int
                       ) -> Optional[Dict[str, object]]:
    """Per-batch oracle gain: the ceiling the REAL decision granularity leaves.

    The bandit decides once per batch, not once per image (``run_dit.py``:
    ``trajectory_id = covr_trajectory_offset + batch_start // bs``, one
    ``apply_strategy`` per trajectory). ``[d]``'s per-image oracle gain is
    therefore not what a per-trajectory bandit could capture; this computes the
    per-batch analogue. Rows are grouped by global_idx
    (``(idx - start_index) // batch_size``), never by row position, because
    ``common`` is a set intersection that can have holes — a hole must not
    shift later rows into the wrong batch.

    Per-batch oracle gain = (mean over ALL rows of the globally best single
    arm's column) - (row-count-weighted mean over batches of each batch's own
    best-mean-cost arm). Returns None when grouping is impossible or
    degenerate (<= 1 batch, or a single batch where the gain is 0 by
    construction).
    """
    n = D.shape[0]
    best_single = float(D.mean(axis=0).min())
    batches: List[Tuple[List[int], float]] = []
    by_batch: Dict[int, List[int]] = {}
    for i, k in enumerate(idxs):
        b = (int(k) - start_index) // batch_size
        by_batch.setdefault(b, []).append(i)
    for b in sorted(by_batch):
        rows = by_batch[b]
        batches.append((rows, float(D[rows].mean(axis=0).min())))
    if len(batches) < 2:
        return None
    gain = best_single - sum(len(r) * g for r, g in batches) / n
    sizes = [len(r) for r, _ in batches]
    return {"batches": len(batches), "gain": gain,
            "best_single": best_single, "per_batch": batches,
            "batch_sizes": sizes,
            "min_batch": min(sizes), "max_batch": max(sizes)}


def _decision_batch_size(results: Dict[str, dict], flag_batch: Optional[int]
                         ) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Recover the bandit's real decision granularity from the arm configs.

    ``run_dit.py`` writes ``batch_size`` and ``generation_start_index`` into
    every results.json ``config``. Both must agree across ALL arms — on
    disagreement [e] is skipped with a message naming the disagreement rather
    than guessing. When the key is absent entirely, ``flag_batch`` (the
    ``--decision-batch-size`` fallback) supplies the batch size and the result
    is flagged ``assumed_bs`` so the renderer labels it as assumed; an absent
    ``generation_start_index`` falls back to 0 and is flagged ``assumed_start``.
    Returns (info, None) or (None, skip_reason).
    """
    bs_vals, si_vals = [], []
    for res in results.values():
        cfg = res.get("config") or {}
        if isinstance(cfg.get("batch_size"), int):
            bs_vals.append(int(cfg["batch_size"]))
        if isinstance(cfg.get("generation_start_index"), int):
            si_vals.append(int(cfg["generation_start_index"]))
    if len(set(bs_vals)) > 1:
        return None, (f"arms disagree on batch_size ({sorted(set(bs_vals))}) "
                      f"— refusing to guess")
    if len(set(si_vals)) > 1:
        return None, (f"arms disagree on generation_start_index "
                      f"({sorted(set(si_vals))}) — refusing to guess")
    assumed_bs = not bs_vals
    if bs_vals:
        bs = bs_vals[0]
    elif flag_batch is not None:
        bs = int(flag_batch)
    else:
        return None, ("no batch_size recorded in arm configs and no "
                      "--decision-batch-size fallback given")
    return {"batch_size": bs, "generation_start_index": si_vals[0] if si_vals else 0,
            "assumed_bs": assumed_bs, "assumed_start": not si_vals}, None


def _gate(row: Dict[str, object]) -> Dict[str, object]:
    """The three-part verdict gate, computed once so renderer and main agree.

    ``p <= 0.05`` alone is not enough, and the reason is structural rather than
    a matter of taste. Permuting the train rows destroys the feature->cost
    relation, so every arm's least squares collapses to its own mean, the argmin
    becomes the globally best arm, and the permuted benefit is EXACTLY zero.
    Measured mass of the null at |benefit| <= 1e-12: 0.56-0.60. With most of the
    null pinned at zero, "p <= 0.05" is very nearly the same statement as
    "benefit > 0" and carries almost no information about effect size.

    So the gate also requires magnitude, calibrated two ways (300-400 trials,
    n=500). Share of the oracle gain the rule recovers when it passes at p<=0.05:

      world                        pass rate   median frac when passing
      no crossover, noise feats        0.007            0.67%
      no crossover, high noise         0.048            1.58%
      weak crossover (0.2)             0.102            1.31%
      REAL crossover (0.8)             0.990           20.90%

    A real crossover shows up ~20x larger than the false positives, so a 5%
    relative floor separates them: it costs 0.963 power on the real interaction
    while cutting the high-noise false-positive rate from 0.048 to 0.003.

    The absolute floor is the physical one: convert the recovered benefit to FID
    with ``_loss_to_fid_slopes`` (max slope, most generous) and require it to
    clear ``2 * sigma_hat`` of the same-arm replica FIDs — the standing
    2x-noise rule this project applies to every other arm-spread claim, with the
    noise measured as an UNBIASED SD instead of a range so the floor does not
    tighten as replicas are added (see ``_noise_fid_stats``).
    """
    oos = row.get("oos") or {}
    frac = oos.get("reg_frac")
    out: Dict[str, object] = {
        "frac_ok": None, "p_ok": None, "rel_ok": None, "abs_ok": None,
        "gain_fid_lo": None, "gain_fid_hi": None, "floor_fid": None,
        "pass": False, "reason": None,
    }
    if oos.get("skip") or frac is None or (isinstance(frac, float)
                                           and math.isnan(frac)):
        out["reason"] = "no out-of-sample result"
        return out
    out["frac_ok"] = bool(frac > 0.0)
    out["p_ok"] = bool(oos.get("reg_p", 1.0) <= 0.05)
    out["rel_ok"] = bool(frac > _REL_FLOOR)

    slopes = row.get("fid_per_loss")
    noise = row.get("noise_fid")
    floor, measured = _noise_floor_fid(noise)
    out["floor_fid"] = floor
    out["floor_measured"] = measured
    out["floor_replicas"] = int(noise["n"]) if noise else 0
    if slopes is not None:
        benefit = float(oos.get("reg_benefit", 0.0))
        out["gain_fid_lo"] = benefit * slopes[0]
        out["gain_fid_hi"] = benefit * slopes[1]
        out["abs_ok"] = bool(out["gain_fid_hi"] > floor)
    # --- [e] veto: the bandit decides per BATCH, so a per-image gain the
    # batch granularity cannot monetize is NOT utilizable even if all three
    # [d] checks pass. The per-batch oracle is the omniscient ceiling for a
    # per-trajectory bandit; if even that sits below the absolute floor, the
    # crossover is not capturable at this granularity. A gain of 0 or less is
    # the strongest possible veto — the batch decision recovers nothing. ---
    out["granularity_ok"] = None
    if (not row.get("granularity_skip")
            and row.get("granularity_bs") is not None
            and row.get("granularity") is not None
            and slopes is not None):
        gain_batch = float(row["granularity"]["gain"])
        fid_batch = gain_batch * slopes[1]
        out["granularity_ok"] = bool(fid_batch > floor)
        out["granularity_fid_hi"] = fid_batch
        out["granularity_bs"] = row["granularity_bs"]["batch_size"]
    checks = [out["frac_ok"], out["p_ok"], out["rel_ok"]]
    if out["abs_ok"] is not None:
        checks.append(out["abs_ok"])
    out["pass"] = all(bool(c) for c in checks)
    if out["granularity_ok"] is False:
        out["pass"] = False
        out["reason"] = (f"per-image crossover not capturable at the bandit's "
                         f"bs={out.get('granularity_bs')}: the per-batch oracle "
                         f"gain is {out.get('granularity_fid_hi', 0.0):.3f} FID, "
                         f"below the same {floor:.2f} FID absolute floor — the "
                         f"bandit decides once per batch, not per image")
    elif not out["pass"]:
        failed = []
        if not out["frac_ok"]:
            failed.append("recovers no positive share")
        elif not out["p_ok"]:
            failed.append("not significant against the permutation null")
        if out["frac_ok"] and not out["rel_ok"]:
            failed.append(f"share {frac * 100:.2f}% is inside the "
                          f"false-positive band (floor {_REL_FLOOR * 100:.0f}%)")
        if out["abs_ok"] is False:
            failed.append(f"gain <= {out['gain_fid_hi']:.3f} FID is below "
                          f"2x the same-arm noise sd ({floor:.2f} FID)")
        out["reason"] = "; ".join(failed) or "magnitude below the floor"
    return out


# --------------------------------------------------------------------------
# Per-budget analysis
# --------------------------------------------------------------------------

def _acquire_budget_payload(budget_dir: str,
                            ref_imgs: Dict[int, np.ndarray],
                            metric: str, device: str) -> Dict[str, object]:
    """Acquisition: everything that touches PNGs, results.json, or the metric
    extractor. Produces a plain payload that ``_analyze_from_payload`` reduces
    to a row.

    Splitting acquisition from statistics is the correctness mechanism behind
    --dump/--from-dump: both the live path and the offline path feed the SAME
    payload through the SAME statistics code, so byte-identical output is
    structural, not a coincidence. Nothing here reads ``args`` — the gate
    parameters all live in the statistics half, which is what keeps
    ``--max-perm``/``--oos-perm``/``--perm-seed``/``--floor-ratio``/
    ``--decision-batch-size`` live under --from-dump.
    """
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

    payload: Dict[str, object] = {
        "budget": budget,
        "num_steps": num_steps,
        "labels": labels,
        "metric": metric,
        "metric_desc": _metric_desc(metric, device),
        # the raw results dicts, not the resolved decision info:
        # _decision_batch_size must run in the STATISTICS half so
        # --decision-batch-size still takes effect offline (results.json is
        # not available under --from-dump).
        "arm_results": {label: results[label] for label in labels},
        "skip": None,
    }
    if not labels:
        payload["skip"] = "no forced-arm runs"
        return payload

    common = set(ref_imgs)
    for label in labels:
        common &= set(arm_imgs[label])
    common = sorted(common)
    if len(common) < 2:
        payload["skip"] = (f"only {len(common)} images shared with reference "
                           "(need >=2)")
        return payload

    ref_al = np.stack([ref_imgs[k] for k in common])
    n = len(common)
    n_arm = len(labels)
    payload["common"] = list(common)

    # Per-arm uint8 stacks, aligned to `common`, shared by every metric below.
    arm_stacks = [np.stack([arm_imgs[label][k] for k in common])
                  for label in labels]

    # loss matrix in the primary metric's units (smaller = better everywhere)
    payload["pixel_mse_crosscheck"] = None
    if metric in _BATCHED:
        try:
            D = _distance_matrix(metric, device, ref_al, arm_stacks)
        except Exception as exc:
            payload["skip"] = (f"{metric} extractor failed at analysis time "
                               f"({type(exc).__name__}: {exc})")
            return payload
        if not np.isfinite(D).all():
            payload["skip"] = f"{metric} produced non-finite values"
            return payload
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
            payload["pixel_mse_crosscheck"] = abs(chk - D[0, 0])
        except Exception:
            payload["pixel_mse_crosscheck"] = None
    else:
        fn, _ = _make_distance(metric, device)
        D = np.empty((n, n_arm))
        for j, label in enumerate(labels):
            for i, k in enumerate(common):
                D[i, j] = fn(ref_imgs[k], arm_imgs[label][k])
        if np.isnan(D).any():
            payload["skip"] = "LPIPS produced NaN (model load failed)"
            return payload
    payload["D"] = D

    # [b]'s quantization floor is ALWAYS in pixel-MSE units: it asks whether
    # the arm IMAGES differ by more than 8-bit rounding, which is a property
    # of the PNGs and not of whatever metric ranks them. For a reference-free
    # metric it therefore answers "do the images differ above rounding", not
    # "does the score differ above rounding" — stated explicitly in the
    # rendering. Computed here (needs the images); compared against _FLOOR in
    # the statistics half.
    if metric == "pixel_mse":
        D_pmse = D
    else:
        D_pmse = np.empty((n, n_arm))
        for j, arm_al in enumerate(arm_stacks):
            D_pmse[:, j] = _pixel_mse_vectorized(ref_al, arm_al)
    payload["D_pmse"] = D_pmse

    payload["fids"] = {
        label: (results[label].get("aggregate") or {}).get("fid")
        for label in labels}
    payload["is_means"] = {
        label: (results[label].get("aggregate") or {}).get("is_mean")
        for label in labels}

    # direct arm-vs-arm pixel MSE (6 pairs for 4 arms)
    pair_mses = []
    for a in range(n_arm):
        for b in range(a + 1, n_arm):
            pair_mses.append(float(_pixel_mse_vectorized(
                arm_stacks[a], arm_stacks[b]).mean()))
    payload["pair_mses"] = pair_mses

    # exogenous features from the REFERENCE image only (see _out_of_sample)
    payload["feat"] = _image_features(ref_imgs, common)
    # same-arm replica FID noise; reads noise_* results.json off disk
    payload["noise_fid"] = _noise_fid_stats(budget_dir)
    return payload


def _analyze_from_payload(payload: Dict[str, object], args
                          ) -> Dict[str, object]:
    """PURE statistics on an acquisition payload: sections [a] [b] [c] [d] [e]
    and every ``row[...]`` key the renderers read. No PNGs, no results.json,
    no metric extractor.

    The rng construction and the null call order live HERE, which is what
    makes the same ``--perm-seed`` produce the same nulls on the live path and
    under --from-dump. A skipped budget is carried as payload["skip"] and
    returns a minimal row, so the dump round-trips SKIPPED lines instead of
    dropping the budget.
    """
    row: Dict[str, object] = {
        "budget": payload["budget"],
        "num_steps": payload["num_steps"],
        "labels": payload["labels"],
        "valid": None,
        "skip": payload.get("skip"),
    }
    if payload.get("skip"):
        return row

    metric = str(payload["metric"])
    labels = row["labels"]
    D = np.asarray(payload["D"])
    D_pmse = np.asarray(payload["D_pmse"])
    common = list(payload["common"])
    n = len(common)
    n_arm = len(labels)

    row["metric"] = metric
    row["reference_free"] = metric in _REFERENCE_FREE
    row["metric_desc"] = payload["metric_desc"]
    row["pixel_mse_crosscheck"] = payload.get("pixel_mse_crosscheck")
    fids = payload["fids"]
    is_means = payload["is_means"]
    row["fids"] = fids
    row["is_means"] = is_means
    row["n_images"] = n
    row["mean_dist"] = {label: float(D[:, j].mean())
                        for j, label in enumerate(labels)}
    row["argmin_counts"] = {label: int((D.argmin(axis=1) == j).sum())
                            for j, label in enumerate(labels)}

    # --- (a) validity: mean per-image loss must rank arms like FID ---
    dist_vals = [float(D[:, j].mean()) for j in range(n_arm)]
    fid_vals = [fids[label] for label in labels]
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in fid_vals):
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

    # --- (c) crossover ---
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
    row["oos"] = _out_of_sample(D, common, np.asarray(payload["feat"]),
                                args.oos_perm, rng)
    # --- [e] granularity: the bandit decides per BATCH, not per image ---
    bs_info, bs_reason = _decision_batch_size(
        payload["arm_results"], getattr(args, "decision_batch_size", None))
    row["granularity_skip"] = bs_reason
    row["granularity_bs"] = bs_info
    if bs_info is not None:
        row["granularity"] = _batch_oracle_gain(
            D, common, bs_info["batch_size"], bs_info["generation_start_index"])
    else:
        row["granularity"] = None
    # The gate needs the recovered benefit in FID units, and the FID scale that
    # the per-image loss maps onto is only knowable from the arms themselves.
    row["fid_per_loss"] = _loss_to_fid_slopes(
        dist_vals, [float(v) for v in fid_vals])  # type: ignore[arg-type]
    row["noise_fid"] = payload.get("noise_fid")

    # --- (b) quantization floor (always in pixel-MSE units) ---
    spread = D_pmse.max(axis=1) - D_pmse.min(axis=1)
    mean_spread = float(spread.mean())
    pair_mses = [float(v) for v in np.atleast_1d(payload.get("pair_mses"))
                 ] if payload.get("pair_mses") is not None else []
    row["floor_ratio"] = mean_spread / _FLOOR
    row["mean_spread_pmse"] = mean_spread
    row["mean_arm_arm_pmse"] = (float(np.mean(pair_mses)) if pair_mses
                                else None)
    row["near_floor_frac"] = float((spread <= 5.0 * _FLOOR).mean())
    return row


def _analyze_budget(budget_dir: str, ref_imgs: Dict[int, np.ndarray],
                    metric: str, device: str, args
                    ) -> Dict[str, object]:
    """Acquire then analyze in one call — the live path. Kept for the tests
    that build synthetic probe trees and call it directly."""
    return _analyze_from_payload(
        _acquire_budget_payload(budget_dir, ref_imgs, metric, device), args)


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
        out.extend(_render_gate(row))

    # --- [e] granularity: what the real decision granularity can capture ---
    out.extend(_render_granularity(row))
    return out


def _render_gate(row: Dict[str, object]) -> List[str]:
    """Print the gate's three checks and what each one is protecting against."""
    g = _gate(row)
    oos = row["oos"]  # type: ignore[index]
    frac = float(oos["reg_frac"])  # type: ignore[index]
    out: List[str] = []
    out.append("    gate check 1/3 SIGN+SIGNIFICANCE: "
               f"frac={frac * 100:.2f}% p={oos['reg_p']:.4g} -> "  # type: ignore[index]
               + ("pass" if g["frac_ok"] and g["p_ok"] else "FAIL"))
    out.append("      p alone is weak evidence here BY CONSTRUCTION: permuting "
               "the train rows destroys feature->cost, every arm's fit collapses "
               "to its own mean, the argmin becomes the globally best arm and the "
               "permuted benefit is EXACTLY 0. Measured null mass at |benefit| "
               "<= 1e-12: 0.56-0.60. So p<=0.05 is nearly the same statement as "
               "frac>0 and says almost nothing about size. Hence checks 2 and 3.")
    out.append(f"    gate check 2/3 RELATIVE SIZE: frac={frac * 100:.2f}% vs "
               f"floor {_REL_FLOOR * 100:.0f}% of the oracle gain -> "
               + ("pass" if g["rel_ok"] else "FAIL"))
    out.append("      calibration (300-400 trials, n=500): constructed worlds "
               "with NO crossover that still reach p<=0.05 report a median "
               "0.67% (noise features) to 1.58% (high noise) recovered share; a "
               "REAL crossover reports ~21%, about 20x larger. The 5% floor "
               "keeps 0.963 power on the real interaction while cutting the "
               "high-noise false-positive rate from 0.048 to 0.003.")
    if g["abs_ok"] is None:
        out.append("    gate check 3/3 ABSOLUTE SIZE: not evaluable (no usable "
                   "loss->FID slope from the arm table)")
    else:
        slopes = row["fid_per_loss"]  # type: ignore[index]
        out.append(f"    gate check 3/3 ABSOLUTE SIZE: recovered gain "
                   f"{g['gain_fid_lo']:.3f}..{g['gain_fid_hi']:.3f} FID "
                   f"(loss->FID slope {slopes[0]:.1f}..{slopes[1]:.1f} from this "  # type: ignore[index]
                   f"budget's own arm table) vs floor {g['floor_fid']:.2f} FID -> "
                   + ("pass" if g["abs_ok"] else "FAIL"))
        out.append("      the floor is 2x the "
                   + ("MEASURED same-arm replica FID noise sd"
                      if g.get("floor_measured")
                      else f"heuristic {_FALLBACK_FID_FLOOR:.1f} FID sd (no "
                           "noise_* replicas found for this budget)")
                   + " — the same 2x-noise-floor rule this project applies to "
                     "every other arm-spread claim, applied to the per-image "
                     "gain. A gain smaller than resampling noise cannot be spent "
                     "even if it is real.")
        out.extend(_render_noise_estimator(row))
    if g["pass"]:
        out.append("    => VERDICT GATE: crossover UTILIZABLE out-of-sample "
                   "(all three checks pass)")
    else:
        out.append(f"    => VERDICT GATE: crossover NOT utilizable — "
                   f"{g['reason']}")
    return out


def _render_noise_estimator(row: Dict[str, object]) -> List[str]:
    """Print how the floor was estimated, from how many replicas, and how
    uncertain that estimate is.

    A floor is a threshold a verdict turns on, so a reader must be able to see
    that at 2 replicas the floor is barely an estimate at all: sigma_hat has a
    75.6% relative sd and a 5-95% span of [0.08 sigma, 2.46 sigma]. Inverted,
    the real 2.4-FID-spread run's floor of 4.25 FID carries a 5-95% band of
    roughly [1.7, 54] FID. Printing the number without that band invites
    treating 4.25 as precise.
    """
    noise = row.get("noise_fid")
    if not noise:
        return ["      floor source: NO same-arm replicas for this budget — "
                f"using the heuristic {_FALLBACK_FID_FLOOR:.1f} FID sd. Run "
                "noise_* replicas (sweep_budget_probe.sh already does at "
                "SEED+1/+2) to measure it."]
    n = int(noise["n"])
    sig = float(noise["sigma_hat"])
    out = [f"      floor source: {n} same-arm replica FIDs "
           f"[{', '.join(f'{f:.2f}' for f in noise['fids'])}] -> "  # type: ignore[union-attr]
           f"sd {float(noise['sd']):.3f} / c4({n})={float(noise['c4']):.4f} = "
           f"sigma_hat {sig:.3f} FID, floor = 2 x sigma_hat = {2 * sig:.2f} FID"]
    out.append(f"      estimator uncertainty at n={n}: sigma_hat has "
               f"{float(noise['rel_sd']) * 100:.1f}% relative sd; the true "
               f"noise sd is plausibly {float(noise['true_lo']):.2f}.."
               f"{float(noise['true_hi']):.2f} FID (5-95%"
               + ("" if noise.get("span_exact") else ", span EXTRAPOLATED "
                  "beyond the tabulated n")
               + f"), i.e. the floor itself is {2 * float(noise['true_lo']):.2f}"
                 f"..{2 * float(noise['true_hi']):.2f} FID. Treat a verdict "
                 "that only just clears it as undecided, and add replicas.")
    if n == 2:
        out.append("      at n=2 the two SD estimators coincide (range/d2(2) == "
                   "s/c4(2) == |f1-f2|/1.12838); what changed is that the "
                   "floor no longer uses the raw RANGE as an sd, which "
                   "overstated it by 1.128x. This budget's floor is therefore "
                   f"{2 * sig:.2f} rather than "
                   f"{2 * float(noise['range']):.2f} FID — LOOSER, not "
                   "tighter: 11.4% lower than the old floor (exactly "
                   "1 - 1/d2(2) at any n=2, so the ratio is fixed).")
    else:
        out.append(f"      NOTE: the range would give {float(noise['range']):.3f}"
                   f" FID here (floor {2 * float(noise['range']):.2f}); sigma_hat "
                   f"is used instead because E[range] grows with n, so a "
                   f"range-based floor tightens as replicas are added while the "
                   f"true noise is unchanged. analyze_teacache_sweeps.py:168 "
                   f"still uses the range — a deliberate divergence, not drift.")
    return out


def _render_granularity(row: Dict[str, object]) -> List[str]:
    """Print [e]: the per-batch ceiling at the bandit's REAL granularity."""
    out: List[str] = []
    if row.get("granularity_skip"):
        out.append("  [e] granularity: SKIPPED — "
                   f"{row['granularity_skip']}")
        return out
    info = row.get("granularity_bs") or {}
    bs = info.get("batch_size")
    start = info.get("generation_start_index")
    if bs is None:
        out.append("  [e] granularity: SKIPPED — no decision granularity "
                   "recoverable")
        return out
    assumed = []
    if info.get("assumed_bs"):
        assumed.append("batch_size ASSUMED from --decision-batch-size "
                       "(not recorded in arm configs)")
    if info.get("assumed_start"):
        assumed.append("generation_start_index ASSUMED 0 (not recorded in arm "
                       "configs)")
    out.append(f"  [e] granularity — per-BATCH oracle at bs={bs} "
               f"(bandit decides once per batch, not per image; "
               f"generation_start_index={start})")
    for note in assumed:
        out.append(f"    [ASSUMED] {note}")
    g = row.get("granularity")
    if g is None:
        out.append("    per-batch oracle gain: n/a (fewer than 2 batches in "
                   f"the shared set; {row.get('n_images')} shared images at "
                   f"bs={bs} yield ~{row.get('n_images', 0) // bs} bandit "
                   "decisions)")
        return out
    sizes = g["batch_sizes"]
    size_s = f"{min(sizes)}..{max(sizes)}"
    if min(sizes) != max(sizes):
        size_s += f" (rows/batch vary: {','.join(str(s) for s in sizes[:8])}"
        if len(sizes) > 8:
            size_s += ",..."
        size_s += ")"
    out.append(f"    batches: {g['batches']} (rows per batch: {size_s})")
    gain_bs1 = float(row.get("benefit", float("nan")))
    ratio_s = ""
    if g["gain"] > 1e-12 and gain_bs1 == gain_bs1 and gain_bs1 > 0:
        ratio_s = f" (ratio to bs=1: {gain_bs1 / g['gain']:.1f}x)"
    out.append(f"    per-batch oracle gain (loss units): {g['gain']:.4g} "
               f"vs bs=1 per-image gain {gain_bs1:.4g}{ratio_s} — "
               f"at bs=1 the two are identical; the gap is the headroom the "
               f"batch decision cannot spend")
    slopes = row.get("fid_per_loss")
    floor, measured = _noise_floor_fid(row.get("noise_fid"))
    if slopes is not None:
        fid_hi = g["gain"] * slopes[1]
        out.append(f"    per-batch gain in FID (max slope "
                   f"{slopes[0]:.1f}..{slopes[1]:.1f}): {fid_hi:.3f} FID vs "
                   f"the {'MEASURED' if measured else 'heuristic'} "
                   f"floor {floor:.2f} FID -> "
                   + ("ABOVE" if fid_hi > floor else "BELOW")
                   + " — a per-image crossover below the floor cannot be "
                     "captured at this granularity")
    else:
        out.append("    per-batch gain in FID: not evaluable (no loss->FID "
                   "slope from the arm table)")
    out.append("    NOTE: n=500 at bs=32 yields ~16 bandit decisions, i.e. "
               "~16 reward samples — not 500. Per-batch oracle is the "
               "OMNISCIENT ceiling; a real bandit gets strictly less.")
    return out


# --------------------------------------------------------------------------
# Dump / offline re-analysis (--dump / --from-dump)
# --------------------------------------------------------------------------

_DUMP_KEYS = ("D", "D_pmse", "common", "feat", "pair_mses")
_JSON_KEYS = ("budget", "num_steps", "labels", "metric", "metric_desc",
              "arm_results", "skip", "pixel_mse_crosscheck", "fids",
              "is_means", "noise_fid")


def _dump_payload(payload: Dict[str, object], dump_dir: str) -> None:
    """Persist one budget's payload as DIR/k<K>.npz + k<K>.json.

    np.savez_compressed for the arrays, JSON for the scalars/dicts/strings.
    Images are never dumped — the payload holds only derived arrays, which is
    exactly what makes --from-dump GPU-free.
    """
    os.makedirs(dump_dir, exist_ok=True)
    tag = f"k{payload['budget']}"
    arrays = {}
    for key in _DUMP_KEYS:
        if key in payload:
            arrays[key] = np.asarray(payload[key])
    np.savez_compressed(os.path.join(dump_dir, tag + ".npz"), **arrays)
    scalars = {key: payload[key] for key in _JSON_KEYS if key in payload}
    with open(os.path.join(dump_dir, tag + ".json"), "w",
              encoding="utf-8") as fh:
        json.dump(scalars, fh, indent=2, sort_keys=True)


def _load_payload(dump_dir: str, budget: int) -> Optional[Dict[str, object]]:
    """Load one budget's payload back from DIR/k<K>.npz + k<K>.json; None when
    the pair is missing or corrupt. The loader never touches PNGs."""
    tag = f"k{budget}"
    npz_path = os.path.join(dump_dir, tag + ".npz")
    json_path = os.path.join(dump_dir, tag + ".json")
    if not (os.path.isfile(npz_path) and os.path.isfile(json_path)):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        with np.load(npz_path, allow_pickle=False) as z:
            for key in _DUMP_KEYS:
                if key in z:
                    payload[key] = z[key]
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None
    if not isinstance(payload, dict) or "budget" not in payload:
        return None
    return payload


def _dump_all(probe: str, dump_dir: str, ref_fid: Optional[float],
              ref_config: dict, metric: str, budgets: List[Tuple[int, str]],
              payloads: List[Dict[str, object]],
              ref_dir: str = "", reference_missing: bool = False) -> None:
    """Write per-budget k<K>.npz/.json plus manifest.json.

    The manifest freezes the metric, the reference FID and config, the source
    probe_dir, the budget list, the analyzing script's argv, and an ISO
    timestamp. --from-dump uses it to reject a dump whose metric disagrees
    with an explicit --metric, and to reproduce the reference-FID lines and
    the budget ordering without the probe tree.

    ``ref_dir`` / ``reference_missing`` are recorded because the live path
    prints a REFERENCE MISSING block naming that directory and the
    RECOMMENDATION branches on it. Without them a dump taken from a probe with
    no reference would re-analyze to different text than the run that produced
    it, which is the one thing --from-dump promises not to do.
    """
    os.makedirs(dump_dir, exist_ok=True)
    for payload in payloads:
        _dump_payload(payload, dump_dir)
    manifest = {
        "metric": metric,
        "metric_desc": str(payloads[0]["metric_desc"]) if payloads else None,
        "reference_free": metric in _REFERENCE_FREE,
        "reference_fid": ref_fid,
        "reference_config": ref_config,
        "reference_dir": ref_dir,
        "reference_missing": bool(reference_missing),
        "probe_dir": probe,
        "budgets": [int(b) for b, _ in budgets],
        "argv": list(sys.argv),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(dump_dir, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"  dump: {os.path.abspath(dump_dir)}/ "
          f"(k<K>.npz + k<K>.json + manifest.json; no images)")


def _load_all(dump_dir: str, probe: str, metric_choice: str, args
              ) -> Tuple[Optional[dict], List[Tuple[int, str]],
                         List[Dict[str, object]]]:
    """Load a dump; (manifest, budgets, payloads). Returns (None, [], []) on
    refusal so main() can return 1 without emitting a partial verdict."""
    manifest = _load(os.path.join(dump_dir, "manifest.json"))
    if not manifest:
        print(f"  ERROR: {os.path.join(dump_dir, 'manifest.json')} is missing "
              f"or unreadable — nothing to re-analyze.")
        return None, [], []
    if not os.path.isdir(probe):
        print(f"  [WARN] source probe dir {probe} is not accessible; the "
              f"reference-FID block and binding block are printed from the "
              f"manifest instead.")
    dump_metric = str(manifest.get("metric") or "pixel_mse")
    if metric_choice not in (None, "auto", dump_metric):
        print(f"  ERROR: --metric {metric_choice} disagrees with the dump's "
              f"frozen metric {dump_metric} (manifest.json). Re-dump with "
              f"--metric {metric_choice} or drop --metric to adopt the "
              f"dump's metric.")
        return None, [], []
    budgets = [(int(b), os.path.join(dump_dir, f"k{b}"))
               for b in (manifest.get("budgets") or [])]
    payloads = []
    for b, _ in budgets:
        payload = _load_payload(dump_dir, b)
        if payload is None:
            print(f"  ERROR: dump file pair k{b}.npz/k{b}.json is missing or "
                  f"corrupt — cannot reproduce a verdict for every budget.")
            return None, [], []
        if str(payload.get("metric") or "") != dump_metric:
            print(f"  ERROR: k{b} payload metric {payload.get('metric')!r} "
                  f"disagrees with manifest metric {dump_metric!r} — the dump "
                  f"is inconsistent; re-dump it.")
            return None, [], []
        payloads.append(payload)
    return manifest, budgets, payloads


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
    ap.add_argument("--decision-batch-size", type=int, default=None,
                    help="fallback batch size for [e] granularity when the arm "
                         "results.json configs do not record batch_size "
                         "(results are then labelled as assumed)")
    ap.add_argument("--dump", metavar="DIR", default=None,
                    help="persist every budget's acquisition payload "
                         "(k<K>.npz + k<K>.json + manifest.json) to DIR so the "
                         "verdict can be reproduced without the probe tree. "
                         "Mutually exclusive with --from-dump.")
    ap.add_argument("--from-dump", metavar="DIR", default=None,
                    help="reproduce the verdict OFFLINE from a --dump DIR: no "
                         "PNG loading, no Inception, no results.json. The "
                         "metric is frozen by the dump (an explicit --metric "
                         "disagreeing with it is an error); the gate "
                         "parameters --max-perm --oos-perm --perm-seed "
                         "--floor-ratio --decision-batch-size stay live.")
    args = ap.parse_args()

    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    probe = args.probe_dir

    if args.dump and args.from_dump:
        print("  ERROR: --dump and --from-dump are mutually exclusive — "
              "produce a dump OR re-analyze one, not both.")
        return 1

    print("== COVR per-image crossover verdict ==")
    print(f"  probe dir: {probe}")

    # --- offline path: no PNGs, no Inception, no results.json ---
    if args.from_dump:
        if args.metric != "auto":
            metric_choice = args.metric
        else:
            metric_choice = None
        manifest, budgets, payloads = _load_all(args.from_dump, probe,
                                                metric_choice, args)
        if manifest is None:
            return 1
        metric = str(manifest["metric"])
        metric_desc = str(manifest.get("metric_desc")
                          or _metric_desc(metric, args.device))
        ref_fid = manifest.get("reference_fid")
        ref_config = manifest.get("reference_config") or {}
        print(f"  offline re-analysis of dump: {args.from_dump}")
        print(f"  source probe (per manifest): {manifest.get('probe_dir')}")
        if manifest.get("reference_missing"):
            print("  REFERENCE MISSING: no generated PNGs under "
                  f"{manifest.get('reference_dir')}/generated/. Re-run the "
                  "sweep with REFERENCE=1 (or point REFERENCE_DIR at an "
                  "existing full-compute reference). Per-image crossover "
                  "cannot be established without it.")
        print(f"  metric: {metric_desc}  (frozen by the dump; an explicit "
              f"--metric that disagrees is refused)")
        # Both of the next two blocks exist on the live path too. They are
        # repeated here rather than skipped because --from-dump promises
        # byte-identical verdict text: omitting them would silently diverge for
        # every reference-free dump, and for every dump taken from a probe whose
        # reference was missing (where the RECOMMENDATION branch differs).
        if metric in _REFERENCE_FREE:
            print("  note: reference-free applies to the LOSS only. The "
                  "reference run is still required — [b]'s quantization floor "
                  "and [d]'s exogenous 1NN features both come from the "
                  "reference image, and arms are still paired to it by "
                  "global_idx.")
        if ref_fid is not None:
            print(f"  reference FID: {_fmt(ref_fid, '.2f')}"
                  f"  (config: seed={ref_config.get('seed')} "
                  f"num_steps={ref_config.get('num_steps')} "
                  f"n_prompts={ref_config.get('n_prompts')})")
        rows: List[Dict[str, object]] = []
        for _, bdir in budgets:
            row = _analyze_from_payload(_load_payload(args.from_dump,
                                                      int(os.path.basename(bdir)[1:])),
                                        args)
            rows.append(row)
            for line in _render_budget(row, args.floor_ratio):
                print(line)
        _render_tail(rows, ref_fid, args,
                     reference_missing=bool(manifest.get("reference_missing")),
                     metric=metric)
        return 0

    ref_dir = os.path.join(probe, "reference")
    ref_fid, ref_config, ref_imgs = _load_reference(ref_dir)
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
    payloads: List[Dict[str, object]] = []
    for _, bdir in budgets:
        payload = _acquire_budget_payload(bdir, ref_imgs, metric, args.device)
        payloads.append(payload)
        row = _analyze_from_payload(payload, args)
        rows.append(row)
        for line in _render_budget(row, args.floor_ratio):
            print(line)

    if args.dump:
        _dump_all(probe, args.dump, ref_fid, ref_config, metric,
                  budgets, payloads, ref_dir=ref_dir,
                  reference_missing=not ref_imgs)

    _render_tail(rows, ref_fid, args, reference_missing=not ref_imgs,
                 metric=metric)
    return 0


def _render_tail(rows: List[Dict[str, object]],
                 ref_fid: Optional[float], args,
                 reference_missing: bool = False,
                 metric: Optional[str] = None) -> None:
    """Everything after the per-budget blocks (binding / per-class /
    recommendation). Shared by the live path and --from-dump so both print
    byte-identical tails."""
    # Resolve metric: prefer the argument, then the first non-skipped row,
    # then any row, so crashed early budgets don't hide what was measured.
    if metric is None:
        for r in rows:
            if "metric" in r:
                metric = str(r["metric"])
                break
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
    elif reference_missing and not usable:
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
        util = [r for r in clean if _gate(r)["pass"]]
        # Passed sign+significance but failed a magnitude floor. This is the
        # single most likely outcome at this N and it must NOT be reported as
        # "utilizable": the permutation null is mostly a point mass at exactly 0,
        # so p<=0.05 is nearly equivalent to frac>0. See _gate.
        marginal = [r for r in clean
                    if r not in util
                    and _gate(r)["frac_ok"] and _gate(r)["p_ok"]]
        # Significant p with a NEGATIVE frac on the label-transfer diagnostic:
        # the features do carry information about which arm wins, but label
        # transfer cannot monetize it. Worth naming, because it is not the same
        # thing as "pure noise" and a reader would otherwise lump them together.
        informative = [r for r in clean
                       if r not in util
                       and r not in marginal
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
                    "in-sample, but no budget clears the out-of-sample gate. An "
                    "in-sample gap of this size is exactly what independent "
                    "per-image measurement noise produces — taking the min over "
                    "4 noisy columns always looks better than any one column. "
                    "On this evidence the gap is NOT established as real "
                    "structure, so do not commit bandit GPU budget to it. Two "
                    "ways forward: stronger per-image features, or more images "
                    "per arm to shrink the per-image noise the oracle is "
                    "harvesting.")
        else:
            print("  NO CROSSOVER FOUND at any budget with a measurable signal: "
                  "the per-image oracle does not even beat the best single arm "
                  "in-sample. One arm is effectively best everywhere -> pick it "
                  "OFFLINE; a per-trajectory bandit has nothing to discover "
                  "here.")
        # Addenda, printed under whichever verdict fired above: a budget can be
        # marginal or label-transfer-informative independently of whether the
        # in-sample gap branch was the one that ran.
        for r in marginal:
            g = _gate(r)
            print(f"  MARGINAL, NOT UTILIZABLE at k{r['budget']}: the cost "
                  f"regression recovered "
                  f"{float(r['oos']['reg_frac']) * 100:.2f}% of the oracle "
                  f"gain at p={r['oos']['reg_p']:.4g} — positive and "
                  f"significant, but {g['reason']}. Read the p with care: "
                  "permuting the train rows makes every arm's fit collapse "
                  "to its own mean, so the permuted benefit is EXACTLY 0 for "
                  "0.56-0.60 of the null draws, and p<=0.05 is nearly the "
                  "same statement as frac>0. Constructed worlds with NO "
                  "crossover that reach p<=0.05 report a median 0.67%-1.58% "
                  "share; a real crossover reports ~21%. This share is in "
                  "the former band. DO NOT spend bandit budget on it.")
        vetoed = [r for r in clean
                  if _gate(r).get("granularity_ok") is False]
        for r in vetoed:
            print(f"  [e] VETOED at k{r['budget']}: per-image crossover is not "
                  f"capturable at the bandit's batch granularity — "
                  f"{_gate(r)['reason']}")
        if len(marginal) > 1:
            print("  AND THE TWO PROXIES DISAGREE: more than one budget is "
                  "marginal, and a per-image structure that is real should "
                  "not appear at one budget under one proxy and a different "
                  "budget under another. Run both --metric inception and "
                  "--metric inception_conf and compare WHICH budget each "
                  "one flags; if they disagree, that is direct evidence the "
                  "flag is sampling noise rather than structure.")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
