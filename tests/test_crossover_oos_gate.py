"""The out-of-sample gate in ``scripts/analyze_crossover.py``.

The gate decides whether COVR gets GPU budget, so its two failure modes both
need pinning down: it must not declare a crossover where none exists, and it
must not throw away one that does. The rule it uses now (per-arm cost
regression) replaced a 1NN label-transfer rule that failed the second way —
significant p, negative recovered share — on worlds that genuinely cross.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import analyze_crossover as ac  # noqa: E402

_ARM_LEVELS = np.array([2.072, 2.912, 3.148, 3.415])  # observed k6 mean_conf
_TILT = np.array([-1.0, -0.3, 0.3, 1.0])


def _world(rng, n, interaction, noise=0.5, informative_features=True):
    """Additive difficulty + arm offsets, optionally with a real interaction.

    ``interaction=0`` means NO per-image crossover exists: the best arm is the
    same arm for every image, so any gate pass is a false positive.
    """
    difficulty = rng.randn(n) * 0.6
    D = (_ARM_LEVELS[None, :] + difficulty[:, None]
         + rng.randn(n, 4) * noise
         + interaction * difficulty[:, None] * _TILT[None, :])
    if informative_features:
        feat = np.stack([difficulty + rng.randn(n) * 0.3,
                         difficulty * 0.7 + rng.randn(n) * 0.3,
                         rng.randn(n) * 0.3], axis=1)
    else:
        feat = rng.randn(n, 3)
    return D, feat


def _run(rng, n=400, **kw):
    D, feat = _world(rng, n, **kw)
    return ac._out_of_sample(D, list(range(n)), feat, 200, rng)


def test_cost_regression_finds_a_real_crossover_that_label_transfer_misses():
    """The reason the gate moved off 1NN label transfer.

    Same D, same features, same split: the cost regression recovers a positive
    share at p<=0.05 while transferring the neighbour's argmin label recovers a
    NEGATIVE share. Both p-values can be small — the label rule's p says the
    features predict the winner, its frac says acting on that prediction by
    committing to one arm costs more than the whole oracle gain.
    """
    out = _run(np.random.RandomState(3), interaction=0.8)
    assert out["skip"] is None
    assert out["reg_frac"] > 0.0
    assert out["reg_p"] <= 0.05
    assert out["nn_frac"] < 0.0        # the former gate would say "no crossover"
    assert out["reg_frac"] > out["nn_frac"]


def test_no_crossover_does_not_pass_the_gate():
    """interaction=0: one arm is best for every image, so frac must not be >0
    at p<=0.05. Checked over several seeds because a single draw says little
    about a false-positive rate."""
    for seed in range(6):
        out = _run(np.random.RandomState(seed), interaction=0.0)
        assert out["skip"] is None
        assert not (out["reg_frac"] > 0.0 and out["reg_p"] <= 0.05), seed


def test_uninformative_features_degenerate_to_the_best_single_arm():
    """With features unrelated to anything, every arm's fit is its own mean, so
    the argmin is the globally best arm and the recovered share is ~0 — not the
    large negative number label transfer produces. This is what keeps the gate
    from being noisy in the harmless direction."""
    out = _run(np.random.RandomState(11), interaction=0.0,
               informative_features=False)
    assert out["reg_frac"] == pytest.approx(0.0, abs=1e-9)
    assert out["nn_frac"] < out["reg_frac"]


def test_cost_regression_never_beats_the_oracle_it_is_measured_against():
    """A share above 1.0 would mean the transfer rule beat per-image omniscience
    on the same half, which is impossible; catching it here guards against the
    train/test halves being crossed by a future edit."""
    for seed in (0, 5, 9):
        out = _run(np.random.RandomState(seed), interaction=0.8)
        assert out["reg_frac"] <= 1.0 + 1e-9
        assert out["const_frac"] <= 1e-9  # feature-free ceiling


def test_features_with_zero_variance_are_reported_not_crashed():
    n = 40
    rng = np.random.RandomState(0)
    D = _ARM_LEVELS[None, :] + rng.randn(n, 4) * 0.5
    out = ac._out_of_sample(D, list(range(n)), np.ones((n, 3)), 10, rng)
    assert out["skip"] and "degenerate" in out["skip"]


def test_zero_oracle_benefit_is_flagged_rather_than_dividing_by_zero():
    """One arm strictly dominating every image leaves nothing to transfer; frac
    must be nan instead of raising or reporting a bogus percentage."""
    n = 40
    D = np.tile(np.array([0.1, 1.0, 2.0, 3.0]), (n, 1))
    rng = np.random.RandomState(0)
    feat = rng.randn(n, 3)
    out = ac._out_of_sample(D, list(range(n)), feat, 10, rng)
    assert out["oracle_benefit_test"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(out["reg_frac"]) and np.isnan(out["nn_frac"])


def test_renderer_labels_the_gate_and_keeps_the_diagnostic_visible(
        tmp_path, monkeypatch):
    """A reader must not mistake the diagnostic for the gate: the k6 real run
    printed p=0.009 with a negative share, and calling that "pure noise" is the
    wrong conclusion.

    The row is produced by ``_analyze_budget`` on a synthetic probe rather than
    hand-written, so the test keeps working when the row schema changes; only
    the out-of-sample result is substituted.
    """
    import json
    from PIL import Image

    rng = np.random.RandomState(3)
    n = 12
    base = (rng.rand(n, 16, 16, 3) * 255).astype(np.uint8)

    def _write(directory, images, fid, is_mean):
        gen = os.path.join(directory, "generated")
        os.makedirs(gen, exist_ok=True)
        for i, arr in enumerate(images):
            Image.fromarray(arr).save(os.path.join(gen, f"{i:06d}_cls.png"))
        with open(os.path.join(directory, "results.json"), "w") as fh:
            json.dump({"config": {"seed": 42, "num_steps": 50,
                                  "n_prompts": len(images)},
                       "aggregate": {"fid": fid, "is_mean": is_mean}}, fh)

    budget = tmp_path / "k6"
    budget.mkdir()
    (budget / "manifest.json").write_text(json.dumps({"num_steps": 50}),
                                          encoding="utf-8")
    for name, scale, fid, ism in [("back_loaded", 12, 170.0, 8.0),
                                  ("uniform", 3, 130.0, 9.0)]:
        noisy = np.clip(base.astype(np.int16) + rng.randn(n, 16, 16, 3) * scale,
                        0, 255).astype(np.uint8)
        _write(str(budget / "equalflops" / f"arm_{name}"), list(noisy), fid, ism)

    crafted = _run(np.random.RandomState(3), interaction=0.8)
    monkeypatch.setattr(ac, "_out_of_sample",
                        lambda *a, **k: dict(crafted))

    class _A:
        perm_seed = 0
        max_perm = 10
        oos_perm = 10

    row = ac._analyze_budget(str(budget), {i: base[i] for i in range(n)},
                             "pixel_mse", "cpu", _A())
    text = "\n".join(ac._render_budget(row, 5.0))
    assert "COST REGRESSION (the gate)" in text
    assert "[diagnostic, NOT the gate] 1NN label transfer" in text
    assert "VERDICT GATE: crossover UTILIZABLE" in text


# --------------------------------------------------------------------------
# The magnitude floors. p<=0.05 is not sufficient here, and the reason is
# structural: the permutation null is mostly a point mass at exactly 0.
# --------------------------------------------------------------------------

def _row(frac, p, oracle=0.5921, dist=(1.113, 1.556, 1.655, 2.502),
         fid=(96.02, 100.86, 112.74, 126.84), noise=2.4):
    """A minimal row carrying only what ``_gate`` reads.

    ``noise`` is the SPREAD between two same-arm replica FIDs — the real run's
    observable. The floor is derived from it as ``2 * sd / c4(2)``, which at
    n=2 equals ``2 * noise / 1.12838``, so the real 2-replica case (2.4 FID
    spread) gives 4.25 rather than the 4.80 a range-based floor gave.
    """
    return {"oos": {"skip": None, "reg_frac": frac, "reg_p": p,
                    "reg_benefit": frac * oracle,
                    "oracle_benefit_test": oracle},
            "fid_per_loss": ac._loss_to_fid_slopes(list(dist), list(fid)),
            "noise_fid": ac._noise_stats_from_fids([100.0, 100.0 + noise])}


def test_permutation_null_is_a_point_mass_at_zero():
    """Why p<=0.05 cannot be the whole gate.

    Shuffling the train rows destroys the feature->cost relation, so every arm's
    least squares collapses to its own mean, the argmin becomes the globally best
    arm, and the permuted benefit is EXACTLY zero. With most of the null pinned
    there, "p<=0.05" degenerates into "benefit>0" and says nothing about size.
    """
    rng = np.random.RandomState(0)
    D, feat = _world(rng, 500, interaction=0.0)
    idxs = np.arange(500)
    tr, te = idxs[idxs % 2 == 0], idxs[idxs % 2 == 1]
    F = (feat - feat.mean(axis=0)) / feat.std(axis=0)
    best = D[te].mean(axis=0).min()
    nulls = np.array([best - ac._cost_regression(
        D[tr][rng.permutation(len(tr))], F[tr], F[te], D[te])
        for _ in range(200)])
    assert (np.abs(nulls) <= 1e-12).mean() > 0.4
    assert (nulls > 1e-12).mean() == 0.0   # permutation can never help


def test_the_two_observed_passing_rows_do_not_clear_the_floors():
    """The real 500-image run: k8 under inception_conf and k6 under inception
    both reported a positive share at p<=0.05. Both must be rejected — their
    shares (0.40%, 1.00%) sit in the band that constructed NO-crossover worlds
    produce when they slip past p, ~20x below a real crossover's ~21%."""
    k8 = ac._gate(_row(0.004, 0.047))
    assert k8["frac_ok"] and k8["p_ok"]
    assert not k8["rel_ok"] and not k8["abs_ok"] and not k8["pass"]

    k6 = ac._gate(_row(0.010, 0.0375, oracle=0.01869,
                       dist=(0.1108, 0.1415, 0.1559, 0.1823),
                       fid=(117.30, 143.75, 156.91, 174.13)))
    assert k6["frac_ok"] and k6["p_ok"]
    assert not k6["rel_ok"] and not k6["abs_ok"] and not k6["pass"]
    # 0.12-0.17 FID recovered against a 4.80 FID floor
    assert k6["gain_fid_hi"] < 1.0


def test_a_real_sized_effect_still_passes_all_three_checks():
    """The floors must not close the gate outright: the ~21% share a genuine
    crossover produces clears both, at this budget's own loss->FID slope."""
    g = ac._gate(_row(0.21, 0.0))
    assert g["frac_ok"] and g["p_ok"] and g["rel_ok"] and g["abs_ok"]
    assert g["pass"] and g["reason"] is None


def test_absolute_floor_uses_the_measured_replica_spread_when_present(tmp_path):
    """``noise_*`` replicas are what the sweep generates them for. When they are
    missing the gate must say so rather than silently using the heuristic."""
    import json
    budget = tmp_path / "k8"
    (budget / "equalflops").mkdir(parents=True)
    assert ac._noise_fid_stats(str(budget)) is None
    for seed, fid in [(43, 100.0), (44, 107.5)]:
        d = budget / "equalflops" / f"noise_{seed}"
        d.mkdir()
        (d / "results.json").write_text(
            json.dumps({"aggregate": {"fid": fid}}), encoding="utf-8")
    stats = ac._noise_fid_stats(str(budget))
    assert stats is not None and stats["n"] == 2
    assert stats["range"] == pytest.approx(7.5)
    # At n=2: s = |f1-f2|/sqrt(2), so sigma_hat = s/c4(2) = |f1-f2|/1.128379
    assert stats["sigma_hat"] == pytest.approx(7.5 / 1.1283791671, rel=1e-9)

    g = ac._gate(_row(0.21, 0.0, noise=7.5))
    assert g["floor_fid"] == pytest.approx(2 * 7.5 / 1.1283791671, rel=1e-9)
    assert g["floor_measured"] and g["floor_replicas"] == 2
    g_missing = dict(_row(0.21, 0.0))
    g_missing["noise_fid"] = None
    out = ac._gate(g_missing)
    assert out["floor_fid"] == pytest.approx(4.0) and not out["floor_measured"]
    assert out["floor_replicas"] == 0


# --------------------------------------------------------------------------
# The noise-floor estimator: sd/c4(n), not max-min. Range grows with n, so a
# range-based floor tightens as replicas are added at unchanged true noise.
# --------------------------------------------------------------------------

def test_sigma_hat_equals_the_range_floor_at_two_replicas():
    """At n=2 the two SD estimators coincide — ``range/d2(2)`` and ``s/c4(2)``
    both reduce to ``|f1-f2|/1.12838`` — so the estimator CHOICE is immaterial
    there. What changes is that the floor no longer uses the raw RANGE as if it
    were an sd: the range overstates sigma by d2(2)=1.128, so the real
    two-replica case (2.4 FID spread) goes 4.80 -> 4.25 FID, LOOSER by 11.4%
    of the old floor (== 1 - 1/d2(2), fixed at n=2)."""
    stats = ac._noise_stats_from_fids([100.0, 102.4])
    # s = |d|/sqrt(2); s/c4(2) = |d|/(sqrt(2)*0.797885) = |d|/1.128379
    assert stats["sigma_hat"] == pytest.approx(2.4 / 1.1283791671, rel=1e-9)
    assert stats["sigma_hat"] == pytest.approx(stats["range"] / 1.1283791671,
                                               rel=1e-9)
    assert 2 * stats["sigma_hat"] == pytest.approx(4.2539, abs=1e-3)
    assert 2 * stats["range"] == pytest.approx(4.80, abs=1e-9)   # the old floor
    # The two ways of quoting the same move, so neither number reads as a typo.
    assert 1.0 - stats["sigma_hat"] / stats["range"] == pytest.approx(0.1138,
                                                                     abs=1e-4)
    assert stats["range"] / stats["sigma_hat"] - 1.0 == pytest.approx(0.1284,
                                                                     abs=1e-4)


def test_the_loosened_floor_flips_no_observed_verdict():
    """The floor moves 4.80 -> 4.25 FID, so it must be shown that no real
    result lands in that window. Both observed passing rows recover 0.284 FID
    (k8) and 0.171 FID (k6) — 15-28x below either floor, so the loosening is
    inert on the evidence in hand and only matters for future gains."""
    k8 = ac._gate(_row(0.004, 0.047))
    assert k8["gain_fid_hi"] == pytest.approx(0.2842, abs=1e-3)
    k6 = ac._gate(_row(0.010, 0.0375, oracle=0.01869,
                       dist=(0.1108, 0.1415, 0.1559, 0.1823),
                       fid=(117.30, 143.75, 156.91, 174.13)))
    assert k6["gain_fid_hi"] == pytest.approx(0.1708, abs=1e-3)
    for g in (k8, k6):
        assert g["gain_fid_hi"] < 4.25 and not g["abs_ok"]


def test_range_floor_would_inflate_with_replicas_but_sigma_hat_does_not():
    """The whole reason for the change. Three replica sets drawn from the SAME
    noise level, at n=2/4/8: the range grows by ~2.5x while sigma_hat stays
    put. A range-based floor would therefore tighten as the sweep adds
    replicas, and a gain that cleared it at n=2 could fail at n=8 with the true
    noise unchanged."""
    rng = np.random.RandomState(7)
    sigma = 3.0
    ranges, sigmas = {}, {}
    for n in (2, 4, 8):
        r, s = [], []
        for _ in range(4000):
            fids = list(100.0 + rng.randn(n) * sigma)
            st = ac._noise_stats_from_fids(fids)
            r.append(st["range"])
            s.append(st["sigma_hat"])
        ranges[n], sigmas[n] = float(np.mean(r)), float(np.mean(s))
    # E[range] = d2(n)*sigma: 1.128 / 2.059 / 2.847 times sigma
    assert ranges[2] == pytest.approx(1.128 * sigma, rel=0.05)
    assert ranges[8] == pytest.approx(2.847 * sigma, rel=0.05)
    assert ranges[8] / ranges[2] > 2.2          # range inflates with n
    # sigma_hat is unbiased at every n, so the floor means the same thing
    for n in (2, 4, 8):
        assert sigmas[n] == pytest.approx(sigma, rel=0.03)


def test_c4_matches_known_values_and_corrects_the_small_n_bias():
    """Bessel's correction alone leaves E[s]低 by 20% at n=2. c4 is the factor
    that removes it; these constants are textbook and load-bearing."""
    for n, expect in [(2, 0.7979), (3, 0.8862), (4, 0.9213), (5, 0.9400),
                      (6, 0.9515), (7, 0.9594), (8, 0.9650)]:
        assert ac._c4(n) == pytest.approx(expect, abs=5e-5)
    rng = np.random.RandomState(3)
    raw = np.array([np.std(rng.randn(2), ddof=1) for _ in range(40000)])
    assert raw.mean() == pytest.approx(ac._c4(2), rel=0.02)   # biased low
    assert (raw / ac._c4(2)).mean() == pytest.approx(1.0, rel=0.02)


def test_noise_estimator_reports_its_own_uncertainty():
    """At n=2 the floor is barely an estimate: 75.6% relative sd and a
    [0.08 sigma, 2.46 sigma] 5-95% span. The renderer must print that band, or
    a reader will treat 4.25 FID as a precise threshold."""
    stats = ac._noise_stats_from_fids([100.0, 102.4])
    assert stats["rel_sd"] == pytest.approx(0.7555, abs=1e-3)
    # inverted to a band on the TRUE sigma: sigma_hat/[hi, lo]
    assert stats["true_lo"] == pytest.approx(stats["sigma_hat"] / 2.456, rel=1e-9)
    assert stats["true_hi"] == pytest.approx(stats["sigma_hat"] / 0.079, rel=1e-9)
    assert stats["span_exact"] is True

    text = "\n".join(ac._render_noise_estimator(
        {"noise_fid": stats}))
    assert "2 same-arm replica FIDs" in text
    assert "sigma_hat" in text and "75.6% relative sd" in text
    assert "LOOSER" in text                 # the n=2 direction-of-change note
    assert "undecided" in text              # tells the reader what to do

    many = ac._noise_stats_from_fids([100.0, 101.0, 103.0, 99.0, 104.0])
    assert many["n"] == 5 and many["span_exact"] is True
    text5 = "\n".join(ac._render_noise_estimator({"noise_fid": many}))
    assert "the range would give" in text5   # names the divergence at n>=3
    assert "analyze_teacache_sweeps.py:168" in text5
    assert "deliberate divergence" in text5

    missing = "\n".join(ac._render_noise_estimator({"noise_fid": None}))
    assert "NO same-arm replicas" in missing and "heuristic 2.0 FID" in missing


def test_noise_estimator_extrapolates_the_span_beyond_the_table():
    """Above the tabulated n the span must be flagged as extrapolated rather
    than silently reusing the last row."""
    stats = ac._noise_stats_from_fids([100.0 + i for i in range(9)])
    assert stats["n"] == 9 and stats["span_exact"] is False
    text = "\n".join(ac._render_noise_estimator({"noise_fid": stats}))
    assert "EXTRAPOLATED" in text


def test_noise_estimator_needs_two_replicas():
    assert ac._noise_stats_from_fids([]) is None
    assert ac._noise_stats_from_fids([100.0]) is None
    assert ac._noise_floor_fid(None) == (4.0, False)


def test_gate_survives_an_unusable_slope():
    """If the arm table cannot yield a positive loss->FID slope the absolute
    check must be skipped, not crash and not silently pass."""
    row = _row(0.21, 0.0)
    row["fid_per_loss"] = ac._loss_to_fid_slopes([1.0, 1.0, 1.0, 1.0],
                                                 [100.0, 100.0, 100.0, 100.0])
    assert row["fid_per_loss"] is None
    g = ac._gate(row)
    assert g["abs_ok"] is None and g["pass"]      # falls back to the two shares


def test_renderer_prints_all_three_checks_and_the_marginal_verdict():
    """A reader must be able to see WHICH check failed; "NOT utilizable" with no
    reason was the ambiguity that let the old gate's negative frac be misread."""
    text = "\n".join(ac._render_gate(_row(0.004, 0.047)))
    assert "gate check 1/3 SIGN+SIGNIFICANCE" in text and "pass" in text
    assert "gate check 2/3 RELATIVE SIZE" in text and "FAIL" in text
    assert "gate check 3/3 ABSOLUTE SIZE" in text
    assert "VERDICT GATE: crossover NOT utilizable" in text
    assert "false-positive band" in text


def _probe(root, budgets, n=24):
    """Minimal on-disk probe main() can read: reference + one arm dir per arm."""
    import json
    from PIL import Image

    rng = np.random.RandomState(5)
    base = (rng.rand(n, 16, 16, 3) * 255).astype(np.uint8)

    def write(directory, images, fid, ism):
        gen = os.path.join(directory, "generated")
        os.makedirs(gen, exist_ok=True)
        for i, arr in enumerate(images):
            Image.fromarray(arr).save(os.path.join(gen, f"{i:06d}_cls.png"))
        with open(os.path.join(directory, "results.json"), "w") as fh:
            json.dump({"config": {"seed": 42, "num_steps": 50,
                                  "n_prompts": len(images)},
                       "aggregate": {"fid": fid, "is_mean": ism}}, fh)

    write(os.path.join(root, "reference"), list(base), 103.45, 40.0)
    for k, fids in budgets.items():
        bk = os.path.join(root, f"k{k}")
        os.makedirs(bk, exist_ok=True)
        with open(os.path.join(bk, "manifest.json"), "w") as fh:
            json.dump({"num_steps": 50}, fh)
        # noise scale ordered to match the FID order so [a] passes
        for (name, scale), fid in zip(
                [("uniform", 4), ("geometric", 7), ("back_loaded", 10),
                 ("front_loaded", 14)], fids):
            noisy = np.clip(
                base.astype(np.int16) + rng.randn(n, 16, 16, 3) * scale,
                0, 255).astype(np.uint8)
            write(os.path.join(bk, "equalflops", f"arm_{name}"), list(noisy),
                  fid, 30.0)
        for seed, fid in [(43, fids[0] - 1.2), (44, fids[0] + 1.2)]:
            noisy = np.clip(
                base.astype(np.int16) + rng.randn(n, 16, 16, 3) * 4,
                0, 255).astype(np.uint8)
            write(os.path.join(bk, "equalflops", f"noise_{seed}"), list(noisy),
                  fid, 30.0)
    return root


def test_main_reports_the_real_runs_pattern_as_marginal_not_utilizable(
        tmp_path, monkeypatch, capsys):
    """End-to-end on the pattern the 500-image sweep actually produced.

    Both budgets report a small positive share at p<=0.05. The old gate called
    that UTILIZABLE and the RECOMMENDATION told the reader to move on to the
    online-feature check — i.e. to keep spending. It must instead land in the
    MARGINAL branch, and because two budgets are marginal the proxy-disagreement
    warning must fire too.
    """
    root = _probe(str(tmp_path / "probe"),
                  {8: [96.02, 100.86, 112.74, 126.84],
                   6: [117.30, 143.75, 156.91, 174.13]})

    def _marginal(D, idxs, feat, n_perm, rng):
        oracle = float(D.min(axis=1).mean())
        best = float(D.mean(axis=0).min())
        gain = max(best - oracle, 1e-6)
        return {"skip": None, "train_n": len(idxs) // 2,
                "test_n": len(idxs) - len(idxs) // 2,
                "best_single_test": best, "oracle_test": oracle,
                "oracle_benefit_test": gain, "rule_const": 0,
                "rule_nn_seed_arms": [0], "const_frac": 0.0,
                "reg_frac": 0.006, "reg_benefit": 0.006 * gain,
                "reg_p": 0.041, "reg_null_mean": 0.0, "reg_null_std": 1e-6,
                "nn_frac": -0.35, "nn_benefit": -0.35 * gain, "nn_p": 0.006,
                "nn_null_mean": -0.48, "nn_null_std": 0.07}

    monkeypatch.setattr(ac, "_out_of_sample", _marginal)
    monkeypatch.setattr(sys, "argv",
                        ["analyze_crossover.py", root, "--metric", "pixel_mse",
                         "--max-perm", "20", "--oos-perm", "20"])
    assert ac.main() == 0
    text = capsys.readouterr().out

    assert "MARGINAL, NOT UTILIZABLE at k8" in text
    assert "MARGINAL, NOT UTILIZABLE at k6" in text
    assert "CROSSOVER REAL AND UTILIZABLE" not in text
    assert "DO NOT spend bandit budget on it" in text
    assert "AND THE TWO PROXIES DISAGREE" in text
    # the point-mass explanation must travel with the verdict, not just the row
    assert "EXACTLY 0 for" in text


# --------------------------------------------------------------------------
# [e] granularity: the bandit decides per BATCH, not per image. A per-image
# crossover that the batch granularity cannot monetize must veto the verdict.
# --------------------------------------------------------------------------

def _batch_row(frac=0.21, p=0.0, dist=(1.113, 1.556, 1.655, 2.502),
               fid=(96.02, 100.86, 112.74, 126.84), noise=2.4,
               bs_info=None, gran_gain=None, gran_batches=16):
    """A row whose [d] checks all pass, with [e] granularity attached."""
    base = _row(frac, p, dist=dist, fid=fid, noise=noise)
    base["granularity_skip"] = None
    base["granularity_bs"] = bs_info or {"batch_size": 32,
                                         "generation_start_index": 0,
                                         "assumed_bs": False,
                                         "assumed_start": False}
    if gran_gain is not None:
        base["granularity"] = {"gain": gran_gain, "batches": gran_batches,
                               "per_batch": [], "batch_sizes": [],
                               "min_batch": 0, "max_batch": 0}
    else:
        base["granularity"] = None
    return base


def test_batch_oracle_gain_groups_by_global_idx_not_by_position():
    """`common` is a set intersection that can have holes; a hole must not
    shift later rows into the wrong batch. idxs [2,3,5,6,8,9] at start=2, bs=2
    have a hole at 4: idx 5 lands alone in batch 1 and idx 6 in its own batch
    2. Position-based grouping would pair (5,6) and (8,9) into clean batches
    and report gain 0; the idx-based grouping must report the true batches
    (sizes 2,1,1,2) and a nonzero gain."""
    D = np.array([[5.0, 1.0],   # batch 0
                  [1.0, 5.0],   # batch 0
                  [5.0, 1.0],   # batch 1 (idx 5, hole at 4)
                  [1.0, 5.0],   # batch 2 (idx 6)
                  [5.0, 1.0],   # batch 3
                  [1.0, 5.0]])  # batch 3
    out = ac._batch_oracle_gain(D, [2, 3, 5, 6, 8, 9], batch_size=2,
                                start_index=2)
    assert out is not None
    assert out["batches"] == 4
    assert out["batch_sizes"] == [2, 1, 1, 2]
    # batch0 best 3.0, batch1 best 1.0, batch2 best 1.0, batch3 best 3.0;
    # best single arm over all rows = 3.0
    expected = 3.0 - (2 * 3.0 + 1.0 + 1.0 + 2 * 3.0) / 6
    assert out["gain"] == pytest.approx(expected)
    # a genuine hole: idxs [0,2,3] at bs=2 -> batches {0} and {2,3};
    # position-based grouping would pair (0,2) and report gain 0
    D2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    out2 = ac._batch_oracle_gain(D2, [0, 2, 3], batch_size=2, start_index=0)
    assert out2 is not None and out2["batches"] == 2
    assert out2["batch_sizes"] == [1, 2]
    # batch0 (row 0) best 0.0; batch1 (rows 1,2) best 0.0; best single = 1/3
    assert out2["gain"] == pytest.approx(1.0 / 3.0)


def test_batch_oracle_gain_equals_per_image_at_bs1():
    """At bs=1 every row is its own batch, so the per-batch oracle gain is
    exactly the per-image oracle gain."""
    rng = np.random.RandomState(0)
    D = _ARM_LEVELS[None, :] + rng.randn(40, 4) * 0.5
    out = ac._batch_oracle_gain(D, list(range(40)), batch_size=1, start_index=0)
    assert out is not None
    per_img = float(D.mean(axis=0).min()) - float(D.min(axis=1).mean())
    assert out["gain"] == pytest.approx(per_img, abs=1e-12)


def test_granularity_veto_overrides_three_passing_checks():
    """A row whose [d] checks 1-3 ALL pass must still be vetoed when the
    per-batch oracle gain sits below the absolute floor."""
    # real-size effect: frac=0.21 passes all three [d] checks
    row = _batch_row(frac=0.21, p=0.0, gran_gain=0.004)
    g = ac._gate(row)
    assert g["frac_ok"] and g["p_ok"] and g["rel_ok"] and g["abs_ok"]
    assert g["granularity_ok"] is False
    assert g["pass"] is False
    assert "bs=32" in g["reason"]
    assert "not capturable" in g["reason"] or "per batch" in g["reason"]


def test_granularity_not_vetoed_when_batch_gain_clears_the_floor():
    """When the per-batch oracle gain clears the absolute floor the veto must
    not fire — checks 1-3 plus [e] all pass."""
    row = _batch_row(frac=0.21, p=0.0, gran_gain=0.10)
    g = ac._gate(row)
    assert g["granularity_ok"] is True
    assert g["pass"] is True


def test_granularity_veto_survives_an_unusable_slope():
    """No loss->FID slope means the veto cannot evaluate — it must neither
    crash nor veto (the [d] checks alone decide)."""
    row = _batch_row(frac=0.21, p=0.0, gran_gain=0.004)
    row["fid_per_loss"] = ac._loss_to_fid_slopes([1.0, 1.0, 1.0, 1.0],
                                                 [100.0, 100.0, 100.0, 100.0])
    g = ac._gate(row)
    assert g["granularity_ok"] is None
    assert g["pass"] is True


def test_arms_disagreeing_on_batch_size_skip_granularity():
    """Disagreeing batch_size across arms must skip [e] with a message naming
    the disagreement, never guess."""
    results = {
        "a": {"config": {"batch_size": 8, "generation_start_index": 0}},
        "b": {"config": {"batch_size": 32, "generation_start_index": 0}},
    }
    info, reason = ac._decision_batch_size(results, None)
    assert info is None
    assert reason and "disagree on batch_size" in reason and "8" in reason


def test_decision_batch_size_fallback_flag():
    """When configs record no batch_size, --decision-batch-size supplies it and
    the result is flagged as assumed."""
    results = {"a": {"config": {"seed": 42, "num_steps": 50}},
               "b": {"config": {"seed": 42, "num_steps": 50}}}
    info, reason = ac._decision_batch_size(results, 32)
    assert reason is None
    assert info["batch_size"] == 32 and info["assumed_bs"] is True
    assert info["generation_start_index"] == 0 and info["assumed_start"] is True
    # without the flag it must skip, not guess
    _, reason = ac._decision_batch_size(results, None)
    assert reason and "no batch_size recorded" in reason


def test_render_granularity_renders_and_main_veto_addendum(tmp_path,
                                                           monkeypatch, capsys):
    """End-to-end: the [e] block renders, and a row passing [d] checks 1-3 but
    failing [e] is reported as vetoed rather than utilizable.

    Arms A and B alternate being the per-image winner (sigma 2 vs sigma 120,
    flipped every 2 global idxs), so the per-image oracle beats the best single
    arm by ~0.037 loss units — a large [d] benefit. But each batch of 4 holds
    2 A-wins and 2 B-wins, so the per-batch oracle (one decision per batch)
    recovers almost nothing: the [e] veto must fire.
    """
    import json
    from PIL import Image

    n = 24
    rng = np.random.RandomState(5)
    base = (rng.rand(n, 16, 16, 3) * 255).astype(np.uint8)

    def _write_arm(root, name, images, fid):
        d = os.path.join(root, "k8", "equalflops", f"arm_{name}")
        gen = os.path.join(d, "generated")
        os.makedirs(gen, exist_ok=True)
        for i, arr in enumerate(images):
            Image.fromarray(arr).save(os.path.join(gen, f"{i:06d}_cls.png"))
        with open(os.path.join(d, "results.json"), "w") as fh:
            json.dump({"config": {"seed": 42, "num_steps": 50,
                                  "n_prompts": n,
                                  "batch_size": 4,
                                  "generation_start_index": 0},
                       "aggregate": {"fid": fid, "is_mean": 30.0}}, fh)

    root = str(tmp_path / "probe")
    ref_dir = os.path.join(root, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    for i, arr in enumerate(base):
        os.makedirs(os.path.join(ref_dir, "generated"), exist_ok=True)
        Image.fromarray(arr).save(
            os.path.join(ref_dir, "generated", f"{i:06d}_cls.png"))
    with open(os.path.join(ref_dir, "results.json"), "w") as fh:
        json.dump({"config": {"seed": 42, "num_steps": 50, "n_prompts": n},
                   "aggregate": {"fid": 103.45, "is_mean": 40.0}}, fh)
    os.makedirs(os.path.join(root, "k8"), exist_ok=True)
    with open(os.path.join(root, "k8", "manifest.json"), "w") as fh:
        json.dump({"num_steps": 50}, fh)

    def _noisy(sigma):
        return np.clip(base.astype(np.int16)
                       + rng.randn(n, 16, 16, 3) * sigma, 0, 255).astype(np.uint8)

    winners = np.array([0 if (i % 4) < 2 else 1 for i in range(n)])
    arm_a, arm_b = [], []
    for i in range(n):
        if winners[i] == 0:
            arm_a.append(_noisy(2)[i]); arm_b.append(_noisy(120)[i])
        else:
            arm_a.append(_noisy(120)[i]); arm_b.append(_noisy(2)[i])
    # FIDs chosen to rank like the mean losses: C(60) < A=B(tie) < D(100)
    _write_arm(root, "uniform", arm_a, 110.0)
    _write_arm(root, "geometric", arm_b, 110.0)
    _write_arm(root, "back_loaded", list(_noisy(60)), 100.0)
    _write_arm(root, "front_loaded", list(_noisy(100)), 125.0)
    # same-arm replica spread, as sweep_budget_probe.sh writes them
    for seed, fid in [(43, 96.02 - 1.2), (44, 96.02 + 1.2)]:
        d = os.path.join(root, "k8", "equalflops", f"noise_{seed}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.json"), "w") as fh:
            json.dump({"aggregate": {"fid": fid}}, fh)

    # force a strong [d] pass (checks 1-3) so only [e] can veto
    def _strong(D, idxs, feat, n_perm, rng):
        oracle = float(D.min(axis=1).mean())
        best = float(D.mean(axis=0).min())
        gain = max(best - oracle, 1e-6)
        return {"skip": None, "train_n": len(idxs) // 2,
                "test_n": len(idxs) - len(idxs) // 2,
                "best_single_test": best, "oracle_test": oracle,
                "oracle_benefit_test": gain, "rule_const": 0,
                "rule_nn_seed_arms": [0], "const_frac": 0.0,
                "reg_frac": 0.21, "reg_benefit": 0.21 * gain,
                "reg_p": 0.001, "reg_null_mean": 0.0, "reg_null_std": 1e-6,
                "nn_frac": 0.0, "nn_benefit": 0.0, "nn_p": 0.5,
                "nn_null_mean": 0.0, "nn_null_std": 1e-6}

    monkeypatch.setattr(ac, "_out_of_sample", _strong)
    monkeypatch.setattr(sys, "argv",
                        ["analyze_crossover.py", root, "--metric", "pixel_mse",
                         "--max-perm", "20", "--oos-perm", "20"])
    assert ac.main() == 0
    text = capsys.readouterr().out

    assert "[e] granularity" in text
    assert "batches:" in text and "bs=4" in text
    assert "~16 bandit decisions" in text or "16 reward samples" in text
    # the row passed [d] checks 1-3, so only the [e] veto can explain
    # NOT utilizable — and the recommendation must say so by name
    assert "[e] VETOED at k8" in text
    assert "CROSSOVER REAL AND UTILIZABLE" not in text
    assert "per batch, not per image" in text or "not capturable" in text




# --------------------------------------------------------------------------
# --dump / --from-dump: the verdict must be reproducible offline with no PNG
# loading and no metric extraction, and the dump must round-trip skips.
# --------------------------------------------------------------------------

def _run_main(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", argv)
    code = ac.main()
    return code, capsys.readouterr().out


def _dump_live_args(probe, dump_dir, metric="pixel_mse"):
    return ["analyze_crossover.py", probe, "--metric", metric,
            "--max-perm", "20", "--oos-perm", "20", "--dump", dump_dir]


def _from_dump_args(probe, dump_dir, extra=()):
    return ["analyze_crossover.py", probe, "--from-dump", dump_dir,
            "--max-perm", "20", "--oos-perm", "20"] + list(extra)


def test_dump_fromdump_byte_identity(tmp_path, monkeypatch, capsys):
    """The live run and the offline re-analysis must print IDENTICAL text.

    Both paths feed the same payload through the same ``_analyze_from_payload``
    with the same ``--perm-seed``, so a difference would mean the dump lost
    data or the offline path computes differently. Excluded lines, and why:
      * ``  probe dir:`` and ``  reference dir:`` — the offline path prints
        the dump directory instead of the probe tree (it does not touch it).
      * ``  dump: ...`` and ``  offline re-analysis of dump: ...`` — the
        action lines that legitimately differ between the two invocations.
      * ``  source probe (per manifest): ...`` — names the source path.

    The ``  metric:`` line is NOT excluded. It differs only by an offline-only
    suffix, so it is normalized by stripping that suffix rather than dropped:
    excluding the whole line would hide a real divergence in the metric name
    itself, which is exactly the thing the dump freezes.
    """
    root = tmp_path
    _probe(str(root / "probe"),
           {8: [96.02, 100.86, 112.74, 126.84],
            6: [117.30, 143.75, 156.91, 174.13]})
    dump_dir = str(root / "dump")
    code, live = _run_main(monkeypatch, capsys,
                           _dump_live_args(str(root / "probe"), dump_dir))
    assert code == 0
    code, offline = _run_main(monkeypatch, capsys,
                              _from_dump_args(str(root / "probe"), dump_dir))
    assert code == 0
    offline_lines = offline.splitlines()
    excluded_prefixes = ("  probe dir:", "  reference dir:", "  dump:",
                         "  offline re-analysis of dump:",
                         "  source probe (per manifest):")
    # The offline metric line carries a suffix the live one cannot: drop the
    # suffix, keep the metric name under comparison.
    frozen_suffix = ("  (frozen by the dump; an explicit --metric that "
                     "disagrees is refused)")

    def _norm(lines):
        out = []
        for ln in lines:
            if ln.startswith(excluded_prefixes):
                continue
            if ln.startswith("  metric: ") and ln.endswith(frozen_suffix):
                ln = ln[:-len(frozen_suffix)]
            out.append(ln)
        return out

    live_lines = _norm(live.splitlines())
    offline_lines = _norm(offline_lines)
    assert live_lines == offline_lines
    # The normalization must actually have something to do — if the offline
    # suffix ever disappears, this test would degrade into a no-op comparison
    # that no longer proves the metric name matched.
    assert any(ln.startswith("  metric: ") for ln in live_lines)
    assert frozen_suffix in offline


def test_fromdump_does_not_load_images_or_extract_metrics(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """--from-dump must not touch PNGs or the metric extractor.

    If the offline path accidentally re-ran acquisition, the patched
    ``_load_images`` and ``_distance_matrix`` would raise and the run would
    fail. It completes with a verdict, so the payload alone carried everything.
    """
    probe = str(tmp_path / "probe")
    _probe(probe, {8: [96.02, 100.86, 112.74, 126.84]})
    dump_dir = str(tmp_path / "dump")
    code, _ = _run_main(monkeypatch, capsys,
                        _dump_live_args(probe, dump_dir))
    assert code == 0

    def _boom(*a, **k):
        raise AssertionError("offline path touched images/metrics")

    monkeypatch.setattr(ac, "_load_images", _boom)
    monkeypatch.setattr(ac, "_distance_matrix", _boom)
    monkeypatch.setattr(ac, "_noise_fid_stats", _boom)
    code, text = _run_main(monkeypatch, capsys,
                           _from_dump_args(probe, dump_dir))
    assert code == 0
    assert "RECOMMENDATION" in text


def test_metric_disagreement_with_dump_is_refused(tmp_path, monkeypatch,
                                                  capsys):
    """--from-dump with an explicit --metric that disagrees with the dump's
    frozen metric must fail with a message naming BOTH metrics.

    Silently re-running the dump's numbers under the requested metric would
    print a verdict that was never computed under it.
    """
    probe = str(tmp_path / "probe")
    _probe(probe, {8: [96.02, 100.86, 112.74, 126.84]})
    dump_dir = str(tmp_path / "dump")
    code, _ = _run_main(monkeypatch, capsys,
                        _dump_live_args(probe, dump_dir))
    assert code == 0
    code, text = _run_main(monkeypatch, capsys,
                           _from_dump_args(probe, dump_dir,
                                           ["--metric", "inception"]))
    assert code == 1
    assert "inception" in text and "pixel_mse" in text


def test_fromdump_missing_manifest_is_refused(tmp_path, monkeypatch, capsys):
    """A dump dir without manifest.json must be an error, not an empty
    verdict. The manifest is the only record of the metric and the budget
    list; analyzing without it would silently drop every budget."""
    os.makedirs(str(tmp_path / "dump"), exist_ok=True)
    code, text = _run_main(monkeypatch, capsys,
                           _from_dump_args(str(tmp_path / "probe"),
                                           str(tmp_path / "dump")))
    assert code == 1
    assert "manifest.json" in text


def test_fromdump_inconsistent_per_budget_metric_is_refused(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    """A budget whose k<K>.json metric disagrees with the manifest must fail
    the whole re-analysis — a partial verdict is worse than none."""
    import json as _json
    probe = str(tmp_path / "probe")
    _probe(probe, {8: [96.02, 100.86, 112.74, 126.84]})
    dump_dir = str(tmp_path / "dump")
    code, _ = _run_main(monkeypatch, capsys,
                        _dump_live_args(probe, dump_dir))
    assert code == 0
    jpath = os.path.join(dump_dir, "k8.json")
    with open(jpath) as fh:
        data = _json.load(fh)
    data["metric"] = "inception"
    with open(jpath, "w") as fh:
        _json.dump(data, fh)
    code, text = _run_main(monkeypatch, capsys,
                           _from_dump_args(probe, dump_dir))
    assert code == 1
    assert "inconsistent" in text and "inception" in text


def test_dump_and_fromdump_together_are_refused(tmp_path, monkeypatch,
                                                capsys):
    """--dump and --from-dump together are contradictory and must be refused
    before anything is analyzed."""
    probe = str(tmp_path / "probe")
    _probe(probe, {8: [96.02, 100.86, 112.74, 126.84]})
    dump_dir = str(tmp_path / "dump")
    code, text = _run_main(
        monkeypatch, capsys,
        ["analyze_crossover.py", probe, "--metric", "pixel_mse",
         "--max-perm", "20", "--oos-perm", "20",
         "--dump", dump_dir, "--from-dump", dump_dir])
    assert code == 1
    assert "mutually exclusive" in text


def test_gate_params_stay_live_offline(tmp_path, monkeypatch, capsys):
    """--floor-ratio must change the offline verdict line: re-tuning the gate
    without GPU is the point of the dump."""
    probe = str(tmp_path / "probe")
    _probe(probe, {8: [96.02, 100.86, 112.74, 126.84]})
    dump_dir = str(tmp_path / "dump")
    code, _ = _run_main(monkeypatch, capsys,
                        _dump_live_args(probe, dump_dir))
    assert code == 0
    code, default_text = _run_main(monkeypatch, capsys,
                                   _from_dump_args(probe, dump_dir,
                                                   ["--floor-ratio", "100000.0"]))
    assert code == 0
    code, lax_text = _run_main(
        monkeypatch, capsys,
        _from_dump_args(probe, dump_dir, ["--floor-ratio", "0.001"]))
    assert code == 0
    assert "QUANTIZATION-LIMITED" in default_text
    assert "QUANTIZATION-LIMITED" not in lax_text
    assert default_text != lax_text


def test_skipped_budget_round_trips_to_the_same_skipped_line(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    """A budget skipped during acquisition (no shared images) must reproduce
    the same SKIPPED line offline — not crash, not silently vanish.

    The dump records the skip reason in k<K>.json; _analyze_from_payload
    returns the same minimal row both paths, so the line is byte-identical.
    """
    import json as _json
    from PIL import Image
    root = str(tmp_path / "probe")
    n = 24
    rng = np.random.RandomState(5)
    base = (rng.rand(n, 16, 16, 3) * 255).astype(np.uint8)

    def _write(directory, images, fid, ism):
        gen = os.path.join(directory, "generated")
        os.makedirs(gen, exist_ok=True)
        for i, arr in enumerate(images):
            Image.fromarray(arr).save(os.path.join(gen, f"{i:06d}_cls.png"))
        with open(os.path.join(directory, "results.json"), "w") as fh:
            _json.dump({"config": {"seed": 42, "num_steps": 50,
                                   "n_prompts": n},
                        "aggregate": {"fid": fid, "is_mean": ism}}, fh)

    def _write_shifted(directory, images, start, fid, ism):
        gen = os.path.join(directory, "generated")
        os.makedirs(gen, exist_ok=True)
        for j, arr in enumerate(images):
            Image.fromarray(arr).save(
                os.path.join(gen, f"{start + j:06d}_cls.png"))
        with open(os.path.join(directory, "results.json"), "w") as fh:
            _json.dump({"config": {"seed": 42, "num_steps": 50,
                                   "n_prompts": n},
                        "aggregate": {"fid": fid, "is_mean": ism}}, fh)

    ref_dir = os.path.join(root, "reference")
    _write(ref_dir, list(base), 103.45, 40.0)
    # k6 arms carry global_idx 1000..1023 while the reference has 0..23, so
    # `common` is empty and the budget is skipped. k8 uses 0..23 and is valid.
    for k, start in [(6, 1000), (8, 0)]:
        bk = os.path.join(root, f"k{k}")
        os.makedirs(bk, exist_ok=True)
        with open(os.path.join(bk, "manifest.json"), "w") as fh:
            _json.dump({"num_steps": 50}, fh)
        for j, (name, scale, fid) in enumerate(
                [("uniform", 4, 96.02), ("geometric", 7, 100.86),
                 ("back_loaded", 10, 112.74), ("front_loaded", 14, 126.84)]):
            noisy = np.clip(
                base.astype(np.int16) + rng.randn(n, 16, 16, 3) * scale,
                0, 255).astype(np.uint8)
            _write_shifted(os.path.join(bk, "equalflops", f"arm_{name}"),
                           list(noisy), start, fid, 30.0)
    dump_dir = str(tmp_path / "dump")
    code, live = _run_main(monkeypatch, capsys,
                           _dump_live_args(root, dump_dir))
    assert code == 0
    assert "SKIPPED: only 0 images shared with reference" in live
    code, offline = _run_main(monkeypatch, capsys,
                              _from_dump_args(root, dump_dir))
    assert code == 0
    assert "SKIPPED: only 0 images shared with reference" in offline
