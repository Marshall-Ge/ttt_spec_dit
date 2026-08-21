# -*- coding: utf-8 -*-
"""Unit tests for accelerators/conf_budget_controller.py (P1 runtime).

CPU-only (numpy; the accelerators package import pulls torch, which the
conda base env has). Run with:

    pytest test_conf_budget_controller.py -v

The equivalence test pins the runtime class against a VERBATIM copy of the
legacy inline loop that produced the [P1-b]/[P1-c] gate numbers — if this
passes, the gate verdicts certify the exact code that runs online.
"""

import math

import numpy as np

from accelerators.conf_budget_controller import ConfBudgetController, Cusum


RUNGS = [8, 6, 4]


def flat_epoch(n=100):
    """An epoch exactly at the calibration target (z == 0 everywhere)."""
    return np.zeros(n)


def make_ctrl(**kw):
    """mu0=0, s0=1 so conf values ARE z-scores."""
    kw.setdefault("delta", 0.5)
    kw.setdefault("h", 5.0)
    kw.setdefault("eps", 0.15)
    kw.setdefault("stable", 2)
    return ConfBudgetController(RUNGS, 0.0, 1.0, **kw)


# ---------------------------------------------------------------------------
# ladder walking
# ---------------------------------------------------------------------------

def test_flat_stream_walks_down_with_cascade_guard():
    """Clean epochs downstep only after `stable` epochs AT EACH RUNG:
    8,8 -> 6,6 -> 4,4,... — never 8,8 -> 6 -> 4 (the cascade bug)."""
    ctrl = make_ctrl()
    trace = []
    for _ in range(10):
        trace.append(ctrl.rung)
        ctrl.end_epoch(flat_epoch())
    assert trace == [8, 8, 6, 6, 4, 4, 4, 4, 4, 4]
    assert ctrl.epochs_per_rung == {8: 2, 6: 2, 4: 6}
    assert ctrl.alarms == 0


def test_bottom_rung_is_absorbing_without_alarm():
    ctrl = make_ctrl()
    for _ in range(6):
        ctrl.end_epoch(flat_epoch())
    assert ctrl.rung == 4
    d = ctrl.end_epoch(flat_epoch())
    assert d.rung_before == 4 and d.rung_after == 4
    assert not d.downstepped and not d.alarmed


def test_downstep_blocked_when_epoch_quality_below_eps():
    """Epoch mean-z of -0.5 (worse than target by half a sigma, eps=0.15)
    must never downstep, but must not alarm either (below CUSUM drift)."""
    ctrl = make_ctrl()
    for _ in range(8):
        d = ctrl.end_epoch(np.full(100, 0.5))   # z = +0.5 each image
    assert ctrl.rung == 8
    assert ctrl.alarms == 0
    assert not d.downstepped


# ---------------------------------------------------------------------------
# alarm path
# ---------------------------------------------------------------------------

def test_alarm_upshifts_one_rung_and_resets_detector_and_history():
    ctrl = make_ctrl()
    for _ in range(4):                       # walk down to rung 4
        ctrl.end_epoch(flat_epoch())
    assert ctrl.rung == 4
    d = ctrl.end_epoch(np.full(100, 10.0))   # grossly degraded epoch
    assert d.alarmed and d.rung_after == 6
    assert ctrl.alarms == 1
    assert ctrl._cusum.s == 0.0              # detector restarted
    # clean-epoch history was cleared: the NEXT clean epoch alone must not
    # downstep (needs `stable`=2 fresh epochs at the new rung)
    d1 = ctrl.end_epoch(flat_epoch())
    assert not d1.downstepped and ctrl.rung == 6
    d2 = ctrl.end_epoch(flat_epoch())
    assert d2.downstepped and ctrl.rung == 4


def test_alarm_to_top_jumps_to_top_rung():
    ctrl = make_ctrl(alarm_to_top=True)
    for _ in range(4):
        ctrl.end_epoch(flat_epoch())
    assert ctrl.rung == 4
    d = ctrl.end_epoch(np.full(100, 10.0))
    assert d.alarmed and d.rung_after == 8


def test_alarm_at_top_stays_at_top():
    ctrl = make_ctrl()
    d = ctrl.end_epoch(np.full(100, 10.0))
    assert d.alarmed and d.rung_before == 8 and d.rung_after == 8


# ---------------------------------------------------------------------------
# construction / edges
# ---------------------------------------------------------------------------

def test_calibrate_matches_manual_moments():
    calib = np.array([1.0, 2.0, 3.0, 4.0])
    ctrl = ConfBudgetController.calibrate(RUNGS, calib)
    assert ctrl.mu0 == calib.mean()
    assert ctrl.s0 == calib.std(ddof=1)


def test_stable_one_downsteps_after_single_clean_epoch():
    """stable=1: se guard (len==1 -> se=0) makes the rule decidable —
    intentional fix of the legacy nan-never-downsteps edge."""
    ctrl = make_ctrl(stable=1)
    d = ctrl.end_epoch(flat_epoch())
    assert d.downstepped and ctrl.rung == 6


def test_rejects_degenerate_ladders():
    import pytest
    with pytest.raises(ValueError):
        ConfBudgetController([8], 0.0, 1.0)
    with pytest.raises(ValueError):
        ConfBudgetController([8, 8, 6], 0.0, 1.0)


# ---------------------------------------------------------------------------
# equivalence with the legacy inline gate loop (verbatim copy)
# ---------------------------------------------------------------------------

def legacy_trial_trace(rng, arm_vals, ladder, epoch, n_epochs,
                       delta, h, eps, stable, alarm_to_top):
    """VERBATIM port of the pre-refactor mode_simulate inner loop
    (scripts/simulate_conf_budget_controller.py) — the code the offline
    gates were originally run against."""
    ref = ladder[0]
    k = ref
    calib = rng.choice(arm_vals[ref], size=epoch, replace=True)
    mu0, s0 = calib.mean(), calib.std(ddof=1)
    cusum = Cusum(mu0, s0, delta, h)
    z_hist = []
    trace, alarms = [], 0
    for _ep in range(n_epochs):
        v = rng.choice(arm_vals[k], size=epoch, replace=True)
        trace.append(k)
        fired = any(cusum.update(x) for x in v)
        if fired:
            alarms += 1
            cusum = Cusum(mu0, s0, delta, h)
            z_hist.clear()
            k = ladder[0] if alarm_to_top else \
                min((b for b in ladder if b > k), default=k)
        else:
            z = (mu0 - v) / s0
            z_hist.append(z.mean())
            if len(z_hist) >= stable:
                recent = z_hist[-stable:]
                se = np.std(recent, ddof=1) / math.sqrt(len(recent))
                if np.mean(recent) + 1.645 * se >= -eps:
                    new_k = max((b for b in ladder if b < k), default=k)
                    if new_k != k:
                        z_hist.clear()
                        k = new_k
    return trace, alarms


def controller_trial_trace(rng, arm_vals, ladder, epoch, n_epochs,
                           delta, h, eps, stable, alarm_to_top):
    calib = rng.choice(arm_vals[ladder[0]], size=epoch, replace=True)
    ctrl = ConfBudgetController.calibrate(
        ladder, calib, delta=delta, h=h, eps=eps, stable=stable,
        alarm_to_top=alarm_to_top)
    trace, alarms = [], 0
    for _ep in range(n_epochs):
        k = ctrl.rung
        v = rng.choice(arm_vals[k], size=epoch, replace=True)
        trace.append(k)
        alarms += ctrl.end_epoch(v).alarmed
    return trace, alarms


def test_equivalence_with_legacy_inline_loop():
    """Same seeded rng stream -> identical per-epoch rung traces and alarm
    counts across regimes that exercise downsteps, alarms and recoveries."""
    base = np.random.default_rng(7)
    arm_vals = {
        8: base.normal(3.00, 0.30, 500),   # reference quality
        6: base.normal(3.08, 0.30, 500),   # mild degradation
        4: base.normal(3.60, 0.35, 500),   # clear degradation -> alarms
    }
    ladder = [8, 6, 4]
    cases = [
        dict(delta=0.5, h=5.0, eps=0.15, stable=2, alarm_to_top=False),
        dict(delta=0.5, h=5.0, eps=0.15, stable=2, alarm_to_top=True),
        dict(delta=0.5, h=8.0, eps=0.30, stable=3, alarm_to_top=False),
    ]
    for case in cases:
        for seed in range(20):
            r1 = np.random.default_rng(seed)
            r2 = np.random.default_rng(seed)
            t_old, a_old = legacy_trial_trace(
                r1, arm_vals, ladder, epoch=50, n_epochs=15, **case)
            t_new, a_new = controller_trial_trace(
                r2, arm_vals, ladder, epoch=50, n_epochs=15, **case)
            assert t_old == t_new, (case, seed, t_old, t_new)
            assert a_old == a_new, (case, seed)
