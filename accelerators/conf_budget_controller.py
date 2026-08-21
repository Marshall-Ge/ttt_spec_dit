# -*- coding: utf-8 -*-
"""Session-level quality-constrained budget controller (P1 v1).

Spec: .claude/covr_quality_constrained_budget_20260820.md §2. The negative
feedback loop for the SPEED axis: reference-free InceptionV3 predictive
entropy of the emitted images ("conf", smaller = better) drives which
forced-mask budget rung the NEXT epoch runs at —

  * upshift (safety):      per-image CUSUM on standardized entropy; an alarm
                           moves one rung up (or straight to the top with
                           ``alarm_to_top``) and resets the detector.
  * downshift (aggression): after ``stable`` consecutive clean epochs at the
                           current rung whose mean-z 95% one-sided lower
                           bound is >= -eps, move one rung down. The
                           clean-epoch clock restarts on EVERY rung change,
                           so stale epochs from a higher budget can never
                           chain into an immediate second downstep.

Plug-and-play boundary: this module owns only the decision state machine —
no tensors, no model, no metric code. The caller owns InceptionV3 scoring
(one forward per image, ~1/200 of a DiT step) and mask forcing (the COVR
forced-strategy path already switches masks per trajectory).

Single source of truth: ``mode_simulate``/``mode_power`` in
scripts/simulate_conf_budget_controller.py consume THIS module, so the
offline [P1-b]/[P1-c] gate numbers certify the exact arithmetic that would
run online.

Typical wiring (closed-loop run):

    ctrl = ConfBudgetController.calibrate([8, 6, 4], calib_conf, h=h_star)
    for each epoch of B images:
        generate the epoch at ``ctrl.rung`` (forced mask of that rung)
        conf = inception_entropy(epoch_images)     # per-image, nats
        decision = ctrl.end_epoch(conf)            # rung for the next epoch
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


class Cusum:
    """Upward CUSUM on standardized entropy (entropy larger = quality worse;
    alarm detects degradation). k=delta is the drift allowance in sigma."""

    def __init__(self, mu0, sigma0, delta=0.5, h=5.0):
        self.mu0, self.s0 = mu0, max(sigma0, 1e-9)
        self.k, self.h, self.s = delta, h, 0.0

    def update(self, x):
        z = (x - self.mu0) / self.s0   # z>0 = worse than calibration
        self.s = max(0.0, self.s + z - self.k)
        return self.s > self.h

    def reset(self):
        self.s = 0.0


@dataclass(frozen=True)
class EpochDecision:
    """What the controller did at one epoch boundary."""
    epoch_idx: int
    rung_before: int
    rung_after: int
    alarmed: bool
    downstepped: bool
    z_mean: Optional[float]   # epoch mean of (mu0 - x)/s0; None on alarm


class ConfBudgetController:
    """CUSUM-guarded budget-ladder walker (see module docstring)."""

    def __init__(self, rungs: Sequence[int], mu0: float, sigma0: float, *,
                 delta: float = 0.5, h: float = 5.0, eps: float = 0.15,
                 stable: int = 2, alarm_to_top: bool = False):
        if len(rungs) < 2:
            raise ValueError("need at least two budget rungs")
        self.rungs: List[int] = sorted(int(r) for r in rungs)[::-1]
        if len(set(self.rungs)) != len(self.rungs):
            raise ValueError("duplicate budget rungs")
        self.mu0 = float(mu0)
        self.s0 = max(float(sigma0), 1e-9)
        self.eps = float(eps)
        self.stable = int(stable)
        self.alarm_to_top = bool(alarm_to_top)
        self._cusum = Cusum(self.mu0, self.s0, delta, h)

        self._idx = 0                 # index into self.rungs (0 = top)
        self._z_hist: List[float] = []
        self._epoch_idx = 0
        self.alarms = 0
        self.epochs_per_rung = {r: 0 for r in self.rungs}

    # -- construction -------------------------------------------------------

    @classmethod
    def calibrate(cls, rungs: Sequence[int],
                  calibration_conf: Sequence[float],
                  **kwargs) -> "ConfBudgetController":
        """Build from the startup calibration epoch generated at the TOP
        rung (spec: the quality target is the session's own k8 epoch)."""
        vals = np.asarray(calibration_conf, dtype=float)
        if vals.size < 2:
            raise ValueError("calibration epoch needs >= 2 conf values")
        return cls(rungs, float(vals.mean()), float(vals.std(ddof=1)),
                   **kwargs)

    # -- state --------------------------------------------------------------

    @property
    def rung(self) -> int:
        """Budget rung the CURRENT epoch should be generated at."""
        return self.rungs[self._idx]

    # -- transition ---------------------------------------------------------

    def end_epoch(self, conf_values: Sequence[float]) -> EpochDecision:
        """Consume one epoch of per-image conf (generated at ``self.rung``)
        and move the ladder for the next epoch. Bit-identical to the
        offline gate harness: per-image CUSUM with first-alarm
        short-circuit; an alarm resets the detector and the clean-epoch
        history; a downshift clears the history too."""
        vals = np.asarray(conf_values, dtype=float)
        if vals.size == 0:
            raise ValueError("empty epoch")
        rung_before = self.rung
        self.epochs_per_rung[rung_before] += 1

        fired = any(self._cusum.update(x) for x in vals)

        downstepped = False
        z_mean: Optional[float] = None
        if fired:
            self.alarms += 1
            self._cusum.reset()
            self._z_hist.clear()
            self._idx = 0 if self.alarm_to_top else max(self._idx - 1, 0)
        else:
            z_mean = float(((self.mu0 - vals) / self.s0).mean())
            self._z_hist.append(z_mean)
            if len(self._z_hist) >= self.stable:
                recent = self._z_hist[-self.stable:]
                se = (float(np.std(recent, ddof=1)) / math.sqrt(len(recent))
                      if len(recent) > 1 else 0.0)
                if float(np.mean(recent)) + 1.645 * se >= -self.eps:
                    if self._idx + 1 < len(self.rungs):
                        # clean-epoch clock restarts at the new rung
                        self._z_hist.clear()
                        self._idx += 1
                        downstepped = True

        decision = EpochDecision(
            epoch_idx=self._epoch_idx, rung_before=rung_before,
            rung_after=self.rung, alarmed=fired, downstepped=downstepped,
            z_mean=z_mean)
        self._epoch_idx += 1
        return decision

    # -- reporting ----------------------------------------------------------

    def stats(self) -> dict:
        return {
            "rung": self.rung,
            "epochs": self._epoch_idx,
            "alarms": self.alarms,
            "epochs_per_rung": dict(self.epochs_per_rung),
            "mu0": self.mu0,
            "sigma0": self.s0,
            "params": {"delta": self._cusum.k, "h": self._cusum.h,
                       "eps": self.eps, "stable": self.stable,
                       "alarm_to_top": self.alarm_to_top},
        }
