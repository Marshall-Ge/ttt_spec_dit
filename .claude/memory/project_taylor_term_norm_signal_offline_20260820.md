---
name: taylor_term_norm_signal_offline_20260820
description: "P3 候选信号'Taylor 各阶 term 范数增长轨迹'在 37,808 条 shadow events 上的离线区分度 gate 结果——负，step+distance 确定性变量解释 99% context 级方差，term norms 边际 ≤0.03%"
type: project
---

Candidate signal evaluated offline (no GPU): per-order Taylor term norms
(`taylor_term_norms`, mean over layers of |term_k|·d^k/k!, batch-level,
observed for free from cached derivatives) as a predictor of cache
"overdraft" — target reframed from current defect to **defect at d+1**
("what happens if we keep skipping") and **explosion indicator**
(defect_{d+1} > p90). Data: the 2026-07-21 SpecA shadow JSONL (37,808
events, 32 batch trajectories, 1,210 contexts, distance 1-4).
Script: `scripts/analyze_taylor_term_signal.py`.

Results (5-fold grouped by trajectory):

- Defect variance is 91.6% between-context. **One-hot step_idx alone:
  OOS R² 0.845 (per-sample T0), 0.903 (T1 defect_{d+1}), 0.991
  (context-mean T1).** Adding distance one-hot: +0.0000 (distance is
  nearly a deterministic function of step under SpecA's own policy).
- Adding term norms + scalars on top: **T0 +0.0003 R², T1 +0.0001 R²,
  explosion AUROC 0.9807 → 0.9813.** Free features alone (no step one-hot):
  R² 0.74 but that is entirely their correlation with step phase.
- Explosion (defect_{d+1} > p90, base rate 0.135) is a **step-phase
  phenomenon**: steps 0-9 mean defect 0.596, 20-39 ~0.10, 40-49 0.494;
  one-hot step AUROC 0.98, top-decile precision 0.905.
- Premise check "term norms grow with overdraft": norms grow with d by
  construction (∝ d^k), and defect does grow with distance within phase
  (phase 0-9: d1=0.34 → d3=1.22) — but (phase, distance) is known a
  priori from the schedule; the observation adds nothing.
- Direction inconsistency: in the only slice with within-step tn2
  variance (steps 5-9, dist≥2), Spearman(tn2, defect) = **−0.687** —
  opposite to the "overdraft precursor" hypothesis. tn3/tn4 have zero
  variance in-slice (higher orders mostly unavailable at d≤4).

**VERDICT: FAIL at the offline distinguishability gate — do not spend
GPU.** Two additional kill shots independent of this data: (i) label
coverage stops at distance 4 (SpecA's own policy), so aggressive-mask
regimes are unmeasured and would need new forced-long-gap instrumentation;
(ii) even perfect defect prediction would not rank quality (reward-proxy
gate failed twice — one-step defect measures local cache error, not where
compute buys quality).

Reopen only if BOTH change: new labels at d>4 AND a defect→quality
validity result (has failed twice: 2026-08-19 active paired result,
reward-proxy Spearman gate).

Why: closes candidate direction P3 from the 2026-08-20 research brief at
the cheapest possible gate (fully offline, existing events).

How to apply: any future "cache-overdraft precursor" signal must be
evaluated against the one-hot (step, distance) baseline in
`scripts/analyze_taylor_term_signal.py` first; beating deterministic
schedule variables is the entry bar, not beating mean-only.
