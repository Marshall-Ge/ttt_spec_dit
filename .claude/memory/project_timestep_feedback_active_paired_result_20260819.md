---
name: timestep feedback active paired quality result
description: 2026-08-19 matched-FLOPs paired comparison of risk-learned vs uniform K=12 SpecA masks; FID/IS deltas are noise-dominated, placement learning does not translate into quality gain
type: project
---

The decisive quality gate for the online timestep-feedback direction
(`--covr-timestep-feedback-active`). DiT/ImageNet, 50 steps, CFG 4.5, 500
images, batch 32, five paired latent offsets, SpecA forced masks via
`scripts/build_paired_masks.py`.

Arms (equal refresh count 12, equal max Taylor gap 4, identical uniform
skeleton [0,5,...,45]):

- `uniform_static`: extra refreshes at steps 2 and 47.
- `risk_learned`: extra refreshes at steps 4 and 6 (chosen by the learned
  one-step-defect risk, which peaks at steps 5-6).

Paired deltas `uniform_static - risk_learned`:

- FID: mean `+0.294`, sd `0.949`, se `0.424`, range `-1.065..+1.355`
  (2/5 offsets favor uniform, 3/5 favor learned);
- IS: mean `-0.153`, sd `0.418`, se `0.187` (sign test p=0.8125).

Both directions fail the 5/5 sign gates; the differences are noise-dominated.
**VERDICT: TIED — risk-driven placement does not improve FID or IS over uniform
placement at equal budget.**

Interpretation chain (all three links now measured):

1. One-step defect feedback learns a real timestep prior (session-held-out MSE
   0.358 -> 0.027) — but exactly equals the unweighted one-hot baseline.
2. The learned risk does change mask placement (steps 4/6 instead of 2/47) —
   verified in active smokes s3/s4.
3. The placement change does NOT move quality (this experiment).

Consistent with prior negative results: the reward-proxy verdict (one-step
defect does not rank arms like FID) and the K=8 arm sweep (the defect-heavy
early steps are where front-loaded compute was WORST, FID 139 vs uniform 119).
The one-step transition defect measures local cache error, not where compute
buys quality.

Decision: the session-level timestep-feedback active direction is closed as a
quality-improvement method. The shadow learner infrastructure (selective
labels, IPW, state persistence, session-held-out analysis) remains reusable.
Do not spend further GPU on learned-placement comparisons at non-binding
budgets; any future attempt must first show a proxy that ranks FID, which the
reward-proxy gate has already failed twice.
