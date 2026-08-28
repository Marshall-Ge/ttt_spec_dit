---
name: timestep feedback prequential smoke result
description: 2026-08-18 two-session shadow learner analysis; one-step timestep prediction improves after session-1 labels, but no terminal or FID claim
type: project
---

Two 16-image DiT/SpecA shadow sessions were analyzed with the same runtime
version and refresh budget 8. Session 1 had 25 selective events; session 2 had
32. The learner predicted each session before ingesting that session's labels.

| session | events | MSE | MAE | top-decile recall |
|---|---:|---:|---:|---:|
| s1, prior only | 25 | 0.3580 | 0.5023 | 0.00 |
| s2, after s1 update | 32 | 0.0270 | 0.1126 | 0.75 |

This is a successful session-held-out sanity check for a timestep-conditioned
one-step defect prior. It is not evidence of better generation quality, does
not compare against a frozen one-hot/static baseline, and does not activate the
learner's decisions in the actual SpecA trajectory. The next gate is active
budget-matched policy evaluation with final latent/decoded metrics.

After adding the unweighted one-hot timestep baseline, learner and baseline
metrics were exactly identical on this smoke (`MSE=0.19250`, `MAE=0.30743`,
top-decile recall `0.375`). Therefore clipped IPW/UCB has no demonstrated
incremental benefit yet; the observed result is only evidence that a timestep
prior can be learned online.

Budget-12 active deployment smoke (session s4, seed 45) confirmed that risk
adaptation is real: with 2 free refresh slots above the 10-refresh safety
minimum, the learned mask kept the uniform skeleton [0,5,...,45] and placed
the extra refreshes at steps 4 and 6 — the second/third highest-risk timesteps
after step 5 (mean 1.26). 48/200 full steps, no NaN. This is the minimal
viable evidence chain (feedback -> cross-session learning -> mask placement
change) but NOT a quality claim: no matched-FLOPs FID/IS paired comparison
has been run yet. The next gate is exactly that comparison (active-learned
mask vs static uniform at equal budget, 5 latent offsets, paired sign test).
