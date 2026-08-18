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
