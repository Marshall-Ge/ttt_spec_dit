---
name: COVR K8 random-null paired result
description: 2026-08-18 DiT/ImageNet K=8 static TeaCache mask sweep; placement changes aggregate FID/IS, but no single static mask dominates plain TeaCache and no adaptive crossover evidence passed
type: project
---

Experiment: DiT/ImageNet, 50 denoising steps, CFG 4.5, 500 images, batch size 32,
five paired latent offsets, K=8 forced TeaCache masks, four deterministic layouts
plus two reproducible random-null masks, and plain TeaCache threshold 1.40.

Aggregate K=8 FID values at offset 0:

| arm | FID | IS |
|---|---:|---:|
| pattern_uniform_k8 | 119.29 | 32.0 |
| pattern_geometric_k8 | 119.82 | 26.9 |
| pattern_random_00_k8 | 119.81 | 33.7 |
| pattern_random_01_k8 | 120.98 | 33.6 |
| pattern_back_loaded_k8 | 128.17 | 27.7 |
| pattern_front_loaded_k8 | 139.17 | 18.1 |

The equal-FLOPs arm spread was 19.87 FID against a measured same-arm noise
floor of 2.00 FID (9.94x). This confirms that placement matters at K=8.

The five-offset paired comparison uniform minus geometric was FID
`-1.768 +/- 0.681 SE` and IS `+4.636 +/- 0.293 SE`; all five offsets favored
uniform and the one-sided exact sign-test p-value was 0.03125 for both metrics.

Against plain TeaCache threshold 1.40 at matched calc count, uniform minus
threshold was FID `-9.270 +/- 0.848 SE` and IS `-5.612 +/- 0.242 SE`. FID
favored uniform, while IS favored plain TeaCache on all five offsets. Therefore
there is no single static winner across both quality metrics.

No reward conclusion is available: the run intentionally used
`SENTINEL_RATE=0`, so terminal reward telemetry was absent. The reference-based
per-image Inception distance also failed the declared rank-validity gate
(Spearman with FID 0.714), so this run does not establish per-image crossover
or justify an adaptive bandit.

Decision: retain uniform K=8 as the FID-oriented static candidate, retain plain
TeaCache threshold 1.40 as the IS-oriented comparator, and do not resume COVR
bandit development. Further work should be held-out static schedule/Pareto
validation, not online controller training.
