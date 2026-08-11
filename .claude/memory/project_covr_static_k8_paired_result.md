---
name: covr-static-k8-paired-result
description: 2026-08-04 严格等计算量配对实验确认 uniform 与 plain TeaCache 在 FID/IS 上形成稳定 Pareto 权衡
metadata: 
  node_type: memory
  type: project
  originSessionId: 8bfc7555-71d4-4e59-801e-051977499260
  modified: 2026-08-04T09:24:49.208Z
---

2026-08-04，DiT/ImageNet、50 steps、500 images、batch size 32、5 个配对 latent offsets 的静态 K=8 实验得到：

- `uniform - geometric`：FID `-1.768 +/- 0.681 SE`，IS `+4.636 +/- 0.293 SE`；5/5 draws 同向，两项单侧 exact sign test 均为 `p=0.03125`。因此 uniform 是所测固定 mask 中的选择。
- 严格同计算量 `uniform K=8 - plain TeaCache threshold=1.40 K=8`：FID `-9.270 +/- 0.848 SE`，IS `-5.612 +/- 0.242 SE`；FID 5/5 偏向 uniform，IS 5/5 偏向 plain TeaCache。两者是稳定 Pareto 权衡，没有同时支配 FID 和 IS 的静态赢家。
- 此结果与此前 threshold=1.25、K=9 对照几乎相同，排除了一个 calc step 的预算偏差作为冲突原因。

**Why:** terminal/H-step reward 都不能按 FID 排序，batch-level controller 又没有可利用的 oracle headroom；稳定的 aggregate-metric 权衡不是继续训练在线 bandit 的依据。

**How to apply:** 不再为该分支增加 replicas 或恢复 COVR bandit。FID 优先时使用 static uniform K=8；IS 优先时保留 plain TeaCache threshold=1.40；若要求两项同时不退化，则关闭该 mask 方向。与 [[project_covr_decision_granularity]]、[[project_covr_bandit_prior_lockin]] 和 [[project_covr_system]] 一起使用。
