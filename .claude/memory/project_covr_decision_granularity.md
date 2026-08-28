---
name: project_covr_decision_granularity
description: COVR bandit 的决策粒度是 batch 不是 image；bs=32 的 batch 含 ~31.5 个不同类，per-image crossover 天花板被批平均抹掉两个数量级（2026-08-03）
metadata: 
  node_type: memory
  type: project
  originSessionId: 8d7c6d99-e3d5-4381-9a01-91907f73c434
  modified: 2026-08-03T09:41:06.649Z
---

**bandit 一个 trajectory = 一个 batch，不是一张图。** `run_dit.py:1936` `trajectory_id = covr_trajectory_offset + batch_start // bs`，每个 trajectory 只有一次 `apply_strategy(...)`。而 `scripts/analyze_crossover.py` 的 `[d]` 量的是 **per-image** crossover —— 两个量不同，per-image 有 headroom 不等于 bandit 能拿到。

**batch 内类几乎全不同。** `dataset/imagenet.py:137-140` 先 shuffle 全部 50k 再截断，`run_dit.py:1915-1917` 取连续 global_idx 区块，所以 bs=32 的 batch 期望含 `1000*(1-(999/1000)^32) = 31.5` 个不同类。per-image 难度 sd 被批平均压到 `1/√32 = 17.7%`，between-batch 变异是被设计掉的，不是数据碰巧如此。

**模拟量级**（`[e]` 段落的动机）：per-image interaction 强度 0.8 时 bs=1 天花板 19.21 FID，bs=32 只有 −0.00 FID；interaction 2.0 时 64.06 vs 0.54。差两个数量级。

`[e]` 段落（`_decision_batch_size` / `_batch_oracle_gain`）在真实 batch 边界上重算 D：按 `(global_idx − start_index) // batch_size` 分组（**不能按行号**，`common` 是有洞的交集），少于 2 个 batch 返回 `None`。它是**纯否决**：`granularity_ok` 默认 `None`，只在 batch size 可恢复且有 loss→FID 斜率时才算，只能把 `pass` 压成 `False`，不会放宽 check 1-3。arm 之间 batch size 不一致时拒绝猜，`--decision-batch-size` 的结果标注为 ASSUMED。

**Why:** 之前所有 arm-spread 实验都在 bs=32 上跑，即使 per-image 真有 context 依赖也测不到 —— 这是和 [[project_covr_bandit_prior_lockin]] 独立的第二个"实验设计把要测的东西设计掉了"。

**How to apply:** 任何要证明 context 依赖的 bandit 实验必须 `BATCH_SIZE=1`。`scripts/sweep_budget_probe.sh:86` 是 `BATCH_SIZE="${BATCH_SIZE:-32}"`，只需环境变量不用改脚本；代价只有 2.2–4.4×（bs=8 已 7.4 img/s vs bs=32 的 8.05 img/s，早就 compute-bound），21 次 n=500 的 sweep 从 0.36h 到 0.79h。若真要 per-batch context 信号，另一条路是给 `dataset/imagenet.py` 加类纯净分批选项。见 [[project_covr_crossover_oos_gate_rule]]。
