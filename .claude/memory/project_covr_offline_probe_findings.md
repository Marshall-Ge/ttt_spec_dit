---
name: project_covr_offline_probe_findings
description: COVR 离线探针结论（2026-07-31）——per-image crossover 真实存在但 causal context 不可用；audit 只测到 cache distance 1-4，激进 mask 的 surrogate 无效
metadata: 
  node_type: memory
  type: project
  originSessionId: 8e47eb08-328e-4617-9d27-3b55e741de17
  modified: 2026-07-31T10:06:48.431Z
---

在唯一的 audit（`output/covr_test/covr/events_20260725-140706-seed42.jsonl`，64 图 / 8 轨迹 / 44 个 observed step）上做的离线探针，用于在花 GPU 前判断"有没有东西可 discover"。见 [[project_covr_system]]。

**四条结论（都配了 permutation null）：**

1. per-image 结构真实。去掉 (traj, step) cell 均值后残差 sd=0.44；lag-1 autocorr +0.828 vs shuffled null −0.046±0.033 (p=0)。
2. **不是稳定 trait，而是 phase**。corr(early 4-15, late 30-43) 的 per-image profile = **−0.481**，permutation null +0.003±0.129, p=0.0000 → 早期难的图后期反而容易。同窗口 split-half 只有 +0.410（可靠性天花板），窗口相关矩阵从 +0.63 单调翻到 −0.52。
3. equal-FLOPs arm **crossover 存在**：front/uniform/back/timestep_prior 四臂，oracle per-image 选择比最佳单臂好 2.27 nats，null 1.11±0.07 (p=0)；odd→even 步 out-of-sample 选择拿到 54% oracle (p=0)。
4. **但 causal context 不可用**：只用 step≤12 的特征 LOO 预测 step≥25 的 arm cost，realized −12% oracle，p=0.84（和随机选一样）。in-sample R²=0.145 vs overfit null 0.095±0.051 (p=0.166)。同一个 target 用 late-window 特征 in-sample R²=0.433 → target 本身可学，只是**早期观测里没有它的信息**。

**Why:** 这直接决定假设 D（contextual bandit）的前提。crossover 存在（有 discover 空间）但前 12 步看不出该选哪臂 —— 与历史"92% 收敛到 timestep_prior"一致：bandit 没有可用 context，只能退化成选全局最优臂。

**How to apply:** 别在现有 audit 上再堆 contextual bandit。要么改 sentinel/reward 让决策点推后到信息出现之后（假设 C），要么先补 audit：n≥79 才有 80% power 检测 R²=0.10 的 causal 效应。另外 audit 只测过 `distance_since_refresh` ∈ 1-4，而 front_loaded/back_loaded 掩码的最大 cache distance 是 29/25 —— **激进 budget（假设 A）的 arm cost 无法用这份 audit 外推**，必须真跑 GPU。
