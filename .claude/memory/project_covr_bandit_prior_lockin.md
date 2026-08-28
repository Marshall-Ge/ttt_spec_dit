---
name: project_covr_bandit_prior_lockin
description: COVR bandit 的 92.1% 收敛是 epsilon-greedy 无学习地板，不是学习结果；alternative_prior_penalty=0.25 比 reward 量级大 4000-12000×（2026-08-03）
metadata: 
  node_type: memory
  type: project
  originSessionId: 8e47eb08-328e-4617-9d27-3b55e741de17
  modified: 2026-08-03T07:39:05.268Z
---

`accelerators/covr_bandit.py:770-771` 的两个先验默认值 **没有任何 CLI 能覆盖**（grep 确认 `main.py` 无对应参数）：

```python
baseline_prior_count: int = 8            # baseline 起始 (count=8, mean=0.0)
alternative_prior_penalty: float = 0.25  # 其他臂起始 (count=1, mean=0.25)
```

reward 走 `_ArmLossStats.update()` = `log1p(terminal_fidelity_loss)` 的 running mean。

**从 50k 实验发表的臂均值反解出真实 reward 量级**（`docs/covr_bandit_experiment_record.md:170-176`，用真实 `_ArmLossStats` 逐点复算验证，四个臂全部 match 到 2e-6）：

| arm | 发表均值 | 收到 reward 数 | 反解 log1p(MSE) |
|---|---|---|---|
| template_01 | 0.050016 | 4 | **2.000e-05 ← 实际最优** |
| template_03 | 0.062526 | 3 | 3.467e-05 |
| timestep_prior | 0.000041 | 69 | 4.575e-05 ← baseline，拿了 92.1% 的 pull |
| template_02 | 0.125030 | 1 | 6.000e-05 |

**penalty 0.25 是最差臂 reward 的 4167×。** 于是任何 alternative 想超过 baseline 需要 `0.25/|log1p gap|` 个 reward：template_01 要 **11,903 个 reward**，在 `sentinel_rate=0.05` + `eps=0.1`/4 臂下等于 **3.05 亿张图**（bs=32）。

**92.1% 不是收敛，是 epsilon-greedy 的无学习地板 0.9+0.1/4 = 92.5%。** 用真实 bandit 类在真实 reward 量级上跑（60 次重复）：

| 配置 | 最优臂份额 | baseline 份额 |
|---|---|---|
| 500 图 | 2.9% | 92.1% |
| 50k 图, sentinel 0.05 | 2.5% | 92.5% |
| 50k 图, sentinel 1.0 | 2.5% | 92.5% |

sentinel 从 0.05 拉到 1.0（20× reward）**完全不动**。三个 alternative 的实测份额 2.6/2.2/3.1% 就是 eps/4=2.5%。

**这解释了 FID 17.48 的来源**：不是 bandit 学出来的，是 `timestep_prior` 这个 baseline 臂本身好（`docs/covr_method_summary.md:130` 已经说固定用它 FID 48.75 就优于 adaptive SpecA 55.54）。bandit 只是把 92.5% 的算力压在了 manifest 作者手选的 baseline 上。而实测 reward 说 **template_01 才是最优**，bandit 从没往那边走。

**修法（三选一，都是一行）：**

| 改动 | 5000 图最优臂份额 | 50k 图（sentinel 0.05） |
|---|---|---|
| 现状 penalty=0.25 | 2.6% | 2.5% |
| penalty=1e-4（2× reward） | 31.2% | 13.1% |
| penalty=5e-5（1× reward） | 66.4% | 52.3% |
| reward×1e4 + penalty 0.25 | 83.4% | 74.5% |

penalty 必须和 reward 同量级，或者把 reward 标准化。**reward 标准化更稳** —— penalty 写死一个绝对值会在换 metric / 换 budget 时再次失配（k6 的 mean loss 比 k8 小 10×，见 [[project_covr_budget_probe_500_results]]）。

**Why:** 之前把 92.1% 读成"bandit 快速收敛，exploration 有限"（`docs/covr_method_summary.md:192`），并据此把 bandit 的角色定义为"发现而非学习"。这个读法是错的 —— 它从来没有学习过，任何 arm-spread 实验只要用默认 penalty 就测不到 bandit 的学习能力，只测到 manifest 作者的手选。

**How to apply:** 跑任何 bandit 实验前先确认 penalty 和 reward 同量级；否则结果只反映 `baseline_strategy_id` 的选择。这也让 `build_budget_manifest.py --baseline-arm uniform` 的默认值更危险（k6 时 uniform 是最差臂，FID 174.13 vs back_loaded 117.30），因为 92.5% 的算力会压在它上面。见 [[project_covr_system]]。
