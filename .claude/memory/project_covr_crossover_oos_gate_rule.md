---
name: project_covr_crossover_oos_gate_rule
description: "COVR [d] 判据两次修正——先换掉 1NN label transfer，再补两道量级地板；p<=0.05 单独不成立因为 permutation null 有 0 点质量（2026-08-03）"
metadata:
  node_type: memory
  type: project
  originSessionId: 8e47eb08-328e-4617-9d27-3b55e741de17
  modified: 2026-08-03T06:05:37.250Z
---

`scripts/analyze_crossover.py` 的 `[d]` out-of-sample 判据（唯一的 verdict gate）在同一天被修了两次。

## 第一次：规则换成 per-arm cost regression

原来用 **1NN label transfer**（train 半区取每张图 argmin 臂作标签，用 reference 图特征找最近邻迁移）。300 trials 标定：真 interaction=0.8 的世界里 **200/200 次给负 frac，95% 的 trial 里 p≤0.05** —— 迁移 argmin 标签强迫对"哪臂赢"下确定性赌注，臂间距只有一个噪声宽度时这个 argmin 基本是噪声。替代规则 `_cost_regression`：每臂用 reference 特征（gradient energy / contrast / luminance）对 cost 做最小二乘，取预测 argmin；特征无信息时每臂拟合塌回自身均值 → argmin = 全局最优臂 → frac 恰好 0。四个对抗构造 n=500/250 误报率全 0.00，强 interaction 功效 0.95/0.83。

## 第二次：p≤0.05 单独不成立，补两道量级地板

**根因（结构性，不是标定问题）**：permutation 打乱 train 行会摧毁 feature→cost，于是每臂拟合塌回自身均值、argmin 变成全局最优臂、**permuted benefit 恰好等于 0**。实测 null 在 |benefit|≤1e-12 的质量 **0.56–0.60**，且 `P(null>0)=0.0000`。所以 `p≤0.05` 几乎等价于 `frac>0`，不含任何量级信息。

**false positive 时报多大（400 trials, n=500）：**

| 世界 | 通过率 | 通过时 frac 中位数 |
|---|---|---|
| 无 crossover + informative 特征 | 0.003 | 0.14% |
| 无 crossover + 噪声特征 | 0.007 | 0.67% |
| 无 crossover + 高噪 | 0.048 | 1.58% |
| 弱 crossover 0.2 | 0.102 | 1.31% |
| **真 crossover 0.8** | 0.990 | **20.90%** |

真信号比误报大 ~20×，所以两道地板：
- **相对**：`_REL_FLOOR = 0.05`（oracle gain 的 5%）。真 interaction 功效仍 0.963，高噪误报从 0.048 降到 0.003。
- **绝对**：用该 budget 自己的 4 个 (mean loss, FID) 点算 loss→FID 斜率（取最大斜率，最宽容），要求 > 2× 同臂 replica FID spread（`_noise_fid_spread` 读 `equalflops/noise_*/results.json`，缺失时回落 2.0 FID 与 `analyze_teacache_sweeps.py` 对齐）。

新增 `_loss_to_fid_slopes` / `_noise_fid_spread` / `_gate` / `_render_gate`；渲染成 `gate check 1/3` / `2/3` / `3/3` 三行并说明各自防什么；RECOMMENDATION 加 `MARGINAL, NOT UTILIZABLE` 分支和 `AND THE TWO PROXIES DISAGREE` 分支。测试 `tests/test_crossover_oos_gate.py` 14 个（含 point-mass 验证、两条实测行必须被拒、端到端 main() 走 marginal 分支），全套 127 passed。

**Why:** 第一次修完后两个 metric 各"通过"一个 budget（见 [[project_covr_budget_probe_500_results]]），差点又给出 GO。这两道地板是唯一能把 0.4%/1.0% 和 21% 分开的东西。

**How to apply:** 读 `gate check 3/3` 那行的 FID 数字 —— 判据现在直接告诉你回收了多少 FID、地板是多少。功效缺口仍在：弱 interaction 只有 0.19 功效，判 NOT utilizable 时要区分"无结构"和"n=500 + 3 特征不够"。见 [[project_covr_crossover_gate_design]]、[[project_covr_offline_probe_findings]]。
