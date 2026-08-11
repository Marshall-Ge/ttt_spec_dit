---
name: project_covr_budget_probe_500_results
description: "COVR 500 图 budget probe 实测结果（2026-08-03）——inception_conf 首次通过 [a]；uniform 臂在 k8 第一 k6 最后，budget 级 crossover 真实"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8e47eb08-328e-4617-9d27-3b55e741de17
  modified: 2026-08-03T06:06:01.009Z
---

新机器上重跑的 `scripts/sweep_budget_probe.sh`（N_PROMPTS=500, BUDGETS=8,6,4, SEED=42, terminal reward），用 `--metric inception_conf` 分析。reference FID 103.45。

**`[a]` validity 首次通过。** 之前三个 budget 在 pixel MSE 下全挂（Spearman −0.400/−0.200/+0.800）。reference-free 的 Inception 预测熵在 k8/k6 给 Spearman(mean_conf, FID) = **1.000**，k4 给 0.800（back/front 换位，但那里 FID 178–262 图已经坏了）。所以 reward family **没有被关掉** —— 之前的失败是 pixel MSE 这个 proxy 不行，不是 fidelity 奖励不行。Spearman(mean_conf, −IS) 三个 budget 都是 1.000。

**臂的 FID（best→worst）：**
- k8: uniform 96.02 < geometric 100.86 < back_loaded 112.74 < front_loaded 126.84
- k6: back_loaded 117.30 < front_loaded 143.75 < geometric 156.91 < **uniform 174.13**
- k4: uniform 178.45 < front_loaded 212.89 < back_loaded 231.55 < geometric 262.75

**uniform 从 k8 第一掉到 k6 最后（96.02 → 174.13，比 back_loaded 差 56.8）**，cross-sampling σ≈2.4。掩码解释得通：k8 uniform = `[0,1,2,3,14,26,38,49]` 均匀覆盖；k6 uniform = `[0,1,2,3,26,49]` 留下两个 ~23 步空洞，而 k6 back_loaded = `[0,1,2,26,38,49]` 把算力压在后段。**这是 per-budget 查表，不是 per-trajectory context**，救不了 COVR 的前提，但说明 `build_budget_manifest.py --baseline-arm uniform` 这个默认值在 k<8 时是错的。

**budget 确实 binding**：k6 best-arm 117.30 vs k8 best-arm 96.02，差 +21.28。k8 best-arm 比 reference 好 7.43（reference 不是 FID 最优点，这条老结论继续成立）。

**`[b]` 远离量化地板**：spread 21056×/14867× floor，0.0% 的图在 5× floor 以内。

**`[d]` 判据结果需要用新规则重跑** —— 这轮打印的是旧的 1NN label transfer（k8: frac −37.4% p=0.2015；k6: frac −34.7% p=0.009），而那个规则在真 crossover 上也会给负 frac，见 [[project_covr_crossover_oos_gate_rule]]。k6 的"p 显著 + frac 负"在无 crossover 世界只有 2.3% 出现率，所以 **不能判定成纯噪声**。

**换 cost regression 后重跑的结果（两个 metric 都跑了）：**

| metric | k8 | k6 |
|---|---|---|
| inception_conf | frac **+0.4%** p=0.047 → 旧判据说 UTILIZABLE | frac −0.0% p=0.89 |
| inception | frac 0.0% p=0.9405 | frac **+1.0%** p=0.0375 → 旧判据说 UTILIZABLE |

**这两个"通过"都是假的，三条理由：**
1. gate 的 permutation null 有 **0 点质量 0.56–0.60**（打乱 train 行 → 每臂拟合塌回均值 → benefit 恰好 0），所以 p≤0.05 几乎等价于 frac>0，不含量级信息。
2. 构造的无 crossover 世界通过 p 时报 0.67%–1.58%，真 crossover 报 ~21%。**0.4% / 1.0% 落在误报带里，比真信号小 20×。**
3. **两个 metric 互相矛盾** —— 都在 k8/k6 通过 `[a]` (Spearman 1.000)，却各点一个不同 budget。真结构不会这样。

换算成 FID（用各 budget 自己的 loss→FID 斜率）：inception_conf/k8 回收 **0.026–0.284 FID**，inception/k6 回收 **0.122–0.171 FID**，而 2× 同臂 replica floor = **4.8 FID**，差 17×–180×。更狠的是 inception/k8 的**完整 per-image oracle 上限**只有 3.79–9.65 FID，即 0.79× of 2σ —— 连全知选择的天花板都刚够摸到噪声地板。

**结论：假设 A（激进 budget）+ 假设 D（contextual）这条组合在 n=500 上被证伪，不是"信号太弱"而是"oracle 天花板本身就在噪声地板附近"。** 要救只有两条：更多图（缩小 oracle harvest 的 per-image 噪声）或更强的外生特征。不要跑 bandit。

**Why:** 这是第一次有 metric 通过 `[a]`，也是第一次拿到 `[d]` 的实测数字。假设 C（H-step reward）原本挂在"inception 通不过就关掉"的条件上，这个条件没触发，所以 C 仍然开着。

**How to apply:** 判据已经在 `scripts/analyze_crossover.py` 里补了两道量级地板，重跑同一份 PNG 现在会打印 `MARGINAL, NOT UTILIZABLE` 而不是 `CROSSOVER REAL AND UTILIZABLE`。见 [[project_covr_system]]。
