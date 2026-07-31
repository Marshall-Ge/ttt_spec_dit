# COVR 方法综述

> 日期：2026-07-29  
> 项目：TTT-DiT / DiT-2-256  
> 全称：COVR — Conservative Online Verification and Reward for Speculative Diffusion

## 1. 动机

### 1.1 问题背景

扩散模型推理的主要瓶颈是去噪循环中重复执行的 Transformer forward pass。以 DiT-2-256 为例，50 步去噪 × 28 blocks × CFG doubling = 大量冗余计算。现有加速方法（SpecA、TeaCache）通过缓存和近似跳过部分计算，但它们的"何时跳过"决策是**固定启发式**：

- **SpecA**：用 `base_threshold * decay^progress` 的衰减公式决定每步是 full forward 还是 Taylor 近似
- **TeaCache**：用调制信号的 relative L1 差异 + poly4 rescale 决定是否跳过整个 block stack

这些固定策略无法利用推理过程中产生的反馈信息。

### 1.2 核心洞察

**"Every Refresh Is a Label"** — 每次 SpecA 执行 full forward 时，自然产生一个免费的反事实标签：

```
Taylor 近似输出 eps_A  vs  真实 full 输出 eps_F
→ counterfactual defect d_t = ||S(x_t, eps_A) - S(x_t, eps_F)|| / ||S(x_t, eps_F) - x_t||
```

这些标签可以用来在线学习更优的刷新策略，而无需：
- 重训骨干模型
- 额外的标注数据
- 离线训练流程

### 1.3 COVR 想解决什么

在**相同 FLOPs 预算**下，通过在线学习找到比固定启发式更优的"何时刷新"决策，从而改善生成质量（FID）或在相同质量下减少计算量。

## 2. 方法原理

### 2.1 研究历程

COVR 经历了两个阶段：

**Phase 1: Full-context 在线控制器（已终止）**

原始设想是训练一个 contextual bandit/ridge regression 控制器，输入丰富的 context（timestep、logSNR、Taylor term norms、cache distance 等），输出 refresh/accept 决策。

Gate C 实验证明此路不通：
- one-hot timestep 单特征已是最强风险预测器（MSE 0.03755, recall 0.827）
- full-context ridge 仅增加 ~4.84% MSE 改善，无 recall 收益
- 结论：不值得维护复杂的在线控制器

**Phase 2: Conservative Template Bandit（当前方向）**

转向更简单但有效的方案：不学习 per-step 决策，而是在**预定义的等 FLOPs refresh template 之间选择**。

### 2.2 Conservative Template Bandit 算法

核心思路：把"50 步中哪些步做 full forward"编码为一个固定 mask（template），用 bandit 在多个候选 template 之间学习选择最优的那个。

**组件：**

1. **Manifest**：一组等 FLOPs 的 refresh template
   - 例如 4 个 template，每个都有 14/50 步做 full forward
   - 步数相同但位置不同（早期密集 vs 均匀分布 vs 后期密集等）
   - 从 audit 数据中通过贪心搜索构建

2. **Bandit (epsilon-greedy)**：
   - 每个 trajectory（一张图的完整去噪）开始时选择一个 template
   - 以 epsilon 概率随机探索，否则选当前最优 arm
   - 使用增量均值更新（log1p 变换后 Welford 更新）

3. **Reward: Terminal Fidelity Loss**：
   - 最后一步额外执行一次 full forward
   - 比较 template 的近似输出和 full 输出的 MSE
   - 比 H-step defect 更好地反映最终图像质量

4. **Safety Table**（可选）：
   - per-(template, step) 的 UCB 安全边界
   - 过滤不满足安全约束的 arm
   - 代价：每步额外 full forward（10% sample rate → 22% FLOPs 增加）

5. **State Persistence**：
   - 每个 trajectory 后序列化 bandit 统计量到 JSON
   - 支持断点续跑和跨 session 继承

### 2.3 方法无关扩展

通过 `AccelerationStrategy` + `StrategyManifest` 抽象，bandit 不限于 SpecA template 选择，也可以在不同 TeaCache threshold 之间选择：

```python
AccelerationStrategy(
    strategy_id="teacache_thresh_0.35",
    method="teacache",
    params={"rel_l1_thresh": 0.35},
    modeled_flops=...,
)
```

## 3. 与已有方法的区别

### 3.1 对比矩阵

| 维度 | Baseline (Full DDIM) | TeaCache | Adaptive SpecA | COVR Bandit |
|------|---------------------|----------|----------------|-------------|
| 决策粒度 | 无（全算） | per-step（全局阈值） | per-block per-step（decay 阈值） | per-trajectory（template 选择） |
| 决策依据 | — | 调制信号相似度 | Taylor vs full 误差探测 | 在线学习的 arm 统计量 |
| 是否学习 | 否 | 否 | 否 | **是（epsilon-greedy bandit）** |
| 缓存对象 | — | blocks 残差 | 各子模块 Taylor 级数 | 同 SpecA（选 template 后用 SpecA 执行） |
| 跳过方式 | — | 整个 block stack | 逐 block Taylor 近似 | 由 template mask 决定 |
| 是否需要 audit | 否 | 否 | 否 | **是（构建 manifest 需要 audit 数据）** |
| 在线开销 | — | ~0 | per-block probe | bandit 选择 + state persist（可忽略） |

### 3.2 COVR 与 Adaptive SpecA 的关系

COVR 不是 SpecA 的替代品，而是**SpecA 之上的策略选择层**：

```
COVR Bandit (选择 template)
    ↓
SpecA (按 template 的 refresh_mask 执行 full/Taylor)
    ↓
DiT forward (block 级计算)
```

Adaptive SpecA 用 `base_threshold * decay^progress` 动态决定每步是否 refresh。COVR Bandit 把这个决策替换为预定义的固定 mask，但在多个 mask 之间在线学习选择。

### 3.3 为什么 COVR 的 FID 更好

Adaptive SpecA 的衰减公式是一个**全局一刀切**的启发式——所有图片、所有类别用相同的 threshold 曲线。而 COVR Bandit 选择的 `timestep_prior` template 可能恰好在 FID 敏感的时间步位置做了更好的 refresh 分配。

关键证据：固定使用 `timestep_prior` 的 FID（48.75）就已经优于 adaptive SpecA（55.54），说明这不是 bandit 学习的功劳，而是**template 本身的刷新分布更优**。Bandit 的贡献在于**发现并稳定选择了更优的 template**。

### 3.4 与 VFL 的区别

| | COVR | VFL |
|---|---|---|
| 改进对象 | 缓存**决策**（何时 refresh） | 缓存**决策**（阈值校准）+ backbone 轨迹 |
| 学习方式 | epsilon-greedy bandit | EMA 阈值 + LoRA 异步训练 |
| 是否改模型权重 | 否 | 是（LoRA） |
| 粒度 | per-trajectory | per-step per-layer |
| 当前状态 | **有效**（FID -7.3） | **无效**（target 信号源缺陷导致 loss=0） |

### 3.5 与 TTT 的区别

| | COVR | TTT |
|---|---|---|
| 改进对象 | 缓存**决策** | 缓存**内容**（hidden state correction） |
| 是否需要额外模型 | 否 | 是（SessionAdaLNModulator, 0.92M） |
| 加速方法依赖 | SpecA (或通用) | TeaCache only |
| 训练时机 | 推理中 arm 统计更新 | 推理中 calc step 蒸馏训练 |
| DiT/PixArt 支持 | 仅 DiT（当前） | 仅 DiT |

## 4. 实验结果汇总

### 4.1 50k CFG=4.5 速度优化实验（最新，2026-07-29）

> safety_sample_rate=0, sentinel_rate=0.02, chain_threshold=5

| 方法 | FID ↓ | IS ↑ | FLOPs(T) | img/s |
|---|---:|---:|---:|---:|
| Baseline | 24.84 | 453.9 | 11.867 | 3.95 |
| Adaptive SpecA | 23.96 | 436.7 | 3.371 | 7.69 |
| **COVR Bandit** | **17.48** | 351.2 | **3.087** | **8.05** |

**相对 Adaptive SpecA：FID -6.48, 速度 +4.7%, FLOPs -8.4%**

### 4.2 50k No-CFG 实验（2026-07-28）

| 方法 | FID ↓ | IS ↑ | FLOPs(T) | img/s |
|---|---:|---:|---:|---:|
| Baseline | 7.28 | 121.8 | 11.867 | 7.47 |
| Adaptive SpecA | 8.04 | 118.9 | 3.089 | 14.45 |
| COVR Bandit | 8.22 | 117.4 | 2.849 | 15.08 |

无 CFG 下 Bandit FID 优势消失，收益仅为速度 +4.4%。

### 4.3 原始 50k CFG=4.5 实验（safety=10%, 2026-07-25）

| 方法 | FID ↓ | IS ↑ | FLOPs(T) | img/s |
|---|---:|---:|---:|---:|
| Baseline | 24.90 | 451.7 | 11.867 | 4.17 |
| Adaptive SpecA | 24.00 | 434.4 | 3.371 | 8.58 |
| COVR Bandit | **17.62** | 353.3 | 3.087 | 4.78 |

此实验暴露了 safety shadow 的速度瓶颈（4.78 vs 8.58 img/s）。

### 4.4 结果解读

| 发现 | 含义 |
|---|---|
| FID -7.3（CFG=4.5） | Bandit 选择的 template 在 CFG 放大下显著优于 adaptive threshold |
| IS -19.6%（CFG=4.5） | 质量改善偏向分布匹配，牺牲了单图辨识度/类别均匀性 |
| 92% 选 timestep_prior | Bandit 快速收敛，exploration 有限 |
| Safety=0 后速度超 SpecA | Safety shadow 是唯一速度瓶颈 |
| No-CFG 下 FID 无优势 | Bandit 的 FID 收益依赖 CFG 放大效应 |

## 5. 当前研究状态

### 5.1 已完成

- ConservativeTemplateBandit 完整实现（epsilon-greedy + safety + state persistence）
- AccelerationStrategy / StrategyManifest 方法无关抽象
- SpecA + TeaCache 双方法 dispatch
- Terminal fidelity reward（替代原始 H-step defect）
- 速度瓶颈诊断与解决（safety shadow 消除）
- 50k 规模 CFG / No-CFG 对比实验
- 完整测试套件

### 5.2 已解决的问题

1. **速度瓶颈**：safety shadow 导致 Bandit 比 SpecA 慢 51%。解决：`safety_sample_rate=0`，速度反超 SpecA 4.7%。
2. **Reward 方向错误**：原始 H-step defect 与 FID 反相关。解决：改用 terminal fidelity loss。
3. **Full-context 控制器不 work**：Gate C 证明 one-hot timestep 即是最强基线。解决：转向 template bandit，不学习 per-step 决策。

### 5.3 未解决的问题

| 问题 | 严重程度 | 可能方向 |
|---|---|---|
| IS 下降 19.6% | 高 | 多目标 reward（FID + IS 联合）；Pareto 选择 |
| 92% 收敛到 timestep_prior | 中 | 更好的初始化；提高 epsilon；UCB 替代 epsilon-greedy |
| TeaCache terminal fidelity 未接入 | 中 | 需绕过 `method == "speca"` 限制 |
| 无 CFG 下无 FID 收益 | 低 | 可能需要 no-CFG 专属 manifest |
| Safety 完全禁用的长期风险 | 低 | 轻量级替代方案（离线校准 + 稀疏 sentinel） |

### 5.4 关键开放问题

1. **IS 退化是否可修复**：如果 `timestep_prior` 本身就导致 IS 下降，那么多目标 reward 可能让 bandit 选择其他 template 从而牺牲 FID。需要实验验证 FID-IS Pareto 前沿。

2. **Template 本身 vs Bandit 学习的贡献**：固定 `timestep_prior` 的 FID 就已经优于 adaptive SpecA，Bandit 的价值更多在"发现"而非"改进"。论文叙事需要区分。

3. **跨模型泛化**：当前只在 DiT-2-256 验证。PixArt-XL-2 的不同架构（3 子模块、T5 encoder）是否有类似的 template 优势未知。

4. **与步数的关系**：当前所有实验都是 50 步。更少步数下 template 的优势空间可能缩小。

## 6. 文件索引

### 核心源码
- `accelerators/covr.py` — 数据模型、在线学习基础设施、recorder
- `accelerators/covr_bandit.py` — Template Bandit、策略、manifest、safety table
- `accelerators/strategy_dispatch.py` — 方法无关 dispatch

### 集成点
- `main.py:131-171` — COVR CLI 参数
- `run_dit.py:1680-2230` — COVR 在生成循环中的集成

### 实验脚本
- `scripts/run_covr_50k_cfg.sh` — CFG=4.5 速度优化实验
- `scripts/run_covr_50k_nocfg.sh` — No-CFG 实验
- `scripts/benchmark_bandit_speed.sh` — 速度瓶颈诊断
- `scripts/build_covr_manifest.py` — 从 audit 构建 manifest

### 文档
- `docs/covr_bandit_experiment_record.md` — 主实验记录（含所有数据）
- `.claude/covr_spec_research_plan.md` — 原始研究计划
- `.claude/covr_spec_research_conclusion.md` — Gate A/B/C 结论
- `.claude/covr_cache_decision_directions.md` — Post-mortem 方向分析
