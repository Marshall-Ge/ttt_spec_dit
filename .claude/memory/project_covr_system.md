---
name: COVR 系统全貌
description: COVR (Conservative Online Verification and Reward) 的动机、方法原理、架构组件、实验发现和当前状态
type: project
originSessionId: fa3b35ee-51e4-4571-9e80-8c0ddb0e5e9e
---
## 原始动机

COVR 的核心问题：能否通过在线学习改进 SpecA 的 per-step refresh 决策，而无需重训骨干模型？

关键洞察："Every Refresh Is a Label" —— 每次 full forward 同时产生一个免费的 counterfactual label（Taylor 近似输出 vs 真实 full 输出的差异），可以利用这些标签在线学习更优的 refresh 策略。

**Why:** 原始 SpecA 用固定 decay threshold 做自适应决策，这是一个简单启发式。如果能利用推理过程中自然产生的 label 学习一个更优策略，就能在相同 FLOPs 预算下获得更好的生成质量。

**How to apply:** COVR 是 SpecA/TeaCache 之上的策略选择层，不改变底层加速器逻辑。当涉及加速器组合实验时需要理解 COVR 的存在。

## 方法原理

### 研究历程（已终止的方向）

原始 Phase 0-5 研究计划提出了全上下文在线控制器，但 Gate C 实验证明：
- **one-hot timestep 已是最强风险预测器**（MSE 0.03755, recall 0.827）
- full-context ridge regression 仅增加 ~4.84% MSE 改善，无 recall 收益
- 结论：STOP 全上下文在线控制器方向

### 当前方向：Conservative Template Bandit

**核心算法**：epsilon-greedy bandit 在等 FLOPs 的 refresh template 之间选择

1. **Manifest**：一组等 FLOPs 的 refresh_mask（例如 4 个 template，每个有 14/50 步做 full forward，但分布位置不同）
2. **Bandit**：每个 trajectory 开始时选择一个 template（arm），epsilon 概率随机探索
3. **Reward**：trajectory 结束后收集 terminal fidelity loss（最后一步的 Taylor vs full MSE）
4. **Safety Table**：per-(template, step) 的 UCB 安全边界，过滤掉不安全的 arm
5. **State Persistence**：每个 trajectory 后序列化到 JSON，支持断点续跑

### 方法无关扩展

通过 `AccelerationStrategy` + `StrategyManifest` 抽象，bandit 可以在不同加速方法间选择（如多个 TeaCache threshold），不限于 SpecA template。

## 关键实验发现

### 50k 速度优化实验（CFG=4.5, safety=0, 50,000 张全量，2026-07-29）

| 方法 | FID ↓ | IS ↑ | FLOPs(T) | img/s | skip% |
|------|-------|------|----------|-------|-------|
| Baseline (full) | 24.84 | 453.9 | 11.867 | 3.95 | — |
| TeaCache | 24.73 | 458.4 | 7.749 | 8.24 | 34.0% |
| Adaptive SpecA | 23.96 | 436.7 | 3.371 | 7.69 | 73.3% |
| COVR Bandit | **17.48** | 351.2 | **3.087** | **8.05** | 74.0% |

**重要结论**：
1. Bandit FID 17.48 最优（vs baseline -7.36, vs TeaCache -7.25, vs SpecA -6.48）
2. **IS 退化是 FID 改善的代价**：TeaCache IS 458.4（最优保持），SpecA 436.7（中度退化），Bandit 351.2（-19.6%）
3. 四项方法形成清晰的 trade-off 光谱：TeaCache 保守 skip 保 IS → SpecA 激进 skip 改善 FID → Bandit 极致 FID 但牺牲 IS
4. Bandit 92.1% 收敛至 `timestep_prior` arm —— bandit 的角色是"发现"而非"学习"更优模板
5. Terminal fidelity reward 成功复现 arm 间 FID 排序（H-step defect 与 FID 反相关）
6. COVR 在线开销 <0.9%（bandit control + terminal sentinel + state persist）

### 50k 初版实验（CFG=4.5, safety=10%, 47,952 张 evaluation, 2026-07-25，历史参考）

| 方法 | FID ↓ | IS ↑ | candidate FLOPs | img/s |
|------|-------|------|-----------------|-------|
| Baseline (full) | 24.90 | 451.75 | 11.87T | 4.17 |
| Adaptive SpecA | 24.00 | 434.43 | 3.37T | 8.58 |
| COVR Bandit | **17.62** | 353.27 | 3.09T | 4.78 (candidate) |

此轮实验 safety shadow 消耗 0.87T FLOPs（22% online），导致 Bandit 在线速度 4.17 img/s 仅为 SpecA 的 48.6%。后续速度优化实验（safety=0）确认 safety 是唯一瓶颈，消除后 Bandit 速度反超 SpecA。

### 2048 张 Profile Validation

| 方法 | FID | img/s | candidate img/s |
|------|-----|-------|-----------------|
| Baseline | 56.66 | 3.94 | — |
| Adaptive SpecA | 55.75 | 7.63 | — |
| Bandit | **49.78** | 6.87 | **8.05** |

确认：bandit candidate 速度（8.05）实际**快于** adaptive SpecA（7.63），safety overhead 是唯一速度瓶颈。

### 速度瓶颈诊断（最新实验）

4 组对照证实：
- `wall_s_safety_total = 0` 时（chain-threshold=5 有效屏蔽），bandit candidate 8.04 img/s > adaptive SpecA 7.64 img/s
- dispatch/persist 开销 < 0.2s/trajectory，可忽略
- **50k 实验 4.78 img/s 的问题是 safety shadow 累积的额外 full forward**

## 当前状态和待解决问题

### 已完成
- ConservativeTemplateBandit 完整实现（epsilon-greedy + safety + state persistence）
- AccelerationStrategy / StrategyManifest 方法无关抽象
- SpecA + TeaCache 双方法 dispatch
- Terminal fidelity reward
- 完整测试套件

### 未完成
1. TeaCache terminal fidelity 仍需 expensive full-baseline fallback（SpecA 有 cheap 1-step path）
2. IS 下降未解决 —— 需要多目标 reward 或 Pareto 选择
3. Arm 过度收敛到 timestep_prior（epsilon=0.1 下）—— 可能需要更好的初始化或 exploration
4. TeaCache 路径的完整端到端 GPU 验证

## 文件路径索引

### 核心源码
- `accelerators/covr.py` — 数据模型、在线学习基础设施、recorder（975 行）
- `accelerators/covr_bandit.py` — Template Bandit、策略、manifest、safety table（1056 行）
- `accelerators/strategy_dispatch.py` — 方法无关 dispatch（79 行）
- `experiments/covr_analysis.py` — 离线分析、Gate 评估、manifest 构建（1184 行）

### 集成点
- `main.py:131-171` — COVR CLI 参数定义
- `run_dit.py:1680-2230` — COVR 在生成循环中的集成

### 脚本
- `scripts/build_covr_manifest.py` — 从 audit JSONL 构建 manifest
- `scripts/analyze_covr.py` — 离线 Gate 评估
- `scripts/run_covr_template_experiment.sh` — 全流程编排（audit → manifest → eval）
- `scripts/run_covr_profile_validation.sh` — 4-GPU 并行 profile
- `scripts/benchmark_bandit_speed.sh` — 速度瓶颈诊断

### 文档
- `docs/covr_bandit_experiment_record.md` — 主实验记录
- `docs/covr_profile_validation_20260727.md` — Profile validation 结果
- `.claude/covr_spec_research_plan.md` — 原始研究计划
- `.claude/covr_spec_research_conclusion.md` — 研究结论（Gates A/B/C）
- `.claude/covr_cache_decision_directions.md` — Post-mortem 方向分析
