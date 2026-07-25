# COVR-Spec 阶段性研究结论

> 日期：2026-07-21
> 状态：停止当前 full-context 在线控制器主线，保留 shadow auditor 作为实验基础设施。
> 原始规格：`.claude/covr_spec_research_plan.md`
> 实验产物：`output/c2i_dit_imagenet_covr-shadow/`
> 保护项：不得修改、覆盖或回滚 `run.log`。

## 1. 实验配置

- 模型：DiT-2-256
- 数据集：ImageNet
- 方法：SpecA
- 图片数：1,000
- batch size：32
- denoising steps：50
- guidance scale：1.0
- seed：42
- COVR 模式：paired Taylor/full one-step shadow audit
- 事件文件：`output/c2i_dit_imagenet_covr-shadow/covr/events_20260721-141029-seed42.jsonl`
- summary：`output/c2i_dit_imagenet_covr-shadow/covr/summary_20260721-141029-seed42.json`
- 原始 gate 输出：`output/c2i_dit_imagenet_covr-shadow/covr/gates.json`

## 2. Shadow purity 结论

普通 SpecA 与 COVR shadow 的生成质量和主轨迹统计完全一致：

| 指标 | SpecA | SpecA + COVR shadow |
|---|---:|---:|
| FID | 63.6191 | 63.6191 |
| IS mean | 51.6837 | 51.6837 |
| IS std | 2.1997 | 2.1997 |
| full steps | 390 | 390 |
| Taylor steps | 1,210 | 1,210 |
| skip ratio | 0.75625 | 0.75625 |
| accelerated FLOPs | 3.1127 T | 3.1127 T |
| probe full blocks | 826 | 826 |

结论：shadow full forward、scheduler counterfactual transition 和 JSONL 写入没有改变 SpecA 决策或生成结果。Phase 0 shadow purity 通过。

性能数据只用于说明采集成本：

| 模式 | wall time | throughput |
|---|---:|---:|
| Baseline | 2.8539 s | 10.95 img/s |
| SpecA | 1.5368 s | 20.33 img/s |
| SpecA + COVR shadow | 4.4643 s | 7.00 img/s |

`flops_accel_T` 不包含额外 shadow full forward，不能作为 shadow 采集模式的真实 FLOPs。

## 3. 数据完整性

- JSONL 行数与 summary 一致：37,808
- 文件大小：约 34 MB
- 样本数：1,000
- 类别数：632
- batch trajectories：32
- 每个样本包含 37 或 38 个 audited Taylor steps
- session、version key、schema version 均唯一
- 实际唯一 `(trajectory_id, step_idx)` context：1,210
- 1,172 个 context 各复制给 32 个样本，38 个 context 各复制给 8 个样本

one-step normalized defect：

| 统计量 | 值 |
|---|---:|
| mean | 0.3115 |
| median | 0.1419 |
| p90 | 0.6600 |
| p95 | 1.0694 |
| p99 | 3.3569 |
| max | 5.0249 |

step 49 的均值为 3.321。当前事件未分别保存 defect numerator 和 scheduler transition denominator，无法判断末步异常是 Taylor 误差突增还是归一化分母缩小。

## 4. Gate 复核结论

正式研究判断：

```text
Gate A: INSUFFICIENT DATA
Gate B: PROVISIONAL PASS, one-step target only
Gate C: FAIL under a competitive timestep baseline
Overall: STOP
```

### 4.1 Gate A

字段覆盖：

- `one_step_defect`：37,808
- `local_probe_error`：25,808
- `h_step_defect`：0
- `terminal_intervention_gain`：0
- 完整 local/one-step/terminal paired samples：0

因此 Gate A 必须停止。

当前 local probe 即使只对 one-step defect 也无有效排序能力：

| 指标 | 值 |
|---|---:|
| per-sample Spearman | -0.0432 |
| AUROC | 0.5679 |
| top-decile lift | 0.9003 |
| batch-mean Spearman | -0.0299 |

826 个 batch-level probe points 只有 44 个不同 probe 值，并被复制给 batch 中所有样本。不能把当前 probe 当成 per-sample terminal risk signal。

### 4.2 Gate B

基于 one-step defect 的 full-information oracle 在 5/5 个预算点获胜：

| refresh budget | relative oracle headroom |
|---:|---:|
| 5% | 14.3% |
| 10% | 30.8% |
| 20% | 46.0% |
| 30% | 52.7% |
| 40% | 60.5% |

该结果说明 timestep/refresh 决策存在可利用结构，但只针对 one-step transition target。它不能替代 terminal quality evidence，因此只能记为 provisional pass。

### 4.3 Gate C

原始 `gates.json` 报告 Gate C pass，但该结论不可接受。

问题一：当前 fold 按 class/sample 分组，同一个 batch-step context 会跨训练折和测试折：

- 1,172 个 context 横跨全部 5 folds
- 38 个 context 横跨 3 folds

问题二：原 timestep-only baseline 仅使用 `[1, progress]`，而 full context 包含 logSNR 和 scheduler step size 等 timestep 的非线性变换，因此 baseline 过弱。

trajectory-held-out 公平消融：

| 模型 | MSE | top-risk recall |
|---|---:|---:|
| linear timestep | 0.32504 | 0.348 |
| rich timestep | 0.12554 | 0.279 |
| current full context | 0.07752 | 0.716 |
| one-hot timestep | 0.03755 | 0.827 |
| one-hot timestep + cache | 0.03573 | 0.822 |

结论：

- one-hot timestep 显著优于当前 full-context 模型。
- cache 特征在 one-hot timestep 上提供约 4.84% MSE 改善，并在 5/5 trajectory folds 获胜。
- cache 特征没有改善 top-risk recall，反而从 0.827 降至约 0.822。
- 证据只支持弱回归增益，不支持 full-context 风险排序器通过 Gate C。

## 5. 已识别的数据与实现问题

1. Context 是 batch-step 粒度，defect 是 per-sample 粒度，37,808 条事件只有 1,210 个独立 context。
2. `previous_defect` 由 recorder 逐事件更新，但下一 batch context 使用上一批最后一个样本的 defect，不是有效的 per-sample history。
3. local probe 是 batch scalar，并被复制给 batch 内每个样本。
4. guidance scale 为 1.0，本次不执行 CFG，因此 `cfg_draft_disagreement` 全零是配置结果，不是采集故障。
5. normalized defect 未保存 numerator/denominator，无法诊断末步重尾。
6. `run_all_gates()` 在 Gate A 失败时仍先计算 B/C，再将总体 decision 设为 stop；B/C 输出容易被误读为正式通过。
7. 只有一个 seed/session，不能证明跨 session 泛化或 prequential 可学习性。

## 6. 研究决策

停止继续扩大当前配置的数据量，不启用完整 COVR 在线策略，不让 policy 改变真实 SpecA 轨迹。

保留以下资产：

- paired shadow full forward
- state-isolated scheduler counterfactual transition
- scalar JSONL recorder
- inference version isolation
- offline analysis框架
- hard budget、ridge/UCB 和 IPW 原型

这些资产可作为新方向的评估基础设施，但不是当前在线飞轮已经成立的证据。

## 7. 推荐新方向

优先探索 timestep-conditioned 静态或分段预算调度，而不是 full-context 在线学习控制器。

核心依据：

- one-hot timestep 是目前最强风险预测基线。
- production SpecA 已实现约 1.86x wall-time speedup。
- Gate B 表明固定预算下仍有显著 oracle headroom。
- cache context 只表现出小幅增量 MSE 价值。

建议候选方向：

1. **Timestep-aware static schedule**：直接学习或搜索固定 refresh step mask。
2. **Segmented budget allocation**：按 early/mid/late 或 scheduler dynamics 分配 full-step budget。
3. **Sparse terminal sentinels**：只在少量候选 step 上执行 H-step/terminal branch，用于校准静态 schedule，而非训练常驻在线控制器。
4. **Scheduler-aware defect target**：分开建模 transition numerator 和 denominator，避免末步归一化伪高风险。

首个新方向必须与以下基线比较：

- 当前静态 SpecA
- periodic refresh
- random budget-matched refresh
- one-hot timestep policy
- full-information one-step oracle

任何新方法都必须使用 trajectory/session-held-out，并报告相同净 FLOPs 下的 FID/IS、wall time 和风险覆盖。
