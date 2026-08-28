---
name: lever-reconstruction-and-structure-aware-design
description: 验证实验:SpecA 时间桶差异化(同预算 MSE-17%)+ TeaCache 线性残差漂移杠杆再造(同 skip 率 MSE-53~66%)+ 预测器族谱与 token 长尾外推
metadata:
  type: project
  modified: 2026-08-27
---

# 结构感知设计验证 + 杠杆再造实验(2026-08-27, RTX 4090D, DiT-2-256, 50 步)

## 背景

[[error-vs-distance-mechanism]] 验证了机制:SpecA 误差随 skip 距离超线性增长(杠杆存在),TeaCache 残差误差平坦(杠杆缺失)。本批实验回答两个问题:
1. 结构感知设计(时间桶差异化调度)能否兑现收益?
2. 给 TeaCache 引入外推(杠杆再造)能否打开自适应空间?

## E5: 预测器族谱(open-loop, 6 图) — block 级 7 种预测结构

| d | reuse_0th | speca_1st(block) | speca_2nd | tc_res | tc_res_lin | tc_res_damp |
|---|-----------|------------------|-----------|--------|------------|-------------|
| 1 | 0.0086 | 0.0054 | 0.0048 | 0.0086 | 0.0054 | 0.0084 |
| 4 | 0.0318 | 0.0290 | 0.0302 | 0.0318 | 0.0290 | 0.0294 |
| 8 | 0.0525 | 0.0622 | 0.0678 | 0.0525 | 0.0622 | 0.0600 |
| 12 | 0.0673 | 0.0945 | 0.1128 | 0.0673 | 0.0945 | 0.0945 |

发现:
- **block 级 0 阶复用 ≈ TeaCache 残差复用**(几乎重合)——TeaCache 本质是 0 阶复用 + 输入漂移项
- **block 级一阶外推只在 d≤4 优于复用**,d≥5 后外推噪声 > 漂移(与子模块级超线性增长相反:28 层门控累积平滑了轨迹)
- 二阶外推更差;阻尼只在 d=4-6 有 ~8% 小增益
- **含义:全局统一外推在多数 token/距离上有害 → WorldCache 必须按 token 分组(Chaotic 才外推)的机制根源**

## E6: token 级长尾分析(4 图) — DiT 上误差接近均匀

| 指标 | 值 | 对比 |
|------|-----|------|
| top1% token 误差占比 | 2.3% | 均匀=1%,WorldCache 世界模型=严重长尾 |
| top10% | 17.1% | 均匀=10% |
| skew(mean/median) | 1.07-1.10 | 接近 1=均匀 |
| oracle per-token 分组增益 | d1: 0.0033 → d8: 0.0009 | 小且随 d 衰减(且是知道未来的上界) |

**关键外推结论:误差异质性的维度因任务而异**
- 世界模型(多模态):token 维度长尾 → token 级方法(WorldCache CHTP)有效
- DiT(单模态类条件):**时间维度(late 40×)+ 层维度(中层)主导,token 均匀** → 时间/层感知的自适应有效,token 级方法无效
- 设计方法论:**先诊断误差结构(哪个维度异质),再选自适应粒度(时间/层/token)**

## E7: SpecA 时间桶差异化调度(closed-loop, 20 图) — 结构感知设计验证

| 配置 | schedule(E,M,L) | MSE | cos | skip% | wall_s |
|------|------|------|-----|-------|--------|
| uniform | (4,4,4) | 0.00088 | 0.00119 | 76.1% | 1.368 |
| **sched_a** | (8,4,2) | **0.00073(-17%)** | 0.00106 | 75.9%(持平) | 1.361 |
| sched_b | (8,4,1) | **0.00055(-37%)** | 0.00079 | 70.0% | 1.399 |
| sched_c | (6,4,2) | 0.00060(-32%) | 0.00086 | 74.0% | 1.380 |

**验证成功**:同 skip 率下 MSE -17%(sched_a);晚期限距到 1 用 6% skip 换 -37% MSE(sched_b)。"同样的 full 预算放在不同位置"有 17-37% 误差差异 → COVR 在 SpecA 上有效的机制证实,且给出了比 bandit 更简单可解释的结构化调度。

## E8: TeaCache 残差漂移变体(closed-loop, 12 图 × 2 组 γ) — 杠杆再造

### γ ∈ {0.20, 0.25, 0.35}(MSE,同 skip 率)

| 配置 | γ=0.20 | γ=0.25 | γ=0.35 |
|------|--------|--------|--------|
| plain | 0.00351 | 0.00510 | 0.00947 |
| **linear(残差+全漂移)** | **0.00171(-51%)** | **0.00240(-53%)** | **0.00432(-54%)** |
| damped(Hermite) | 0.00325(-7%) | 0.00453(-11%) | 0.00728(-23%) |

### γ ∈ {0.45, 0.55, 0.70}(等质量换速度)

| 配置 | γ=0.45 | γ=0.55 | γ=0.70 |
|------|--------|--------|--------|
| plain | 0.01356 | 0.01729 | 0.02673 |
| **linear** | **0.00588(-57%)** | **0.00686(-60%)** | **0.00919(-66%)** |
| damped | 0.00929 | 0.01028 | 0.01379 |

**决定性对比**:
- linear γ=0.70 的 MSE(0.00919)≈ plain γ=0.35(0.00947),但 **skip 88.2% vs 80.7%,wall 0.306 vs 0.431(快 29%)**
- 线性外推增益随 γ 增大而增大(高 skip → 更长 streak → 外推价值更大)

**为什么 closed-loop 与 open-loop 结论不同**:TeaCache 累积触发机制限制实际 streak 只有 2-3 步,恰好落在线性外推最优的短距离窗口(d≤4);open-loop 里长距离(d≥7)线性劣化是脱离实际工作窗口的。

## 总结论

1. **结构感知设计兑现收益**:SpecA 时间桶差异化 = 免费 17%(同预算)或 37%(+6% full)
2. **杠杆再造打开 TeaCache 自适应空间**:线性残差漂移同 skip -53~-66% 误差;γ 可推到 0.70 快 29% 且质量持平
3. **外推维度结论**:DiT 误差结构 = 时间主导 + 层主导 + token 均匀;WorldCache 的 token 级动机不适用于 DiT,但"预测器升级 + 自适应"的组合机制在两类任务都成立
4. **论文叙事更新**:自适应有效性的判据 = 误差-距离结构(§4);设计准则 = 诊断维度 → 匹配粒度 → 必要时再造杠杆

## 待办(下一步候选)

- [ ] linear 变体 + per-class γ 组合(杠杆再造 × 档位自适应叠加)
- [ ] linear 变体 50k FID 确认(skip 88% 的 FID vs plain 81%)
- [ ] 中层 check_layer(7-17)的 SpecA 变体
- [ ] sched_a/b 的 50k FID 确认

## 文件

- 脚本(远程): `scripts/diag_predictor_family.py`, `scripts/diag_token_longtail.py`,
  `scripts/run_speca_bucket_schedule.py`, `scripts/run_teacache_residual_variants.py`
- 数据(本地): `experiments/diag_error_vs_distance/{diag_predictor_family,diag_token_longtail,speca_bucket_schedule,teacache_residual_variants,teacache_residual_variants_hi}.json`
- 相关: [[error-vs-distance-mechanism]], [[final-framework-conclusion]], [[project_covr_system]]
