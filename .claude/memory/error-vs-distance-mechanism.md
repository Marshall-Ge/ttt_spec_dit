---
name: error-vs-distance-mechanism
description: 误差-距离结构诊断:SpecA 误差随 skip 距离超线性增长,TeaCache 残差误差亚线性饱和——COVR 收益差异的机制根因
metadata:
  type: project
  modified: 2026-08-27
---

# 误差-距离结构:COVR 在 SpecA 有效、TeaCache 失效的机制根因

## 背景问题

用户观察:COVR 自适应(等 FLOPs refresh mask / bandit 决策)在 SpecA 上 FID 显著提升(50k: COVR Bandit FID 17.48 vs Adaptive SpecA 23.96 vs Baseline 24.84),但迁移到 TeaCache 上收益消失。

**机制假说**:两种加速器的误差对"在哪 refresh"(skip 距离 d)的敏感度不同——
- SpecA(Taylor 外推):误差随 d 增长(1 阶截断 ~ d²·sup‖y''‖)→ refresh 位置是决定性变量 → 自适应有杠杆
- TeaCache(残差复用):误差在 d 上平坦/饱和(无外推项)→ refresh 位置无杠杆 → 自适应无收益空间

## 实验(2026-08-27, RTX 4090D, DiT-2-256, 50 步 DDIM)

### E1: Open-loop oracle 模拟(4-6 图,全 full 轨迹,模拟"距上次缓存 d 步"的预测误差)

| d | SpecA attn cos(层平均) | SpecA ff cos | SpecA block cos | TeaCache res cos | TC rel_l1 |
|---|----------------------|--------------|-----------------|------------------|-----------|
| 1 | 0.0046 | 0.0040 | 0.0054 | 0.0086 | 0.104 |
| 5 | 0.0649 | 0.0686 | 0.0372 | 0.0377 | 0.354 |
| 10 | **0.2190** | **0.2145** | 0.0792 | **0.0605** | 0.547 |
| 增长 | **48×**(超线性≈d^1.7) | 54× | 15×(近线性) | **7×(亚线性, d≥6 饱和)** | 5× |

- SpecA 子模块级误差超线性增长;block 级(门控累积后)近线性增长
- TeaCache 残差误差亚线性且 d≥6 后饱和(0.043→0.048→0.053→0.057→0.061,增量递减)

### E2: 层结构(6 图,全 28 层)

| layer | attn cos d=1 | d=5 | d=10 | ratio |
|-------|-------------|-----|------|-------|
| 0 | 0.0004 | 0.006 | 0.043 | 112× |
| 7 | 0.0025 | 0.056 | **0.263** | 105× |
| 14 | 0.0033 | 0.101 | **0.311** | 95× |
| 17 | 0.0033 | 0.108 | **0.317** | 96× |
| 20(check_layer) | 0.0023 | 0.069 | 0.219 | 94× |
| 27 | 0.0192 | 0.051 | 0.150 | 7.8× |

- **误差增长最猛的是中层(7-17),check_layer=20 不是最优探测点**
- 深层(27)单步误差就大(0.019)但增长慢——深层本身波动大

### E3: 时间结构(per-bucket: 去噪早/中/晚期各 1/3)

| bucket | SpecA block cos d=1→d=10 | TeaCache res cos d=1→d=10 |
|--------|--------------------------|---------------------------|
| early | 0.0001 → 0.0047 | 0.0001 → 0.0047 |
| mid | 0.0002 → 0.0173 | 0.0005 → 0.0151 |
| **late** | **0.0161 → 0.1775** | **0.0256 → 0.1332** |

- **误差高度集中在去噪晚期**(late 是 early 的 ~40 倍)
- 这解释了 COVR 的 one-hot timestep 是最强风险预测器(prior lockin 92%)

### E4: Closed-loop 验证(5 图,真实 SpecA 运行时 check_layer=20 探测)

| d | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| err | 0.0014 | 0.0021 | 0.043 | 0.123 | 0.165 | **0.206** |

- 实际运行时误差同样随距离加速增长(d=3→8: 150×)
- by_bucket: early 0.001 / mid 0.002 / **late 0.335**(closed-loop 下晚期误差更大,输入漂移放大)

## 结论

1. **机制假说验证成立**:SpecA 误差是"距离敏感 + 层敏感 + 时间敏感"的(杠杆空间大);TeaCache 误差是"距离迟钝 + 时间敏感"的(杠杆只在 γ 档位)。
2. **COVR 收益差异的根因**:COVR 的 refresh-mask 决策优化的是"距离/位置"——在 SpecA 上这是决定性变量(误差 ×48 随 d),在 TeaCache 上残差复用误差饱和(×7),所以 SpecA 上收益大、TeaCache 上消失。
3. **时间结构统一了所有观测**:误差主战场在去噪晚期(40×)→ COVR timestep_prior 胜出、per-class γ 收益、TeaCache 晚期强制 calc 的原因。
4. **WorldCache(ICML 2026)在 TeaCache 类方法上成功的原因**:它把 TeaCache 从"纯复用"升级为"外推型"(线性/阻尼预测器),重新引入了距离相关的误差结构,于是 CAS 自适应有了杠杆。**自适应需要"随距离增长的误差"才有空间**。

## 对下一步的含义

- **TeaCache 上做自适应**:不要调 mask/γ 档位,而是先引入"随距离的预测"(残差线性外推/阻尼,类似 WorldCache CHTP),再套在线决策
- **SpecA 上做自适应**:杠杆已经确认;可优化方向是(1)按时间桶差异化 max_taylor_steps(晚期限制距离),(2)check_layer 移到误差增长最猛的中层(7-17),(3)按层差异化阈值
- **研究叙事**:误差-距离结构是可发表的机制分析——"自适应缓存加速的有效性取决于误差对 skip 距离的敏感度",附 COVR SpecA/TeaCache 对照 + 本诊断

## 文件

- `scripts/diag_error_vs_distance.py` — v1 open-loop(层平均)
- `scripts/diag_error_vs_distance_v2.py` — v2 open-loop(per-layer + per-bucket)
- `scripts/diag_error_vs_distance_v3.py` — v3 closed-loop(真实 SpecA 探测)
- 输出: `/tmp/diag_error_vs_distance.json`, `/tmp/diag_error_vs_distance_v2.json`, `/tmp/diag_error_vs_distance_v3.json`
- 相关: [[project_covr_system]], [[final-framework-conclusion]], [[project_covr_static_k8_paired_result]]
