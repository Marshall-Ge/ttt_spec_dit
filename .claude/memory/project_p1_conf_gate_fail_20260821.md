---
name: P1 conf 信号 [P1-a] gate 失败
description: "inception 预测熵在 K8 static 6 arm × 5 offsets 上完美追踪 IS（5/5 Spearman=1.000）但对 FID 轴致盲；FID-IS decoupling 下 conf 跟 IS 走，P1 质量约束预算控制器按预注册规则关闭"
type: project
---

2026-08-21，GPU 机 `output/covr_static_k8_random`（08-18 的 K8 static run，
6 mask arm × 5 latent offsets × 500 图，19,001 PNG），工具链
`scripts/run_p1_offline_gates.sh`（conf 定义与通过 [a] 的
`analyze_crossover --metric inception_conf` 逐位一致）。

**结果：Gate [P1-a] FAIL，P1 整条线停（spec 预注册规则，不调参挽救）。**

- Spearman(mean_entropy, **−IS**) = **+1.000 在全部 5 个 offset**（各 p=1/720，
  offset 为独立 latent 抽样）——conf 是完美的 session 级 IS proxy。
- Spearman(mean_entropy, **FID**) = 0.60/0.37/0.60/0.37/0.31——FID 排序碎裂。

**致死 pair（above-floor，非噪声）**：geometric vs back_loaded。两臂 IS 几乎
并列（~27-29），FID 相差 ~8（same-arm floor 2.0 的 4 倍）：

| offset | ent(back) | ent(geo) | conf 判 | FID(back) | FID(geo) | FID 判 |
|---|---|---|---|---|---|---|
| 0 | 1.660 | 1.682 | back 好 | 128.17 | 119.82 | geo 好 ✗ |
| 1 | 1.539 | 1.667 | back 好 | 127.72 | 120.07 | geo 好 ✗ |
| 2 | 1.577 | 1.561 | geo 好 | 126.07 | 120.61 | geo 好 ✓(熵差仅0.016) |
| 3 | 1.635 | 1.665 | back 好 | 127.66 | 119.82 | geo 好 ✗ |
| 4 | 1.596 | 1.675 | back 好 | 127.08 | 117.81 | geo 好 ✗ |

IS 并列而 FID 差 4×地板时 conf 看不见（甚至反向）——**conf 不含 IS 之外的
任何 FID 信息**。random null 两臂被 conf/IS 排最好（IS 33-34）而 FID 略差于
uniform（Δ0.5-1.7，地板内），同一机制。

**为什么 budget probe 的 [a] 曾通过**：probe 每 budget 只有 4 臂且 FID 与 IS
恰好同序（placement 变差两者一起崩），任何 IS-proxy 都会"顺便"通过 FID 排序。
static run 的 random nulls + geometric 打破 FID-IS 一致性后 conf 立即跟 IS 走。
08-18 记忆"无双指标静态赢家"（uniform FID 最优 / random+threshold IS 最优）
就是这个 decoupling 的先兆，当时没有连到 conf 上。

**gate 设计教训**：6 臂中 4 臂（uniform/geo/random_00/random_01）FID 互差
0.5-1.7，全在 2.0 地板内——严格 Spearman=1.000 对这些 pair 原则上不可达
（oracle FID 估计器也过不了）。但本次 FAIL 不依赖这些 pair：geo/back 单独
致死。未来同类 gate 应只要求 above-floor pair 排序正确。

**波及**：P2（漂移跟踪）中把 conf 当质量哨兵的部分同样受限——conf 只能做
IS 轴哨兵，不能支撑 "FID 非劣" 声明。[P1-b]/[P1-c] 未跑（moot；且 budget
probe 数据已不在当前 GPU 机，无需寻找或重生成）。

**Reopen 条件 / 唯一自然后继（未跑，未预注册）**：换信号不换骨架——
session/epoch 级 **mini-FID 哨兵**：每 epoch 用同一 InceptionV3 的 2048 pool
特征对预缓存 real 统计量算小样本 FID（n=100 bias 大但恒定，作相对哨兵）。
可用同一批 static-run PNG 离线验证 6-arm 排序有效性（[P1-a'] gate），成本
≈ 再过一遍 InceptionV3。gate 判据须预注册且只计 above-floor pair。

**How to apply**：任何"reference-free 质量信号"提案先对着本条打——必须在
FID-IS decoupled 的 arm 集上证明含 IS 之外的 FID 信息；budget probe 式
FID-IS 同序的 arm 集上的通过不算数。
