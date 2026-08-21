---
name: mini-FID 哨兵 [P1'-a] gate 失败
description: "a1 有效性 PASS（5/5 组 0 排错，mini500 与真 FID 逐位相等）但 a2 功率 FAIL（85 个 gated pair 中 17 个 acc 80.5-88.5% < 90% bar）；死因纯粹是 n=100 小样本方差（mini100 sd≈3.2-5.2 FID → 单 epoch 可靠检测下限 ~7 FID）；与 conf 死法互补；方向按预注册规则关闭"
type: project
---

2026-08-21，GPU 机同批 K8 static run（`output/covr_static_k8_random`，
特征 npz 由 `scripts/extract_inception_feats.py` 提取，判决
`scripts/analyze_minifid_rank_validity.py`，预注册判据
`.claude/covr_minifid_sentinel_20260821.md` §2：floor=2.0，gated pair =
|ΔFID_true|>4.0，n=100，R=200，min-acc=0.90）。

**结果（完整 log，2026-08-21）：**

- **[P1'-a1] 有效性 PASS**：5/5 组、0 个 above-floor pair 排错。且
  mini500 与真 FID **逐位相等**（128.17↔128.18、119.82↔119.82……）——
  同特征同 real 侧下全量 mini-FID 就是重算真 FID，实现忠实度封顶。
- **[P1'-a2] 功率 FAIL**：85 个 gated pair（|ΔFID|>4.0）中 **17 个**
  acc 80.5-88.5% < 90% bar → **GATE [P1'-a] FAIL，方向按预注册关闭，
  不调参。**

用户实跑未加 `--arm-regex`（含 threshold/reference/noise_lat 等预注册外
arm），但 FAIL 在预注册 6 mask arm 宇宙内独立成立——纯 6-arm 的不合格
gated pair 有 6 个：

| offset | pair | ΔFID_true | n=100 acc |
|---|---|---|---|
| rep_1 | random_00 vs uniform | +4.65 | 87.5% |
| rep_1 | random_01 vs uniform | +4.78 | 84.0% |
| rep_3 | back_loaded vs front_loaded | −9.80 | 86.0% |
| rep_3 | back_loaded vs random_01 | +4.54 | 85.5% |
| rep_3 | random_01 vs uniform | +4.99 | **80.5%** |
| rep_4 | back_loaded vs front_loaded | −10.48 | 84.0% |

**死因定性（a1 PASS 后完全坐实）：功率不足，不是信息错误。**
- geo/back（杀死 conf 的试金石 pair，ΔFID 5.5-9.3）**5/5 offset 全部
  ≥90%**（91.5-99.5%）——信息轴上 mini-FID 完胜 conf。
- 不合格的 17 个集中在 ΔFID 4.5-10.5；ΔFID>14 的 pair 全部 ≥99%。
- 噪声标尺：mini100 的 sd ≈ 3.2-5.2 FID（随 arm/offset 波动），两臂独立
  比较差值 sd ≈ √2×4 ≈ 5.7 → 90% 单侧正确率对应检测下限
  |ΔFID| ≳ 1.28×5.7 ≈ **7 FID**——与失败分布吻合（也解释了同一
  back/front pair 在不同 offset 84-94% 的波动：offset 间特征离散度不同）。
- 这正是规格 §4 风险 1 预言的死法。两个信号死法互补：**conf = 信息轴
  错误（只含 IS）；mini-FID = 信息对但单 epoch 功率不足**。

**How to apply**：
- 任何"每 epoch 一次小样本 FID"的哨兵提案，先对本条打：n=100 在本模型
  本数据上的单 epoch 可靠检测下限 ≈ 7 FID（ΔFID 4.5-10.5 区间 17/85
  不达 90%，>14 全部 ≥99%）。要过 90% bar 只能加大 n（同一 npz 可
  零成本算 acc-vs-n 功率曲线，探索性分析已合法，gate 已关不算调参）
  或改决策规则（跨 epoch 累积证据）——两者都是**新方向**，须新预注册
  规格，不得复用本 gate 的名义。
- 预算梯控制器骨架（`accelerators/conf_budget_controller.py`）与本次
  死因无关，继续保留为即插即用件。
