---
name: mini-FID 哨兵 [P1'-a] gate 失败
description: "n=100 mini-FID 方向正确率 80.5-88.5%，低于预注册 90% bar；连 ~5×floor 的 back/front pair（ΔFID≈10）都只有 84-86%——死因是小样本功率不足（非信号信息错误，与 conf 的死法互补）；方向按预注册规则关闭"
type: project
---

2026-08-21，GPU 机同批 K8 static run（`output/covr_static_k8_random`，
特征 npz 由 `scripts/extract_inception_feats.py` 提取，判决
`scripts/analyze_minifid_rank_validity.py`，预注册判据
`.claude/covr_minifid_sentinel_20260821.md` §2：floor=2.0，gated pair =
|ΔFID_true|>4.0，n=100，R=200，min-acc=0.90）。

**结果：GATE [P1'-a] FAIL（a2 功率腿），方向按预注册规则关闭，不调参。**

用户实跑未加 `--arm-regex`（包含了 threshold 等预注册外 arm），但 **FAIL
在预注册的 6 mask arm 宇宙内独立成立**——可见输出中纯 6-arm 的不合格
gated pair 已有 ≥4 个：

| offset | pair | ΔFID_true | n=100 acc |
|---|---|---|---|
| rep_3 | back_loaded vs front_loaded | −9.80 | 86.0% |
| rep_3 | back_loaded vs random_01 | +4.54 | 85.5% |
| rep_3 | random_01 vs uniform | +4.99 | **80.5%** |
| rep_4 | back_loaded vs front_loaded | −10.48 | 84.0% |

（threshold 混入的 gated pair 同样 83-88.5% 不达标，方向一致。）

**死因定性：功率不足，不是信息错误。** acc 80-88% 远高于随机 50%——
mini-FID 确实携带 FID 方向信息（对比 conf 在致死 pair 上 4/5 排反）；
但 n=100 的抽样方差使它连 ΔFID≈10（**5×floor**）的 pair 都到不了
90% 单 epoch 可靠度。这正是规格 §4 风险 1 预言的死法。两个信号的
死法互补：**conf = 信息轴错误（只含 IS）；mini-FID = 信息对但单 epoch
功率不足**。

**待补（完整 log 尚未贴回，用户只贴了尾部）**：[P1'-a1]（n=500 有效性腿）
的判决与 geo/back pair 的 acc；gated pair 总数/不合格数。拿到后补记——
a1 若 PASS 则"信号有效但欠功率"的定性完全坐实。

**How to apply**：
- 任何"每 epoch 一次小样本 FID"的哨兵提案，先对本条打：n=100 在本模型
  本数据上对 ΔFID≤10 的 pair 单次方向可靠度只有 ~80-88%。要过 90% bar
  只能加大 n（同一 npz 可零成本算 acc-vs-n 功率曲线，探索性分析已合法，
  gate 已关不算调参）或改决策规则（跨 epoch 累积证据）——两者都是
  **新方向**，须新预注册规格，不得复用本 gate 的名义。
- 预算梯控制器骨架（`accelerators/conf_budget_controller.py`）与本次
  死因无关，继续保留为即插即用件。
