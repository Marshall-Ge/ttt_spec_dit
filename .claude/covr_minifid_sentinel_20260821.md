# P1' 设计规格：mini-FID 哨兵（session 级小样本 FID 负反馈信号）

> 日期：2026-08-21
> 状态：**ACTIVE — [P1'-a] 判据已预注册，等 GPU 特征提取后裁决**
> 前驱：P1（conf 信号）Gate [P1-a] FAIL 关闭
> （`.claude/covr_quality_constrained_budget_20260820.md`、
> `memory/project_p1_conf_gate_fail_20260821.md`）。本方向是其条目里预告的
> 唯一自然后继：**换信号不换骨架**——预算梯控制器
> `accelerators/conf_budget_controller.py` 原样复用（已单测 + 与 gate 认证
> 逻辑逐 epoch 等价），只把 per-image conf 换成 per-epoch mini-FID。
> 对照教训（该 memory 条 How-to-apply）：conf 死于"不含 IS 之外的 FID
> 信息"；mini-FID 直接构造在 FID 的特征空间里，by construction 含 FID
> 信息——本 gate 验证的是**小样本噪声下排序是否保留**，这是它唯一可能的
> 死法，也正是 [P1'-a2] 要裁决的。

## 1. 信号定义（全部口径逐位固定，与真 FID 同源）

mini-FID(arm, n) = 该 arm n 张生成图的 InceptionV3 特征高斯拟合
(μ_f, Σ_f) 对该 arm **自己 run dir 的 `real_299/`** 全体图 (μ_r, Σ_r) 的
Fréchet 距离：

    FID = ||μ_f − μ_r||² + tr(Σ_f) + tr(Σ_r) − 2·tr((Σ_f Σ_r)^{1/2})

- **特征**：torch-fidelity `inception-v3-compat` 的 `"2048"` pool 特征——
  与各 run results.json 的 FID **同一网络同一权重**（复用
  `analyze_crossover._inception_features`，即 conf 提取共用的 extractor）。
- **预处理**：无损 PNG → uint8 → PIL BICUBIC 299×299，与 eval/fid_is.py
  add() / compute_mixture_fid.py 逐位一致（extractor 内部 299 interpolate
  对 299 输入是恒等，不引入第二次 resize 差异）。
- **real 侧**：每个 arm 用**自己 run dir 的 `real_299/` 全体**——与该 arm
  results.json 的 FID real 侧完全一致；不同 offset 的类集不同，天然按
  offset 对齐。协方差 ddof=1。
- **n=100 的 bias**：小样本 FID 偏大，但 n 固定时 bias 近似恒定——哨兵
  只做**相对比较**（arm 间 / epoch 间），绝对值无意义，不与 n=500 值混比。

## 2. Gate [P1'-a] 排序有效性 + 判别力（预注册，先于数据）

数据：K8 static run（`output/covr_static_k8_random`，6 mask arm × 5
offsets × 500 图 + 各 arm results.json 的真 FID(n=500)）。**零新图**，
成本 ≈ 全部 PNG 再过一遍 InceptionV3（含 real_299）。

自由度全部在此固定：floor = 2.0（同臂 replica floor，08-18 实测）；
n_epoch = 100；重采样 R = 200（不放回，独立 rng，种子 0）；特征/预处理/
real 侧如 §1；分组 = 每个 offset 一组，6 arm，15 pair。

- **[P1'-a1] 有效性（n=500 全量）**：每 offset 内，对全部 above-floor
  pair（|ΔFID_true| > 2.0），mini-FID_500 排序方向与真 FID 一致。
  **5/5 offset 全部 pair 正确 → PASS；任何一个反向 → FAIL。**
  （n=500 mini-FID 与真 FID 特征同源、real 同侧，差异只剩高斯拟合近似；
  这层过不了说明实现或口径有 bug，或高斯近似本身失效。）
- **[P1'-a2] 判别力（n=100）**：每 offset 每 arm 独立抽 R=200 次
  n=100 mini-FID；对每个 |ΔFID_true| > 4.0（2×floor）的 pair，
  方向正确率 = mean_r[ sign(mini_r(a)−mini_r(b)) == sign(FID(a)−FID(b)) ]。
  **全部此类 pair 在 5/5 offset 上 ≥ 0.90 → PASS，否则 FAIL。**
  floor < |ΔFID| ≤ 2×floor 的 pair 只报告不裁决（哨兵用途是挡大退化）。
  致死试金石 = geo/back pair（ΔFID≈8 ≈ 4×floor，conf 曾 4/5 排反）。
- **总判决 = a1 AND a2**；FAIL → 本方向关闭，不调参挽救（floor、R、n
  均不得事后改动）。Spearman(mini-FID_500, FID_true) 每 offset 报告但
  **不作为判据**（floor 内 pair 原则上不可排序——conf gate 的设计教训）。

## 3. 通过后的路线（沿用 P1 骨架，只列不展开）

- [P1'-b] 检测力：mini-FID 是 per-epoch 单标量 → CUSUM 退化为每 epoch
  单点更新；校准分布 (μ0, σ0) 用校准 epoch 特征的 bootstrap mini-FID 分布
  （同 n、同类分布抽样）。`ConfBudgetController` 不改一行：
  `end_epoch([minifid_value])`。检测延迟/误报率 gate 沿用 P1 判据
  （k8→k6 ≤2 epoch、平稳 FA ≤5%）。
- [P1'-c] 闭环模拟 + 混合 FID：`simulate_conf_budget_controller.py` 的
  conf 表换成 per-epoch mini-FID 流（工具适配届时再定）；quality leg
  仍走 `compute_mixture_fid.py`。
- 三 gate 全过才 GPU 闭环真跑（配对 5 offsets + exact sign test）。

## 4. 已知风险

1. **n=100 方差**：ΔFID 2-4 的 pair 大概率分不开——判据已按此设计
   （只 gate >2×floor）；若连 4×floor 的 geo/back 都分不开则方向死。
2. **类分布波动**：epoch 间类分布不同会动 mini-FID 基线（real 侧固定时
   fake 侧类混合变化 = 分布真移动）。c2i 下类已知：闭环里校准/比较都用
   **同类分布抽样**；[P1'-a] 阶段 static run 每 arm 同 offset 类集配对，
   天然免疫。
3. **协方差奇异**：n=100 < 2048 → Σ_f 低秩。Fréchet 公式本身不要求满秩
   （tr((Σ_f Σ_r)^{1/2}) 用谱等价 Gram 技巧数值稳定），但低秩拟合是
   bias 的主要来源——靠"n 固定 bias 恒定"假设消化，[P1'-a2] 直接检验
   其排序后果。
4. **与 IS 轴的关系**：mini-FID 不看 IS。若未来闭环要求 IS 非劣，quality
   leg（混合 FID + IS）仍在 [P1'-c] 把关——哨兵只负责在线降档决策。

## 5. 工具与命令（GPU 机）

```bash
# 0) 特征提取（fake 全部 arm + 每 arm sibling real_299；只读）
python3 scripts/extract_inception_feats.py output/covr_static_k8_random feats_k8static.npz

# 1) Gate [P1'-a]（真 FID 自动从各 arm results.json 取）
python3 scripts/analyze_minifid_rank_validity.py feats_k8static.npz \
    --results-root output/covr_static_k8_random
#    （--arm-regex/--group-regex 可调 arm 过滤与 offset 分组；
#      判据参数 floor/n/R 有默认=预注册值，改动即违反预注册）
```
