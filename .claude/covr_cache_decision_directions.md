# COVR 缓存决策改进方向——研究与比较

## 现状基准

Shadow 实验已确认：one-hot timestep 是最强风险预测器 (MSE 0.03755, recall 0.827)；full-context ridge 仅加 4.84% MSE 无 recall 收益；local probe 无排序能力 (Spearman -0.04)。VFL target 无效 (no-op 起点 loss=0)。Gate B 确认 oracle headroom 存在 (5% budget→14.3% defect 降)。以下方案均基于相同 prequential holdout 和匹配净 FLOPs。

---

**方向 1: Per-timestep UCB bandit**. 假设 defect 跨步独立，无需跨 timestep 泛化。50 步各自维护 UCB 估计，refresh 当 UCB > price×cost。支持：one-hot 即是 per-timestep 统计量的证据。反对：distance-since-refresh 无法捕获。区别于 one-hot：加在线 UCB 更新和 hard budget；区别于 periodic：学习真实排序。开销：50 floats+PrimalDual。最小实现：shadow 数据上预估步均 defect，离线线性搜索阈值。证伪：per-timestep UCB 能否优于 per-timestep mean。停止：若在线更新无额外收益。新颖性低但工程价值高。

**方向 2: Scheduler-aware 分段预算**. 假设 logSNR 曲率/step size 定义自然分段，段内 defect 稳定。支持：3-bucket 已被 VFL 使用。反对：跨 scheduler 泛化需验证。区别于 one-hot：分段压缩 (3-6 段 vs 50 步)。开销：3-6 个分数。最小实现：shadow data 上按 logSNR 聚类。新颖性中，工程价值中。

**方向 3: Budget-matched 固定 mask 搜索**. 假设最优刷新步集是确定性的。离线做 budget-constrained 子集选择。支持：Gate B headroom 可转化；贪心搜索 O(50 choose k) 可行。反对：不感知 cache state。区别于 periodic：search-based 而不是均匀。开销：离线启发式搜索，在线 O(1)查询。最小实现：贪心 forward selection。证伪：贪心 mask 能否在 10/20/30% budget 下优于 periodic。新颖性低但工程价值极高。

**方向 4: 稀疏 H-step sentinel 校准**. 假设 one-step defect 在某些步不能反映 terminal impact。在 strategic step 做 H-step rollout 校准。支持：Gate A 因无 terminal label 失败；末步 defect 异常 (均值 3.321)。反对：H-step 计入 budget，统计功效低。区别于 one-hot/oracle：sentinel 提供校准而非新策略。开销：每 sentinel H full steps (5 个 sentinel, H=10 ≈ 1% budget)。最小实现：step 1/10/20/30/40 各 1 次 rollout。新颖性高。

**方向 5: Numerator/denominator 分离建模**. 假设 defect d=||x_prev_A-x_prev_F||/||x_prev_F-x_t|| 的分母末步缩小制造虚假高风险。分离建模。支持：step 49 均值 3.321 是 denominator 效应。反对：numerator 可能更噪声。开销：多存 1 float/步。最小实现：重处理 shadow raw data。新颖性中，诊断价值高。

**方向 6: 共形风险控制**. 用 calibration set 构造 per-step 共形预测集，得上界。支持：分布自由，有覆盖保证。反对：exchangeability 在 prequential 下不严格成立；无 trajectory 适应能力。开销：每步 1 分位数。最小实现：80% shadow data calibration。新颖性中高。

---

**评估方法论**: Random audits 必须保留——无随机化则无可识别选择性偏差。IPW 必须用于跨 session 更新 (COVRPolicy 中 stabilized_numerator=p_min, clip=20 已实现)。Prequential holdout 需 trajectory-level: 按 sample_id 哈希 80/20 分 session，同上 batch 的事件必须同侧。Hard budget 用 BudgetLedger + PrimalDual 匹配净 FLOPs (含探索和 rollout 摊销)。

**推荐: 方向 1 (Per-timestep UCB) + 方向 3 (Budget-matched mask) 组合**。
理由：(1) 都不需要 H-step/terminal label，Gate A 瓶颈不阻碍；(2) 直接利用 one-hot 已证的优势；(3) 顺序阶段化——先离线 mask 做确定性的 baseline，再叠加在线 UCB 做持续校准；(4) 完全不依赖 VFL LoRA；(5) 在 COVRPolicy 框架中已有原型。具体落地方案：Phase 1 shadow data 贪心搜索 mask；Phase 2 mask 初始化 + p_min=0.02 探索 + IPW；Phase 3 5+ session prequential 验证。不推荐方向 4 作为主路线——sentinel 成本尚未证明与收益匹配。
