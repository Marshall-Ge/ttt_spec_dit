# COVR-Spec 研究与落地规格

> 状态：候选主线，须先通过三项一票否决实验，再实现完整在线飞轮。
> 基线：`dev3@4c5bad4`。保留现有 selective-recompute 代码作为已结束实验记录。
> 保护项：`run.log` 是用户运行日志，不得覆盖、回滚或纳入代码改动。

## 1. 论文定位

**工作标题**：Every Refresh Is a Label: Counterfactual Online Learning for Budgeted Speculative Diffusion

**方法名**：COVR-Spec（Counterfactual Online Verification and Refresh for Speculative Diffusion）

**中心命题**：SpecA 的精确刷新既是计算动作，也是同状态反事实标签。通过概率化验证、选择偏差校正和预算约束在线学习，可在不更新生成模型权重的情况下，跨 session 持续改善精确计算的分配。

主创新是完整闭环，而非单独的 Taylor、IPW、bandit 或 replay：

1. scheduler-aligned whole-step counterfactual feedback；
2. exact refresh 作为 revealing action；
3. 概率化审计与 selective-label bias 校正；
4. 固定净 FLOPs 下的在线预算控制；
5. 严格 prequential、可证伪的跨 session 飞轮。

## 2. 已排除路线

- 不继续 probe-only hidden correction：5k ImageNet 无稳定收益，`reject` 仅改善 0.004 FID，`always` 反而恶化。
- 不做 learned probe correction policy、probe target 神经预测器或更大 probe-only 实验。
- 不以当前 VFL LoRA checkpoint 为研究基线：现有 target 在 LoRA no-op 起点产生零/错误方向梯度。
- 不用单层 full hidden 作为质量修正 target。
- learner 首版只学习 refresh 决策，不训练 backbone、LoRA 或 feature corrector。

## 3. 反事实负反馈

对去噪步 `t` 的同一个预决策状态 `s_t=(x_t,C_t)`：

- Taylor draft：`eps_A = f_A(x_t, C_t, t)`；
- full teacher：`eps_F = f_Theta(x_t, t)`；
- 两者必须共享相同 scaled latent、timestep、class/CFG 条件和 scheduler。

通过同一个 scheduler transition 得到：

```text
x_prev_A = S_t(x_t, eps_A)
x_prev_F = S_t(x_t, eps_F)
```

主标签为归一化一步 counterfactual defect：

```text
d_t = ||x_prev_A - x_prev_F||_2 /
      (||x_prev_F - x_t||_2 + epsilon)
```

该标签衡量 whole-denoiser 近似对下一 latent transition 的影响，不等同于 FID，也不宣称是终态感知质量。

低频 sentinel audit 使用 `H` 步 twin rollout：当前步分别采用 Taylor/full，后续共享同一动作序列，用于验证一步 defect 对延迟影响的排序能力。它不是常规训练标签，必须计入净 FLOPs。

## 4. 在线学习问题

动作：

- `ACCEPT`：提交 Taylor draft；不获得 full 标签；
- `REFRESH`：执行 full，提交 full 输出并刷新 cache；同时获得 Taylor/full 标签。

这是带成本的 contextual revealing-action / partial-monitoring 问题。目标是在硬预算 `B` 下最小化被接受近似的稳定性加权 defect：

```text
min_pi E[sum_t 1[a_t=ACCEPT] * w_t * d_t]
s.t.   sum_t cost(a_t) <= B
```

预算必须包含 draft、policy refresh、探索 refresh、sentinel rollout、在线更新和 canary 的摊销成本。

## 5. Context、模型与控制器

首版只使用 full 前向之前可免费获得的低维 context：

- timestep、logSNR、scheduler step size；
- 距离上次 refresh 的步数；
- Taylor 各阶 term norm；
- order-2/order-4 prediction disagreement；
- 各层 attn/MLP curvature 的聚合统计；
- CFG draft disagreement；
- 上一条已观测 defect；
- trajectory 剩余预算。

首版风险模型使用 online ridge/UCB：

```text
risk_hat = phi_t^T theta_hat
risk_ucb = risk_hat + beta * sqrt(phi_t^T V_inv phi_t)
REFRESH iff risk_ucb > lambda_t * incremental_cost
```

`lambda_t` 用 primal-dual 更新；另设硬 token bucket，保证单 trajectory 不超预算。小型 MLP 仅在 ridge 明显欠拟合后作为扩展。

## 6. 概率化审计与偏差校正

对策略原本会 `ACCEPT` 的位置，以已记录概率 `p_t >= p_min > 0` 转为探索性 `REFRESH`。概率必须在观察 full label 前确定。

在线更新使用 stabilized IPW；理论分析使用未裁剪 IPW，实践可报告裁剪权重的偏差—方差权衡。必要假设包括 sequential randomization、已知 propensity、正概率覆盖和有界损失。

禁止只审计高风险位置而不记录 propensity，否则飞轮会形成不可识别的选择性标签偏差。

## 7. 双层飞轮

**trajectory 快环**：draft → 风险/UCB → accept 或随机 refresh → defect → 风险模型与预算价格更新。

**session 慢环**：

1. 用上一版策略在未见 class/seed 上先评估；
2. 再将当前 session 的事件加入训练；
3. 持久化 risk sufficient statistics、propensity、版本、分层标量事件和 canary 结果；
4. 候选策略仅在相同净 FLOPs 下通过 held-out gate 后晋升；
5. 始终保留 `p_min`，防止策略锁死；
6. backbone、scheduler、步数或 CFG 配置变化时隔离版本并降权/重置统计。

飞轮成立标准：相同净 FLOPs 下，连续 session 的 held-out paired trajectory error 下降，或相同质量下审计/refresh 成本下降；不能以 checkpoint 数量或训练次数作为证据。

## 8. 理论边界

若 full transition 对状态是 `L_t`-Lipschitz，终态偏差可由局部 defect 的传播和上界：

```text
||x_0 - xbar_0|| <= sum_{t:ACCEPT} (prod_{j<t} L_j) * delta_t
```

可研究选择反馈下的风险估计一致性，以及在线 ridge/UCB + primal-dual 的次线性学习项和预算违反。正式定理必须在证明完成后再写具体阶数。

不得声称：FID 理论保证、一步 defect 等于感知质量、full refresh 对每个样本必然改善、IPW 本身是新算法、参数跨 session 继承本身构成理论创新。

## 9. 三项一票否决实验

### Gate A：Signal validity

对随机 layer/time 位置执行 branch-to-end，比较 block-20 cosine、一步 defect、`H` 步 defect与 terminal intervention gain。若 whole-step defect 对 terminal gain 的 top-decile lift/AUROC/Spearman 不显著优于 local probe，停止本方向。

### Gate B：Oracle headroom

采集候选步骤的 Taylor/full 对，让 full-information oracle 在固定 full-step-equivalent 预算下选择 refresh。若 oracle 不能 Pareto 支配 static SpecA，说明没有可学习的决策空间，停止本方向。

### Gate C：Learnability

在 class/seed held-out 上比较 timestep-only、cache-distance heuristic、static threshold 与完整 context risk model。若完整 context 不能稳定优于 timestep-only，说明 learner 只是在重发现固定课表，不进入完整飞轮开发。

建议顺序：A → B → C。任一失败即停止，不以扩大样本量挽救明确的机制失败。

## 10. 实验协议

严格 prequential：session `s` 的策略先在 `s+1` 未见数据上评估，再用该数据更新。所有方法匹配净 FLOPs，而非只匹配推理 refresh 次数。

必备基线：vanilla full、static SpecA、random refresh、periodic refresh、现有 OnlineCalibrator、timestep-only offline policy、full-information oracle。

必备消融：无 IPW、无 uncertainty exploration、每 session 重置、无 cross-session replay、无 sentinel calibration、固定阈值替代 primal-dual。

主指标：同 seed accelerated-vs-full final latent MSE、decoded LPIPS、terminal intervention gain、净 FLOPs、独立 wall time、budget violation、calibration error、top-risk recall、prequential regret。最终结论再跑修复后的 50k FID/IS。

## 11. 工程约束与阶段化落地

先实现只记录、不改变主轨迹的 shadow auditor。不能直接重复调用当前 SpecA forward，因为 `speca_cal_type()`、cache、controller counters 和 VFL hooks 均有副作用。

推荐重构接口：

```text
draft(state) -> eps_A, draft_context
shadow_full(same_input) -> eps_F        # 无副作用
policy.decide(context, budget) -> action
commit_accept() | commit_refresh()
learner.observe(context, propensity, defect, cost)
```

shadow full 必须绕过 SpecA state/cache 更新、VFL event 写入和 controller 计数；使用相同 CFG 输入与 scheduler。事件只保存低维 context、propensity、标量 defect、cost 和版本，不保存大 hidden tensor。

预计触及：`accelerators/compute_controller.py`、`accelerators/speca.py`、`models/dit.py`、`run_dit.py`，并新增独立的 counterfactual learner/event 模块及聚焦测试。旧 VFL 可复用 replay/version/eval-gate 思路，但不沿用 L3 LoRA loss。

阶段：

1. Phase 0：paired shadow recorder + Gate A/B 数据；
2. Phase 1：离线 oracle 与 learnability 分析；
3. Phase 2：online ridge/UCB + 硬预算，单 session；
4. Phase 3：随机审计 + IPW + prequential 多 session；
5. Phase 4：canary/versioning 与 50k 最终评估；
6. Phase 5：仅在证据支持时扩展 PixArt、非线性 learner 或更长 horizon。

任何实现都必须补充状态纯净性、CFG 一致性、scheduler counterfactual、propensity、预算守恒和跨 session 隔离测试。