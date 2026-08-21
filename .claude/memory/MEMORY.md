1	# 项目 memory 索引（canonical，维护在此）
2	
3	> **维护位置**：本目录 `.claude/memory/` 是本项目 memory 的**唯一权威来源**（2026-08-11 从用户级 `~/.claude/projects/-Users-marshall-Projects-ttt-spec-dit/memory/` 迁入，源端已退役）。新增 / 更新 memory 一律写回这里，**不要写回用户级**。
>
> **入口**：先读上级 `.claude/AGENTS.md`（Agent 入口，对标 `lttta-main/.codex/AGENTS.md`），它把背景 / 流程 / 代码 / 原子事实四类资产分工指清楚。本文件只是**原子事实**的索引，不放正文，也不放结构性文档。

## 结构性文档（不在本目录，避免重复）

| 类型 | 位置 | 内容 |
|---|---|---|
| Agent 入口 | `.claude/AGENTS.md` | 一句话定位、关键事实速查、资产分工、入口指引 |
| 项目背景 / 主线思路 | `.claude/skills/ttt-dit-context/SKILL.md` | 在线学习负反馈的高层 primer |
| 标准工作流 | `.claude/skills/ttt-dit-workflow/SKILL.md` | 开发 / 评测 / 标定 / 记录流程（大体） |
| 代码结构 / 各文件职责 / 坑 / 命令 | 仓库根 `CLAUDE.md` | 唯一代码交接文档 |
| 阶段研究结论 / 规格 | `.claude/covr_*.md` | COVR/VFL 大块研究文档 |

## 原子事实 memory（每条一个 `.md` 文件）

---

## 维护约定

- 一条原子事实 = 一个 `.md` 文件（带 frontmatter），并在上面索引加一行。
- 大块研究规格 / 阶段结论 / 方向对比 → 进 `.claude/` 根的 `covr_<topic>_<YYYYMMDD>.md`，不进本目录。
- 代码结构变更 → 改仓库根 `CLAUDE.md`，不在本目录维护代码文档。
- 新增结构性文档前先确认是否与 `AGENTS.md` / 两个 skill / `CLAUDE.md` 重复，避免多头维护。
4	
5	---
6	
7	- [VFL 已知实现缺陷清单](project_vfl_known_gaps.md) — D1-D5 非紧急缺陷；CLAUDE.md 关于 PixArt+VFL 的描述与实际不符
8	- [VFL 训练信号源根因诊断](project_vfl_signal_source_flaw.md) — 50k 实验证明 LoRA 训练无效；target 是 base forward 输出导致 no-op 起点对称性 loss=0；P0 必须修信号源
9	- [COVR 系统全貌](project_covr_system.md) — 动机/方法/架构/实验发现/文件索引；bandit FID↓7.3但IS↓18%；safety shadow是速度瓶颈；50k实验92%选timestep_prior
10	- [COVR 离线探针结论](project_covr_offline_probe_findings.md) — crossover 真实(p=0)但 causal context 不可用(p=0.84)；audit 只测到 cache distance 1-4，激进 mask 不能外推
11	- [COVR crossover 判据设计](project_covr_crossover_gate_design.md) — in-sample oracle gap 与三个 permutation null 全不可用；OOS 特征取自 D 会泄漏，n=500 时纯噪声误报 99%
12	- [COVR OOS 判据两次修正](project_covr_crossover_oos_gate_rule.md) — 先换掉 1NN label transfer，再补两道量级地板；permutation null 有 0 点质量 0.56-0.60，p≤0.05 单独等价于 frac>0
13	- [COVR 500 图 budget probe 结果](project_covr_budget_probe_500_results.md) — inception_conf 首次通过 [a] Spearman=1.000；两个 metric 的"UTILIZABLE"都是误报，回收 0.03-0.28 FID vs 4.8 FID 地板
14	- [COVR static K=8 配对结论](project_covr_static_k8_paired_result.md) — uniform 胜 geometric；严格 K=8 对照中 uniform FID 优先、plain TeaCache IS 优先，形成稳定 Pareto 权衡
15	- [COVR bandit 先验锁死](project_covr_bandit_prior_lockin.md) — 92.1% 收敛=epsilon-greedy 无学习地板 92.5%；penalty 0.25 比 reward 大 4167×，最优臂需 3 亿张图才能翻盘
16	- [COVR 决策粒度是 batch](project_covr_decision_granularity.md) — trajectory=batch 不是 image；bs=32 含 ~31.5 类，per-image 天花板被批平均抹掉两个数量级；必须 BATCH_SIZE=1
17	- [COVR contextual+efficiency bandit 已 GPU 验证=负](project_covr_contextual_efficiency_bandit.md) — deferred-commit LinUCB + λ·FLOPs reward；500图首跑两个维度 contextual gain 都≤0（fidelity -6.4e-6, combined -2.6e-4），per-image Pareto 异质性不存在；STOP，不扩大 GPU
18	- [COVR K8 random-null 配对结果](project_covr_static_k8_random_result_20260818.md) — K=8 mask placement 影响 aggregate FID/IS；uniform 稳定优于确定性布局但对 random null 形成 FID/IS 反向权衡；无多指标静态赢家，reward 未启用，per-image crossover gate 未通过，不启动 bandit
19	- [timestep feedback shadow 集成 smoke](project_timestep_feedback_shadow_smoke_20260818.md) — 真实 DiT SpecA 16图/4 trajectory 成功产生 25 个 selective contexts、100 labels 并持久化 state；仅验证 wiring，不代表质量或学习收益
20	- [timestep feedback prequential smoke](project_timestep_feedback_prequential_smoke_20260818.md) — 两 session held-out one-step MSE 0.358→0.027、MAE 0.502→0.113、top-decile recall 0→0.75；与 unweighted one-hot baseline 完全相同，尚无 IPW/UCB 增量收益，不代表 FID 或 active policy 收益
21	- [timestep feedback active 配对质量结果](project_timestep_feedback_active_paired_result_20260819.md) — K=12 等 FLOPs 配对：risk-learned(4,6) vs uniform(2,47) FID +0.294±0.949、IS -0.153±0.418，全噪声内；三环证据链闭合，learned placement 不改善质量，该方向按质量方法关闭
22	- [Taylor term 范数信号离线 gate=负](project_taylor_term_norm_signal_offline_20260820.md) — P3 候选信号在 37,808 条 shadow events 上对 defect_{d+1}/爆炸目标的边际 ≤0.03% R²（one-hot step+dist 已解释 99% context 级方差）；切片内 tn2 与 defect 呈 -0.687 负相关，方向相反；离线关闭，不花 GPU；P1（conf 质量约束预算校准）规格见 .claude/covr_quality_constrained_budget_20260820.md
- [P1 conf 信号 [P1-a] gate 失败](project_p1_conf_gate_fail_20260821.md) — K8 static 6 arm × 5 offsets：conf 完美追踪 IS（5/5 Spearman=1.000）但对 FID 轴致盲（geo/back pair IS 并列、FID 差 4×地板，conf 4/5 排反）；FID-IS decoupling 下 conf 跟 IS 走；P1 按预注册规则关闭；后继候选 = mini-FID 哨兵（同批 PNG 可离线 gate）
23	- [简短提供测试命令](feedback_concise_test_commands.md) — 远程测试场景只给从真实入口开始的核心命令
24	- [input_tokens 错误后自动续做](feedback_resume_after_input_tokens_error.md) — 遇到客户端 token 错误时拆分调用并持续完成任务，不输出空响应
