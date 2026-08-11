# 项目 memory 索引（canonical，维护在此）

> **维护位置**：本目录 `.claude/memory/` 是本项目 memory 的**唯一权威来源**（2026-08-11 从用户级 `~/.claude/projects/-Users-marshall-Projects-ttt-spec-dit/memory/` 迁入，源端已退役）。新增 / 更新 memory 一律写回这里，**不要写回用户级**。每条 memory 一个 `.md` 文件（带 frontmatter）；本文件只是索引，不放 memory 正文。相关研究规格 / 结论 / 方向在上级 `.claude/*.md`（见 skill `ttt-dit-context` §3）。

---

- [VFL 已知实现缺陷清单](project_vfl_known_gaps.md) — D1-D5 非紧急缺陷；CLAUDE.md 关于 PixArt+VFL 的描述与实际不符
- [VFL 训练信号源根因诊断](project_vfl_signal_source_flaw.md) — 50k 实验证明 LoRA 训练无效；target 是 base forward 输出导致 no-op 起点对称性 loss=0；P0 必须修信号源
- [COVR 系统全貌](project_covr_system.md) — 动机/方法/架构/实验发现/文件索引；bandit FID↓7.3但IS↓18%；safety shadow是速度瓶颈；50k实验92%选timestep_prior
- [COVR 离线探针结论](project_covr_offline_probe_findings.md) — crossover 真实(p=0)但 causal context 不可用(p=0.84)；audit 只测到 cache distance 1-4，激进 mask 不能外推
- [COVR crossover 判据设计](project_covr_crossover_gate_design.md) — in-sample oracle gap 与三个 permutation null 全不可用；OOS 特征取自 D 会泄漏，n=500 时纯噪声误报 99%
- [COVR OOS 判据两次修正](project_covr_crossover_oos_gate_rule.md) — 先换掉 1NN label transfer，再补两道量级地板；permutation null 有 0 点质量 0.56-0.60，p≤0.05 单独等价于 frac>0
- [COVR 500 图 budget probe 结果](project_covr_budget_probe_500_results.md) — inception_conf 首次通过 [a] Spearman=1.000；两个 metric 的"UTILIZABLE"都是误报，回收 0.03-0.28 FID vs 4.8 FID 地板
- [COVR static K=8 配对结论](project_covr_static_k8_paired_result.md) — uniform 胜 geometric；严格 K=8 对照中 uniform FID 优先、plain TeaCache IS 优先，形成稳定 Pareto 权衡
- [COVR bandit 先验锁死](project_covr_bandit_prior_lockin.md) — 92.1% 收敛=epsilon-greedy 无学习地板 92.5%；penalty 0.25 比 reward 大 4167×，最优臂需 3 亿张图才能翻盘
- [COVR 决策粒度是 batch](project_covr_decision_granularity.md) — trajectory=batch 不是 image；bs=32 含 ~31.5 类，per-image 天花板被批平均抹掉两个数量级；必须 BATCH_SIZE=1
- [COVR contextual+efficiency bandit 已 GPU 验证=负](project_covr_contextual_efficiency_bandit.md) — deferred-commit LinUCB + λ·FLOPs reward；500图首跑两个维度 contextual gain 都≤0（fidelity -6.4e-6, combined -2.6e-4），per-image Pareto 异质性不存在；STOP，不扩大 GPU
- [简短提供测试命令](feedback_concise_test_commands.md) — 远程测试场景只给从真实入口开始的核心命令
- [input_tokens 错误后自动续做](feedback_resume_after_input_tokens_error.md) — 遇到客户端 token 统计错误时拆分调用并持续完成任务，不输出空响应
