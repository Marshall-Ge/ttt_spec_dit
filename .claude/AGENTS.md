# TTT-DiT 项目记忆 — Agent 入口

> 本目录是本项目的工作记忆（对标 `lttta-main/.codex/`）。**先读本文件**，再按需深入 `memory/`、`skills/` 与仓库根 `CLAUDE.md`。

## 一句话定位

对 **DiT-2-256** 与 **PixArt-XL-2-512x512** 两种扩散生成模型做**推理加速**研究：评估 SpecA / TeaCache / DDIM 三种基础加速器对生成质量（FID/IS/CLIP）与效率（FLOPs/Latency）的影响，并在此基础上探索**在线学习改进缓存**的两条正交路线 —— **TTT**（改缓存内容，DiT-only）与 **VFL/COVR**（改缓存决策）。

## 关键事实速查

- **模型与权重**：DiT-2-256（675M, 256², adaLN-Zero, class-conditional）在 `~/autodl-fs/models/dit_2_256/`；PixArt-XL-2（2.5B, 512², adaLN-Single + T5）在 `~/autodl-fs/models/models--PixArt-alpha--PixArt-XL-2-512x512/`。SD VAE scaling_factor=0.18215。
- **数据**：ImageNet val `~/autodl-fs/data/imagenet/val/`（1000 类 × 50 张 = 50k），COCO `~/autodl-fs/data/coco/`。ImageNet 类 ID 需经 `ilsvrc2012_to_dit_id.json` 翻译（ILSVRC2012_ID → DiT WNID 字母序）。
- **运行环境**：AutoDL GPU（路径前缀 `~/autodl-fs/`）；HF 走镜像 `HF_ENDPOINT=https://hf-mirror.com`（`config.py` 已默认设置）。
- **三个基础加速器**（可叠加、互不依赖）：SpecA（per-block Taylor 缓存）、TeaCache（per-step 残差缓存）、DDIM（步数压缩）。
- **两条在线学习路线**（不影响基础加速器，可叠加）：
  - **TTT** — DiT-only，~0.92M `SessionAdaLNModulator` 插件，蒸馏 teacher 信号改进 TeaCache 的 stale cached state（缓存**内容**）。
  - **VFL/COVR** — 三层反馈环（L1 在线阈值校准 / L2 分层回放 / L3 异步 LoRA）+ 反事实在线学习，改进缓存**决策**。
- **当前研究状态**（2026-08-20 更新）：COVR full-context 在线控制器主线 **STOP**（`covr_spec_research_conclusion.md`）；timestep feedback active 方向按质量 gate 关闭（2026-08-19，三环证据链闭合）；"Taylor term 范数前兆信号"离线 gate **负**（2026-08-20，`memory/project_taylor_term_norm_signal_offline_20260820.md`）。**当前活跃方向：P1 质量约束的在线预算校准**（inception-confidence session 闭环，规格与 gate 见 `.claude/covr_quality_constrained_budget_20260820.md`；2026-08-21 全链路五脚本定稿并通过本地合成数据冒烟——extract_inception_conf / analyze_conf_rank_validity（三态裁决+精确置换）/ build_conf_costs / simulate_conf_budget_controller（[P1-b]/[P1-c] 判决行+级联降档 bug 修复）/ compute_mixture_fid（与 fid_is.py 逐位一致的混合 FID）——待 GPU 机恢复后按规格 §5 执行三个离线 gate）。VFL LoRA 训练信号源根因缺陷见 `memory/project_vfl_signal_source_flaw.md`。

## 三类资产分工（避免重复）

| 资产 | 位置 | 写什么 | 不写什么 |
|---|---|---|---|
| **代码交接文档** | 仓库根 `CLAUDE.md` | 目录结构、各文件职责、forward 分支、超参表、坑、已验证组合、常用命令 | 研究 trial / gate 结论 / 候选方向 |
| **背景 primer skill** | `.claude/skills/ttt-dit-context/SKILL.md` | 大背景与主线思路（在线学习负反馈），稳定高层 | 方法细节、实验日志 |
| **工作流 skill** | `.claude/skills/ttt-dit-workflow/SKILL.md` | 开发/评测/标定/记录的标准流程，大体即可 | 细枝末节参数 |
| **原子事实 memory** | `.claude/memory/*.md` + `MEMORY.md` 索引 | 一条结论 / 一个偏好 / 一个坑 = 一个文件 | 大块研究规格 |
| **大块研究文档** | `.claude/covr_*.md` 等根文件 | 阶段规格 / 结论 / 方向对比 | 代码结构（去 `CLAUDE.md`） |

## 入口指引

- **开工前**：读本文件 → 按需读 `skills/ttt-dit-context/SKILL.md`（背景）与 `skills/ttt-dit-workflow/SKILL.md`（流程）→ 读 `memory/MEMORY.md` 索引定位相关原子事实 → 必要时读 `CLAUDE.md`（代码细节）与 `.claude/covr_spec_research_conclusion.md`（确认 STOP/GO 与已排除路线）。
- **记录时**：写回本目录，不写用户级 auto-memory。原子事实 → `memory/`（一文件一条 + 更新 `MEMORY.md` 索引）；大块研究 → `.claude/` 根的 `covr_<topic>_<YYYYMMDD>.md` 或对应文件。

详细说明：
- 项目背景与主线思路：`skills/ttt-dit-context/SKILL.md`
- 标准工作流：`skills/ttt-dit-workflow/SKILL.md`
- 代码结构 / 各文件职责 / 坑 / 命令：仓库根 `CLAUDE.md`
- 原子事实索引：`memory/MEMORY.md`
- 阶段研究结论：`.claude/covr_spec_research_conclusion.md`、`.claude/covr_spec_research_plan.md`、`.claude/covr_cache_decision_directions.md`
