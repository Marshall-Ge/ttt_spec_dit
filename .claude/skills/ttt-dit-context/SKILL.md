---
name: ttt-dit-context
description: Use when starting or resuming work on the TTT-DiT diffusion inference-acceleration project, or when discussing COVR / VFL / TTT / SpecA / TeaCache / DDIM / online-learning-for-caching, or when recording new experimental findings. Provides only the high-level project background and the central thesis (online learning via counterfactual negative feedback). Specific method specs, trial logs, gate verdicts and candidate directions are NOT here — they live in project .claude/. Read .claude/ before acting; write back to .claude/ when recording.
---

# TTT-DiT 项目背景（高层 primer）

> 这个 skill **只讲大背景和主线思路**。具体的方法分支、实验过程、gate 结论、候选方向、已证伪路线**都不在这里**——它们在项目 `.claude/` 下（见 §3）。**开工前先读 `.claude/`，记录时写回 `.claude/`。** 本 skill 保持稳定、高层，不要把 trial 细节塞进来。

## 1. 项目在做什么

扩散生成模型的**推理加速**研究。对两个模型评估"加速方法对生成质量与效率的影响"：

- **DiT-2-256**（class-conditional，256×256，675M，adaLN-Zero）
- **PixArt-XL-2-512x512**（text-to-image，2.5B，adaLN-Single + T5）

三个**基础加速器**（可叠加、互不依赖）：

- **SpecA** — per-block Taylor 级数缓存（每步决定 full 重算 vs Taylor 外推）
- **TeaCache** — per-step 残差缓存（相邻步调制信号相似则跳过整个 block stack）
- **DDIM** — 步数压缩

评估维度：质量（FID / IS / CLIP / LPIPS）vs 效率（FLOPs / latency / wall time）。代码架构、目录结构、各加速器实现细节见仓库根 `CLAUDE.md`（那是**代码交接文档**，不是研究 trial memory）。

## 2. 主线思路：在线学习的负反馈信号

基础加速器的缓存决策是"盲"的——固定阈值、固定 schedule、固定 decay。本项目的核心赌注：

> **把每一次缓存决策变成一个带标签的学习机会。** 当廉价的缓存/草稿输出被昂贵的 full 计算验证时，两者的偏差（defect / residual error）就是一个**负反馈信号**——它告诉你"这次缓存错得有多离谱"——可以用来**在线改进未来的缓存，且不更新生成模型权重**。

这个负反馈能改进两个**正交的轴**：

| 轴 | 改什么 | 对应工作 |
|---|---|---|
| **缓存内容** | cached 值本身（skip step 修正 stale state） | **TTT**（Test-Time Training，DiT-only，~0.92M 微型插件蒸馏 teacher 信号） |
| **缓存决策** | 什么时候 / 哪里 refresh | **VFL**（三层反馈环：在线阈值校准 / 分层回放 / 异步 LoRA）、**COVR**（反事实在线学习 + bandit） |

## 3. 当前状态与细节：在项目 `.claude/` 里，先读再动手

**具体的方法规格、实验过程、gate 结论、候选方向、已证伪路线，全部在项目 `.claude/` 下。** 动手前必须先读，避免重走已证伪的路。当前关键文件（若有更新日期的同类文件，以**最新日期**为准）：

**原子事实 memory** 在 `.claude/memory/`（13 条 + `MEMORY.md` 索引）

**所有 memory **（从用户级 `~/.claude/.../memory/` 迁入并退役源端）：原子事实进 `.claude/memory/`，大块研究文档进 `.claude/` 根。**不要再往用户级 auto-memory 写本项目的研究结论。**

## 4. 工作流约定

- **开工前**：读 `.claude/covr_spec_research_conclusion.md` 确认当前 STOP/GO 与已排除路线；按需读 plan / directions 和 `.claude/memory/` 里相关条目。不要凭印象假设某方向"还没试过"。
- **记录时**：**写回项目，不写用户级**。原子事实（一条结论 / 一个偏好 / 一个坑）→ 进 `.claude/memory/`（一个文件一条 + 更新 `MEMORY.md` 索引）；大块研究规格 / 阶段结论 / 方向对比 → 进 `.claude/` 根的对应文件或新建 `covr_<topic>_<YYYYMMDD>.md`。不要把 trial memory 散落到对话或代码注释里。
- **本 skill 保持稳定、高层**。只有当大背景或主线思路本身变化时才改它；方法细节、实验日志、特定方向一律进 `.claude/` 或 `.claude/memory/`。