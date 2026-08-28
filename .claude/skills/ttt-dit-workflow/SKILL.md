---
name: ttt-dit-workflow
description: TTT-DiT 扩散推理加速项目的标准工作流（开发 / 评测 / 标定 / 记录）。只讲大体流程，细枝末节参数与代码细节见仓库根 CLAUDE.md。
---

# TTT-DiT 标准工作流

## 何时使用

在 TTT-DiT 项目上进行代码开发、跑评测、标定加速器系数、记录实验发现时使用。
先读 `.claude/AGENTS.md` 与背景 skill `ttt-dit-context`，本技能只讲流程，不重复背景与代码细节。

## 工作流步骤

### 1. 环境就绪

- AutoDL GPU 环境，模型/数据在 `~/autodl-fs/`（路径由 `config.py` 解析，不写死机器）。
- HF 走镜像：`config.py` 已默认 `HF_ENDPOINT=https://hf-mirror.com`。
- 验证 CUDA 可用、权重与数据集就位（DiT/PixArt 权重、ImageNet val、COCO）。

### 2. 改动前的状态确认

- 读 `.claude/covr_spec_research_conclusion.md` 确认当前 **STOP/GO** 与已排除路线，不重走已证伪方向。
- 按 `memory/MEMORY.md` 索引定位相关原子事实（VFL 信号源缺陷、COVR bandit 锁死、决策粒度等）。
- 代码结构 / 各文件职责 / 坑查仓库根 `CLAUDE.md`，不在本技能重复。

### 3. 代码改动原则

- **无 monkeypatch**：加速逻辑只走两条路 —— 模型内部显式 forward 分支（SpecA `current`/`cache_dic`、TeaCache `teacache_state`、TTT `ttt_state` 作为可选参数传入 `forward()`），或采样循环层（`teacache_step()` / `ttt_train_step` / `ttt_record_skip`）。
- **VFL 不走 forward 参数**：通过 `feedback.vfl/vfl_state.py` 进程级单例钩入，不改 forward 签名。
- `eval/` 与 `dataset/` 目录不动（`dataset/imagenet.py` 的类 ID 翻译除外）。
- Generator 职责缩小为管理 VAE/scheduler/device/dtype/encode_prompt，不参与 forward 逻辑。

### 4. 标定（改步数必跑）

```bash
python scripts/calibrate_teacache.py --model dit --num_steps 50 --num_runs 10
```

TeaCache 多项式系数对步数敏感，换 `num_steps` 必须重新标定，否则训练范围外振荡产生负值。

### 5. 单组合评测

```bash
python main.py --model dit --task c2i --dataset imagenet \
    --method <baseline|teacache|speca|ddim> \
    --metrics fid is latency flops speed \
    --seed 42 --num_steps 50 --n_prompts 80 --guidance_scale 4.5 --batch_size 32
```

在线学习扩展叠加：`--ttt --ttt_lr 1e-4 --ttt_micro_epochs 3`（DiT-only，须配 `--method teacache`）；`--vfl`（完整三层）或 `--vfl-no-train`（仅 L1 阈值校准）。

### 6. 评测流程要点（大体）

`main.py:parse_args → validate_args → run_dit/run_pixart`：
1. `Generator.load()` 加载 VAE + transformer。
2. 加载数据集（ImageNet 自动做类 ID 翻译）。
3. **FLOPs 测量必须在加速器初始化之前**（走 vanilla forward 路径）。
4. 创建加速器状态 → 逐 batch 去噪循环（CFG doubling → 逐步 method dispatch → learned-sigma 切割 → scheduler.step → unchunk）→ VAE decode。
5. `FIDISComputer.add/compute/cleanup`，聚合指标存 `results.json`。

### 7. 独立入口（不走 main.py）

- `run_ttt_benchmark.py` — TTT 单类/跨类全评估，plugin 跨 image 持续训练。
- `continual_inference_runner.py` — Session 3 单类 N 图流，γ 课表 + CSV 遥测。
- `run_session2_flywheel.py` — Session 2 VFL 飞轮（加载 Session 1 LoRA + exploit + SpecA，无 TTT）。
- `ttt_baseline.py` — Phase 1 历史遗留原型（无 TTT）。

### 8. COVR 探针与 smoke

```bash
bash scripts/run_covr_forced_smoke.sh uniform        # forced COVR smoke
bash scripts/run_covr_v2_viability_probe.sh          # batch=1 OOS viability probe
```

`viability_report.json` 的 `recommendation=PASS` 只表示有逐图 headroom 值得进入确认实验，**不是** FID/IS 结论；`STOP`/`INSUFFICIENT_DATA` 时不要扩大 GPU 实验。

### 9. 记录与归档

- **写回项目，不写用户级 auto-memory**。
- 原子事实（一条结论 / 一个偏好 / 一个坑）→ `.claude/memory/` 一文件一条 + 更新 `MEMORY.md` 索引。
- 大块研究规格 / 阶段结论 / 方向对比 → `.claude/` 根的 `covr_<topic>_<YYYYMMDD>.md`。
- 代码结构变更 → 更新仓库根 `CLAUDE.md`（唯一代码交接文档）。
- 不要把 trial memory 散落到对话或代码注释里。

## 常见坑速查

- 改 TeaCache `num_steps` 不重新标定 → 多项式越界振荡 → rescaled accumulation 异常。
- DiT CFG 用 `model_out[:, :in_channels]` 切割（in_channels=4），硬编码 `:3` 是 bug。
- TTT plugin 必须 fp32、骨干 fp16（hidden-state MSE 可超 fp16 上限 65504 → NaN）。
- `_denoise_loop_ttt` 有意无 `@torch.no_grad()`（autograd 只流过 plugin）；骨干冻结由 runner 负责。
- SpecA `current['step']` 从 `num_steps-1` 递减到 0（reverse denoising order）。
- VFL checkpoint 自动加载但主流程未过 EvalGate 闸门（生产化前需补）。
- COVR 决策粒度是 batch 不是 image，per-image 天花板被批平均抹掉 → 必须 `BATCH_SIZE=1`。
