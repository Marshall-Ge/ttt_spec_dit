# TTT-DiT 项目上下文 — 给新 Agent 的完整交接文档

## 1. 项目目标

对 DiT-2-256 和 PixArt-XL-2-512x512 两种扩散生成模型进行推理加速，评估加速方法对生成质量（FID/IS/CLIP）和效率（FLOPs/Latency）的影响。加速方法包括 SpecA（Per-block Taylor 缓存）、TeaCache（Per-step 残差缓存）、DDIM 步数压缩。

在此基础上项目扩展了两个**在线学习**方向（不影响基础加速器，可叠加使用）：

- **TTT (Test-Time Training)** — DiT-only，在 TeaCache 之上挂载一个 ~0.92M 参数的 `SessionAdaLNModulator` 插件，**改进缓存内容**（hidden state correction）：calc step 用 teacher 信号蒸馏训练插件，skip step 用插件调制 stale cached state，绕过 28 个 block。
- **VFL (Verification Feedback Loop)** — 三层递进架构（L1 在线阈值校准 / L2 分层回放缓冲 / L3 LoRA 异步微调），**改进缓存决策**（SpecA threshold / backbone 轨迹平滑度）。可与 TTT 同时启用，互不依赖。

## 2. 模型和权重路径

| 模型 | 参数量 | 分辨率 | 架构 | 权重路径 |
|------|--------|--------|------|----------|
| DiT-2-256 | 675M | 256×256 | adaLN-Zero, class-conditional | `~/autodl-fs/models/dit_2_256/` |
| PixArt-XL-2 | 2.5B | 512×512 | adaLN-Single, T5 text encoder | `~/autodl-fs/models/models--PixArt-alpha--PixArt-XL-2-512x512/` |

- DiT: 28 blocks, 16 heads × 72 dim, in_channels=4, out_channels=8 (learned sigma: noise + variance)
- PixArt: 28 blocks, cross_attention_dim=1152, T5 caption_channels=4096, 3 submodules (attn1/attn2/ff)
- SD VAE: `~/autodl-fs/models/dit_2_256/vae/`, scaling_factor=0.18215
- ImageNet val: `~/autodl-fs/data/imagenet/val/` (1000 个类目录, 每类 50 张, 共 50k)
- COCO: `~/autodl-fs/data/coco/`
- devkit: `~/autodl-fs/data/imagenet/ILSVRC2012_devkit_t12/`

## 3. 目录结构和各文件职责

```
~/ttt_spec_dit/
├── config.py                  # 全局路径、默认超参 (DIT_REPO, IMAGENET_DIR, SPECA_DEFAULTS...)
├── main.py                    # CLI 入口: parse_args() + validate_args() → 分发到 run_dit/run_pixart
├── utils.py                   # CudaTimer, VAE decode, save_image, ensure_real_299()
├── run_dit.py                 # DiTGenerator + run_c2i(args) — DiT 编排器/采样入口，集成 TTT/VFL
├── run_dit_shared.py          # 纯生命周期辅助: _GenerationProfiler + _covr_* canonical JSON / resume / sentinel (被 run_dit 与 COVR runtime 共享)
├── run_pixart.py              # PixArtGenerator + run_t2i/run_c2i — 集成 VFL（无 TTT）
├── dit_coef.json              # DiT TeaCache 标定系数 (poly4, 50 步标定)
├── pixart_coef.json           # PixArt TeaCache 标定系数
├── continual_inference_runner.py  # Session 3 入口: 单类 N 图流 TTT session, γ 课表 + CSV 遥测
├── run_session2_flywheel.py       # Session 2 入口: 加载 Session 1 LoRA + VFL exploit 模式 + SpecA 推理
├── run_ttt_benchmark.py           # TTT 独立 benchmark: 单类/跨类全评估 (FID/IS), plugin 跨 image 持续训练
├── ttt_baseline.py                # Phase 1 原型: PixArt 特征探测 + 开环线性推测 baseline (历史遗留, 无 TTT)
├── test_checkpoint_manager.py     # VFL checkpoint manager (retention) 单测
├── models/
│   ├── __init__.py            # 导出 DiTTransformer2D, PixArtTransformer2D
│   ├── dit.py                 # DiTTransformer2D — 显式 forward, SpecA/TeaCache/TTT 分支可见
│   │                          #   forward(ttt_state=...) → _forward_ttt(); CFG 路由 forward_with_cfg_ttt
│   ├── pixart.py              # PixArtTransformer2D — SpecA/TeaCache 分支 (无 TTT)
│   └── ttt_plugin.py          # SessionAdaLNModulator (~0.92M params) + ttt_init/train_step/record_skip
│                              #   数学: Z_out = Z_cached*(1+Δγ) + Δβ; 零初始化保证 Image 1 step 0 恒等
├── accelerators/
│   ├── __init__.py            # 导出所有纯函数
│   ├── speca.py               # SpecA: speca_init, speca_cal_type, taylor_cache_init,
│   │                          #   derivative_approximation, taylor_formula, cache_step_dit/pixart,
│   │                          #   compute_error_gate (cosine/l1/l2/relative_l1/relative_l2)
│   ├── teacache.py            # TeaCache: teacache_init/decide/cache_residual/apply_residual/step/reset
│   │                          #   + compute_modulated_input(_dit) — 调制信号提取
│   ├── covr_runtime.py        # COVR runtime 边界 (可选插件): COVRMode/COVRRuntimeConfig/
│   │                          #   COVRRuntime facade + Forced/ExperimentalBandit backend + recorder
│   │                          #   Phase 1 仅 DiT 非 TTT 主去噪循环; 禁用时不构造任何对象
│   └── covr_bandit.py         # ConservativeTemplateBandit (EXPERIMENTAL, 语义冻结) + manifests
├── verification_feedback_loop/    # VFL 子系统 (三层架构, 详见 §12)
│   ├── __init__.py            # 导出所有公共符号
│   ├── config.py              # VFLConfig: accept_sample_rate, buffer_capacity_per_stratum, loRA_rank...
│   ├── verification_hook.py   # VerificationEvent + record_event + make_timestep_bucket (3 桶)
│   ├── online_calibration.py  # L1: OnlineCalibrator — (layer_id, bucket) 级 EMA 阈值 mean+k*std
│   ├── replay_buffer.py       # L2: StratifiedReplayBuffer + AnchorSample, reservoir sampling
│   ├── lora_adapter.py        # L3 适配器: attach_lora_all_layers 给 6 个 Linear/层 挂 LoRA (B 矩阵零初始化)
│   ├── curvature_loss.py      # L3 损失: trajectory_curvature_loss (低阶多项式拟合残差) + compute_training_loss
│   ├── async_trainer.py       # L3: AsyncTrainer/AsyncTrainingWorker — daemon 线程, deepcopy fp32 模型训练
│   ├── eval_gate.py           # Canary 发布闸门: 质量回归 + reject 率下降双检查
│   ├── version_registry.py    # AdapterStatus (ACTIVE/STALE/UNKNOWN) + 版本匹配校验
│   ├── vfl_state.py           # 进程级全局单例: set_vfl_buffer/calibrator, record_speca/teacache_event
│   ├── demo_e2e.py            # VFL 端到端 demo (EvalGate 自动发布流程的参考实现)
│   └── tests/                 # 单测: buffer, calibration, curvature, async_worker, hotpath, e2e
├── eval/                      # 指标模块 (不动)
│   ├── fid_is.py              # FIDISComputer — torch-fidelity 封装，add()+compute()+cleanup()
│   ├── latency.py             # FLOPsMetric — _profile_once() 调用 transformer() 测 FLOPs
│   │                          #   add_generation(teacache) 需要 .decisions 属性 (list of "calc"/"skip")
│   ├── clip_score.py, lpips.py, mse.py, image_reward.py, gen_eval.py
│   └── base.py                # Metric ABC
├── dataset/                   # 数据集 (不动)
│   ├── imagenet.py            # ImageNetDataset: 自动加载 ilsvrc2012_to_dit_id.json 做类 ID 翻译
│   ├── coco.py, drawbench.py, geneval.py
│   └── base.py
└── scripts/
    ├── run.sh                 # 20 combo benchmark 脚本
    ├── run_full_smoke.sh      # 完整冒烟测试
    ├── run_covr_forced_smoke.sh # 一键生成 manifest 并执行 forced COVR smoke
    ├── check_covr_forced_smoke.py # 检查 forced smoke 的结果与 schema
    ├── run_covr_bandit_resume_smoke.sh # experimental bandit 首段+恢复段 smoke
    ├── check_covr_bandit_resume_smoke.py # 检查 state 连续性/resume window/schema
    └── calibrate_teacache.py  # TeaCache 多项式系数标定脚本
```

## 4. 架构核心原则

**无 monkeypatch**：旧代码 (`pipelines/t2i.py`, `pipelines/c2i.py`, `models/base.py`) 已删除。加速逻辑只有两种方式：

1. **模型内部显式分支** — SpecA 的 `current`/`cache_dic`、TeaCache 的 `teacache_state`、TTT 的 `ttt_state` 都作为可选参数传入 `forward()`，在 pos_embed 和 blocks 之间做决策
2. **采样循环层** — `teacache_step()` 在循环中计数；TTT 的 `ttt_train_step` / `ttt_record_skip` 也在循环层调用

**VFL 不走 forward 参数**，而是通过 `verification_feedback_loop/vfl_state.py` 的进程级全局单例（`_vfl_buffer` / `_vfl_calibrator` / `_vfl_step_idx`）钩入。`models/dit.py` 和 `models/pixart.py` 在 SpecA check_layer 和 TeaCache calc 步后直接调用 `_vfl_record_speca_event` / `_vfl_record_teacache_event`（通过 vfl_state 全局 hook），不改 forward 签名。这是有意为之：VFL 是旁路观测/校准，不参与前向计算。

**Generator 保留**，但职责缩小为：管理 VAE/scheduler/device/dtype/encode_prompt。不参与 forward 逻辑。

**eval/ 和 dataset/ 目录不动** — 但 dataset/imagenet.py 增加了 ILSVRC2012_ID → DiT class_id 翻译（通过 `ilsvrc2012_to_dit_id.json`）

## 5. 推理流程 (以 DiT c2i 为例)

```
main.py:parse_args()
  → validate_args()
  → run_dit.run_c2i(args)
    → DiTGenerator.load()           # 加载 VAE + DiTTransformer2D
    → ImageNetDataset(n, seed)      # 加载数据，shuffle，返回 (img_path, prompt, DiT_class_id)
    → FLOPsMetric(generator).profile()  # 必须在加速器之前测！
    → 创建加速器状态: teacache_init() 或 speca_init()
    → for batch in dataset:
        generator.generate(prompts, seeds, method=...)
          → _denoise_loop():
            init latents → CFG doubling
            for t in scheduler.timesteps:
              → method dispatch:
                teacache: transformer(x, t, teacache_state=state, ...) + teacache_step(state)
                speca:    transformer(x, t, current=cur, cache_dic=dic, ...)
                baseline: transformer(x, t, ...)
              → learned-sigma split: noise_pred[:, :in_channels]
              → scheduler.step()
            → unchunk (CFG)
          → VAE decode → image tensor
        → FIDISComputer.add(img, tag=class_name)
        → 保存图片 (限 img_save_limit 张，含类名)
    → FIDISComputer.compute() → cleanup()  # 删除 temp generated_299/
    → 聚合指标 → 保存 results.json
```

## 6. SpecA 实现细节

### 6.1 核心概念

Per-block per-submodule 的 Taylor 级数缓存。每步决定是 `full`（计算全部 blocks 并缓存各子模块输出+有限差分导数）还是 `Taylor`（用缓存预测，跳过 attention/MLP 计算）。

### 6.2 关键函数 (accelerators/speca.py)

- `speca_init(num_steps, base_threshold, decay_rate, min/max_taylor_steps, max_order, num_layers, error_metric, check_layer)` → `(cache_dic, current)`
- `speca_cal_type(cache_dic, current)` — 根据 error history + decay 决定当前步是 `full` 还是 `Taylor`。Side-effect: 设置 `current['type']`
- `taylor_cache_init(cache_dic, current)` — 第一步 (step=num_steps-1) 分配 cache slot
- `derivative_approximation(cache_dic, current, feature)` — 有限差分计算各阶导数，存在 `cache_dic['cache'][-1][layer][module]`
- `taylor_formula(module_list, distance)` — 用 1/n! 系数计算 Taylor 预测
- `cache_step_dit(x, attn_list, mlp_list, gate_msa, gate_mlp, distance)` — DiT 2 子模块预测
- `cache_step_pixart(x, attn1_list, attn2_list, ff_list, gate_msa, gate_mlp, distance)` — PixArt 3 子模块预测（attn2 无 gate）
- `compute_error_gate(x, full_x, metric)` — 最后 block 的 Taylor vs full 误差，用于阈值决策

### 6.3 DiT forward 中的 SpecA 分支 (models/dit.py)

```
forward(x, t, current, cache_dic, teacache_state, class_labels):
  use_speca = current is not None and cache_dic is not None
  speca_cal_type(cache_dic, current)    # 决定 step_type
  x = pos_embed(x)
  for layer, block in enumerate(blocks):
    norm_out, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(x, t, class_labels)
    if step_type == 'full':
      current['module'] = 'attn'; taylor_cache_init(); attn_out = block.attn1(norm_out)
      derivative_approximation(cache_dic, current, attn_out)
      x += gate_msa * attn_out
      # same for 'mlp' via norm3 + block.ff
    elif step_type == 'Taylor':
      x = cache_step_dit(x, cache[-1][layer]['attn'], cache[-1][layer]['mlp'], ...)
      if do_check (last block + accumulated >= min_taylor):
        计算 full block 做 error probe → compute_error_gate
  tail: norm_out + proj_out_1/2 + unpatchify
```

### 6.4 关键超参

| 参数 | DiT | PixArt | 含义 |
|------|-----|--------|------|
| check_layer | 20 | 24 | 在哪个 block 做 Taylor vs full 误差探测 |
| error_metric | cosine_similarity | cosine_similarity | 误差度量 |
| base_threshold | 0.01 | 0.01 | 基础阈值，随 progress decay |
| decay_rate | 0.01 | 0.01 | 阈值衰减率 |

### 6.5 PixArt 3 子模块结构

```
每个 block 的 forward:
  1. attn1: x = x + gate_msa * attn1( norm1(x) * (1+scale_msa) + shift_msa )
  2. attn2: x = x + attn2( x, encoder_hidden_states )          ← raw hidden, 无 gate
  3. ff:    x = x + gate_mlp * ff( norm2(x) * (1+scale_mlp) + shift_mlp )
```

## 7. TeaCache 实现细节

### 7.1 核心概念

Per-step 的残差缓存。在 pos_embed 输出处比较 block0 的调制信号，如果相邻步骤的信号相似，跳过整个 block stack，直接用上次的残差 `(blocks_output - blocks_input)` 近似当前输出。尾部和 unpatchify 始终计算。

### 7.2 关键函数 (accelerators/teacache.py)

- `teacache_init(num_steps, rel_l1_thresh, coefficients)` → state(dict)
- `teacache_decide(state, modulated_input)` → (should_calc, raw_diff) — 核心决策：首尾强制 calc，中间根据 accumulated rescale 判断
- `teacache_cache_residual(state, out, ori)` — 存 `(out - ori)`
- `teacache_apply_residual(state, hidden_states)` — `x + residual`
- `teacache_step(state)` — cnt += 1
- `teacache_reset(state)` — 重置运行时状态
- `compute_modulated_input(transformer, hidden_states, timestep_emb)` — PixArt: block0.norm1 调制
- `compute_modulated_input_dit(transformer, hidden_states, timestep, class_labels)` — DiT: block0.norm1 第一个返回值
- `teacache_stats(state)` — 计算 skip_ratio 等聚合统计

### 7.3 DiT forward 中的 TeaCache 分支 (models/dit.py)

```
forward(x, t, current, cache_dic, teacache_state, class_labels):
  use_teacache = teacache_state is not None and not use_speca
  x = pos_embed(x)
  if use_teacache:
    modulated = compute_modulated_input_dit(self, x, t, class_labels)
    should_calc = teacache_decide(teacache_state, modulated)
    if not should_calc:
      x = teacache_apply_residual(teacache_state, x)  # skip blocks!
      goto tail  (直接返回)
    ori_hidden = x.clone()  # 保存输入用于 residual 计算
  # Block loop (full mode)
  for block in blocks: ...
  if use_teacache:
    teacache_cache_residual(teacache_state, x, ori_hidden)
  tail...
```

### 7.4 TeaCache 决策算法

```
raw_diff = |modulated - prev|.mean() / |prev|.mean()     # relative L1
rescaled = max(0, poly4(raw_diff))                       # poly4 clamp ≥ 0
accumulated += rescaled
should_calc = (cnt == 0 or cnt == num_steps-1) or (accumulated >= threshold)
if should_calc: accumulated = 0
```

### 7.5 系数标定

`scripts/calibrate_teacache.py --model dit --num_steps 50 --num_runs 10`

脚本收集 N 条 denoising trajectory 的 raw_diff 序列，拟合 poly4 使 rescale 缩放后 accumulation 达到 target skip rate。**步数必须与推理一致**，否则多项式在训练范围外振荡产生负值。

### 7.6 FLOPsMetric 与 TeaCache 的交互

`FLOPsMetric.add_generation(teacache)` 需要 `.decisions` 属性（list of "calc"/"skip"）。在 run_*.py 中通过 `SimpleNamespace(decisions=state["decisions"])` 传入。

## 8. 关键注意事项和坑

### 8.1 ImageNet 类 ID 翻译
dataset 目录用 ILSVRC2012_ID 序（0000=kit fox=278），DiT 用 WNID 字母序（0=tench）。`dataset/imagenet.py` 通过 `ilsvrc2012_to_dit_id.json` 做自动翻译。`imagenet_class_index.json` 提供类名映射。警告 "No class-name mapping found" 如果出现说明缺少这些文件。

### 8.2 DiT CFG 的 learned-sigma 通道
DiT 输出 8 通道：前 `in_channels=4` 是 noise prediction，后 4 是 variance。CFG 只对 noise 通道做外推，用 `model_out[:, :config.in_channels]` 切割。以前硬编码 `:3` 是 bug（只 CFG 了 3 个 noise channel）。

### 8.3 forward_with_cfg 的 latent 处理
DiT `forward_with_cfg` 接受已经 doubled 的 latent `[cond, null]`，但只取 cond 拷贝一份 `[cond, cond]` 通过模型，分别用 cond/null class 得到两个 noise pred，然后 CFG 外推。

### 8.4 PixArt attn2 无 gate 无 norm2
Cross-attention 入口是 raw hidden_states（不是 norm2 调制后的），输出直接加回去无 gate。这在 SpecA 的 `cache_step_pixart` 中有体现。

### 8.5 SpecA step 计数方向
`current['step']` 必须从 `num_steps-1` 递减到 0（reverse denoising order）。在采样循环中设置：`current['step'] = len(timesteps) - 1 - step_idx`

### 8.6 文件保存策略
- `generated/`: 原始分辨率 PNG，受 `--img_save_limit` 限制，含类名
- `generated_299/`: FID 临时目录，`FIDISComputer.add()` 写入，`compute()` 后 `cleanup()` 删除
- `real_299/`: symlink 到 `val_299_cache/`（一次性预处理全量 50k）
- 所有文件名格式: `{idx:06d}_{class_name}.png`

### 8.7 FLOPs 测量必须在加速器之前
`FLOPsMetric.profile()` 调用 `transformer(latent_input, timestep=..., class_labels=..., return_dict=False)` —— 这个 diffusers 风格的调用在新 forward 签名下兼容（`current=None, cache_dic=None` 走 vanilla 路径）。

### 8.8 TTT plugin 必须 fp32，骨干 fp16
`models/ttt_plugin.py:503-510`：hidden-state MSE 可超 65504 (fp16 max)，导致 NaN。plugin 仅 <1M 参数，fp32 开销可忽略。runner 在初始化时显式 `plugin = plugin.float()`。

### 8.9 `_denoise_loop_ttt` 有意无 `@torch.no_grad()`
`models/dit.py:431, 484-488` 注释：这是有意为之，autograd 只流过 plugin（loop 内显式用 `torch.no_grad()` 包裹非 plugin 部分）。给整个 loop 加 `@torch.no_grad()` 会切断 plugin 的训练梯度。

### 8.10 骨干冻结由 runner 负责
`run_dit.py:680-689`, `continual_inference_runner.py:168-177`：`requires_grad_(False)` 是 runner 层级的设置，model 文件不负责。TTT 启动时必须由 runner 显式冻结骨干，否则 autograd 会撑爆显存。

### 8.11 backbone NaN 安全网
`models/dit.py:528-536`：teacher `z_true` 中出现 NaN 时 fallback 到 stale cache 并跳过训练。这是为了对齐 fp16 推理的数值稳定性，不要删除。

### 8.12 TTT FLOPs 计算包含 plugin 开销
`run_dit.py:977-991`：每个 micro-epoch 约 19M FLOPs (fwd~6M + bwd~12M + opt~1M)。`--ttt` 启用后 FLOPs 数值会比纯 TeaCache 略高。

### 8.13 VFL checkpoint 自动加载但不经 EvalGate
`run_dit.py:759-769`：推理启动时 `find_latest_checkpoint()` 加载历史 LoRA，但主流程未调用 `EvalGate.evaluate()` 验证。生产化前需补 canary 闸门（参考 `verification_feedback_loop/demo_e2e.py`）。

### 8.14 VFL 与 TTT 可同时启用
两者互不依赖：TTT 改进缓存**内容**，VFL 改进缓存**决策**。`run_dit.py:920` 检查 `args.ttt` 和 `vfl_buf` 可组合使用。但实测组合较小，建议先单独验证。

### 8.15 VFL buffer 的 PixArt 内存陷阱
`replay_buffer.py:163`：`max_encoder_hidden_states_events=200` 限制 PixArt 的 T5 hidden states（约 600KB/event）写入数量。超过后新 event 丢弃 `encoder_hidden_states`。如果 L3 训练需要完整 context，注意这个上限。

### 8.16 TeaCache coefficients 对步数敏感
`continual_inference_runner.py:196-198`：换 `num_steps` 需重新跑 `scripts/calibrate_teacache.py`。多项式在训练范围外振荡产生负值，会导致 rescaled accumulation 异常。

## 9. 已验证的组合

### 基础加速器

| Model | Method | 状态 |
|-------|--------|------|
| DiT | baseline | ✅ FID 正常 |
| DiT | teacache | ✅ skip~48%, FID~225 (50步, γ=0.25) |
| DiT | ddim | ✅ |
| DiT | speca | ✅ FLOPs -70% |
| PixArt | baseline | ✅ |
| PixArt | teacache | ✅ skip~45% |
| PixArt | speca | ✅ FLOPs -65% |

### 在线学习扩展

| 组合 | 状态 | 备注 |
|------|------|------|
| DiT + teacache + TTT | ✅ | DiT-only；plugin fp32, 骨干 fp16；FLOPs 含 plugin 开销 |
| DiT + speca + VFL (--vfl-no-train) | ✅ | 仅 L1 阈值校准，无训练开销 |
| DiT + speca + VFL (完整三层) | ✅ | L3 daemon 线程异步训练 LoRA rank=4 |
| PixArt + speca + VFL | ✅ | 注意 `max_encoder_hidden_states_events=200` 内存阀 |
| DiT + teacache + TTT + VFL | ⚠️ 实验性 | 可同时启用但实测组合少，建议先单独验证 |
| PixArt + TTT | ❌ 不支持 | TTT 仅 DiT 实现 |

## 10. 常用命令

```bash
# 标定 TeaCache 系数（改步数必跑）
python scripts/calibrate_teacache.py --model dit --num_steps 50 --num_runs 10

# 单组合验证
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --seed 42 --num_steps 50 --n_prompts 80 \
    --guidance_scale 4.5 --batch_size 32

# 一键生成 version-key/manifest 并执行 forced COVR smoke
bash scripts/run_covr_forced_smoke.sh uniform

# 检查已完成的 forced COVR smoke（不重新跑 GPU）
python scripts/check_covr_forced_smoke.py /tmp/covr_forced_smoke

# experimental bandit 持久化/恢复 smoke（默认 2+2 张；不评估自适应有效性）
bash scripts/run_covr_bandit_resume_smoke.sh

# 全 20 组合 benchmark
N_PROMPTS=50 bash scripts/run.sh

# DiT + TeaCache + TTT 插件 (DiT-only)
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --ttt --ttt_lr 1e-4 --ttt_micro_epochs 3 \
    --seed 42 --num_steps 50 --n_prompts 80

# DiT + SpecA + VFL 完整三层 (L1+L2+L3)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops speed \
    --vfl --vfl_output_dir ./vfl_runs/exp1 \
    --seed 42 --num_steps 50 --n_prompts 80

# DiT + SpecA + VFL 轻量模式 (仅 L1 阈值校准, 无训练)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops speed \
    --vfl --vfl-no-train \
    --seed 42 --num_steps 50 --n_prompts 80

# TTT 独立 benchmark (跨类持续训练评估)
python run_ttt_benchmark.py --help

# Session 3 持续推理 (单类 N 图, γ 课表)
python continual_inference_runner.py --help
```

## 11. TTT (Test-Time Training) 实现细节

### 11.1 核心概念

DiT-only 的扩展。在 TeaCache 之上挂载一个 ~0.92M 参数的微型网络 `SessionAdaLNModulator` (`models/ttt_plugin.py:48`)，**修改 TeaCache 的 stale cached hidden state**，而非修改模型权重（骨干 Θ 被 `requires_grad_(False)` 冻结）。

数学形式 (`models/ttt_plugin.py:150-151`)：

```
Z_out = Z_cached * (1 + Δγ) + Δβ
```

Δγ、Δβ 由时间步嵌入和池化缓存状态驱动。`_init_zero_output` (`:104-114`) 保证初始映射为恒等（Δγ=Δβ=0），即 Image 1 step 0 输出与静态 TeaCache 一致。

工作流：
- **calc step** (TeaCache 决定要算) — 跑完整 28 个 block 得到 teacher 信号 `Z_true`，用 MSE loss 蒸馏训练 plugin 参数，做 `micro_epochs`（默认 3）次前向+反向以榨干 teacher 信号
- **skip step** (TeaCache 决定跳过) — 用 plugin 调制 stale cached state，绕过 28 个 block

### 11.2 关键函数 (models/ttt_plugin.py)

- `ttt_init(plugin, optimizer, lr, micro_epochs)` → `ttt_state` dict
- `ttt_train_step(ttt_state)` — 顶层调用（loop 层调用），实际检查 `z_pred/z_true` 是否被 `_forward_ttt` 置 None，是则返回 0.0
- `ttt_record_skip(ttt_state)` — skip step 计数
- `SessionAdaLNModulator` — `nn.Module`，~0.92M 参数

### 11.3 DiT forward 中的 TTT 分支

DiT `forward` (`models/dit.py:224-251`) 新增可选参数 `ttt_state: Optional[dict]` (`:232`)。激活条件：

```
use_ttt = use_teacache and ttt_state is not None    # 必须叠加在 TeaCache 之上
```

激活后路由到 `self._forward_ttt()` (`:291-294, 470-608`)，该函数在 TeaCache `should_calc` 决策之上运行。CFG 路径用特化的 `forward_with_cfg_ttt` (`:636-661`)，**不**走通用 `forward_with_cfg`。

**PixArt 不支持 TTT** — `models/pixart.py:200-210` 的 forward 签名无 `ttt_state` 参数，也无 `_forward_ttt` 方法。

### 11.4 训练触发与 micro_epochs

`_denoise_loop_ttt` (`models/dit.py:538-571`) 每步：
- calc step (`:554-559`) — 调 `ttt_train_step(ttt_state)`；实际训练在 `_forward_ttt` 内部 (`:547-559`) 完成，loop 层调用是双保险
- skip step (`:560-561`) — 调 `ttt_record_skip(ttt_state)`，仅统计计数

**micro_epochs** (`models/ttt_plugin.py:191-195`)：每个 calc step 重用一次昂贵的 28-block teacher 信号 `Z_true` 进行多次（默认 3 次）plugin 前向+反向训练，提升样本效率。

### 11.5 关键超参

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--ttt` | False | 启用 TTT 插件 (DiT-only, 必须配合 `--method teacache`) |
| `--ttt_lr` | 1e-4 | AdamW 作用于 plugin 参数的学习率 |
| `--ttt_micro_epochs` | 3 | 每个 calc step 内重复训练次数；1=单次, 3-5=更高效 |
| `--thresh` (γ) | 0.25 | TeaCache 阈值，控制 calc/skip 比率（TTT 依赖它） |

### 11.6 TTT 独立运行入口（不走 main.py）

| 文件 | 角色 |
|------|------|
| `run_ttt_benchmark.py` | 独立 benchmark：单类/跨类 TTT 全评估 (FID/IS)，plugin 跨所有 image 持续训练 |
| `continual_inference_runner.py` | Session 3：单类 N 图流 TTT session，γ 课表 (0.35→0.55→0.75)，baseline 参考预缓存，CSV 遥测 |
| `run_session2_flywheel.py` | Session 2：加载 Session 1 LoRA + VFL exploit-mode + SpecA 推理（**不含 TTT**，是 VFL 飞轮流水线） |
| `ttt_baseline.py` | Phase 1 原型：PixArt 特征探测 + 开环线性推测 baseline，历史遗留代码（无 TTT） |

四个文件都有独立 `main()` / `parse_args()`，是 main.py 之外的并行入口。

## 12. VFL (Verification Feedback Loop) 实现细节

### 12.1 三层架构

| 层 | 模块 | 是否有梯度 | 触发条件 | 产出 |
|----|------|-----------|---------|------|
| L1 | `online_calibration.py` | 否 (常驻) | 每个 VerificationEvent 到达 | 动态阈值 `mean + k*std`，替换 SpecA 固定 decay 公式 |
| L2 | `replay_buffer.py` | 否 (被动存储) | 每个事件/anchor 自动写入 | 分层 reservoir sampling 数据，供 L3 读取 |
| L3 | `async_trainer.py` | 是 (异步) | buffer_ready_min_strata≥2, total≥200, anchor≥10 | LoRA checkpoint (lora_candidate_vNNN.pt) |

**L1/L2/L3 数据流**：
1. 推理中 SpecA check_layer 或 TeaCache calc 步计算 `(predicted_feature, true_feature, error_value, latent_input)`
2. `record_speca_event` / `record_teacache_event` (`vfl_state.py:171-273`) 打包成 `VerificationEvent`
3. 同时写入 `StratifiedReplayBuffer` (L2) 和 `OnlineCalibrator.update()` (L1)
4. L3 daemon 线程从 buffer `sampleTraining_batch()`，跑 `compute_training_loss` 更新 LoRA
5. L3 checkpoint 存盘，下一轮推理通过 `load_lora_checkpoint()` 加载

### 12.2 "改进缓存决策" 具体含义

与 TTT 改进缓存**内容**相对，VFL 改进缓存**决策**：

- **SpecA threshold** — L1 的 EMA 动态阈值替换固定公式 `base_threshold * decay^progress` (`online_calibration.py:6-9`)。`speca_cal_type` 通过 `get_threshold()` / `get_threshold_cached()` 查询 (`models/dit.py:258`, `models/pixart.py:231`)
- **Backbone 轨迹平滑度** — L3 的 `trajectory_curvature_loss` 惩罚真实 hidden states 轨迹偏离低阶多项式 (Vandermonde 最小二乘, order=2 与 SpecA Taylor 阶数对齐)，鼓励 backbone 输出更易被低阶外推预测
- **TeaCache rescale** — `record_teacache_event` 收集预测-真实对比，但当前代码未发现直接修改 rescale 参数的逻辑（L1 尚未覆盖的领域）

### 12.3 关键模块速查

| 模块 | 关键符号 | 职责 |
|------|---------|------|
| `verification_hook.py` | `VerificationEvent`, `record_event`, `make_timestep_bucket`, `NUM_TIMESTEP_BUCKETS=3` | 事件结构 + 早/中/晚三桶映射；reject 100% 记录, accept 2% 采样 |
| `online_calibration.py` | `OnlineCalibrator`, `_EMAThreshold` | (layer_id, bucket) 级 EMA mean/var，输出 `mean+k*std`；TTL 缓存；`on_base_model_swap` 缩短窗口；`set_exploit_mode` 把 k 从 3.0 降到 1.5 |
| `replay_buffer.py` | `StratifiedReplayBuffer`, `AnchorSample` | 按 (layer_id, bucket) 分 `_Stratum`，每层 `capacity_per_stratum=1000` reservoir；anchor 全局保留最近 500；`max_encoder_hidden_states_events=200` 内存安全阀 |
| `lora_adapter.py` | `attach_lora_all_layers`, `LoRALinear` | 给每层 6 个 Linear 挂 LoRA (`attn1.to_q/k/v/to_out.0`, `ff.net.0.proj`, `ff.net.2`)；Phase 2 实际 rank=4 (config 写的 8)；B 矩阵零初始化；全 28 层 ≈ 5.2M 参数 |
| `curvature_loss.py` | `trajectory_curvature_loss`, `compute_training_loss` | curvature = 低阶多项式拟合残差 MSE；`L_total = L_supervised + λ_curvature * L_curvature + λ_anchor * L_anchor` |
| `async_trainer.py` | `AsyncTrainer`, `AsyncTrainingWorker` | **daemon 线程**（不是进程）；deepcopy fp32 模型训练；AdamW lr=1e-4；`trainer_steps_per_trigger=50`；`_train_once` 整段 try/except 容错 |
| `eval_gate.py` | `EvalGate`, `GateStatus` (8 种状态) | Canary 发布闸门：质量回归 (FID 退化 ≤ `quality_epsilon=5.0`) + 效果 (reject 率下降 ≥ `reject_delta=0.05`) 双通过 → `CANARY_READY`，复制到 `good_checkpoints/` |
| `version_registry.py` | `VersionRegistry`, `AdapterStatus (ACTIVE/STALE/UNKNOWN)` | base_model_version + adapter_version；版本切换时旧 adapter 自动 stale；`is_adapter_valid()` 严格校验 |
| `vfl_state.py` | 进程级单例 | `set_vfl_buffer/calibrator`, `record_speca/teacache_event`, `set_vfl_step_info/sample_id` |

### 12.4 集成点

- **CLI**: `--vfl` 启用完整三层 (cal + buf 都注册)；`--vfl-no-train` 仅 L1 阈值模式 (buf=None)
- **采样循环**: 每步开头调 `set_vfl_step_info(step_idx, len(timesteps), timestep_actual=int(t))` (`run_dit.py:347`, `run_pixart.py:356`)
- **SpecA 事件**: `models/dit.py:387`, `models/pixart.py:406` 调 `_vfl_record_speca_event`
- **TeaCache 事件**: `models/dit.py:409`, `models/pixart.py:424` 调 `_vfl_record_teacache_event`
- **Anchor 样本**: `run_dit.py:405-413`，每 5 步收集一次，anchor 数 < 50 时收集

### 12.5 `--vfl-no-train` 轻量模式省了什么

- 不构造 `VerificationEvent`，不 `detach().half().cpu()` 搬运 tensor（走 `_ScalarCalibEvent` 只传三个标量）
- 不创建 `StratifiedReplayBuffer`（省约 2GB+ 内存）
- 不 deepcopy 模型（省约 2.7GB fp32 显存）
- 不挂 LoRA（省 5.2M 参数）
- 不启动后台训练线程（零 CPU/GPU 训练开销）

**唯一代价**：`OnlineCalibrator` 的标量 EMA 更新（每次 event O(1) 浮点运算），对推理延迟几乎不可测。**仍会加载历史 LoRA checkpoint** (`run_dit.py:730-740`)，但不启动 worker。

### 12.6 L3 canary 发布现状

`EvalGate` 定义了双闸门，但 `run_dit.py` 主流程中**未实际调用 `EvalGate.evaluate()`** 自动发布——当前 checkpoint 自动加载 (`find_latest_checkpoint()`) 但跳过 gate 验证。完整 canary 自动发布流程仅在 `verification_feedback_loop/demo_e2e.py` 中实现，是参考代码。生产化时需要补这个环节。

## 13. COVR runtime 边界（可选插件）

**边界原则**：`accelerators/covr_runtime.py` 是 COVR 的 runtime facade（可选插件）。禁用时 `build_covr_runtime_config()` 返回 None，run_dit 路径不构造任何 COVR 对象、不产生 aggregate.covr_* 字段、不改变普通加速决策/前向计数。

- **Phase 1 范围**：仅 DiT 非 TTT 主去噪循环。PixArt / TTT 在 `validate_covr_capabilities` 提前失败（main.py 校验层）。
- **Runtime 职责**：初始化（version/resume/backend/recorder）、trajectory begin/end（sentinel 选择、assignment、feedback 物化、bandit update/persist）、adapter FLOPs、aggregate/config payload。
- **采样循环职责**：保持 tensor 计算在 run_dit；terminal/H-step/safety/audit 反馈仍写在 loop 内（写入 `COVRTrajectoryAssignment.feedback`）。
- **纯辅助**：`run_dit_shared.py` 持有 `_GenerationProfiler` + `_covr_*` canonical JSON / resume / sentinel 纯函数；run_dit 从其中 import 并保持历史 `_covr_*` 名称可导入（scripts/tests 依赖）。
- **StrategyManifest version_key 对称校验**：forced 与 bandit 两个 mode 都走 `load_strategy_manifest`（错误消息含 "runtime"）。
- **Bandit 标记 EXPERIMENTAL**：`ConservativeTemplateBandit` 语义冻结，未做算法改动；`--covr-strategy-bandit` help 已标注。
- **VFL 独立**：VFL 是旁路观测/校准，不参与 forward；COVR runtime 与其互不依赖，既有有效组合不变。
