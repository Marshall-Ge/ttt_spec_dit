# TTT-DiT 项目上下文 — 给新 Agent 的完整交接文档

## 1. 项目目标

对 DiT-2-256 和 PixArt-XL-2-512x512 两种扩散生成模型进行推理加速，评估加速方法对生成质量（FID/IS/CLIP）和效率（FLOPs/Latency）的影响。加速方法包括 SpecA（Per-block Taylor 缓存）、TeaCache（Per-step 残差缓存）、DDIM 步数压缩。

## 2. 模型和权重路径

| 模型 | 参数量 | 分辨率 | 架构 | 权重路径 |
|------|--------|--------|------|----------|
| DiT-2-256 | 675M | 256×256 | adaLN-Zero, class-conditional | `/root/autodl-fs/models/dit_2_256/` |
| PixArt-XL-2 | 2.5B | 512×512 | adaLN-Single, T5 text encoder | `/root/autodl-fs/models/models--PixArt-alpha--PixArt-XL-2-512x512/` |

- DiT: 28 blocks, 16 heads × 72 dim, in_channels=4, out_channels=8 (learned sigma: noise + variance)
- PixArt: 28 blocks, cross_attention_dim=1152, T5 caption_channels=4096, 3 submodules (attn1/attn2/ff)
- SD VAE: `/root/autodl-fs/models/dit_2_256/vae/`, scaling_factor=0.18215
- ImageNet val: `/root/autodl-fs/data/imagenet/val/` (1000 个类目录, 每类 50 张, 共 50k)
- COCO: `/root/autodl-fs/data/coco/`
- devkit: `/root/autodl-fs/data/imagenet/ILSVRC2012_devkit_t12/`

## 3. 目录结构和各文件职责

```
/root/ttt_spec_dit/
├── config.py                  # 全局路径、默认超参 (DIT_REPO, IMAGENET_DIR, SPECA_DEFAULTS...)
├── main.py                    # CLI 入口: parse_args() + validate_args() → 分发到 run_dit/run_pixart
├── utils.py                   # CudaTimer, VAE decode, save_image, ensure_real_299()
├── run_dit.py                 # DiTGenerator + run_c2i(args) — DiT 的编排器和采样入口
├── run_pixart.py              # PixArtGenerator + run_t2i(args) + run_c2i(args)
├── dit_coef.json              # DiT TeaCache 标定系数 (poly4, 50 步标定)
├── pixart_coef.json           # PixArt TeaCache 标定系数
├── models/
│   ├── __init__.py            # 导出 DiTTransformer2D, PixArtTransformer2D
│   ├── dit.py                 # DiTTransformer2D — 显式 forward，SpecA+TeaCache 分支可见
│   └── pixart.py              # PixArtTransformer2D — 同上，3 子模块 (attn1/attn2/ff)
├── accelerators/
│   ├── __init__.py            # 导出所有纯函数
│   ├── speca.py               # SpecA: speca_init, speca_cal_type, taylor_cache_init,
│   │                          #   derivative_approximation, taylor_formula, cache_step_dit/pixart,
│   │                          #   compute_error_gate (cosine/l1/l2/relative_l1/relative_l2)
│   └── teacache.py            # TeaCache: teacache_init/decide/cache_residual/apply_residual/step/reset
│                              #   + compute_modulated_input(_dit) — 调制信号提取
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
    └── calibrate_teacache.py  # TeaCache 多项式系数标定脚本
```

## 4. 架构核心原则

**无 monkeypatch**：旧代码 (`pipelines/t2i.py`, `pipelines/c2i.py`, `models/base.py`) 已删除。加速逻辑只有两种方式：

1. **模型内部显式分支** — SpecA 的 `current`/`cache_dic` 和 TeaCache 的 `teacache_state` 作为可选参数传入 `forward()`，在 pos_embed 和 blocks 之间做决策
2. **采样循环层** — `teacache_step()` 在循环中计数

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

## 9. 已验证的组合

| Model | Method | 状态 |
|-------|--------|------|
| DiT | baseline | ✅ FID 正常 |
| DiT | teacache | ✅ skip~48%, FID~225 (50步, γ=0.25) |
| DiT | ddim | ✅ |
| DiT | speca | ✅ FLOPs -70% |
| PixArt | baseline | ✅ |
| PixArt | teacache | ✅ skip~45% |
| PixArt | speca | ✅ FLOPs -65% |

## 10. 常用命令

```bash
# 标定 TeaCache 系数（改步数必跑）
python scripts/calibrate_teacache.py --model dit --num_steps 50 --num_runs 10

# 单组合验证
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --seed 42 --num_steps 50 --n_prompts 80 \
    --guidance_scale 4.5 --batch_size 32

# 全 20 组合 benchmark
N_PROMPTS=50 bash scripts/run.sh
```
