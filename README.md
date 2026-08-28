# TTT-Spec-DiT: Diffusion Transformer Inference Acceleration

Accelerating [DiT-2-256](https://arxiv.org/abs/2212.09748) and [PixArt-α](https://arxiv.org/abs/2310.00426) inference with **TeaCache** (per-step residual caching), **SpecA** (per-block Taylor caching), and **DDIM** (step compression). Evaluates quality (FID/IS/CLIP) vs. efficiency (FLOPs/latency) trade-offs across 20 model×method×dataset combinations.

## Table of Contents

- [Models & Weights](#models--weights)
- [Directory Structure](#directory-structure)
- [Architecture](#architecture)
- [Inference Pipeline](#inference-pipeline)
- [Acceleration Methods](#acceleration-methods)
  - [TeaCache](#teacache)
  - [SpecA](#speca)
  - [DDIM](#ddim)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Benchmark Suite](#benchmark-suite)
- [Key Design Decisions & Pitfalls](#key-design-decisions--pitfalls)
- [Verified Results](#verified-results)

---

## Models & Weights

| Model | Params | Resolution | Architecture | Weight Path |
|-------|--------|------------|--------------|-------------|
| DiT-2-256 | 675M | 256×256 | adaLN-Zero, class-conditional | `~/autodl-fs/models/dit_2_256/` |
| PixArt-XL-2 | 2.5B | 512×512 | adaLN-Single, T5 text encoder | `~/autodl-fs/models/models--PixArt-alpha--PixArt-XL-2-512x512/` |

**Key specs:**
- Both: 28 transformer blocks, 16 attention heads × 72 dim, hidden_dim=1152, in_channels=4, out_channels=8 (learned sigma)
- PixArt: 3 submodules per block (attn1/attn2/ff), cross_attention_dim=1152, T5 caption_channels=4096; attn2 has **no gate**
- SD VAE: `~/autodl-fs/models/dit_2_256/vae/`, scaling_factor=0.18215
- ImageNet val: `~/autodl-fs/data/imagenet/val/` (1000 classes, 50 images/class, 50k total)
- COCO 30K: `~/autodl-fs/data/coco/`

---

## Directory Structure

> 完整分层规范见 `.claude/project-structure.md`(强制)。

```
~/ttt_spec_dit/
├── config.py                  # 全局路径、默认超参、系数加载
├── main.py                    # CLI 入口(薄): parse_args → validate_args → dispatch
├── run_dit.py / run_pixart.py # 顶层编排入口(run_c2i / run_t2i;Generator 在 pipelines/)
├── pipelines/                 # 编排层
│   ├── dit.py                 #   DiTGenerator(类条件;含 TTT 训练路径)
│   ├── pixart.py              #   PixArtGenerator(t2i/c2i;T5 编码)
│   └── hooks/covr_hook.py     #   COVR 采样循环钩子(需要 transformer/scheduler 的 _covr_* 函数)
├── utils/                     # 通用工具包(纯函数,最底层)
│   ├── io.py / timing.py      #   图像 I/O、计时/Profiler
│   ├── serialization.py       #   JSON 安全化、_covr_* 哈希/哨兵/记账
│   ├── common.py              #   latent_seed_for_index、VFL checkpoint 辅助
│   └── flops.py               #   tail-only FLOPs 剖析
├── models/                    # 显式 Transformer 实现(无 monkeypatch)
│   ├── dit.py / pixart.py     #   forward() 内 SpecA/TeaCache/TTT 分支
│   └── ttt_plugin.py          #   SessionAdaLNModulator(0.92M, TTT)
├── accelerators/              # 加速器(纯函数 + 显式状态)
│   ├── speca.py / teacache.py #   SpecA(Taylor 缓存) / TeaCache(残差缓存,residual_mode)
│   ├── covr*.py               #   COVR runtime / bandit / viability
│   ├── registry.py            #   加速器适配器注册表(COVR init/reward/flops 三职责)
│   └── strategy_dispatch.py   #   AccelerationStrategy → 加速器状态 分发
├── feedback/vfl/              # VFL 在线学习子系统(L1 阈值校准 / L2 回放缓冲 / L3 LoRA)
├── eval/                      # 指标:FID/IS, CLIP, LPIPS, MSE, ImageReward, GenEval, Latency, FLOPs
├── dataset/                   # 数据集:ImageNet(类 ID 翻译), COCO, DrawBench, GenEval
├── scripts/                   # 脚本按用途分类
│   ├── calibrate/             #   TeaCache 系数标定
│   ├── diagnose/              #   误差-距离诊断、预测器族谱、token 长尾、类偏移探针
│   ├── experiment/            #   20-combo benchmark、COVR smoke、时间桶/残差变体验证
│   └── analyze/               #   COVR 分析、manifest 构建、P1 gate 工具链
├── tests/                     # 单元测试
└── docs/                      # 文档(教师报告、方法总结、诊断报告)
```

## Architecture

### Core Principles

1. **Zero monkeypatching.** Old pipeline code (`pipelines/t2i.py`, `pipelines/c2i.py`, `models/base.py`) has been deleted. All acceleration logic lives in exactly two places:
   - **Model-internal explicit branches**: `current`/`cache_dic` (SpecA) and `teacache_state` (TeaCache) are optional parameters threaded through `forward()`. The block loop contains visible `if step_type == 'full' / 'Taylor'` and `if use_teacache` branches.
   - **Loop-level**: `teacache_step()` is called in the sampling loop, not inside the model.

2. **Separated concerns.** The `Generator` class manages VAE/scheduler/device/dtype/encode_prompt only — it does not participate in forward logic.

3. **Pure-function accelerators.** All functions in `accelerators/` are stateless (except for the plain-dict state they mutate). No classes, no `nn.Module` subclasses.

4. **Checkpoint compatibility.** Models are built by instantiating diffusers `Transformer2DModel` and borrowing its submodule tree. `state_dict` keys match the released checkpoints exactly, so `load_state_dict(strict=True)` works.

### Model Forward Signatures

```python
# DiT
DiTTransformer2D.forward(
    hidden_states,         # (B, 4, 32, 32)
    timestep,              # (B,) int
    current=None,          # SpecA state dict
    cache_dic=None,        # SpecA cache dict
    teacache_state=None,   # TeaCache state dict
    class_labels=None,     # (B,) ImageNet class indices
)
# Returns: (output,) or Transformer2DModelOutput(sample=output)

# PixArt
PixArtTransformer2D.forward(
    hidden_states,              # (B, 4, 64, 64)
    encoder_hidden_states,      # (B, seq, 1152) T5 embeddings
    timestep,                   # (B,) int
    current=None, cache_dic=None, teacache_state=None,
    added_cond_kwargs=None, attention_mask=None, encoder_attention_mask=None,
)
```

---

## Inference Pipeline

### DiT c2i (ImageNet) — Full Flow

```
main.py: parse_args() → validate_args()
  → run_dit.run_c2i(args)
    ├─ [1] ImageNetDataset(n, seed)         # shuffle, return (path, prompt, DiT_class_id)
    ├─ [2] DiTGenerator.load()              # VAE + DiTTransformer2D + DDIM scheduler
    ├─ [3] FLOPsMetric(generator).profile() # ★ MUST run before accelerator setup
    ├─ [4] Accelerator init: teacache_init() or speca_init()
    │
    ├─ [5] for batch in dataset:
    │     generator.generate(prompts, seeds, method, teacache_state, cache_dic, current)
    │       → _denoise_loop():
    │           init latents → CFG doubling → [cond, null]
    │           for t in scheduler.timesteps:
    │             method dispatch:
    │               teacache: transformer(x, t, teacache_state=state) + teacache_step(state)
    │               speca:    transformer(x, t, current=cur, cache_dic=dic)
    │               baseline: transformer(x, t)
    │             → noise_pred[:, :in_channels]        # learned-sigma split
    │             → scheduler.step(noise_pred, t, latents)
    │           → unchunk (CFG): take cond half
    │         → VAE decode → image tensor (B, 3, 256, 256)
    │     → FIDISComputer.add(img, tag=class_name)
    │     → Save PNG (limit: img_save_limit, filename: {idx:06d}_{class_name}.png)
    │
    ├─ [6] FIDISComputer.compute() → cleanup()  # removes temp generated_299/
    └─ [7] Aggregate → save results.json
```

### PixArt t2i — Key Differences

- Uses DPMSolverMultistepScheduler (not DDIM)
- Prompt encoding via T5 text encoder (max 120 tokens, padded to batch max)
- CFG: `[uncond, cond]` — cond is index 1 (not 0 as in DiT)
- `adaln_single` returns `(timestep_emb, embedded_timestep)` — the 6-way modulation vector

---

## Acceleration Methods

### TeaCache

**Mechanism:** Per-step residual caching. At pos_embed output, compares block0's modulation signal against the previous step. If similar, skips the entire block stack and reuses the cached residual `(blocks_output - blocks_input)`.

**Decision algorithm:**
```
raw_diff = |modulated - prev|.mean() / |prev|.mean()    # relative L1
rescaled = max(0, poly4(raw_diff))                       # calibrated polynomial
accumulated += rescaled
should_calc = (cnt == 0 or cnt == num_steps-1) or (accumulated >= threshold)
if should_calc: accumulated = 0
```

**Key functions** (`accelerators/teacache.py`):
| Function | Role |
|----------|------|
| `teacache_init(num_steps, rel_l1_thresh, coefficients)` | Allocate state dict |
| `compute_modulated_input(_dit)()` | Extract block0.norm1 modulation |
| `teacache_decide(state, modulated)` | Decision: calc or skip? |
| `teacache_cache_residual(state, out, ori)` | Store `out - ori` |
| `teacache_apply_residual(state, x)` | `x + residual` |
| `teacache_step(state)` | `cnt += 1` |
| `teacache_reset(state)` | Clear runtime state (keep config) |
| `teacache_stats(state)` | Aggregate skip_ratio, calc/skip counts |

**Coefficient calibration:** `scripts/calibrate_teacache.py` collects raw_diff trajectories from N denoising runs, fits a poly4 so accumulated rescale hits target skip rate. **Must re-run when changing `--num_steps`** — poly4 oscillates outside its training range.

**FLOPs interaction:** `FLOPsMetric.add_generation()` requires a `.decisions` attribute (list of `"calc"`/`"skip"`). Passed via `SimpleNamespace(decisions=state["decisions"])`.

**Defaults:** `threshold=0.25`, `num_steps=20`, coefficients from `dit_coef.json`/`pixart_coef.json`.

### SpecA

**Mechanism:** Per-block per-submodule Taylor caching. Each step is either `full` (compute all blocks, cache submodule outputs + finite-difference derivatives) or `Taylor` (predict via Taylor series, skipping attention/MLP computation). At `check_layer`, re-computes a full block to measure error and feed it into the threshold decision.

**Key functions** (`accelerators/speca.py`):
| Function | Role |
|----------|------|
| `speca_init(num_steps, base_threshold, decay_rate, ...)` | Allocate `(cache_dic, current)` |
| `speca_cal_type(cache_dic, current)` | Decide full vs Taylor for this step |
| `taylor_cache_init(cache_dic, current)` | Allocate cache slot (first step only) |
| `derivative_approximation(cache_dic, current, feature)` | Finite-difference derivatives |
| `taylor_formula(module_list, distance)` | Σ (1/n!) · dⁿ · featureₙ |
| `cache_step_dit/pixart(...)` | Apply Taylor prediction per block |
| `compute_error_gate(x, full_x, metric)` | Error probe at check_layer |

**Error metrics:** `l1`, `l2`, `relative_l1`, `relative_l2`, `cosine_similarity` (default).

**DiT submodules:** `attn`, `mlp` (2 per block)
**PixArt submodules:** `attn1`, `attn2`, `ff` (3 per block — attn2 has no gate)

**Key hyperparameters:**
| Param | DiT | PixArt | Meaning |
|-------|-----|--------|---------|
| check_layer | 20 | 24 | Error-probe layer index |
| error_metric | cosine_similarity | cosine_similarity | Error type for threshold |
| base_threshold | 0.01 | 0.01 | Cosine error scale [0,1] |
| decay_rate | 0.01 | 0.01 | Threshold decay with progress |
| min_taylor_steps | 1 | 1 | Min consecutive Taylor steps before checking |
| max_taylor_steps | 4 | 4 | Force full after N Taylor steps |
| max_order | 4 | 4 | Taylor expansion order |

### DDIM

Simple step-count reduction. Uses DDIMScheduler with `num_steps` steps. No caching, no state — pure scheduler-based acceleration. The FLOP-matched equivalent of TeaCache at ~50% skip rate is 10 DDIM steps (vs 20 DPM-Solver steps on PixArt).

---

## Quick Start

### 环境

```bash
# conda 环境(远程 AutoDL 已配置 realpde)
conda activate realpde
# 权重路径在 config.py 中配置(DIT_REPO / PIXART_REPO / IMAGENET_DIR)
```

### 典型启动方式

**① 快速验证单组合(小规模,无 FID)**

```bash
# DiT + TeaCache(默认 γ=0.25)
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics latency flops speed \
    --num_steps 50 --n_prompts 8 --batch_size 8 --img_save_limit 2

# DiT + SpecA
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics latency flops speed \
    --num_steps 50 --n_prompts 8 --batch_size 8 --img_save_limit 2

# PixArt t2i(DrawBench)
python main.py --model pixart --task t2i --dataset drawbench \
    --method teacache --metrics imagereward latency flops speed \
    --num_steps 20 --n_prompts 10
```

**② 完整质量评估(5k/50k, 含 FID/IS)**

```bash
# DiT + TeaCache 50k FID
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --thresh 0.25 \
    --metrics fid is latency flops speed \
    --seed 42 --num_steps 50 --n_prompts 50000 --batch_size 32
```

**③ 新特性 1:杠杆再造(TeaCache 残差线性漂移)——已验证严格支配原版**

```bash
# 同速度下质量更好(skip 32%→70%, FID 34.82→34.40, 速度×2)
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --thresh 0.70 --teacache-residual-mode linear \
    --metrics fid is latency flops speed \
    --seed 42 --num_steps 50 --n_prompts 5000 --batch_size 32

# 变体:damped(Hermite 阻尼漂移)/ plain(原版)
#   --teacache-residual-mode damped | plain
```

**④ 新特性 2:结构感知调度(SpecA 时间桶差异化 max_taylor)**

```bash
# 同预算 latent MSE -17%(晚期误差主导的机制驱动设计)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --speca-bucket-schedule 8,4,2 \
    --metrics latency flops speed \
    --num_steps 50 --n_prompts 20 --batch_size 16
# schedule 格式:(early, mid, late) 三个桶的 max_taylor_steps
```

**⑤ 新特性 3:per-class γ 查表(离线标定,零开销)**

```bash
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --thresh 0.25 --per-class-gamma dit_per_class_gamma.json \
    --metrics latency flops speed --num_steps 50 --n_prompts 100
```

**⑥ 在线学习扩展**

```bash
# TTT 插件(DiT-only, 叠加在 TeaCache 上)
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --ttt --ttt_lr 1e-4 --ttt_micro_epochs 3 \
    --num_steps 50 --n_prompts 80

# VFL(SpecA + 在线阈值校准;--vfl-no-train 仅 L1,零训练开销)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops speed \
    --vfl --vfl-no-train \
    --num_steps 50 --n_prompts 80
```

**⑦ 诊断脚本(机制分析,不跑完整评估)**

```bash
# 误差-距离曲线(判断加速器属于外推型还是复用型)
python scripts/diagnose/diag_error_vs_distance.py --n_images 4 --num_steps 50

# 预测器族谱(0阶/1阶/2阶 Taylor vs 残差复用/线性/阻尼)
python scripts/diagnose/diag_predictor_family.py --n_images 6 --num_steps 50

# 类分布偏移探针(静态 γ 在类子集上的失衡证据)
python scripts/diagnose/probe_class_shift_gamma.py --n_per_class 30 --gammas 0.15,0.25,0.40

# 时间桶调度 / 残差变体验证(closed-loop)
python scripts/experiment/run_speca_bucket_schedule.py --n_images 20
python scripts/experiment/run_teacache_residual_variants.py --n_images 12
```

**⑧ COVR 在线策略控制(DiT + 加速器之上,可选)**

```bash
# 一键 forced COVR smoke(manifest 构建 → 实际生成 → 结果/策略校验)
bash scripts/experiment/run_covr_forced_smoke.sh uniform

# experimental bandit 持久化/恢复 smoke(2+2 张)
bash scripts/experiment/run_covr_bandit_resume_smoke.sh

# COVR shadow 审计(记录 counterfactual events,不影响决策)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics latency flops speed \
    --covr-shadow --covr-output-dir ./covr_audit \
    --num_steps 50 --n_prompts 20 --batch_size 4

# forced 策略(指定 manifest 中的 arm)
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics latency flops speed \
    --covr-strategy-manifest /tmp/covr_forced_smoke/teacache_k8.json \
    --covr-force-strategy-id uniform \
    --num_steps 50 --n_prompts 8 --batch_size 1

# timestep feedback shadow 学习(session 级缺陷学习者)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics latency flops speed \
    --covr-shadow --covr-timestep-feedback \
    --num_steps 50 --n_prompts 20 --batch_size 4
```

> COVR 边界:Phase 1 仅 DiT 非 TTT 主去噪循环;禁用时不构造任何对象、
> 不产生 aggregate.covr_* 字段、零开销。Bandit 标记 EXPERIMENTAL。

### TeaCache 系数标定

**改 `--num_steps` 必跑**(多项式在标定范围外会振荡):

```bash
python scripts/calibrate/calibrate_teacache.py --model dit --num_steps 50 --num_runs 10
python scripts/calibrate/calibrate_teacache.py --model pixart --num_steps 20 --num_runs 10
```

### 20 组合全量 benchmark

```bash
N_PROMPTS=50 bash scripts/experiment/run.sh
```

---

## CLI Reference

```
python main.py --help
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `pixart` | `pixart` or `dit` |
| `--task` | str | (required) | `t2i` or `c2i` |
| `--dataset` | str | (required) | `drawbench`, `geneval`, `coco`, `imagenet` |
| `--method` | str | `teacache` | `baseline`, `teacache`, `ddim`, `speca` |
| `--n_prompts` | int | dataset default | Number of prompts/images |
| `--num_steps` | int | `20` | Denoising steps |
| `--thresh` | float | `0.25` | TeaCache threshold γ |
| `--guidance_scale` | float | `4.5` | CFG scale (1.0 = no CFG) |
| `--batch_size` | int | `4` | Max prompts per `generate()` call |
| `--metrics` | list | `latency flops speed` | Metrics to compute |
| `--seed` | int | `42` | Random seed |
| `--img_save_limit` | int | `50` | Max PNGs to save to disk |
| `--output_dir` | str | auto | Output directory |
| **SpecA flags** | | | |
| `--speca_base_threshold` | float | `0.01` | Base threshold |
| `--speca_decay_rate` | float | `0.01` | Threshold decay rate |
| `--speca_min_taylor_steps` | int | `1` | Min Taylor steps before error check |
| `--speca_max_taylor_steps` | int | `4` | Max consecutive Taylor steps |
| `--speca_error_metric` | str | `cosine_similarity` | Error metric for gating |

### Valid Task/Dataset/Method Combinations

| Model | Task | Dataset | Methods |
|-------|------|---------|---------|
| dit | c2i | imagenet | baseline, teacache, ddim, speca |
| pixart | t2i | drawbench | baseline, teacache, ddim, speca |
| pixart | t2i | geneval | baseline, teacache, ddim, speca |
| pixart | c2i | coco | baseline, teacache, ddim, speca |
| pixart | c2i | imagenet | baseline, teacache, ddim, speca |

### Valid Metrics per Dataset

| Task | Dataset | Valid Metrics |
|------|---------|---------------|
| t2i | drawbench | imagereward, latency, flops, speed |
| t2i | geneval | geneval, latency, flops, speed |
| c2i | coco | fid, is, clip, lpips, mse, latency, flops, speed |
| c2i | imagenet | fid, is, latency, flops, speed |

---

## Benchmark Suite

```bash
# Full 20-combo benchmark (DiT×4 + PixArt c2i×8 + PixArt t2i×8)
N_PROMPTS=50 bash scripts/run.sh

# Quick smoke test (smaller N)
bash scripts/run_full_smoke.sh
```

The 20 combinations are:

| # | Model | Task | Dataset | Method |
|---|-------|------|---------|--------|
| 1-4 | DiT | c2i | imagenet | baseline, teacache, ddim, speca |
| 5-8 | PixArt | c2i | coco | baseline, teacache, ddim, speca |
| 9-12 | PixArt | c2i | imagenet | baseline, teacache, ddim, speca |
| 13-16 | PixArt | t2i | drawbench | baseline, teacache, ddim, speca |
| 17-20 | PixArt | t2i | geneval | baseline, teacache, ddim, speca |

---

## Key Design Decisions & Pitfalls

### 1. ImageNet Class ID Translation
Dataset directories use ILSVRC2012_ID order (0000=kit fox→WNID n02119789). DiT uses WNID alphabetical order (0=tench→n01440764). `dataset/imagenet.py` auto-translates via `ilsvrc2012_to_dit_id.json`.

### 2. DiT CFG Channel Split (Learned Sigma)
DiT outputs 8 channels: first `in_channels=4` are noise prediction, last 4 are variance. CFG extrapolation ONLY applies to the noise channels: `model_out[:, :config.in_channels]`. The old hardcoded `:3` was a bug.

### 3. `forward_with_cfg` Latent Handling (DiT)
Accepts already-doubled latent `[cond, null]`, but only takes the cond half and duplicates it `[cond, cond]` through the model with cond/null class labels, then CFG-extrapolates.

### 4. PixArt attn2 Has No Gate
Cross-attention takes raw hidden_states (not norm2-modulated), output is added directly without a gate. This is reflected in `cache_step_pixart` where attn2 has no `gate *` multiplier.

### 5. SpecA Step Counting Direction
`current['step']` counts DOWN from `num_steps-1` to 0 (reverse denoising order). Set in the sampling loop: `current['step'] = len(timesteps) - 1 - step_idx`.

### 6. FLOPs Profiling Must Run Before Accelerator Setup
`FLOPsMetric.profile()` calls `transformer(latent, timestep, class_labels, return_dict=False)` — compatible with the new forward signature because `current=None, cache_dic=None` defaults to vanilla path.

### 7. File Save Strategy
- `generated/`: Original resolution PNGs, limited by `--img_save_limit`, with class name in filename
- `generated_299/`: FID temp directory, `FIDISComputer.add()` writes, `compute()` then `cleanup()` deletes
- `real_299/`: Symlinks to `val_299_cache/` (one-time full-dataset 299×299 preprocess)
- Filename format: `{idx:06d}_{class_name}.png`

### 8. TeaCache Coefficients Are Step-Count Specific
The poly4 is fitted on raw_diff distributions from a specific step count. Using 50-step coefficients with 20-step inference produces negative rescaled values (polynomial oscillation outside training range). Always recalibrate after changing `--num_steps`.

### 9. Batch Size Semantics
`--batch_size` controls how many **different prompts** are processed in parallel per `generate()` call, each with independent random seeds. Padding is applied to T5 text embeddings across different-length prompts.

---

## Verified Results

| Model | Method | Status | Key Metric |
|-------|--------|--------|------------|
| DiT | baseline | ✅ | FID normal |
| DiT | teacache | ✅ | skip~48%, FID~225 (50-step, γ=0.25) |
| DiT | ddim | ✅ | — |
| DiT | speca | ✅ | FLOPs -70% |
| PixArt | baseline | ✅ | — |
| PixArt | teacache | ✅ | skip~45% |
| PixArt | speca | ✅ | FLOPs -65% |

### GPU Memory Budget (RTX 4090 D, 23.5 GB)

| Model | CFG | Max Batch Size |
|-------|-----|----------------|
| DiT-2-256 (675M) | on (gs=4.5) | 64 |
| PixArt-XL-2 (2.5B) | on (gs=4.5) | 4 |

---

## Session-Level TTT (Test-Time Training) — Phase 3

Session-Level TTT is a **closed-loop continual learning** plugin that attaches to the TeaCache acceleration path. During a sequential stream of semantically-related images, a tiny persistent network ($\phi$) learns to correct stale cached residuals on-the-fly — enabling the **Flywheel Effect**: later images sustain higher skip ratios without fidelity collapse.

### Architecture

```
                    ┌──────────────────────────────┐
                    │   SessionAdaLNModulator (φ)   │
                    │   ┌─────┐   ┌─────┐          │
  timestep_emb ────►│   │ fc_t│   │fc_h │          │
                    │   └──┬──┘   └──┬──┘          │
                    │      │         │              │
  cached_state ────►│ pooled│────────┼──► fc_out ─►│──► Δγ, Δβ
                    │      │    SiLU │     ↑        │
                    │      └─────────┘     │        │
                    └──────────────────────┼────────┘
                                           │ zero-init
                                           ▼
            Z_out = Z_cached ⊙ (1 + Δγ) + Δβ
```

**Key properties:**
- 0.92M parameters (fc_out zero-initialized → identity mapping at step 0)
- Plugin runs in **fp32** for numerical stability; backbone stays fp16
- Backbone ($\Theta$) is frozen via `requires_grad_(False)` — gradients flow **only** through $\phi$

### Dual-Mode Forward

| Mode | Trigger | Backbone | Plugin | Output |
|------|---------|----------|--------|--------|
| **Teacher (calc)** | TeaCache decides "calc" | Runs 28 blocks under `no_grad` → Z_true | Runs on stale cache → Z_pred | Emits Z_true; trains φ via MSE(Z_pred, Z_true) |
| **Student (skip)** | TeaCache decides "skip" | Bypassed | Runs on stale cache → Z_pred | Emits Z_pred |

### Dynamic γ Curriculum

The TeaCache threshold γ is scheduled per image to force exploration before exploitation:

| Image Range | γ | Skip Ratio (8-step) | Purpose |
|-------------|---|---------------------|---------|
| 1–3 (burn-in) | 0.35 | ~37.5% | Many calc steps → distillation ground truth |
| 4–10 (transition) | 0.55 | ~50.0% | Balanced training/inference |
| 11+ (exploitation) | 0.75 | ~62.5% | Plugin sustains high acceleration |

### Usage

TTT plugin is invoked via `main.py` with `--ttt` flag (DiT-only). See the
[CLI Reference](#cli-reference) section for the full flag list.

```bash
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --ttt --ttt_lr 1e-4 --ttt_micro_epochs 3 \
    --num_steps 20 --n_prompts 20 --guidance_scale 4.5 \
    --metrics fid is latency flops speed
```

### Output

- `generated/{idx:06d}_{class_name}.png` — per-image output (subject to `--img_save_limit`)
- `results.json` — aggregated metrics (FID/IS/latency/FLOPs/speed)
- Telemetry (skip ratio, plugin loss) is emitted via the standard metrics pipeline

### Design Decisions & Caveats

1. **DiT has no text encoder.** "Semantic manifold" is realised as a fixed ImageNet class label with independent Gaussian noise seeds — the plugin learns the denoising-trajectory manifold of that class.

2. **fp32 mixed precision.** Plugin loss can reach ~10⁵, which overflows fp16 (max 65504). Plugin runs in fp32; the teacher signal (`z_true`) is computed under `torch.amp.autocast('cuda', dtype=torch.float32)`.

3. **CFG is orthogonal to the plugin.** The plugin operates in the pre-tail hidden space (pos_embed output). `forward_with_cfg_ttt` handles CFG extrapolation on noise channels identically to the vanilla path.

4. **Separate codepath, zero regression risk.** `generate_ttt` / `_denoise_loop_ttt` are dedicated methods under `DiTGenerator`. The existing `generate` / `_denoise_loop` / `run_c2i` / `forward` / `forward_with_cfg` are untouched when `ttt_state=None`.

---

## Project Memory

Additional context, experiment logs, and design rationale are recorded in the project's [memory directory](.claude/projects/-root-ttt-spec-dit/memory/MEMORY.md). Key entries include:

- PixArt baseline setup & loading pitfalls
- Phase 2 TeaCache baseline (74% skip, 3.19× speedup)
- SpecA integration & root-cause fixes (3-layer fix: check_layer, cosine metric, 1.9× MSE improvement)
- Batch size semantics & padding strategy
- Phase 3 TTT Closed-Loop Tracker handoff

See `CLAUDE.md` for the complete agent-facing handoff document.
