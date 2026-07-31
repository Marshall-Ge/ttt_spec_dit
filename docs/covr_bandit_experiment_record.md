# COVR Bandit 实验记录

> 记录日期：2026-07-27  
> 项目：TTT-DiT / DiT-2-256 / SpecA COVR Template Bandit  
> 数据来源：2026-07-25 的 `20260725-160333` 实验归档、当前工作区代码和本轮讨论中的结果日志

本文档集中保存 COVR Template Bandit 从 SpecA 模板选择到通用加速策略调度器的实验数据、实现状态和未解决问题。这里的“50k”实验需要特别注意：ImageNet 总样本数为 50,000，但实验把前 2,048 张用于 audit，实际 evaluation 使用后续 47,952 张，因此结果表中的 50k 实际是 47,952 张评估结果。

## 1. 实验配置

- 模型：DiT-2-256，675M 参数，28 个 Transformer blocks
- 任务：class-conditional ImageNet (`c2i`)
- 分辨率：256x256
- 去噪步数：50
- `guidance_scale`：4.5
- seed：42
- 评估数据：ImageNet
- audit 数据区间：`[0, 2048)`
- evaluation 数据区间：`[2048, 50000)`，共 47,952 张
- batch size：主要实验为 32
- COVR bandit：epsilon-greedy，按 trajectory/batch 选择策略
- SpecA 候选：4 个相同建模 FLOPs 的 refresh mask
- 原始 bandit reward：H-step transition defect
- 当前实验 reward：最后一步 terminal fidelity loss

## 2. 方法定义

### 2.1 SpecA 模板

每个模板是一条 50 步去噪时间表，用 `refresh_mask` 指示每个时间步是否执行 full refresh：

- `True`：执行完整 block 计算并刷新缓存
- `False`：使用 Taylor cache 外推
- 当前候选模板通常为 13 个 full refresh、37 个 Taylor step
- 候选建模 FLOPs 约为 `3.087T`

候选 ID：

- `timestep_prior`
- `template_01`
- `template_02`
- `template_03`

模板之间计算预算接近，但 refresh 的时间位置不同。相同的 full step 数并不代表相同的最终图像质量，因为早期高噪声阶段和后期低噪声阶段的误差影响不同。

### 2.2 Bandit

每条 trajectory 开始时选择一个候选 arm，trajectory 结束后使用反馈更新该 arm 的统计量。

当前实现使用增量均值，不是 EMA。损失先做 `log1p` 变换，再通过 Welford 方式更新均值：

```python
logged = math.log1p(loss)
mean += (logged - mean) / count
```

选择逻辑是：

- 以 epsilon 概率随机探索
- 否则选择当前平均 loss 最低的 arm
- SpecA 还可以结合 safety table 过滤不满足安全约束的 arm

### 2.3 Terminal fidelity reward

原始 H-step reward 只反映固定窗口内的 latent transition defect，和最终 FID 出现反相关，因此改为最后一步的局部 fidelity：

```text
x_prev_approx = scheduler.step(noise_pred_template, t, x_t)
x_prev_full   = scheduler.step(noise_pred_full,     t, x_t)
loss = MSE(x_prev_approx, x_prev_full)
```

最后一步额外执行一次 full forward。相对于完整 baseline rollout 的 50 次 full forward，这条路径每个 sentinel trajectory 只增加一次 full forward。

`TemplateFeedback.bandit_loss` 的优先级为：

1. `terminal_fidelity_loss`
2. `h_step_numerator`
3. `terminal_quality_loss`

注意：当前 `run_dit.py` 的 terminal fidelity 代码仍带有 `method == "speca"` 和 `current is not None` 条件，因此设计上虽是方法无关的 reward，实际 TeaCache 路径还没有完全获得该 cheap terminal reward。这是通用化后的待修复项。

## 3. 原始 H-step reward 的问题

2048 张验证结果暴露出原始 reward 与 FID 的方向相反：

| 模板 | H-step defect | FID |
|---|---:|---:|
| `template_02` | 0.299 | 55.52 |
| `timestep_prior` | 0.391 | 48.75 |

H-step defect 越低并没有带来更低的 FID。原因是该指标把不同去噪阶段的误差近似等权处理，而 terminal 图像质量对后期步骤更敏感。这个结果是采用 terminal fidelity reward 的直接动机。

## 4. 2,048 张验证结果

以下方法使用同一批验证样本和相同的基础配置：

| 方法 | FID ↓ | IS ↑ | candidate FLOPs | online FLOPs | img/s |
|---|---:|---:|---:|---:|---:|
| Baseline | 56.66 | 125.79 | 11.867T | 11.867T | 3.85 |
| Adaptive SpecA | 55.54 | 122.85 | 3.288T | 3.288T | 6.96 |
| `timestep_prior` | 48.75 | 102.95 | 3.087T | 3.087T | 7.43 |
| `template_01` | 49.32 | 107.74 | 3.087T | 3.087T | 7.36 |
| `template_02` | 55.52 | 122.08 | 3.087T | 3.087T | 7.39 |
| `template_03` | 51.70 | 117.41 | 3.087T | 3.087T | 7.46 |
| COVR Bandit | 49.35 | 107.30 | 3.087T | 3.949T | 6.19 |

### 4.1 2,048 张 bandit 分配

| 策略 | 分配次数 | 占比 |
|---|---:|---:|
| `timestep_prior` | 157 | 61.3% |
| `template_01` | 36 | 14.1% |
| `template_02` | 36 | 14.1% |
| `template_03` | 27 | 10.5% |
| 合计 | 256 | 100.0% |

Terminal fidelity reward 让 bandit 从原先偏向 `template_02` 转向 `timestep_prior`，并且固定模板的 FID 排名基本得到反映。

## 5. 47,952 张 evaluation 结果

### 5.1 Baseline

| 指标 | 数值 |
|---|---:|
| FID | 24.8961 |
| IS | 451.7488 |
| FLOPs | 11.8667T |
| img/s | 4.1720 |
| batch wall time | 7.8455s |

### 5.2 Adaptive SpecA

| 指标 | 数值 |
|---|---:|
| FID | 23.9988 |
| IS | 434.4256 |
| candidate FLOPs | 3.3705T |
| FLOPs reduction vs baseline | 71.60% |
| img/s | 8.5797 |
| skip ratio | 73.29% |
| probe blocks | 35,361 |

### 5.3 COVR Bandit

| 指标 | 数值 |
|---|---:|
| FID | 17.6218 |
| IS | 353.2727 |
| candidate FLOPs | 3.0867T |
| online FLOPs | 3.9594T |
| safety FLOPs | 0.8727T |
| candidate img/s | 4.7813 |
| online img/s | 4.1692 |
| skip ratio | 74.00% |
| trajectories | 1,499 |

### 5.4 47,952 张 bandit 分配

| 策略 | 分配次数 | 占比 |
|---|---:|---:|
| `timestep_prior` | 1,381 | 92.1% |
| `template_01` | 39 | 2.6% |
| `template_02` | 33 | 2.2% |
| `template_03` | 46 | 3.1% |
| 合计 | 1,499 | 100.0% |

### 5.5 47,952 张 arm 平均 loss

| 策略 | 平均 loss |
|---|---:|
| `timestep_prior` | 0.000041 |
| `template_01` | 0.050016 |
| `template_02` | 0.125030 |
| `template_03` | 0.062526 |

### 5.6 三组直接对比

| 方法 | FID ↓ | IS ↑ | candidate FLOPs | online FLOPs | img/s |
|---|---:|---:|---:|---:|---:|
| Baseline | 24.8961 | 451.7488 | 11.8667T | 11.8667T | 4.1720 |
| Adaptive SpecA | 23.9988 | 434.4256 | 3.3705T | 3.3705T | 8.5797 |
| COVR Bandit | **17.6218** | 353.2727 | **3.0867T** | 3.9594T | 4.1692 |

相对 baseline：

- FID 改善 `7.2743`
- candidate FLOPs 约下降 `74.0%`
- online FLOPs 约下降 `66.6%`
- IS 下降约 `21.8%`
- online 速度基本没有提升：`4.1692` vs `4.1720 img/s`

相对 adaptive SpecA：

- FID 改善 `6.3770`
- candidate FLOPs 约低 `8.4%`
- 由于 safety shadow，online FLOPs 反而高约 `17.5%`
- IS 下降约 `18.7%`
- online 速度慢约 `51.4%`：`4.1692` vs `8.5797 img/s`

## 6. 当前结论

### 已确认

1. Terminal fidelity reward 比 H-step defect 更能复现固定模板的 FID 排名。
2. Bandit 在当前 reward 下稳定偏向 `timestep_prior`，47,952 张 evaluation 中占 92.1%。
3. Bandit 的 FID 明显优于 baseline 和 adaptive SpecA。
4. Bandit 的 IS 明显低于 baseline 和 adaptive SpecA，说明它不是全面质量提升，而是明显偏向 FID。
5. ~~Adaptive SpecA 仍然是当前更好的效率基线：约 8.58 img/s，且 IS 更高。~~ **已更新**：速度优化后 Bandit 8.05 img/s 快于 SpecA 7.69 img/s（见 §6.1）。
6. Safety observation 是 Bandit online FLOPs 增加的明确来源；50k Bandit safety FLOPs 为 `0.8727T`。**消除 safety 后速度问题完全解决。**

### 6.1 速度优化后 CFG=4.5 实验 (2026-07-29)

> 配置：safety_sample_rate=0, sentinel_rate=0.02, chain_threshold=5  
> 50,000 张全量评估，guidance_scale=4.5

| 方法 | FID ↓ | IS ↑ | FLOPs(T) | img/s | cand img/s | skip% |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 24.84 | 453.9 | 11.867 | 3.95 | 3.95 | — |
| Adaptive SpecA | 23.96 | 436.7 | 3.371 | 7.69 | 7.69 | 73.3% |
| COVR Bandit | **17.48** | 351.2 | **3.087** | **8.01** | **8.05** | 74.0% |

Stage Profiling (COVR Bandit):

| Stage | 时间(s) |
|---|---:|
| generation_online | 6216.4 |
| denoise_loop | 5801.1 |
| speca_full | 3374.1 |
| speca_taylor | 2378.1 |
| image_save_metrics | 1487.5 |
| vae_decode | 412.8 |
| cuda_sync_wait | 406.2 |
| bandit_state_persist | 23.9 |

COVR overhead: safety=0.0s, terminal=3.5s, control=26.4s

**核心结论：消除 safety shadow 后，COVR Bandit 同时实现了：**
- **FID 大幅优于 baseline（-7.36）和 SpecA（-6.48）**
- **速度快于 Adaptive SpecA（8.05 vs 7.69 img/s，+4.7%）**
- **FLOPs 更低（3.087T vs 3.371T，-8.4%）**

唯一代价是 IS 下降（351.2 vs 436.7，-19.6%），这与之前的 safety-on 实验一致，说明 IS 退化是 template 选择（偏向 `timestep_prior`）的固有特性，与 safety 无关。

### 6.2 与原始 50k 实验（§5）的速度对比

| 指标 | §5 原始 (safety=10%) | §6.1 速度优化 (safety=0) |
|---|---|---|
| Bandit candidate img/s | 4.78 | **8.05** (+68%) |
| Bandit online img/s | 4.17 | **8.01** (+92%) |
| Bandit vs SpecA 速度 | 慢 51.4% | **快 4.7%** |
| Bandit FID | 17.62 | 17.48 (一致) |
| Bandit IS | 353.3 | 351.2 (一致) |

结论：safety shadow 是唯一的速度瓶颈。消除后 FID/IS 不变，速度从 4.78 提升到 8.05 img/s。

### 不能直接下结论

- 不能说 Bandit 全面改进了 SpecA（IS 明显更差）。
- 不能把 FID 降低解释为生成质量所有维度都提升。
- 不能把当前 50k 结果直接和其他数据切片上的”50k FID < 10”结果比较。需要核对数据切片、真实参考集、模型权重、CFG、评估实现和样本命名是否完全一致。

## 7. 速度差异已解决

原始 §5 实验中 Bandit 和 adaptive SpecA 的 skip ratio 接近（73.29% vs 74.00%），但速度差异很大（8.58 vs 4.78 img/s）。

**根因**：safety shadow（10% sample rate × per-step full forward）累积 0.87T FLOPs（22% online），是唯一瓶颈。

**验证路径**（benchmark_bandit_speed.sh, 2048 张 4 组对照）：
- B (no safety) candidate = 8.04 img/s ≈ D (adaptive SpecA) = 7.64 img/s → safety 是唯一瓶颈
- B vs C 差异 <0.2s → bandit dispatch 开销可忽略
- C vs D 差异 <3% → speca_init reinit 开销可忽略

**最终确认**（§6.1, 50k CFG=4.5, safety=0）：Bandit 8.05 img/s > SpecA 7.69 img/s，问题完全解决。

## 8. 方法无关策略抽象状态

当前代码已经开始把 SpecA 专属 `RefreshTemplate` 抽象为通用 `AccelerationStrategy`：

```python
AccelerationStrategy(
    strategy_id="...",
    method="speca" | "teacache",
    params={...},
    modeled_flops=...,
    source="...",
)
```

已实现或已接入的部分：

- `accelerators/covr_bandit.py`
  - `AccelerationStrategy`
  - `StrategyManifest`
  - `ConservativeTemplateBandit.from_strategies()`
  - 旧 `RefreshTemplate` / `TemplateManifest` 兼容路径
- `accelerators/strategy_dispatch.py`
  - `speca` 策略分发到 `speca_init`
  - `teacache` 策略分发到 `teacache_init`
- `scripts/build_covr_manifest.py`
  - SpecA refresh mask manifest
  - TeaCache threshold manifest
- `main.py`
  - 新参数：
    - `--covr-strategy-manifest`
    - `--covr-strategy-bandit`
    - `--covr-force-strategy-id`
  - 旧 template 参数保留，并映射到新接口后发 deprecation warning
- `tests/test_covr_strategy.py`
  - 覆盖 strategy 数据结构、manifest、TeaCache strategy bandit 生命周期和旧接口兼容

当前尚未完成：

- TaylorSeer 分发和真实初始化接口
- TeaCache 的真实 GPU smoke/evaluation
- TeaCache 的 terminal fidelity reward 接入
- 不同 TeaCache threshold 的实际 FLOPs/速度标定
- Bandit 在不同计算预算下的公平比较

TeaCache 策略可以用不同 `rel_l1_thresh` 作为 arm，例如 `0.15/0.25/0.35/0.45`。但 threshold 越高通常意味着更多 skip，候选之间不天然等 FLOPs，因此不能直接沿用 SpecA 的 equal-FLOPs 结论。

## 9. CLI 与实验脚本

旧参数仍可用，但建议新实验使用：

```bash
python main.py --model dit --task c2i --dataset imagenet --method teacache --metrics fid is flops speed --seed 42 --num_steps 50 --n_prompts 2048 --batch_size 8 --guidance_scale 4.5 --covr-strategy-bandit --covr-strategy-manifest manifest_teacache.json --covr-sentinel-horizon 0 --output_dir output/teacache_bandit_test
```

SpecA 分阶段实验脚本：

```bash
bash scripts/run_covr_template_experiment.sh validate
```

复用已有 audit 和 manifest 时，应使用 `eval` phase，并将 evaluation 起点设置为 audit 之后的样本位置。脚本当前的设计口径是：audit 使用前 2,048 张，evaluation 从 `DATASET_START_INDEX=2048` 开始。

常用环境变量：

- `OUTPUT_ROOT`：实验输出根目录
- `MANIFEST`：策略/模板 manifest 路径
- `BANDIT_STATE`：bandit 状态路径
- `SESSION_ID`：COVR 会话 ID
- `SKIP_EXISTING=1`：在 `eval` phase 跳过已有 `results.json`
- `AUDIT_N=2048`：audit 样本数量

注意：脚本的 `eval` phase 有结果跳过逻辑；直接调用 `baseline`、`speca`、`template` 或 `bandit` phase 时，当前脚本中的 `_maybe_skip ... || true` 不会真正阻止后续 Python 命令执行，这部分仍需修正。

## 10. 评估口径与风险

### 10.1 FID 与 IS 是不同目标

当前结果显示 Bandit 的 FID 更低但 IS 更低。后续报告必须同时列出 FID 和 IS，不能只用 FID 宣称质量全面改善。

### 10.2 Terminal fidelity 的局限

Terminal fidelity 只比较最后一步、同一个 `x_t` 下的局部 transition，不直接衡量前 49 步误差的累积。如果某个策略最后一步固定执行 full forward，它的 terminal loss 可能天然接近 0，从而被 bandit 偏好，即使它前面的误差并不小。后续应验证：

- terminal loss 与最终 FID/IS 的相关性
- 最后一步为 full 的策略是否存在 reward 偏置
- 是否需要加入少量多时间点 fidelity 或累计误差

### 10.3 Safety 开销

SpecA 的 safety observation 通过 Taylor/full 对照估计 step-level defect。50k Bandit 中 safety FLOPs 为 `0.8727T`，约占 online FLOPs 的 22.0%，且会显著影响速度。TeaCache 当前计划使用方案 A：不做逐步 shadow safety，仅靠 terminal fidelity。

### 10.4 旧 bandit state 不可直接复用

reward 从 H-step defect 切换到 terminal fidelity 后，损失尺度和含义都发生变化。重新实验时应使用新的 `BANDIT_STATE`，不要直接复用旧 reward 产生的 arm statistics。

## 11. 后续验证优先级

1. 修正并验证 TeaCache 的 terminal fidelity reward 路径，使 reward 不再被 `method == "speca"` 限制。
2. 运行 TeaCache strategy manifest 的 GPU smoke，确认 dispatch、state reset、threshold 生效和结果落盘。
3. 对 Bandit 与 adaptive SpecA 做统一计时，分别记录主路径、sentinel、safety、CPU 同步和 VAE/指标时间。
4. 修正实验脚本 standalone phase 的 `SKIP_EXISTING` 行为。
5. 用相同样本切片和相同评估配置核对“历史 FID < 10”与当前 FID 24 左右的差异。
6. 评估多目标 reward 或 Pareto 选择，避免只优化 FID 而牺牲 IS 和速度。
7. TaylorSeer 只有在明确其初始化和运行时控制接口后再接入 dispatcher，不应先添加空实现。

## 12. 50k No-CFG 实验结果 (2026-07-28)

> guidance_scale=1.0, DiT-2-256, 50 steps, seed=42, 50,000 张全量评估  
> COVR Bandit 速度优化配置：safety_sample_rate=0, sentinel_rate=0.02, chain_threshold=5

### 12.1 三组对比

| 方法 | FID ↓ | IS ↑ | FLOPs(T) | img/s | cand img/s | skip% |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.28 | 121.8 | 11.867 | 7.47 | 7.47 | — |
| Adaptive SpecA | 8.04 | 118.9 | 3.089 | 14.45 | 14.45 | 75.8% |
| COVR Bandit | 8.22 | 117.4 | 2.849 | 14.95 | 15.08 | 76.0% |

### 12.2 Stage Profiling (COVR Bandit)

| Stage | 时间(s) |
|---|---:|
| generation_online | 3318.2 |
| denoise_loop | 2903.1 |
| image_save_metrics | 1696.0 |
| speca_full | 1578.4 |
| speca_taylor | 1276.4 |
| vae_decode | 412.5 |
| cuda_sync_wait | 405.6 |
| bandit_state_persist | 24.4 |

COVR overhead: safety=0.0s, terminal=2.6s, control=26.9s

### 12.3 与 CFG=4.5 实验的关键对比

| 维度 | CFG=4.5 (§5) | No-CFG (§12) |
|---|---|---|
| Bandit vs Baseline FID | -7.27 (显著更好) | +0.94 (略差) |
| Bandit vs SpecA 速度 | 4.78 vs 8.58 img/s (慢55%) | 15.08 vs 14.45 img/s (快4.4%) |
| IS 退化 | 452→353 (暴跌22%) | 121.8→117.4 (仅-3.6%) |
| Safety overhead | 0.87T (22% online FLOPs) | 0.0s (完全消除) |

### 12.4 结论

1. **速度优化有效**：消除 safety shadow 后，Bandit candidate 15.08 img/s 确实快于 Adaptive SpecA 14.45 img/s（+4.4%），确认之前瓶颈诊断结论。
2. **无 CFG 下 Bandit FID 优势消失**：CFG=4.5 时 Bandit FID 改善 7.3 点，但 no-CFG 时反而略差 0.94 点。说明 Bandit 的 FID 优势依赖 CFG 放大效应。
3. **IS 退化在无 CFG 下可忽略**：无 CFG 条件下三组 IS 差异仅 ~3.6%，不像 CFG 实验那样出现 22% 暴跌。
4. **无 CFG 下 Bandit 的收益主要是速度**（+4.4%）和略低的 FLOPs（2.849T vs 3.089T），FID/IS 几乎持平。
5. **需要在 CFG=4.5 条件下重新验证速度优化配置**，确认消除 safety 后是否仍保持 FID 优势。

## 13. 数据完整性说明

本记录保存的是当前已经得到并在讨论中确认的汇总数据。原始逐样本图像、完整 JSONL audit、manifest 和 bandit state 仍以实验输出目录/归档为准；如果重新运行实验，应在结果目录中同时保留：

- `results.json`
- `manifest.json`
- `bandit_state.json`（如果启用持久化 state）
- `audit/covr/events_<session_id>.jsonl`
- 本次运行的完整 CLI 参数和 git commit

报告中的“50k”统一解释为 `2,048` 张 audit 加 `47,952` 张 evaluation，不能把 evaluation 命令写成从 `50,000` 开始再请求 50,000 张，否则会超过 ImageNet 数据集边界。
