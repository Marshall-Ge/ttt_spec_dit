# VFL (Verification Feedback Loop) 项目进度与诊断报告

> **目的**：请外部专家评估 VFL L3 LoRA 方向是否值得继续，以及最优发表策略。
> **日期**：2026-07-16
> **项目**：TTT-DiT (`/Users/marshall/Projects/ttt_spec_dit/`)

---

## 1. 项目背景

对 DiT-2-256 和 PixArt-XL-2-512x512 扩散生成模型做**推理加速**。已实现三个加速器（正交可叠加）：

| 加速器 | 机制 | DiT FLOPs 削减 |
|--------|------|---------------|
| SpecA | Per-block Taylor 级数缓存 | 70%-78% |
| TeaCache | Per-step 残差缓存（block0 调制信号） | 45%-50% |
| DDIM | 步数压缩 | 取决于步数 |

配置：50 步 DDPM、cfg=2.0、ImageNet 50k、batch=32。Baseline FID=**11.67**、IS=229.56、skip_ratio=0.78、4.5× FLOPs reduction。

---

## 2. VFL 提出的初衷

SpecA 用**静态阈值**决定每步是 `full`（全量计算 + 缓存）还是 `Taylor`（缓存预测 + 跳过）：

```python
threshold = base_threshold * decay_rate ** progress
```

这个公式有几个问题：
1. **离线、无数据意识**：阈值不随输入分布变化，对所有 prompt、所有 timestep 用同一曲线
2. **无反馈闭环**：推理时不验证 Taylor 预测对不对，错了也不修正
3. **手工调参**：base_thresh=0.01、decay_rate=0.01 是经验值，迁移到新模型/数据集要重调

VFL 想做的是：**把"是否缓存"的决策变成 verification-driven 的在线学习问题**。三层递进架构：

| 层 | 机制 | 梯度 | 触发 | 改进对象 |
|----|------|------|------|---------|
| **L1** | Online EMA calibrator | 否 | 每个验证事件 | **缓存决策阈值**（替换固定 decay 公式） |
| **L2** | Stratified replay buffer | 否 | 每事件被动存储 | 为 L1/L3 提供分层样本 |
| **L3** | 异步 LoRA 微调（daemon thread） | 是 | buffer 满 ≥2 strata / 200 events / 10 anchors | **backbone 轨迹平滑度** |

L1/L2/L3 互不依赖，可单独启用：
- `--vfl-no-train`：仅 L1（轻量，零训练开销）
- `--vfl`：完整三层（L3 daemon 线程异步训练 LoRA，推理时 hot-swap）

详见 `CLAUDE.md` §12。

---

## 3. 三次 50k 实验结果（核心数据）

所有实验：DiT-2-256、SpecA 加速、50 步、cfg=2.0、ImageNet 50k、seed=42。

| 配置 | FID | IS | ||B||_F 终值 | LoRA 实际作用 | 相对 baseline |
|------|-----|-----|------------|--------------|--------------|
| Baseline (无 VFL) | 11.67 | 229.56 | N/A | N/A | — |
| **L1-only (v3)** | **11.45** | 229.56 | 0.000 | 静默 no-op（代码 bug） | **-0.22** |
| L1+L2+L3 curvature (v4) | 16.79 | 229.56 | 3.682 | 训练正常，每 cycle hot-swap | +5.12 |
| L1+L2+L3 homing+identity (v5) | **19.0** | 229.56 | ? | L_anchor 驱动 | +7.33 |

**关键观察**：
- **每次启用 L3 LoRA 训练都让 FID 单调恶化**
- v3 的 -0.22 FID 完全来自 L1 阈值校准（LoRA 因 bug 是 no-op）
- v4 修了 bug，LoRA 真的训练了，FID 立刻变差 5.12
- v5 重新设计 loss，FID 进一步变差到 19.0

---

## 4. 三次 L3 失败的根因分析

### 4.1 v3 (no-op bug)

`async_trainer.py:_ensure_train_model` 调 `attach_lora_all_layers` 在已 attach LoRA 的深拷贝上重新 attach，`_resolve_path` 拒绝 `LoRALinear`（继承 `nn.Module` 不是 `nn.Linear`），抛 TypeError 被 `except` 静默吞掉。`_layer_wrappers` 变成空字典，LoRA `B` 矩阵始终零。FID 11.45 的改善完全归因于 L1。

**已修复**（用 `_collect_existing_lora_wrappers` 扫描已有 LoRALinear）。

### 4.2 v4 (curvature loss 设计错配)

Loss：`L = L_supervised + λ_curv * L_curvature + λ_anchor * L_anchor`

- `L_curvature`：对 true_feature 时序拟合低阶多项式，惩罚残差——目标是让 backbone 输出"更平滑、更易被 Taylor 外推"
- **问题**：模型自然去噪轨迹本来就不该是低阶多项式，强行平滑化破坏生成语义
- `||B||_F` 单调增长 0.25 → 3.68 无饱和，max 长到 9.89，对 backbone 是显著扰动
- 193 次 hot-swap 破坏 FID 评估的统计一致性

### 4.3 v5 (homing loss target 空间错配) —— 最关键

我们设计的新 loss：
```
L_homing = MSE(Backbone(x_drift) + LoRA(x_drift), Backbone(x_golden))
L_identity = ||LoRA(x_golden)||²
```

但 **buffer 里存的 `true_feature` 是 `base_block(x_drift)`，不是 `base_block(x_golden)`**。

证据：`models/dit.py:370-390`
```python
_block_input = hidden_states.clone()    # ← Taylor 漂移后的输入 x_drift
full_hidden = _block_input
# 跑 full block on full_hidden (= x_drift)
_vfl_record_speca_event(..., full_hidden=full_hidden, block_input_hidden=_block_input)
```

Agent 实现的 `compute_training_loss` 实际计算的是：
```
MSE(block_with_lora(x_drift), base_block(x_drift)) = ||LoRA(x_drift)||²
```

这是 **L_identity 作用在漂移输入上**，不是 L_homing。No-op 起点处（B=0）梯度 = 0。

**实际驱动训练的只有 L_anchor**（real anchor 数据上的标准 diffusion loss），等价于 LoRA fine-tuning：
- 50 个 anchor（buffer 容量）
- 9650 步训练无衰减
- anchor overfit → 分布外退化 → FID 19

### 4.4 共同根因

L1 改进**缓存决策**（per-layer per-bucket EMA 阈值），不修改 backbone。FID 改善真实。

L3 改进**缓存内容**（用 LoRA 修正 backbone 输出），引入 distribution shift。三次设计都因 **target signal 错配**导致 LoRA 学到错误目标：
- v4：target = 低阶多项式（人为平滑）
- v5：target = `base_block(x_drift)`（自指，no-op 对称性）
- 真正需要的 `base_block(x_golden)` **不在 buffer 里**，需要并行 teacher forward，OOM 风险

---

## 5. 当前代码状态

### 已修复
- L3 LoRA 训练链断裂 bug（`_collect_existing_lora_wrappers`）
- `||B||_F` 监控遍历对象错误（改用 `self._train_model.modules()`）

### 已实现但效果差
- `curvature_loss.py` 重写为 homing + identity + anchor 三项
- `--vfl-lambda-identity` CLI flag
- 8 个单元测试通过

### 未做
- `EvalGate` 自动发布闸门未接入主流程（CLAUDE.md §8.13/§12.6 已知缺陷）
- 真正的 `base_block(x_golden)` 录制路径（需并行 teacher）

---

## 6. 三个候选方向（请专家评估）

### A. 固化 L1-only，放弃 L3

**论文卖点**：Verification-Driven Adaptive Caching
- L1 per-layer per-bucket EMA 阈值（novel，多数 cache 方法用静态阈值）
- L2 stratified buffer 作为 L1 的基础设施
- **Negative result section**："Why learned correction fails" — 三次 loss 设计分析

**优点**：FID 锁定 11.45（需 A/B 确认显著性）；零推理开销；下行风险低
**缺点**：创新性中等；L3 的三次失败要写成诚实分析而非主贡献
**风险**：L1 的 -0.22 FID 可能不显著，需 1k A/B 控制

### B. 实现 Plan B：真 golden teacher

异步训练线程跑小批量无 skip forward 录 `base_block(x_golden)`，构造真 L_homing。

**优点**：创新性高（online LoRA distillation from parallel teacher）；如果 work，strong paper
**缺点**：+3GB 显存、+50% 训练时间、OOM 风险；收益不确定（"对齐到 golden"不等于"生成更好"）
**风险**：三次失败后的第四次赌注；deadline 紧时不建议

### C. 重新框定问题：条件激活 LoRA

LoRA 只在 SpecA error_value 高的 step 激活（gating），正常 step 严格 no-op。

**优点**：缩小 LoRA 影响范围；下界 = L1-only 11.45；推理零开销
**缺点**：gating 是已知技术，novel 性有限；需改 LoRA forward 加 gating 逻辑
**风险**：中等工作量，中等收益

---

## 7. 关键问题请教专家

1. **L1 的 -0.22 FID 改善**在 50k ImageNet 上是否够发表？还是噪声？需要多少 seed 才能确认？

2. **L3 的三次失败**是工程问题（loss 没设计对）还是方向问题（learned correction 本质上引入 distribution shift）？是否有第四种 loss 设计能绕开？

3. **negative result section** 在扩散加速领域的发表价值？顶会（NeurIPS/ICML/CVPR）会接受吗？

4. 三次 L3 实验的 **FID 单调恶化轨迹**是否已经足够支撑"learned cache correction 在 diffusion 上不可行"的论点？还是需要更多控制实验？

5. 从论文策略看，**A / B / C 哪个 ROI 最高**？考虑：投稿 deadline、当前数据、3 次失败的事实。

6. 是否有我们没想到的第四条路？例如：
   - 改 LoRA target 空间（从 hidden 到 noise prediction）
   - 用 score matching loss 而不是 MSE
   - 在 latent space 而不是 hidden space 训练
   - 完全放弃 learned correction，转做 SpecA 阈值的 meta-learning

---

## 8. 关键文件

| 文件 | 作用 |
|------|------|
| `CLAUDE.md` | 项目全文档，§12 是 VFL |
| `verification_feedback_loop/curvature_loss.py` | L3 loss（v5 实现） |
| `verification_feedback_loop/async_trainer.py` | L3 daemon 线程 |
| `verification_feedback_loop/replay_buffer.py` | L2 buffer 存储 |
| `verification_feedback_loop/online_calibration.py` | L1 EMA 阈值 |
| `models/dit.py:370-411` | SpecA check_layer 录 `true_feature` 的位置 |
| `run_dit.py:821` | 推理模型预 attach LoRA 的入口 |
| `vfl.log` | v5 实验完整日志（4188 行） |

---

## 9. 实验复现命令

```bash
# Baseline
python main.py --model dit --task c2i --dataset imagenet \
    --method baseline --metrics fid is flops latency \
    --guidance_scale 2.0 --seed 42 --num_steps 50 \
    --n_prompts 50000 --batch_size 32

# L1-only (推荐固化方向)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is flops latency \
    --guidance_scale 2.0 --seed 42 --num_steps 50 \
    --n_prompts 50000 --batch_size 32 \
    --vfl --vfl-no-train

# L1+L2+L3 (当前 v5 实现)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is flops latency \
    --guidance_scale 2.0 --seed 42 --num_steps 50 \
    --n_prompts 50000 --batch_size 32 \
    --vfl --vfl-lambda-identity 1.0
```

---

## 10. 三句话总结

1. **VFL L1（online 阈值校准）work**：FID -0.22，零推理开销，是真实可发表贡献。
2. **VFL L3（learned LoRA correction）三次失败**：FID 11.45 → 16.79 → 19.0，根因是 target signal 错配（buffer 存的是 `base_block(x_drift)` 不是 `base_block(x_golden)`）。
3. **关键决策点**：固化 L1 + negative result（A，低风险），还是赌 Plan B 真 golden teacher（B，高风险高回报），还是条件激活 LoRA（C，中风险）。
