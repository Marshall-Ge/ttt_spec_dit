# VFL (Verification Feedback Loop) 现状总结

> 编写日期：2026-07-06 | 目标读者：外部专家 | 比代码更抽象，比论文更具体

---

## 1. 项目背景

### 1.1 我们要做什么

对 **DiT-2-256**（675M 参数，class-conditional ImageNet 256×256 扩散模型）进行推理加速。加速方法叫 **SpecA**（Speculative Acceleration）。

### 1.2 SpecA 原理（极简版）

扩散模型去噪时，每一步都要跑 28 层 transformer blocks。SpecA 的核心思路：

- **Full step**（全量步）：正常跑所有 28 层，同时**缓存每层每个子模块的输出 + 有限差分导数**
- **Taylor step**（外推步）：不跑 attention/MLP 计算，用 **Taylor 级数外推**预测当前输出（根据缓存的上次 full step 输出 + 导数 × 步距）

```
x_t ≈ x_prev + distance × derivative + (distance²/2!) × second_derivative + ...
```

- 每步根据误差历史 + 阈值衰减决定是 full 还是 Taylor
- 在 `check_layer`（DiT 用 layer 20）做一次 **Taylor vs full 误差探测**：如果在 Taylor 步中检测到第 20 层的预测误差超过阈值，下一步强制 full

**核心超参**：`base_threshold=0.01`, `decay_rate=0.01`, `min_taylor_steps=1`, `max_taylor_steps=4`, `error_metric=cosine_similarity`

### 1.3 SpecA 的效果和问题（DiT-2-256, 50k ImageNet, 50 steps）

| 指标 | Baseline（50 步全量 DDIM） | SpecA |
|---|---|---|
| FID ↓ | 7.32 | 8.06 |
| IS | 229.6 | 229.6 |
| FLOPs | 11.87 T | 2.87 T (**-75.8%**) |
| 跳过率（Taylor 步占比） | — | 76% |

**问题**：FLOPs 减少 75% 但 FID 差了 0.74。我们希望在不损失更多质量的前提下进一步提高跳过率，或者在同一跳过率下缩小与 baseline 的 FID 差距。

---

## 2. VFL 的设计目标

VFL（Verification Feedback Loop）的核心假设：

> SpecA 的 Taylor 预测误差是可以学习的。如果能**在线收集** Taylor-vs-full 的误差数据，用这些数据**训练一个小型 LoRA adapter** 修正 Transformer 输出，使其去噪轨迹更"光滑"（更易被低阶多项式外推），就能在**相同或更低误差下容忍更多 Taylor 步**。

两层收益路径：
1. **M2 OnlineCalibrator**（无梯度）：从误差分布中学习更好的**阈值**（per-layer × per-timestep-bucket），在不损害质量的前提下多跳过一些步
2. **M5 LoRA 训练**（有梯度）：用收集的误差事件训练 LoRA adapter，使 Taylor 预测更准 → 同样的阈值下误差更小 → 可以更激进地跳过

**与 TTT（Phase 3）的区别**：TTT 改进缓存的**内容**（hidden state correction），VFL 改进缓存的**决策**（threshold / trajectory smoothness）。

---

## 3. VFL 系统架构（8 个模块）

### 整体数据流

```
推理循环 (run_dit.py)
  │
  ├─→ SpecA Taylor step 中 do_check 触发
  │     │
  │     ▼
  │   [M1] 在 check_layer=20 做 Taylor vs full 比较
  │     计算 error_value (cosine_similarity)
  │     │
  │     ├─→ [M2] OnlineCalibrator.update(error_value)
  │     │      EMA 维护 per-(layer, bucket) 误差分布
  │     │     → speca_cal_type() 查询 get_threshold_cached()
  │     │     在线阈值替代静态 base_threshold * decay^progress
  │     │
  │     └─→ [M1] 构造 VerificationEvent
  │           (predicted_hidden, true_hidden, latent_input, class_labels, ...)
  │           所有 tensor detach + half + cpu → GPU→CPU 同步传输
  │           │
  │           ▼
  │         [M3] StratifiedReplayBuffer.add(event, kind="hard_negative")
  │           按 (layer_id, timestep_bucket) 分层 reservoir sampling
  │           每 stratum 容量 1000，全局 3 桶 × 28 层 = 最多 84 strata
  │           实际非空: ~3 strata (只有 layer 20 有 event, 3 个 timestep bucket)
  │
  ├─→ 每 5 步: [M3] 收集 anchor sample
  │     (latent, timestep, noise_pred_target) 独立存储，最多 50 个
  │
  └─→ 后台线程 (daemon=True)
        │
        [M6] AsyncTrainingWorker._train_loop()
          每 5s 轮询 buffer readiness
          条件: ≥2 strata 非空, ≥200 总 event, ≥10 anchor
          │
          ▼
        _train_once() — 一次训练周期 (~45s)
          │
          ├─ [M4] 第一次触发时 lazy init:
          │     transformer deepcopy → fp32 → attach_lora_all_layers(rank=4)
          │     28 层 × 6 Linear/层 × (4×1152×2) ≈ 5.2M LoRA params
          │     backbone requires_grad=False
          │
          ├─ buffer.sample_training_batch(batch_size=16)
          │     hard_negative:normal:anchor = 0.5:0.3:0.2
          │
          ├─ [M5] compute_training_loss()
          │     L_total = L_supervised + λ_curv * L_curvature + λ_anchor * L_anchor
          │     │
          │     ├─ L_supervised: 对每个 event, 重跑 forward(带 LoRA),
          │     │   hook 捕获 layer 20 输出, MSE(lora_hidden, event.true_feature)
          │     │
          │     ├─ L_curvature: 按 (sample_id, layer) 分组,
          │     │   对同组不同 step 的 hidden states 拟合 order=2 多项式,
          │     │   惩罚拟合残差
          │     │
          │     └─ L_anchor: 在 anchor samples 上做标准 diffusion MSE loss
          │
          └─ 50 步梯度更新 (max_events_per_step=2)
               → save_lora_checkpoint() to disk
```

### 各模块详细说明

#### M1: Verification Hook (`verification_hook.py` + `vfl_state.py`)
- `VerificationEvent` dataclass: 包含 layer_id, timestep, predicted_feature, true_feature, error_value, latent_input, class_labels 等
- `make_speca_event()`: 在 SpecA 的 do_check 路径（layer 20）构造事件
- `make_teacache_probe_event()`: 在 TeaCache calc 步构造事件（当前运行仅 SpecA，未使用）
- `record_event()`: reject 事件全部记录，accept 事件按 2% 采样率随机记录
- **DiT 端**：`models/dit.py:_vfl_record_speca_event()` — 两层模式：
  - **full path**（`--vfl`，buf 已注册）：per-sample 拆分 (for i in range(B))，GPU→CPU 传输
  - **scalar path**（`--vfl --vfl-no-train`，buf=None）：只传一个标量 error_value 给 calibrator，零 GPU→CPU 开销

#### M2: Online Calibrator (`online_calibration.py`)
- 维护 `(layer_id, timestep_bucket)` → `_EMAThreshold` 的字典
- `_EMAThreshold`: EMA of error mean + variance → threshold = mean + k * std (k=3.0)
- `get_threshold_cached()`: TTL=5 的缓存，避免每步重复计算
- 在 `speca_cal_type()` 中被调用：如果 calibrator 返回更宽松的阈值，覆盖静态公式
- 本次运行产生 **1,399,968** 次 update，覆盖 3 个 (layer=20, bucket=0/1/2) strata

#### M3: Stratified Replay Buffer (`replay_buffer.py`)
- 按 `(layer_id, timestep_bucket)` 分 stratum，每 stratum reservoir sampling ring buffer
- 容量：`capacity_per_stratum=1000`，全局约 84 strata（28 层 × 3 桶），但实际有数据的只有 3 个（layer 20 × 3 buckets）
- 全局 buffer 总容量约 3000 event（本次运行始终饱和）
- `sample_training_batch(batch_size, ratio)`: 按 ratio 从各 strata 采样
- 带 `threading.RLock` 保护并发读写
- `max_encoder_hidden_states_events=200`：PixArt 内存安全阀（DiT 不受影响）
- Anchor samples 独立存储，最多 500 个

#### M4: LoRA Adapter (`lora_adapter.py`)
- `LoRALinear`: 标准 LoRA — y = Wx + (alpha/r) * (B @ A) @ x，B 零初始化
- 每 block 6 个 target path: `attn1.to_q/k/v`, `attn1.to_out.0`, `ff.net[0].proj`, `ff.net[2]`
- `attach_lora_all_layers()`: Phase 2 用，挂载全部 28 层，rank=4, alpha=1.0
- 总计约 5.2M 参数（相比 backbone 675M，不到 1%）
- `save_lora_checkpoint()` / `load_lora_checkpoint()` / `find_latest_checkpoint()`: checkpoint 管理

#### M5: Curvature Loss (`curvature_loss.py`)
- **`L_supervised`**: 对每个 event 重跑 forward（带 LoRA），hook 捕获目标层输出，MSE 逼近 true_feature
- **`L_curvature`**: `trajectory_curvature_loss()` — 对同一 (sample_id, layer) 的不同 step 的 hidden states 序列，用最小二乘拟合 order=2 多项式，惩罚拟合残差。**不依赖 predicted_feature — 只衡量真实轨迹的"可外推性"**
- **`L_anchor`**: 标准扩散 loss（noise prediction MSE），在 anchor samples 上计算，防止 catastrophic forgetting
- 权重：`lambda_curvature=1e-4`, `lambda_anchor=1.0`

#### M6: Async Training Worker (`async_trainer.py`)
- 后台 daemon 线程，独立于推理循环
- `_train_loop()`: 每 5s poll buffer → 如果 ready 则 `_train_once()`
- 训练在 `self._train_model`（原模型的 fp32 deepcopy）上进行，推理模型 fp16 不变
- 每次 `_train_once()`: sample batch → 50 步梯度更新 → save checkpoint
- 本次运行: **162 个训练周期, 8100 步梯度更新, 约 2 小时 GPU 训练**
- 产出 checkpoint: `lora_candidate_v001.pt` ~ `v162.pt`，保留最近 5 个
- 容错：任何异常被捕获，不传播到推理线程

#### M7: Eval Gate (`eval_gate.py`)
- Canary 发布闸门：LoRA candidate 在发布前需通过质量回归测试（FID/CLIP + quality_epsilon）和效果验证（reject 率下降 > reject_delta）
- **当前未接入实际流程**

#### M8: Version Registry (`version_registry.py`)
- 确保 LoRA adapter 不会跨 base model 版本静默复用
- 检测版本变化时说标记 adapter 为 stale
- **当前未完全接入**

---

## 4. 实验结果

### 4.1 50k 全样本测试（ImageNet, DiT-2-256, 50 steps）

| 指标 | Baseline | SpecA | SpecA+VFL (full) | SpecA+VFL (no-train) |
|---|---|---|---|---|
| **FID ↓** | 7.32 | 8.06 | 8.07 | 8.07 |
| **IS** | 229.6 | 229.6 | 229.6 | 229.6 |
| **FLOPs** | 11.87 T | 2.87 T | 2.61 T | 2.61 T |
| **FLOPs 减少** | 0% | 75.8% | 78.0% | 78.0% |
| **跳过率** | — | 76% (38T/12F) | 78% (39T/11F) | 78% (39T/11F) |
| VFL events 收集 | — | — | 3000 | — |
| VFL 训练周期 | — | — | 162 | — |
| VFL 训练步数 | — | — | 8100 | — |
| VFL calibrator updates | — | — | 1,399,968 | 43,751 |
| VFL checkpoint | — | — | v162 | — |
| 训练 loss (最终) | — | — | ~6.9e-5 | — |

**关键观察**：
- VFL full 和 no-train **FID 完全相同** (8.068 vs 8.068)，说明 162 个训练周期产出的 LoRA 对本次运行精度**零贡献**
- Calibrator 将跳过率从 76% 提升到 78%（+2pp），FID 未变——这是一个正向结果，验证了 M2 的有效性
- 但 +2pp 的收益远小于预期

### 4.2 500-image AB 测试（对比 SPECA_DEFAULTS 旧 vs 新配置，CFG=1.0 vs 4.5）

| 配置 | FID | Wall (s/batch) | 跳过率 |
|---|---|---|---|
| SpecA (no CFG) | 100.6 (500 img) | 2.29 | 74% |
| SpecA (CFG=4.5) | 102.7 (500 img) | 5.15 | 72% |
| SpecA VFL new (no CFG) | 100.6 (500 img) | 2.31 | 74% |
| SpecA VFL new (CFG=4.5) | 102.7 (500 img) | 5.15 | 72% |

500 图 FID 噪声大（~100 vs 50k 的 ~8），AB 测试主要用于验证配置等价性。结论：new/old 配置产生相同结果，CFG 降低跳过率约 2pp。

### 4.3 训练 loss 轨迹

```
cycle #1:   loss_mean=0.000384
cycle #5:   loss_mean=0.005126  (峰值)
cycle #10:  loss_mean=0.000686
cycle #20:  loss_mean=0.000440
cycle #50:  loss_mean=0.000073
cycle #100: loss_mean=0.000051
cycle #150: loss_mean=0.000173
cycle #162: loss_mean=0.000069  (最终)
```

- Loss 在前几个周期迅速从 ~4e-4 降到 ~7e-5，之后 150+ 个周期几乎不降，在 3e-5 ~ 5e-4 之间波动
- 每个周期 50 步，每步只采样 2 个 event（`max_events_per_step=2`）
- Buffer 始终保持 3000 event 饱和（前 ~100 张图就填满了）

---

## 5. 根因分析

### 5.1 直接原因：LoRA 权重在当前运行中未被加载

这是**有意为之的设计**而非 bug。`AsyncTrainingWorker` 在 fp32 深拷贝上训练，checkpoint 写磁盘。推理用的 fp16 模型只在**启动时**通过 `find_latest_checkpoint` 加载历史权重：

```python
# run_dit.py 第 759-766 行
prev_ckpt = find_latest_checkpoint(vfl_output_dir)
if prev_ckpt:
    load_lora_checkpoint(generator.transformer, prev_ckpt)
```

本次是首次 VFL 全样本运行，无历史 checkpoint → 推理模型裸奔。这是"跨 run 飞轮"设计——训练收益延迟到下次运行兑现。本次运行中唯一影响推理的 VFL 组件是 **M2 OnlineCalibrator**。

但这也意味着：**即使训练信号是完美的，本次运行也看不到精度改善**。要验证 LoRA 是否有效，必须先跑一次积累 checkpoint，再跑第二次加载它。

### 5.2 训练信号问题 1：`sample_id` 粒度过粗，curvature loss 收到随机噪声

```python
# run_dit.py 第 911 行
set_vfl_sample_id(batch_start)  # batch_start = 0, 32, 64, ...
```

每 batch（最多 32 张图）共享同一个 `sample_id`。在 `curvature_loss.py` 中：

```python
curvature_by_layer[(event.sample_id, hook_layer)].append(
    (event.step_idx, lora_hidden))
```

curvature loss 按 `(sample_id, layer)` 分组后拟合多项式。但 `sample_id=0` 包含 **32 张不同图片** × 每张 39 个 Taylor step = **~1248 个 event**，来自完全不同图片的 hidden states 被混在一起按 step_idx 排序拟合多项式——这等价于把 32 个不同人的身高按年龄排成一条"生长曲线"。

**影响**：curvature loss 的梯度方向是随机的（不同图片的 hidden states 之间不存在可外推的多项式关系）。这直接解释了为什么训练 loss 极小（3e-5 ~ 5e-4）且 162 个周期不下降——curvature loss 在被随机噪声驱动。

**修复方向**：每张独立图片需要唯一的 `sample_id`。应在 per-sample 循环中设置 `set_vfl_sample_id(global_idx + i)`。

### 5.3 训练信号问题 2：只有 layer 20 有监督信号

SpecA 只在 `check_layer=20` 做 error probe 并记录 event。`compute_training_loss` 的 supervised MSE 也只比较 layer 20 的输出：

```
梯度流: loss → layer_20_hidden → layers 19,18,...,0 (反传) 和 layers 21-27 (前传影响很小)
```

28 层共享一个来自 layer 20 的 MSE 梯度，每层约 185K LoRA 参数的梯度非常微弱。更重要的是，**layers 21-27（check_layer 之后的层）几乎没有有效的 supervised 信号**——它们只在 anchor loss（标准扩散 loss，50 个 anchor）中有间接信号。

**修复方向**：在 Taylor step 中对多层（至少均匀采样 4-7 层）都做一次 full vs Taylor 比较并记录 event，给每层直接监督。

### 5.4 训练信号问题 3：`max_events_per_step=2`

```python
# config.py 第 105 行
max_events_per_step: int = 2
```

每个梯度步从 3000 event 中随机采 2 个。这导致：
- 梯度噪声极大（等效 batch_size=2）
- 50 步 × 2 event × 162 周期 = 总共只有 16,200 次 forward，且每次只见 2 个样本
- 与其他 DL 训练的典型 batch size（16-64）相差一个数量级

**修复方向**：提高到至少 8-16。如果 GPU 内存紧张，可以减少 `trainer_steps_per_trigger`（从 50 降到 20）作为补偿。

### 5.5 训练信号问题 4：Buffer 饱和过早，数据多样性差

- Buffer 容量 3000 event（3 个非空 strata × 1000）
- 每张图产生约 39 个 Taylor step × 1 event/step = 39 event
- **前约 77 张图就填满 buffer**（3000 / 39 ≈ 77），但全量测试有 50k 张图
- Buffer 饱和后 reservoir sampling 保留的是随机替换的混合（早期 + 后期 event），并不代表多样的图片分布
- 所有 event 都来自相同的 layer=20，没有跨层多样性

### 5.6 训练信号问题 5：Anchor 数量太少

- 每 5 步收集 1 个 anchor（`step_idx % 5 == 0`），最多 50 个
- 50 个 anchor ÷ 1000 个 ImageNet 类 = 平均每类 0.05 个 anchor
- 每个训练 batch 只采 `int(16 × 0.2) = 3` 个 anchor，覆盖非常有限

### 5.7 一个更深层的问题：layer 20 error 是"症状"而非"病因"

Taylor 预测在 layer 20 的误差，是 layers 0-19 累计 Taylor 近似误差的结果。Layer 20 自己的 block 可能完全正常。把 supervised MSE 加在 layer 20 上，相当于"症状处用药"——LoRA 被训练去修正 layer 20 的输出以匹配真实值，但误差的根源在前面的层。

如果能收集每层的 Taylor-vs-full 误差，就可以让每层的 LoRA 学习修正**本层的** Taylor 近似误差，而不是让 layer 20 的 LoRA 去承担 layers 0-19 累积的所有误差。

---

## 6. 已确认正常的部分

以下组件经过 50k 全样本测试验证，工作正常：

1. **M1 事件采集**：稳定产生 3000 event，per-sample 拆分逻辑正确，CFG dedup 正确
2. **M2 OnlineCalibrator**：140 万次 update，TTL cache 有效，将跳过率从 76% 提升到 78%，FID 不降
3. **M3 Replay Buffer**：reservoir sampling 正常，并发读写无死锁，3 strata 均正常填充
4. **M4 LoRA 挂载**：fp32 deepcopy + attach_lora_all_layers 正常，5.2M params，B 零初始化正确
5. **M6 AsyncTrainingWorker**：后台线程稳定运行约 2 小时，162 周期无 crash（crash_count=0），checkpoint 正常写入
6. **SpecA + VFL 共存**：speca_cal_type 正确调用 calibrator.get_threshold_cached()，不影响 baseline 路径

---

## 7. 待专家回答的关键问题

### 7.1 架构层面

1. **跨 run 飞轮 vs 运行内生效**：当前设计是 LoRA 训练产出只在下一次推理启动时加载。是否需要改成运行内热切换（比如每 N 个 batch 用最新 checkpoint 更新推理模型）？有没有更优雅的方式让训练收益在当前 run 内就体现？

2. **全层 LoRA vs Top-K**：当前 Phase 2 设计挂载全部 28 层（rank=4），理由是 buffer 稀疏时 layer selection 不可靠。但实际上事件只来自 layer=20，其他 27 层的 LoRA 几乎收不到 supervised 梯度。是否应该回到 Top-K 策略，只挂载有高误差信号的层？

3. **Curvature loss 的样本分组**：当前按 `(sample_id, layer)` 分组，要求同一张图同一层的不同 step hidden states。但同一步内不同层的 hidden states 序列也可以拟合多项式（cross-layer trajectory）。哪种分组更有物理意义？

### 7.2 训练策略层面

4. **Buffer 早饱和问题**：3000 event 在前 100 张图就满了，50k 图的数据多样性完全没被利用。是否应该：(a) 增大 buffer 容量（如 10k-50k）？(b) 采用 sliding window 按时间淘汰？(c) 改为 per-image 采样（每张图只保留最后几个 event）？

5. **Loss 设计**：当前 L_supervised 只用 MSE(lora_hidden, true_feature)。是否应该加入 contrastive loss（让 Taylor 预测更接近 full 而非偏离？）或者其他更适合"修正近似误差"的 loss？

6. **curvature loss 的 λ 调优**：当前 `lambda_curvature=1e-4`，在 loss 中占比很小（~1e-9 vs supervised ~1e-4）。但增大 λ 在 sample_id 修复之前只会放大随机噪声。修复 sample_id 后，合理的 λ 范围应该是什么？

### 7.3 实验设计层面

7. **验证飞轮假设的最简实验**：在修复上述问题之前，是否应该先做一个"作弊实验"——在 LoRA checkpoint 产出的同一次运行中手动加载，验证 LoRA 是否存在有意义的精度提升？如果作弊实验的 FID 也不改善，那问题就不在飞轮设计而在训练信号本身。

8. **SpecA vs TeaCache**：当前 VFL 主要针对 SpecA 设计，但 event 采集框架也支持 TeaCache。TeaCache 的"skip vs calc"决策是否更适合 VFL？（TeaCache 的 skip 步产生更大的 hidden state 跳变，可能有更强的训练信号）

---

## 8. 关键文件索引

| 文件 | 行数 | 职责 |
|---|---|---|
| `verification_feedback_loop/__init__.py` | 119 | 公共 API 导出 |
| `verification_feedback_loop/vfl_state.py` | 274 | 全局状态 + record_speca/teacache_event |
| `verification_feedback_loop/verification_hook.py` | 269 | M1: VerificationEvent + make_*_event + record_event |
| `verification_feedback_loop/online_calibration.py` | 344 | M2: _EMAThreshold + OnlineCalibrator |
| `verification_feedback_loop/replay_buffer.py` | 392 | M3: _Stratum + StratifiedReplayBuffer |
| `verification_feedback_loop/lora_adapter.py` | 447 | M4: LoRALinear + attach/detach/checkpoint |
| `verification_feedback_loop/curvature_loss.py` | 407 | M5: trajectory_curvature_loss + compute_training_loss |
| `verification_feedback_loop/async_trainer.py` | 691 | M6: AsyncTrainingWorker (Phase 2) + AsyncTrainer (legacy) |
| `verification_feedback_loop/config.py` | 113 | VFLConfig dataclass |
| `verification_feedback_loop/eval_gate.py` | ~120 | M7: EvalGate (not wired) |
| `verification_feedback_loop/version_registry.py` | ~120 | M8: VersionRegistry (not wired) |
| `models/dit.py` | 716 | DiT forward 中的 VFL hook 点 + _vfl_record_* helpers |
| `run_dit.py` | 1208 | VFL 初始化的入口 + anchor 收集 + VFL stats 聚合 |
| `config.py` (root) | 101 | SPECA_DEFAULTS 等全局默认值 |
| `accelerators/speca.py` | 461 | speca_cal_type 中 calibrator 查询点 |
| `output/c2i_dit_imagenet_speca_vfl/results.json` | — | 50k VFL 全量结果 |
| `output/c2i_dit_imagenet_speca_vfl_notrain/results.json` | — | 50k VFL calibrator-only 结果 |
| `output/c2i_dit_imagenet_speca/results.json` | — | 50k 纯 SpecA 结果 |
| `output/c2i_dit_imagenet_baseline/results.json` | — | 50k baseline 结果 |
| `vfl.log` | — | VFL 全量运行的完整 log |

---

## 9. 复现命令

```bash
# 纯 SpecA baseline
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops \
    --seed 42 --num_steps 50 --n_prompts 50000 \
    --guidance_scale 4.5 --batch_size 32

# SpecA + VFL (full: calibrator + LoRA training + buffer)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops \
    --seed 42 --num_steps 50 --n_prompts 50000 \
    --guidance_scale 4.5 --batch_size 32 --vfl

# SpecA + VFL (calibrator-only, no training overhead)
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops \
    --seed 42 --num_steps 50 --n_prompts 50000 \
    --guidance_scale 4.5 --batch_size 32 --vfl --vfl-no-train

# 第二次 VFL 运行（加载上一轮的 LoRA checkpoint 做飞轮验证）
# 将 --vfl-output-dir 指向上一轮的 vfl/ 目录
python main.py --model dit --task c2i --dataset imagenet \
    --method speca --metrics fid is latency flops \
    --seed 42 --num_steps 50 --n_prompts 50000 \
    --guidance_scale 4.5 --batch_size 32 --vfl \
    --vfl_output_dir output/c2i_dit_imagenet_speca_vfl/vfl
```
