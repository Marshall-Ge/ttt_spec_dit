---
name: VFL 训练信号源根因诊断
description: VFL L3 LoRA 训练在 50k 图实验中完全无效（v1/v2/no-train FID 完全一致）的根因 — supervised 和 anchor loss 的 target 都是 base forward 输出，导致 no-op 起点对称性下 loss 天然为 0
type: project
originSessionId: 7d3813aa-6029-49d6-9c28-650c01470017
---
## 现象（2026-07-11 实验数据）

50k ImageNet val + SpecA + DiT 实验：

| 实验 | FID | IS | VFL 训练步 | LoRA 更新次数 |
|------|-----|-----|-----------|--------------|
| SpecA 基线（无 VFL） | 8.07 | 229.56 | — | — |
| VFL v1（完整三层） | 8.07 | 229.56 | 8100 | 162 |
| VFL no-train（仅 L1） | 8.07 | 229.56 | — | — |
| VFL time (v2) | 8.07 | 229.56 | 8100 | 162 |
| VFL time_100 | 13.75 | 229.56 | 8200 | 164 |

v1 / v2 / no-train 三个 50k 实验的 FID 和 IS **完全一致**，证明 L2/L3 的 LoRA 训练对推理输出零影响。

## 根因

**核心问题：supervised 和 anchor loss 的 target 都是 base forward 输出，no-op 起点下 loss 天然为 0，B 永远不更新**

### 根因 1：anchor loss 是 self-distillation

`run_dit.py:475`：
```python
target=noise_pred[0:1].detach(),   # 推理时 base forward 的输出
```

`curvature_loss.py:400`：
```python
anchor_losses.append(F.mse_loss(model_out[:, :ch], a_target[:, :ch]))
```

当 LoRA 是 no-op（B=0）起点时：
- `model_out = base_forward(latent, t) = noise_pred`
- `a_target = noise_pred`
- → `anchor loss = 0`，梯度 = 0，B 不动

### 根因 2：supervised loss 方向反了 + no-op 起点为 0

`curvature_loss.py:345`：
```python
target = event.true_feature.to(...)   # SpecA do_check 时 base block 重算输出
```

`curvature_loss.py:335`：
```python
_run_transformer_forward(transformer, latent, timestep, ...)  # vanilla forward，无 SpecA state
```

两层问题：
- **方向反了**：target = SpecA 路径下 block 重算输出（base 重算 with Taylor-predicted input），input = vanilla forward（base 真实计算）。LoRA 学到的是「让 vanilla 输出 ≈ SpecA 输出」=「复制 SpecA 偏差」，与「补偿 SpecA 偏差」相反
- **no-op 起点为 0**：当 LoRA B=0 时，`lora_hidden = base_forward(real_input)`，`target = base_forward(taylor_input)`。两者差异 = Taylor 预测误差经 base_forward 的传播。Taylor 准时 loss ≈ 0；Taylor 误差大时 loss > 0 但方向反

### 根因 3：curvature loss 太弱

`λ_curvature = 1e-4`，curvature loss 量级 ~1e-2，加权后 ~1e-6，对 rank=4 LoRA 梯度不足以让 B 在 162 次 update 内显著偏离零。

## 现象链

```
LoRA B 零初始化（no-op 起点）
   ↓
supervised loss ≈ 0    （根因 2）
anchor loss = 0         （根因 1）
curvature × 1e-4 ≈ 0    （根因 3）
   ↓
梯度 ≈ 0 → B 仍 ≈ 0
   ↓
checkpoint 里 B ≈ 0
   ↓
下一轮 load_lora_checkpoint 加载 B ≈ 0
   ↓
推理 forward 时 LoRA 仍 no-op
   ↓
FID 与 no-train 完全一致
```

`time_100` 退化（8.07 → 13.75）是反向证据：超参变化让 B 收到非零但方向错误的梯度，让 forward 更差。

## 已接通但无效的部分（不是 bug）

- L1 EMA 阈值替换 decay 公式：✅ skip 76% → 78%（多 skip 1 步）
- L2 buffer 写入：✅
- L3 daemon 启动 + 训练：✅ 162 次 update 都成功
- LoRA checkpoint 保存/加载链路：✅（time_conditioned 也支持）
- 这些都没 bug，但因为 loss = 0 训不出有用的 LoRA

## 修复优先级（与 project_vfl_known_gaps.md D1-D5 独立）

VFL 要真正起作用，必须从**信号源**入手，而不是改 LoRA 架构（包括 AdaLN-LoRA）：

- **P0**: 训练 input forward 必须与 target 走不同路径
  - 推荐方案：训练时 forward 用 SpecA state（带 cache_dic/current），target = event.true_feature（base 重算输出）
  - 这要求 event 序列化 SpecA state snapshot，`_run_transformer_forward` 支持加速器参数
  - 不修这个，所有 LoRA 架构改进（包括 AdaLN-LoRA）都无梯度可训
- **P0**: anchor target 改为非 self-distillation 来源
  - 选项：DDIM 反推的 x̂_0、或暂时禁用 anchor
  - anchor loss 在推理时本来就缺真实 ε，self-distillation 是数学必然，需要重新设计
- **P1**: λ_curvature 提到 1e-2 ~ 1e-1
- **P1**: EvalGate 接主流程（D2）
- **P2**: rank 提到 8 或 16（容量问题）

**Why**: 这是 2026-07-11 通过交叉读 `curvature_loss.py:345,400` + `run_dit.py:475` + 实验数据得出的判断。代码本身不显式记录「target 是 self-distillation」，需要审计才能发现。

**How to apply**:
- 任何「VFL LoRA 训不出」的调试先核对这份诊断
- 不要在没修信号源前调 LoRA 架构（rank/AdaLN/MoE 等）—— 都是浪费
- 验证修复成功的标准：100 图小实验后 B 矩阵范数 > 1e-6（说明梯度回流了）
- 与 project_vfl_known_gaps.md D1-D5 独立：那些是工程缺陷，本条是设计缺陷，两者不重叠
