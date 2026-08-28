---
name: VFL 已知实现缺陷清单
description: VFL (Verification Feedback Loop) 在主流程中的未修复缺陷，2026-07-10 审计得出；非紧急，但触碰 VFL 时需先看这份清单而非盲信 CLAUDE.md
type: project
originSessionId: 7d3813aa-6029-49d6-9c28-650c01470017
---
VFL 各层在 DiT 主流程"基本接通"，但以下缺陷截至 2026-07-10 未修复，标记为非紧急：

**D1 (P0 但非紧急): PixArt + VFL 完全没接通**
- `run_pixart.py` 只调 `set_vfl_step_info`（5 处），没有 `set_vfl_buffer` / `set_vfl_calibrator` / `AsyncTrainingWorker` 任何初始化
- `models/pixart.py:64-132` 写好的 `_vfl_record_*` hook 调用永远走 no-op（buffer=None, calibrator=None）
- CLAUDE.md §9 表格写 "PixArt + speca + VFL ✅" 与 "PixArt + speca + VFL 完整三层 ✅" 是**误导性描述**，实际是 ❌

**D2 (P0 但非紧急): EvalGate canary 闸门未接主流程**
- `EvalGate.evaluate()` 仅在 `feedback.vfl/demo_e2e.py` 中被调用
- `run_dit.py` 主流程：AsyncTrainingWorker 训完直接 `save_lora_checkpoint`，下一轮通过 `find_latest_checkpoint`（按 mtime）直接加载，无质量校验
- 后果：LoRA 越训越差也会被自动加载

**D3 (P1): VersionRegistry 未接主流程**
- `run_dit.py:752` `set_vfl_buffer(buf, model_version="dit-v1")` 硬编码字符串
- base 模型权重切换后，旧 LoRA checkpoint 仍会被 find_latest_checkpoint 加载，无版本隔离
- `OnlineCalibrator.on_base_model_swap` 也从未触发

**D4 (P1): L1 阈值被 `max(online, default)` 砍掉一半方向**
- `online_calibration.py:215` 默认保守模式：L1 阈值永远 ≥ SpecA 静态阈值
- L1 只能纠偏"过激进"方向（让加速器更保守），无法让加速器更激进
- 只有 `set_exploit_mode(True)` 才解锁双向，但主流程从不调（只有 `run_session2_flywheel.py` 调）

**D5 (P2): L1 EMA 矩阵 28×3 大部分是死空间**
- SpecA 只在 check_layer（DiT=20）做误差探测；TeaCache 只在 _VFL_PROBE_LAYER 单点写
- 实际只有 1 layer × 3 bucket = 3 条 EMA 真正参与决策
- 不是 bug（SpecA 本来就单点探测），但与 CLAUDE.md §12.2 "per-(layer_id, bucket) EMA" 描述存在落差

**已接通的部分（确认正常工作）**：
- L1 EMA 写入 (`record_speca_event` / `record_teacache_event`) → calibrator.update
- L1 EMA 查询 (`speca_cal_type:406-423`, `teacache_decide:166-170`) 真传入 calibrator
- L2 buffer reservoir + anchor（每 5 步采，限 50）+ ehs 阀
- L3 AsyncTrainingWorker daemon 线程 + lazy deepcopy + LoRA attach 全 28 层

**Why**: 这些是 2026-07-10 通过交叉读 `run_dit.py` / `run_pixart.py` / `online_calibration.py` / `eval_gate.py` / `version_registry.py` + grep 主流程调用点得出的判断。代码本身不显式记录"我没接 EvalGate"或"CLAUDE.md 描述与实际不符"——需要审计才能发现。

**How to apply**:
- 任何"VFL 已经验证可用"的假设都要先核对这份清单
- 改 VFL 相关代码前先 `grep EvalGate|VersionRegistry|set_vfl_buffer run_dit.py run_pixart.py` 确认当前接通情况（这份清单可能已过时）
- 优先修复顺序：D2 → D1 → D3 → D4 → D5（D2 改动最小、收益最大）
- 触碰 `models/pixart.py` 的 VFL hook 时，要么补完 D1，要么删掉死代码 hook
