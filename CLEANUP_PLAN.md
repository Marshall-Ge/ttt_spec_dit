# TTT-DiT 项目代码清理计划

> 生成时间：2026-07-03
> 当前分支：`dev1`（工作区干净）
> 协议：10 步精简协议，等待人工审阅确认后再执行删除动作

---

## 0. 执行原则

- **安全第一**：所有删除动作前已确认 Git 工作区干净，可随时回滚。
- **按模块分组提交**：A/B/C/D/E 五类分别独立 commit，便于回滚和 review。
- **不触碰红线**：
  - 不修改对外暴露的 Public API 签名（如 `main.py` 的 CLI 参数、`DiTGenerator.generate()` 的入参）。
  - 不合并语义相反的代码块。
  - 不删除领域模型字段。
  - 不使用奇技淫巧简化（如位运算替代布尔）。
- **每步验证**：删除后运行 `python -c "import main"` 做导入冒烟，并跑 `verification_feedback_loop/tests/` 下相关测试。

---

## 1. 依赖图谱概览

### 1.1 入口点

```
main.py:parse_args()
  → validate_args()
  → 分发:
    dit   → run_dit.run_c2i(args)
    pixart → run_pixart.run_t2i(args) | run_pixart.run_c2i(args)
```

旁路 CLI 入口（不在 `main.py` 分发链中）：

| 脚本 | 入口形式 | 文档化情况 |
|------|---------|----------|
| `continual_inference_runner.py` | `if __name__ == "__main__": main()` | README 第 465 行有示例 |
| `run_ttt_benchmark.py` | `if __name__ == "__main__": run_ttt_benchmark(args)` | 无 |
| `run_session2_flywheel.py` | `if __name__ == "__main__"` | 仅 `async_trainer.py` 注释提及 |
| `ttt_baseline.py` | `if __name__ == "__main__"` | 无任何引用 |
| `scripts/calibrate_teacache.py` | CLI 脚本 | CLAUDE.md 第 10 节有命令示例 |

### 1.2 跨包耦合图（生产路径）

```
run_dit.py ─┬─ models/dit.py (DiTTransformer2D)
            ├─ accelerators/ (speca_init, teacache_*)
            ├─ dataset/ (COCO30KDataset, ImageNetDataset)
            ├─ eval/ (CLIPScorer, FIDISComputer, LatencyMetric, FLOPsMetric,
            │        LPIPSScorer, MSEMetric)
            ├─ models/ttt_plugin.py (SessionAdaLNModulator, ttt_state_init, ...)
            ├─ verification_feedback_loop/ (StratifiedReplayBuffer, OnlineCalibrator,
            │                                AsyncTrainingWorker, VFLConfig,
            │                                load_lora_checkpoint, set_vfl_*, ...)
            └─ utils.py, config.py

run_pixart.py ─┬─ models/pixart.py (PixArtTransformer2D)
               ├─ accelerators/ (speca_*, teacache_*, cache_step_pixart, ...)
               ├─ dataset/ (COCO30KDataset, ImageNetDataset, DrawBenchDataset, GenEvalDataset)
               ├─ eval/ (CLIPScorer, FIDISComputer, GenEvalScorer, ImageRewardScorer,
               │        LatencyMetric, FLOPsMetric, LPIPSScorer, MSEMetric)
               ├─ verification_feedback_loop/ (set_vfl_*, record_*_event, get_vfl_*)
               └─ utils.py, config.py

models/dit.py ─┬─ accelerators/ (teacache_*, speca_*, taylor_*, cache_step_dit, ...)
               └─ verification_feedback_loop.vfl_state (get_vfl_*, record_*_event, set_vfl_sample_id)

models/pixart.py ─┬─ accelerators/ (同上但 cache_step_pixart)
                  └─ verification_feedback_loop.vfl_state (同上但无 set_vfl_sample_id)

accelerators/speca.py ──→ verification_feedback_loop.OnlineCalibrator (类型注解)
accelerators/teacache.py ──→ verification_feedback_loop.OnlineCalibrator (类型注解)

eval/latency.py ──→ models.dit.DiTGenerator (通过 models/dit.py 的 __getattr__ 延迟导入)
                  ──→ models.pixart.PixArtGenerator (直接 import)
```

### 1.3 VFL 包内部依赖

```
verification_feedback_loop/
├── __init__.py             ← 批量 re-export（当前过度导出）
├── vfl_state.py            ← 全局 hooks（set_vfl_*, get_vfl_*, record_*_event）
│                             被 models/dit.py, models/pixart.py, run_dit.py, run_pixart.py 使用
├── online_calibration.py  ← OnlineCalibrator（生产使用，被 run_dit.py、accelerators/* 引用）
├── replay_buffer.py        ← StratifiedReplayBuffer（生产使用，被 run_dit.py 引用）
├── async_trainer.py        ← AsyncTrainingWorker（生产）+ AsyncTrainer（deprecated，仅 demo 用）
├── lora_adapter.py         ← LoRA 工具集（被 async_trainer.py、run_dit.py 使用）
├── curvature_loss.py       ← compute_training_loss（生产）+ 2 个 trajectory_* 函数（死）
├── verification_hook.py    ← VerificationEvent, record_event（生产部分使用）
├── config.py               ← VFLConfig, DEFAULT_VFL_CONFIG
├── eval_gate.py            ← 整文件死代码（仅 demo_e2e.py 用）
├── version_registry.py     ← 整文件死代码（仅 demo_e2e.py 用）
├── demo_e2e.py             ← 独立 demo，仅自身使用上面两个死模块
└── tests/                  ← 7 个测试文件，全部直接 import 子模块
```

---

## 2. 待清理清单（分级）

### A. 高置信度死代码（零外部引用，可直接删除）

> **修订记录（2026-07-03）**：初版基于 Explore agent 报告，未亲自核实，存在严重误判。
> 已对每一项执行 `grep -rn` 全项目验证，结果如下：
> - A1-A5 经核实确为死代码，保留。
> - A6（`SpeedupMetric`）虽无 runner import，但属于 `eval/__init__.py` 显式导出的公共 API 且 README 提及，**移出删除清单**，仅作记录。
> - A7（`trajectory_curvature_loss`）误判 —— 被 `compute_training_loss` (line 344) 调用，alive，**移除**。
> - A8（`trajectory_curvature_loss_from_buffer`）无生产引用但属公共 API 导出，**移出删除清单**，仅作记录。
> - A9（9 个私有函数）**全部误判**，全部 alive，**整表移除**。

| # | 文件:行 | 类型 | 估行 | 证据 |
|---|--------|------|------|------|
| A1 | `config.py: COCO_NUM_PROMPTS` | 未引用常量 | 3 | 全项目 grep 零结果（已核实） |
| A2 | `config.py: default_pixart_coefficients` | 未引用函数 | ~15 | 全项目 grep 零结果（已核实） |
| A3 | `config.py: default_dit_coefficients` | 未引用函数 | ~15 | 全项目 grep 零结果（已核实） |
| A4 | `utils.py: load_real_image` | 未引用函数 | ~15 | 仅 `utils.py:77` 自身定义，无外部 import（已核实） |
| A5 | `accelerators/teacache.py: teacache_export_trace` | 未引用函数 | ~10 | 仅 `teacache.py:250` 自身定义，不在 `__init__.py`，无外部调用（已核实） |

**A 类合计：约 58 行**（仅 A1-A5）

---

### A' . 仅作记录，不删除（公共 API / 待人工决策）

| # | 文件:行 | 类型 | 状态 | 不删原因 |
|---|--------|------|------|---------|
| A'1 | `eval/latency.py: SpeedupMetric` (line 329) | 未引用类 | 保留 | `eval/__init__.py` 显式导出 + README 提及；属于对外公共 API，删除需先确认无外部调用方依赖 |
| A'2 | `verification_feedback_loop/curvature_loss.py: trajectory_curvature_loss_from_buffer` (line 118) | 未引用函数 | 保留 | `__init__.py` 显式导出的公共 API；包装 `trajectory_curvature_loss`，可能给外部调用方使用 |

> 这两项如需清理，建议改为「标记 deprecated + 文档说明」而非直接删除。

---

### B. 中置信度死模块（仅 demo / 自身使用，无生产引用）

| # | 位置 | 类型 | 行数 | 证据 |
|---|------|------|------|------|
| B1 | `verification_feedback_loop/eval_gate.py` 整文件 | 死模块 | 375 | 仅 `demo_e2e.py` 引用（4 处：line 26, 55, 250, 174/336）；无测试覆盖；无生产 runner 引用（已核实） |
| B2 | `verification_feedback_loop/version_registry.py` 整文件 | 死模块 | 401 | 仅 `demo_e2e.py` 引用（2 处：line 26, 52, 248）；无测试覆盖；无生产 runner 引用（已核实） |
| B3 | `verification_feedback_loop/demo_e2e.py` 整文件 | 死 demo | 371 | 仅 `async_trainer.py:35, 437` 注释提及，无任何 import（已核实） |

**B 类合计：约 1147 行**（3 文件）

---

### B' . `__init__.py` 精简（修订）

> **修订记录（2026-07-03）**：初版 B4 列了 13 个「仅 demo/tests 用」的 symbol，逐个 grep 核实后**严重误判** —— 11 个 symbol 实际被 `async_trainer.py` 或 `lora_adapter.py` 内部使用，是 alive 的。真正可从 `__init__.py` 移除的只有 B1/B2/D1 删除后失去 consumer 的 import。

#### B'.1 B1/B2/D1 删除后必须同步移除的 import（强制）

| `__init__.py` 行 | 移除内容 | 原因 |
|-----------------|---------|------|
| line 14-19 | 无需动 | `VerificationEvent`、`record_event`、`make_timestep_bucket`、`NUM_TIMESTEP_BUCKETS` 全部 alive（被 `vfl_state.py`、`verification_hook.py` 内部使用） |
| line 48-52 | `from verification_feedback_loop.eval_gate import EvalGate, GateStatus, GateResult` | B1 文件删除后 import 失败 |
| line 53-57 | `from verification_feedback_loop.version_registry import VersionRegistry, AdapterStatus, AdapterRecord` | B2 文件删除后 import 失败 |
| line 44-47 中 `AsyncTrainer` | `AsyncTrainer` 的 import 和 `__all__` 项 | D1 类删除后无法 import |

#### B'.2 可选精简（保守策略，**不强制**）

以下 symbol 在 `__init__.py` 顶层导出但生产代码（`async_trainer.py`）通过子模块路径 `from verification_feedback_loop.lora_adapter import ...` 直接访问，不依赖顶层导出。从「公共 API 收窄」角度可移除，但**不属于死代码**，建议**保留**以维持 API 兼容性：

| Symbol | 状态 | 生产调用方 |
|--------|------|----------|
| `LoRALinear` | alive | `lora_adapter.py` 内部类型注解 |
| `attach_lora_all_layers` | alive | `async_trainer.py:270`（通过子模块 import） |
| `detach_lora` | alive | `async_trainer.py:625`（通过子模块 import） |
| `get_lora_params` | alive | `async_trainer.py:346, 372, 521, 537`（通过子模块 import） |
| `freeze_backbone` | alive | `async_trainer.py:275, 602, 613`（通过子模块 import） |
| `select_top_k_layers` | alive | `async_trainer.py:592`（通过子模块 import） |
| `AnchorSample` | alive | `replay_buffer.py` 内部大量使用 |
| `trajectory_curvature_loss` | alive | `curvature_loss.py:344` 内部调用（A7 已修订） |
| `trajectory_curvature_loss_from_buffer` | 见 A'2 | 仅 `__init__.py` 导出，待人工决策 |

> **结论**：B' 类实际强制改动只有 B'.1 表中的 3 处 import 移除（约 6 行），不是「精简 __init__.py」的大动作。

---

### C. 待确认的孤立脚本（无生产 import 路径）

| # | 位置 | 行数 | 状态（已核实） | 建议 | 等待确认 |
|---|------|------|--------------|------|---------|
| C1 | `ttt_baseline.py` | 439 | 全项目 grep 零引用，README/CLAUDE/scripts 均未提及 | 强烈疑似废弃实验，建议删 | ❓ |
| C2 | `run_session2_flywheel.py` | 202 | 仅 `async_trainer.py:36, 437` 注释提及作为 deprecated `AsyncTrainer` 的 consumer；无任何 import | 若废弃，可连带清理 D1 | ❓ |
| C3 | `run_ttt_benchmark.py` | 366 | 全项目 grep 零引用，README/CLAUDE/scripts 均未提及 | 待人工确认是否还在跑 | ❓ |
| C4 | `continual_inference_runner.py` | 394 | README 第 465 行文档化 + `models/ttt_plugin.py:167` 注释提及 | **保留**，不删 | ✅ |

**C 类待确认删除：约 1007 行**（C1+C2+C3）

---

### D. 已废弃类（需先迁移 consumer）

| # | 位置 | 行数 | 证据（已核实） |
|---|------|------|--------------|
| D1 | `verification_feedback_loop/async_trainer.py: AsyncTrainer`（line 445） | ~200 | 代码注释标记 `deprecated`；仅 `demo_e2e.py:138, 322` 和 `run_session2_flywheel.py:64` 使用；`AsyncTrainingWorker`（line 75）是替代品，已被 `run_dit.py` 采用 |

**前置条件**：D1 删除需先确认 C2（`run_session2_flywheel.py`）和 B3（`demo_e2e.py`）都删除。

**D 类合计：约 200 行**

---

### E. 冗余逻辑简化（修订）

> **修订记录（2026-07-03）**：初版 E1 主张「收窄 `__all__` 到 14 个 symbol」，但 B4 核实后发现绝大多数 export 都是 alive 的，强行收窄属于改公共 API，违反红线。改为保守策略：仅清理 B1/B2/D1 删除后失去 consumer 的 import。

| # | 位置 | 操作 | 行数 |
|---|------|------|------|
| E1 | `verification_feedback_loop/__init__.py` | 移除 B'.1 表中 3 处失效 import（`eval_gate`、`version_registry`、`AsyncTrainer`） | ~6 |
| E2 | 全项目 | A 类删除后跑 `pyflakes` 清理未引用 import | ~10 |

#### E1 后 `__all__` 实际改动

仅移除以下 7 项：
```python
# 从 __all__ 移除：
"EvalGate", "GateStatus", "GateResult",       # B1 删除
"VersionRegistry", "AdapterStatus", "AdapterRecord",  # B2 删除
"AsyncTrainer",                                # D1 删除
```

其余 27 项 export 全部保留（已核实均为 alive）。

---

## 3. 总计影响预估（修订）

| 类别 | 删除行数（估） | 删除文件数 | 风险等级 | 备注 |
|------|-------------|----------|---------|------|
| A（高置信度死代码，A1-A5） | ~58 | 0 | 低 | 仅删 5 个未引用函数/常量 |
| A'（仅记录，不删） | 0 | 0 | — | `SpeedupMetric`、`trajectory_curvature_loss_from_buffer` 公共 API 保留 |
| B1+B2+B3（死模块 + 死 demo） | ~1147 | 3 | 中 | 需确认 demo 不再需要 |
| C1+C2+C3（孤立脚本，待确认） | ~1007 | 3 | 中 | 需人工确认 |
| D1（废弃 AsyncTrainer，依赖 B3+C2） | ~200 | 0 | 低 | 前置满足后 |
| E1+E2（import 清理） | ~16 | 0 | 低 | B1/B2/D1 删除后强制清理 |
| **合计（A+B+D+E，不含 C）** | **~1421** | **3** | — | 推荐执行 |
| **合计（含 C 全删）** | **~2428** | **6** | — | 全量执行 |

> **与初版差异**：初版估计 ~2649 行/6 文件，主要差异来自 A9（9 个误判私有函数 ~80 行）、A7（误判 ~80 行）、A6（移出 ~30 行）、A8（移出 ~50 行）、B4（绝大多数 export 实为 alive）。修订后更保守、更准确。

---

## 4. 执行计划（按模块分组提交）

每个步骤独立 commit，便于 review 和回滚。提交信息格式：`refactor(cleanup): <模块> <动作>`。

### Step 1 — A 类删除（高置信度死代码，5 项）

**操作**（仅删 5 项，已逐项 grep 核实）：
- 删除 `config.py` 中 `COCO_NUM_PROMPTS`、`default_pixart_coefficients`、`default_dit_coefficients`
- 删除 `utils.py` 中 `load_real_image`
- 删除 `accelerators/teacache.py` 中 `teacache_export_trace`

**不删**（A' 类，公共 API 保留）：`SpeedupMetric`、`trajectory_curvature_loss`、`trajectory_curvature_loss_from_buffer`、所有 A9 私有函数。

**验证**：
```bash
python -c "import main; from config import *; import utils; import accelerators; import eval; from verification_feedback_loop import curvature_loss"
python -m pyflakes config.py utils.py accelerators/teacache.py
```

**Commit**：`refactor(cleanup): remove 5 dead symbols (config/utils/teacache)`

---

### Step 2 — B 类删除（demo + 死模块）

**前置确认**：B3（`demo_e2e.py`）是否还需要保留作为演示？

**操作**：
- 删除 `verification_feedback_loop/eval_gate.py`
- 删除 `verification_feedback_loop/version_registry.py`
- 删除 `verification_feedback_loop/demo_e2e.py`

**验证**：
```bash
python -c "import verification_feedback_loop"
python -m pytest verification_feedback_loop/tests/ -x --tb=short
```

**Commit**：`refactor(cleanup): remove dead VFL modules (eval_gate, version_registry, demo_e2e)`

---

### Step 3 — C 类删除（孤立脚本，待人工确认）

**前置确认**：C1/C2/C3 是否可删？

**操作**（按确认结果执行）：
- 若 C1 确认可删：删除 `ttt_baseline.py`
- 若 C2 确认可删：删除 `run_session2_flywheel.py`
- 若 C3 确认可删：删除 `run_ttt_benchmark.py`

**验证**：
```bash
python -c "import main"  # 确认主入口未受影响
```

**Commit**：`refactor(cleanup): remove orphan standalone scripts`

---

### Step 4 — D 类删除（废弃 AsyncTrainer）

**前置**：Step 2（B3）和 Step 3（C2）均已执行。

**操作**：
- 删除 `verification_feedback_loop/async_trainer.py` 中 `AsyncTrainer` 类（line 445 起，约 200 行）
- 删除 `verification_feedback_loop/__init__.py` 中 `AsyncTrainer` 的 import 和 `__all__` 项

**验证**：
```bash
python -c "from verification_feedback_loop import AsyncTrainingWorker"
python -m pytest verification_feedback_loop/tests/ -x --tb=short
```

**Commit**：`refactor(cleanup): remove deprecated AsyncTrainer class`

---

### Step 5 — E1 精简（`__init__.py` 失效 import 清理）

**操作**（仅强制 3 处，不收窄其他 export）：
- 移除 `verification_feedback_loop/__init__.py` 中：
  - `from verification_feedback_loop.eval_gate import EvalGate, GateStatus, GateResult`（line 48-52）
  - `from verification_feedback_loop.version_registry import VersionRegistry, AdapterStatus, AdapterRecord`（line 53-57）
  - `AsyncTrainer` 的 import 和 `__all__` 项（line 45 + line 97）
- 其余 27 项 export 全部保留

**验证**：
```bash
python -c "import verification_feedback_loop; print(verification_feedback_loop.__all__)"
python -m pytest verification_feedback_loop/tests/ -x --tb=short
```

**Commit**：`refactor(cleanup): remove broken VFL __init__ imports after module deletion`

---

### Step 6 — E2 收尾（pyflakes 全量扫描）

**操作**：
- 跑 `pyflakes` 全项目，清理 A/D 删除后产生的未引用 import
- 复查无遗漏

**验证**：
```bash
python -m pyflakes *.py accelerators/ models/ eval/ dataset/ verification_feedback_loop/*.py
python -m pytest verification_feedback_loop/tests/ -x --tb=short
```

**Commit**：`refactor(cleanup): purge unused imports post-cleanup`

---

## 5. 验证与修复策略

### 5.1 冒烟测试矩阵

每步执行后必跑：

| 测试 | 命令 | 预期 |
|------|------|------|
| 主入口导入 | `python -c "import main"` | 无报错 |
| VFL 包导入 | `python -c "import verification_feedback_loop"` | 无报错 |
| 加速器导入 | `python -c "import accelerators; from accelerators import speca_init, teacache_init"` | 无报错 |
| 模型导入 | `python -c "from models import DiTTransformer2D, PixArtTransformer2D"` | 无报错 |
| eval 导入 | `python -c "import eval; from eval import FIDISComputer, FLOPsMetric, LatencyMetric"` | 无报错 |
| VFL 测试套件 | `python -m pytest verification_feedback_loop/tests/ -x --tb=short` | 全绿 |

### 5.2 端到端冒烟（最终步骤后）

```bash
# DiT baseline 最小冒烟
python main.py --model dit --task c2i --dataset imagenet \
    --method baseline --metrics fid is latency flops speed \
    --seed 42 --num_steps 5 --n_prompts 4 --batch_size 4

# DiT teacache 最小冒烟
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --thresh 0.25 --metrics fid is latency flops speed \
    --seed 42 --num_steps 5 --n_prompts 4 --batch_size 4

# PixArt baseline 最小冒烟
python main.py --model pixart --task t2i --dataset drawbench \
    --method baseline --metrics imagereward latency flops speed \
    --seed 42 --num_steps 5 --n_prompts 2 --batch_size 2
```

> **注**：端到端冒烟需要 GPU + 模型权重，可能无法在本地完整跑通。若环境受限，至少跑导入冒烟和单元测试。

### 5.3 失败回滚

任一步骤测试失败：
1. 立即停止后续步骤
2. `git diff` 检查改动
3. 修复后重新跑该步骤验证
4. 若无法快速修复，`git reset --hard HEAD~1` 回滚到上一步 commit

---

## 6. 红线复查

| 红线 | 本计划是否触碰 |
|------|--------------|
| 删除领域模型字段 | ❌ 不触碰（不动 `dataset/`、`models/` 的数据字段） |
| 合并语义相反代码块 | ❌ 不触碰 |
| 修改对外 Public API 签名 | ❌ 不触碰（不动 `main.py` CLI、不动 `DiTGenerator.generate()` 签名；VFL `__init__.py` 仅移除 B1/B2/D1 删除后失效的 import，不主动收窄其他 export） |
| 奇技淫巧简化 | ❌ 不触碰 |
| 删除 CLAUDE.md 标注的已验证组合（第 9 节）相关代码 | ❌ 不触碰（baseline/teacache/speca/ddim 主路径全部保留） |
| **新增**：误判删除 alive 私有函数 | ❌ 不触碰（A9 表 9 个私有函数全部 alive，已从清单移除） |
| **新增**：误判删除 alive 公共函数 | ❌ 不触碰（A7 `trajectory_curvature_loss` alive，已移除；A6/A8 移到 A' 仅记录） |

---

## 7. 待人工确认清单（修订）

请逐项回复 ✅ 同意 / ❌ 保留 / ⚠️ 修改：

### A 类（~58 行死代码，已逐项 grep 核实）
- [ ] **A1** `config.py: COCO_NUM_PROMPTS` 删除？
- [ ] **A2** `config.py: default_pixart_coefficients` 删除？
- [ ] **A3** `config.py: default_dit_coefficients` 删除？
- [ ] **A4** `utils.py: load_real_image` 删除？
- [ ] **A5** `accelerators/teacache.py: teacache_export_trace` 删除？

### B 类（死模块，~1147 行，3 文件）
- [ ] **B1** `verification_feedback_loop/eval_gate.py` 删除？
- [ ] **B2** `verification_feedback_loop/version_registry.py` 删除？
- [ ] **B3** `verification_feedback_loop/demo_e2e.py` 删除？

### C 类（孤立脚本，~1007 行，3 文件，待人工确认）
- [ ] **C1** `ttt_baseline.py` 删除？
- [ ] **C2** `run_session2_flywheel.py` 删除？
- [ ] **C3** `run_ttt_benchmark.py` 删除？
- [ ] C4 `continual_inference_runner.py` 保留（已确认）

### D 类（废弃类，~200 行，依赖 B3+C2）
- [ ] **D1** `AsyncTrainer` 删除（仅当 B3+C2 都同意删除时）？

### E 类（强制 import 清理，~16 行）
- [ ] **E1** 移除 `__init__.py` 中 B1/B2/D1 失效 import？（强制，无选项）
- [ ] **E2** pyflakes 全量扫描清理？

### A' 类（仅记录，不删）
- [ ] A'1 `SpeedupMetric` 保留？（建议保留，公共 API）
- [ ] A'2 `trajectory_curvature_loss_from_buffer` 保留？（建议保留，公共 API）

### 执行方式
- [ ] 按 Step 1-6 分 6 次 commit，OK？

---

## 8. 修订日志

| 时间 | 修订内容 | 原因 |
|------|---------|------|
| 2026-07-03 初版 | 基于 Explore agent 报告生成 | — |
| 2026-07-03 修订 1 | A9 表 9 个私有函数全部移除（误判）、A7 移除（误判）、A6/A8 移到 A' 仅记录 | 用户指出 `_profile_tail_flops` 实际有调用，逐项 grep 核实后发现 Explore agent 报告多处错误 |
| 2026-07-03 修订 2 | B4 表 13 个 symbol 中 11 个误判为「仅 demo/tests 用」，实际被 `async_trainer.py` 内部使用；改为 B'.1 强制清理 + B'.2 保守保留 | 逐项 grep 核实每个 export 的真实引用 |
| 2026-07-03 修订 3 | 总计从 ~2649 行修订为 ~2428 行（含 C）/ ~1421 行（不含 C） | A/B/E 行数纠正 |

确认后我按顺序执行。
