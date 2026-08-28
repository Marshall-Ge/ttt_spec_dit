# 项目结构规范(强制)

> 本文件是 TTT-DiT 项目**代码结构的总规定**。任何 agent / 开发者在本仓库新增、修改、移动代码时,必须先阅读本文件并遵守其中的分层、命名与依赖规则。违反本规范的改动不应被合入。
>
> 状态:2026-08-27 定稿(配合 dev3 分支重构)。

---

## 1. 总原则

1. **单向依赖**:`scripts → pipelines/eval/dataset/accelerators/models/utils`,`pipelines → models/accelerators/eval/dataset/utils`,禁止反向依赖(模型不得 import 编排器;accelerators 不得 import scripts)。
2. **薄入口、厚模块**:`main.py` 只做参数解析与分发,不做业务逻辑;编排逻辑在 `pipelines/`。
3. **一文件一职责**:单个 .py 文件不超过 ~600 行;超过必须拆。巨型函数(>300 行)必须拆成具名辅助函数。
4. **纯函数与副作用分离**:可复用的纯计算(序列化、统计、时间换算、mask 构造)放 `utils/`;有状态/有副作用的逻辑(采样循环、指标累积、后台线程)放对应模块。
5. **集成点显式化**:COVR / VFL / TTT 等扩展对采样循环的注入,必须通过 `pipelines/hooks/` 中显式的钩子函数完成,禁止把扩展逻辑散落在编排器函数体内。
6. **scripts 只做"编排实验"**:scripts 下的脚本负责组装参数、调用库代码、读写结果文件;算法逻辑必须位于库代码(accelerators / models / pipelines / utils)中,scripts 内不得出现核心算法实现。

---

## 2. 目录结构与职责

```
ttt_spec_dit/
├── main.py                    # CLI 入口(薄):parse_args + validate_args + dispatch
├── config.py                  # 全局路径与默认超参(唯一配置源)
├── pipelines/                 # ★ 编排层(生成器 + 采样循环 + 扩展钩子)
│   ├── __init__.py            #   导出 DiTGenerator, PixArtGenerator
│   ├── base.py                #   Generator 基类:VAE/scheduler/device/dtype/编码管理
│   ├── dit.py                 #   DiTGenerator(类条件)+ DiT 采样循环
│   ├── pixart.py              #   PixArtGenerator(t2i/c2i)+ 采样循环
│   └── hooks/                 #   扩展集成点(COVR/VFL/TTT 的循环钩子)
│       ├── __init__.py
│       ├── covr_hook.py       #   COVR runtime 生命周期(轨迹 begin/end、sentinel、reward)
│       ├── vfl_hook.py        #   VFL 事件记录 / 校准器更新
│       └── ttt_hook.py        #   TTT 插件训练 / skip 计数
├── models/                    # 模型定义(显式 forward,无 monkeypatch)
│   ├── dit.py                 #   DiTTransformer2D(SpecA/TeaCache/TTT 分支)
│   ├── pixart.py              #   PixArtTransformer2D(SpecA/TeaCache 分支)
│   └── ttt_plugin.py          #   SessionAdaLNModulator
├── accelerators/              # 加速器(纯函数 + plain 状态,由调用方持有)
│   ├── speca.py               #   SpecA:SpecACache/SpecAState + Taylor 工具
│   ├── teacache.py            #   TeaCache:状态机 + 决策 + 残差应用
│   ├── compute_controller.py  #   计算决策抽象(ProbeCorrect 等)
│   ├── registry.py            #   加速器适配器注册表(COVR 的 init/reward/flops 三职责)
│   ├── strategy_dispatch.py   #   策略 → 加速器状态 分发
│   ├── covr*.py               #   COVR runtime / bandit / viability
│   └── timestep_feedback.py   #   session 级 per-timestep 缺陷学习
├── feedback/                  # ★ 在线学习子系统
│   ├── vfl/                   #   Verification Feedback Loop(L1 校准 / L2 缓冲 / L3 LoRA)
│   └── ...                    #   未来新增在线学习方向(如 TTT 训练器)放这里
├── eval/                      # 指标(不动):fid_is / latency / clip / lpips / mse ...
├── dataset/                   # 数据集(不动):imagenet / coco / drawbench / geneval
├── utils/                     # ★ 通用工具包(纯函数)
│   ├── __init__.py            #   统一导出
│   ├── io.py                  #   图像 I/O、tensor↔PIL、VAE decode
│   ├── timing.py              #   CudaTimer、GenerationProfiler、_record_profile_stage
│   ├── serialization.py       #   _clean(JSON 安全化)、_covr_canonical_json、hash/sentinel 纯函数
│   └── common.py              #   其他无归属的纯函数
├── scripts/                   # ★ 脚本按用途分子目录(只做编排,不做算法)
│   ├── calibrate/             #   标定:teacache 系数、per-class γ
│   ├── diagnose/              #   诊断:误差-距离曲线、预测器族谱、token 长尾、类偏移探针
│   ├── experiment/            #   实验运行:时间桶调度、残差变体、COVR smoke 等
│   ├── analyze/               #   分析:COVR 结果、静态 mask、timestep feedback 等
│   └── smoke/                 #   冒烟/检查:forced/bandit/resume smoke + 检查脚本
├── tests/                     # 单元测试(与库代码一一对应)
├── docs/                      # 文档(报告、方法总结、诊断报告)
├── experiments/               # 实验数据(gitignore,不入库)
└── .claude/                   # agent 工作区
    ├── project-structure.md   #   本文件(结构规范)
    ├── AGENTS.md              #   agent 工作规范(引用本文件)
    └── memory/                #   实验结论 memory(唯一权威来源)
```

---

## 3. 命名与文件放置规则

| 类别 | 规则 | 反例(禁止) |
|------|------|------------|
| 编排器 | `pipelines/<model>.py`,类名 `<Model>Generator` | run_dit.py 里堆 3000 行 |
| 加速器 | `accelerators/<name>.py`,纯函数 + 显式状态 | 加速器内部 import run_dit |
| 模型 | `models/<name>.py`,forward 显式分支 | 模型 import eval/scripts |
| 钩子 | `pipelines/hooks/<ext>_hook.py`,函数签名 `hook_*(generator, state, step_info)` | 在采样循环体内写扩展逻辑 |
| 工具 | `utils/<domain>.py`,纯函数 | 编排器私有 helper 重复造轮子 |
| 脚本 | `scripts/<category>/<name>.py`,`if __name__ == "__main__"` 可运行 | scripts 根目录堆 40 个平铺脚本 |
| 指标 | `eval/<name>.py`,实现 `Metric` ABC | 指标逻辑写在 run_* 里 |
| 测试 | `tests/test_<module>.py` | 测试 import scripts 下的脚本 |

## 4. 依赖方向(禁止违反)

```
main.py
  └─ pipelines/{dit,pixart}.py
        ├─ models/*            (显式 forward)
        ├─ accelerators/*      (纯函数状态机)
        ├─ pipelines/hooks/*   (COVR/VFL/TTT 集成)
        ├─ eval/* , dataset/*  (指标与数据)
        └─ utils/*
scripts/* → 上述任意库层(但库层不得 import scripts)
utils/    → 不依赖任何项目内模块(最底层)
```

## 5. 新增功能的流程(agent 必须遵守)

1. **判断归属层**:新功能是 算法(→accelerators/models)/ 编排(→pipelines)/ 指标(→eval)/ 工具(→utils)/ 脚本(→scripts/<category>)?
2. **查重**:先 `grep` 确认没有现成实现;纯计算逻辑必须放 `utils/` 而不是复制。
3. **集成走钩子**:若需在采样循环中注入行为,在 `pipelines/hooks/` 新增/扩展钩子,并在 `pipelines/<model>.py` 的循环中**只调用钩子**,不写实现。
4. **脚本分类**:新增脚本放入 `scripts/<category>/`,category 不存在时新建并在本文件 §2 登记。
5. **遵守依赖方向**:新代码不得引入反向 import(用 `grep -rn "import run_dit" accelerators/ models/` 等自查)。
6. **规模红线**:新文件 >600 行、新函数 >300 行时必须拆分;拆出的纯函数进 `utils/`。
7. **验证**:`python -c "import main"` 冒烟 + 新增/更新 `tests/`。

## 6. 已知待迁移项(渐进,不阻塞新功能)

- `run_dit.py` / `run_pixart.py` 的历史遗留 `_covr_*` 辅助函数正在迁移至 `utils/serialization.py` 与 `pipelines/hooks/covr_hook.py`(run_dit 顶部 ~350 行已迁完的标记见文件头注释)。

- `eval/latency.py` 内嵌的 tail-profiler 复制逻辑待抽至 `utils/flops.py`。
