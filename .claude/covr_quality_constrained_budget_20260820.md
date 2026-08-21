# P1 设计规格：质量约束的在线预算校准（inception-confidence session 闭环）

> 日期：2026-08-20（工具链更新：2026-08-21）
> 状态：**SPEC — 离线 gate 待跑**（数据在 GPU 机，本机不可达；
> 四个脚本已全部就绪并通过本地合成数据冒烟测试，机器恢复即可按 §5 执行）
> 上游事实：inception_conf 是唯一通过过 [a] validity gate 的信号
> （k8/k6 内 Spearman(mean_conf, FID)=1.000，Spearman(mean_conf, −IS)=1.000，
> 见 `memory/project_covr_budget_probe_500_results.md`）。被否决的只是
> per-image crossover 回收（天花板贴噪声地板），session 级使用从未测过。

## 1. 问题重定义

不再问"固定预算下打败最优静态 arm"，改问：

> **在质量非劣约束下最小化期望计算。** 静态 uniform K=8 是"什么都不做"的
> null；成功标准 = 同质量（FID/IS 非劣）下净 FLOPs 下降。

反馈信号：每张生成图的 InceptionV3 预测置信度（reference-free，推理侧
只需一次 InceptionV3 前向，成本 ~DiT 一步的 1/200）。决策粒度：
**session/epoch 级**（每 B 张图一次预算决策），不是 per-image（已证伪）
也不是 batch 级（已证伪）。

## 2. 控制器（v1 规格，尽量简）

- 预算档位 k ∈ {8, 6, 4}（TeaCache forced uniform mask，每档内的 placement
  用该档静态最优布局；k6 及以下 uniform 不再默认最优，用 budget probe 实测
  best-arm 布局：k6 = back_loaded）。
- 每 B 张图（建议 B=100，500 图 session → 5 个决策点）：
  - **降档（aggression）**：连续 S=2 个 epoch 无警报，且 μ̂ 的 95% 单侧下界
    ≥ target − ε → k 下调 2。target 来自启动时在 k8 上的校准 epoch。
  - **升档（safety）**：对标准化 −conf 跑 CUSUM（δ=0.5σ, h=5），警报 →
    k 上调 2（或回 k8）。
- 误报控制：CUSUM 参数由离线 power analysis 标定（gate 2），不拍脑袋。

## 3. 离线验证（不生成任何新图，只复用已有 PNG）

已有资产：budget probe 12 arm × 500 图（k8/k6/k4 × uniform/geometric/
back/front）+ K8 static-mask run 6 arm × 500 图（含 2 random null），
全部在 GPU 机 `output/` 下。

- **Gate [P1-a] 信号加强**（先做，最便宜）：对 K8 static-mask 6 arm 重算
  per-image conf，验证 6-arm 排序 Spearman（把 [a] 从 4 点加强到 6 点；
  两个 metric 都要）。失败 → 整条线停。
  - 命令：`python3 scripts/analyze_conf_rank_validity.py <conf_table.csv>
    --results-root <run_root> --arm-regex <gate子集>`（脚本自动从各 arm 的
    results.json 展开 aggregate.fid/is_mean，arm 标签按 run 目录相对路径
    对齐，无需手工 metrics.csv；也可 `--metrics-csv` 手工提供）。
  - 裁决规则：严格 gate 要求 Spearman(entropy, FID) 与
    Spearman(entropy, −IS) 都 = +1.000（n=6，精确全排列 p_min=1/720）。
    若仅 IS 在**彼此相差 < noise floor (~2.4 FID / ~0.5 IS) 的 arm 对内**
    反转而 FID 排序完好，记 PASS-FID-only 并写入结论——控制器的用途是
    检测大退化，不是在近邻 arm 间排序；除此之外的任何反转都是 FAIL。
    （脚本已实现三态裁决 PASS / PASS-FID-ONLY / FAIL，floors 可用
    `--fid-floor/--is-floor` 调整。）
- **Gate [P1-b] 检测力**：`simulate_conf_budget_controller.py <conf> power`：
  per-(budget,arm) conf 分布的分离度（summary 模式打 Cohen's d）；
  bootstrap epoch 流上 CUSUM 的检测延迟（换档后几个 epoch 报警）与
  平稳段误报率。判据：k8→k6 退化 ≤2 epoch 检出（脚本操作化为 ≥90% 的
  trial 内检出，`--detect-frac` 可调）、平稳 k8 每 epoch 误报 ≤5%
  （h 由脚本自动标定取最小合格值）。gate arm 自动选取"参考档下一档中
  mean conf 最优的 arm"（即控制器实际会降入的 arm），`--gate-arm` 可覆写。
  失败 → 停，不调参挽救。
- **Gate [P1-c] 闭环模拟 + 混合 FID**：`simulate` 模式跑控制器轨迹（消费
  bootstrap 流）并打印 FLOPs leg 判决，`--mixture-out plan.json` 输出
  arm 混合比例；`scripts/compute_mixture_fid.py` 按 plan **从现有 PNG 里
  按比例抽图拼成混合流**（每个共享 global_idx 恰好分配到一个 arm，
  largest-remainder + seeded shuffle，`--replicates` 重抽暴露分配噪声），
  用 torch-fidelity 算混合 FID/IS（同 real_299 reference；预处理与
  eval/fid_is.py 逐位一致，判决锚定同管线重算的 reference arm）。
  判据：混合 FID ≤ k8-uniform FID + 2.4（同臂 replica floor），IS 非劣
  （≥ ref − 0.5），每个 replicate 都须满足；且期望净 FLOPs/图 < k8 − 15%
  （留出 InceptionV3 前向 + 校准 epoch 的摊销开销）。失败 → 停。
- 三 gate 全过才做 **GPU 闭环真跑**（2 个 session，配对 5 latent offsets +
  exact sign test，对照静态 uniform k8，沿用既有配对纪律）。

## 4. 已知风险（写进结论时要对照）

1. **[a] 证据基数小**（每 budget 4 arm）：Gate [P1-a] 就是为此设的。
2. **类分布漂移会动 conf 基线**：c2i 下类别已知，可用类内标准化 conf
   （class-conditional z-score）做主分析；绝对阈值作敏感性分析。
3. **conf 与 IS 的关系**：budget probe 里两个 metric 都 1.000，但基数同
   为 4 arm；[P1-a/c] 同步验。
4. **k4 档 FID 178+ 已坏**：控制器若降到 k4 应被迅速打回——这正是检测力
   gate 要验的（k8→k4 是最坏情形，d 最大，最容易；k8→k6 是难点）。
5. 与 P2（分布漂移跟踪）的关系：本 gate 全在平稳分布上验证"省计算"；
   漂移下的"何时切换"是 P2 的独立问题，共用本控制器骨架。

## 5. GPU 机恢复后的执行清单（2026-08-21 定稿，全链路已本地冒烟）

五个脚本：`extract_inception_conf.py`（GPU，需 torch）→
`analyze_conf_rank_validity.py`（[P1-a]）→ `build_conf_costs.py` →
`simulate_conf_budget_controller.py`（[P1-b] power / [P1-c] FLOPs leg）→
`compute_mixture_fid.py`（GPU，[P1-c] quality leg）。

```bash
# 0) 提取 per-image conf（两次 run 的全部 arm PNG；只读，不生成）
python3 scripts/extract_inception_conf.py output/<k8_static_run> conf_k8static.csv
python3 scripts/extract_inception_conf.py output/covr_budget_probe conf_probe.csv

# 1) Gate [P1-a] —— 6 mask arm（offset-0），regex 排除 rep_/reference/threshold
python3 scripts/analyze_conf_rank_validity.py conf_k8static.csv \
    --results-root output/<k8_static_run> \
    --arm-regex 'arm_pattern_[a-z0-9_]+_k8$'
#    （增强版可免费加做：对 rep_1..rep_4 各 offset 分别跑同一 gate）

# 2) costs（k6 用实测 best-arm back_loaded，不是 uniform）+ Gate [P1-b]
python3 scripts/build_conf_costs.py output/covr_budget_probe \
    --arm 8=k8/equalflops/arm_pattern_uniform \
    --arm 6=k6/equalflops/arm_pattern_back_loaded \
    --arm 4=k4/equalflops/arm_pattern_uniform -o costs.json
python3 scripts/simulate_conf_budget_controller.py conf_probe.csv summary \
    --headroom-arm k6/equalflops/arm_pattern_back_loaded   # Cohen's d 分离度
python3 scripts/simulate_conf_budget_controller.py conf_probe.csv power
#    （power 自动标定 h*，打印 GATE [P1-b] 判决行）

# 3) Gate [P1-c]：闭环模拟（FLOPs leg）→ 混合 FID（quality leg）
python3 scripts/simulate_conf_budget_controller.py conf_probe.csv simulate \
    --costs costs.json --h <power选出的h*> --mixture-out plan.json
python3 scripts/compute_mixture_fid.py plan.json \
    --run-root output/covr_budget_probe --workdir /tmp/p1c_mixture
#    （--real-dir 默认取 reference arm 的 real_299；两个 GATE 行都要 PASS）
```

工具链事实（2026-08-21 本地验证）：conf 提取与 [a] 通过时的定义逐位一致
（analyze_crossover 的 logits_unbiased 熵）；compute_mixture_fid 的 299
预处理与 eval/fid_is.py add() 逐位一致（PNG 无损重载 + 同一 uint8 →
BICUBIC 链路），故混合 FID 与各 run results.json 直接可比；simulate 修复了
降档后 z_hist 未清空导致的级联降档 bug（回归测试：三档同分布 →
occupancy {8:2, 6:2, 4:6}）。
