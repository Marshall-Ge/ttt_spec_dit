---
name: project-covr-crossover-gate-design
description: analyze_crossover.py 的判据为何必须是「外生特征 + 样本外迁移」，以及三个 permutation null 全部不可用的原因
metadata: 
  node_type: memory
  type: project
  originSessionId: 8e47eb08-328e-4617-9d27-3b55e741de17
  modified: 2026-07-31T15:43:08.207Z
---

COVR per-image crossover 的判据只能是「外生特征 + 样本外迁移」，in-sample oracle gap 和三个 permutation null 都不能当门禁。相关：[[project-covr-offline-probe-findings]] [[project-covr-system]]。

**Why:**
- `oracle - best_single`（每图取 min）会把 iid 测量噪声变成正 benefit：构造数据上纯噪声给出的 gap（占最优 arm loss 的 24%）比真实 difficulty-driven crossover（18%）还大。这个统计量本身没有零假设意义。
- null A（每图内 shuffle arm label）结构退化：`sort(D, axis=1)` 不变 → oracle 项恒定不动。
- null B（各 arm 列独立置换）的 p 只反映跨 arm 耦合的符号，不反映可利用性：正耦合（共享图片难度）→ null mean 远高于观测、p≈1（即使 crossover 真实，实测 obs +0.098 vs null mean +0.30）；反相关 → p≈0。
- null R（剥掉两个 main effect 后置换残差）居中但两个方向都会错：真实交互上 p=0.47（漏检），纯噪声上 p=0.065（近似误报）。
- **最关键的坑**：OOS 迁移特征若取自 `D` 本身（如 train best arm 的 per-image 距离当"难度"），会泄漏 —— 同一份噪声实现同时决定特征值、哪个 arm 赢、以及 test half 的成本。实测纯 iid 噪声误报率 n=40:25% / n=100:61% / n=250:93% / **n=500:99%**，正好在这个 sweep 自己的 N 上几乎必然误报。

**How to apply:**
- 迁移特征只能来自 reference 图像本身（`_image_features`: 梯度能量 / 对比度 / 亮度），绝不碰 `D`。改这段代码时先问「这个特征见过 arm 输出吗」。
- 换判据前必须做校准扫描：纯噪声 / 真实 crossover / 纯 difficulty main effect / arm-specific 噪声尺度，四种构造下测 reject 率。当前实现：噪声 4-6%（名义 5%），真实 crossover 97.5-100% power，两种对抗构造 0-7.5%。
- 即使 OOS 通过，结论也只是「per-image 结构存在且可从图像内容预测」。reference 特征是 oracle-side（需要 full-compute 输出），上 bandit 前还得单独验证一个 online 可见信号（early-step latent 统计 / class embedding / 第一个 calc step 的 TeaCache raw_diff）能复现同样的迁移。
