---
name: project-covr-contextual-efficiency-bandit
description: COVR contextual LinUCB + efficiency reward (P1-P8) GPU-verified 2026-08-11: per-image contextual signal ABSENT on both fidelity (gain -6.4e-6 vs best static) and combined/efficiency (gain -2.6e-4 vs best static) dimensions; STOP — per-image Pareto heterogeneity does not exist
metadata: 
  node_type: memory
  type: project
  originSessionId: 60855cf1-763c-45d2-9cbf-5e3668ce2149
  modified: 2026-08-11T11:56:41.320Z
---

The deferred-commit contextual LinUCB bandit with efficiency-aware reward was **implemented and unit-tested (237 tests green)** as the structural fix for the non-contextual bandit collapse documented in [[project_covr_bandit_prior_lockin]] and [[project_covr_decision_granularity]]. **As of 2026-08-11 it has been GPU-verified on 500 ImageNet images (bs=1, λ=1e-3, K=3) and the contextual signal is absent on BOTH dimensions — the direction is STOP.** See the verdict section below.

**Why the non-contextual bandit collapsed (and this fixes it):**
1. Non-contextual = one global best arm → on c2i terminal fidelity that IS the baseline (threshold_0p25), so gain ≈ 0 by construction.
2. Pure terminal-fidelity reward is monotone in γ → can never reward a faster arm.

**The two-part fix:**
- **Deferred-commit contextual selection**: arm chosen AFTER a forced-calc prefix of K steps, using causal-prefix features (14 scalars/step × K) as LinUCB context. `begin_trajectory` returns a *pending* assignment (template_id=None); `commit_arm` picks the arm via per-arm LCB `score = θ·x − α·√(xᵀA⁻¹x)`, epsilon-greedy on top for non-degenerate IPS propensity.
- **Efficiency-aware reward**: `combined = terminal_MSE + λ·measured_flops` (cost = measured/vanilla FLOPs ∈ [0,1]). Bandit optimizes combined; analyzer still *evaluates* on terminal_fidelity.

**Key files**: `accelerators/covr_bandit.py:ContextualLinUCBBandit` (schema v2), `accelerators/covr_runtime.py` (commit_arm + `_efficiency_labels` + all-calc prefix strategy), `run_dit.py` (commit hook after learned-sigma trim), `scripts/run_covr_teacache_contextual.sh`, `scripts/analyze_covr_teacache_bandit.py`.

**Non-obvious gotcha resolved during handoff verification (2026-08-11)**: plan P3 lists "restore baseline γ after teacache_reset" as a required step, and `tests/test_teacache_prefix_mask.py::test_gamma_leaks_across_reset_documented_hazard` documents the hazard at the module level. But in the actual contextual run path there is **no leak and no restore** — each trajectory re-inits `teacache_state` via `apply_strategy → TeaCacheAdapter.init_state → teacache_init(**kwargs)` (registry.py:177), and `run_dit.py:2154` overwrites the loop's `teacache_state` with that fresh dict. So `teacache_reset` at run_dit.py:2112 is a no-op on the contextual path (its output is immediately replaced). The mask branch (teacache.py:218-305) also never reads `rel_l1_thresh` while `refresh_mask` is set. Don't go looking for a γ-restore line in run_dit.py — it doesn't exist and doesn't need to.

**Non-obvious gotcha #2 — refresh-mask arms are incompatible with deferred-commit (2026-08-11, found on first GPU run)**: the first AutoDL run crashed at trajectory 6 with `RuntimeError: contextual commit selected a strategy without rel_l1_thresh`. Root cause: `run_covr_teacache_contextual.sh` (P7) inherited `REFRESH_COUNTS=8` from the non-contextual runner, so `build_budget_manifest.py` emitted a *mixed* manifest — 5 threshold arms (params has `rel_l1_thresh`) plus refresh-mask arms (params has `refresh_mask`, no threshold). `ContextualLinUCBBandit._strategies = list(manifest.strategies)` fed all arms into LinUCB; the first 5 trajectories happened to pick threshold arms, the 6th picked a mask arm, and `commit_arm` (run_dit.py:1019) raised because `commit_arm` drops `refresh_mask` and switches onto the `rel_l1_thresh` dynamic path — a mask arm has no threshold value. Fix: (1) the contextual runner no longer passes `--refresh-counts`; (2) `ContextualLinUCBBandit._assert_threshold_only_strategies` rejects mixed manifests at construction (fail-fast at startup, not after burning GPU). The non-contextual `ConservativeTemplateBandit` still accepts mask arms (it applies them directly in `begin_trajectory`) — the guard is contextual-only. **Lesson**: the GPU run was the first time the actual `build_budget_manifest` output flowed into the contextual bandit; the local test manifest was hand-built threshold-only, so the mixed-manifest path was never exercised locally.

**Defaults**: CLI `--covr-efficiency-lambda 0.0` (preserves today's behavior); runner script defaults λ=1e-3, α=1.0, K=3. Schema v1↔v2 resume fails loudly (identity gate).

**Success criterion for the first GPU run** (`output/covr_teacache` namespace): `linucb_theta_norm` differs across arms AND chosen-arm distribution is NOT concentrated on one arm (no second collapse) AND `estimated_contextual_gain_vs_best_static > 0`. If the per-image Pareto heterogeneity is weak, contextual gain may still be small — that's the scientific bet; the analyzer decides.

**Non-obvious gotcha fixed during P8**: `ExperimentalBanditBackend.begin_trajectory` originally called `self.bandit.active_strategy` unconditionally, which raises for a pending contextual arm. Fixed to return `strategy=None` when `assignment.template_id is None` so the runtime can install the prefix strategy.

**First GPU run verdict (2026-08-11, 500 imgs, bs=1, λ=1e-3, K=3) — STOP:**

| criterion | result | pass? |
|---|---|---|
| linucb_theta_norm differs across arms | norms distinct (but all ~1e-4, near-zero) | yes (weakly) |
| chosen-arm distribution not concentrated | 5 arms all used, top share 30.8% | yes (no second collapse) |
| estimated_contextual_gain_vs_best_static > 0 (fidelity) | **−6.37e-6** | **no (≈0, noise) |
| estimated_contextual_gain_vs_best_static > 0 (combined) | **−2.58e-4** | **no (significantly negative) |

Per-arm SNIPS terminal fidelity is **monotone increasing in γ** (threshold_0p15 best → threshold_1 worst) — i.e. there is **no per-image crossover**: the per-image ranking of arms is the same as the global ranking, so a contextual per-image selector has nothing to exploit. theta norms ~1e-4 confirm LinUCB learned essentially nothing from the causal-prefix features.

The combined-dimension result (added to the analyzer post-hoc without re-burning GPU) is the decisive falsification: even on the efficiency-aware objective the bandit *actually optimizes*, contextual loses to the best static arm by −2.58e-4 (≈40× the fidelity gap, policy ESS=50 so not noise). best_static_combined = threshold_1 (cheapest/most-aggressive arm), but contextual selected threshold_1 only 9.8% of the time (picked threshold_0p6 most, 30.8%) — the policy is "weak LinUCB preference + eps=0.2 exploration," which structurally cannot converge to the single best static arm because there is no per-image signal to steer it.

**Conclusion**: per-image Pareto heterogeneity — the plan's core motivation — does not exist. Adding samples (2000-img resume) or sweeping λ cannot manufacture a signal source that is absent. This is the final validation of [[project_covr_bandit_prior_lockin]] and [[project_covr_decision_granularity]]: even after fixing prior-lockin (penalty=0) and decision-granularity (bs=1), contextual selection still cannot beat a static arm, because the problem was never the bandit machinery — it was the (non-existent) per-image signal. **Do not expand GPU experiments on this direction.** Stage C (FID/IS evaluation) is also not warranted.

**How to apply**: The plan at `~/.claude/plans/prancy-wiggling-pike.md` is resolved NEGATIVE. The contextual bandit code stays in-tree (it works correctly and the analyzer's combined-dimension reporting is reusable), but treat it as a closed investigation, not a pending rollout. The fix that DID land and is worth keeping: `_assert_threshold_only_strategies` (mixed-manifest fail-fast, gotcha #2) and the combined-dimension analyzer keys.
