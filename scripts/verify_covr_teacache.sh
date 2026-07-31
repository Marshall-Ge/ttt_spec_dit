#!/usr/bin/env bash
# =============================================================================
# COVR-on-TeaCache 验证脚本（等-FLOPs 固定 mask 方案）
#
# 覆盖计划验证方式的第 2、3 步：
#   [2] 从真实 SpecA action-audit 构建 teacache-mask manifest，校验等-FLOPs
#   [3] 端到端小样本：COVR strategy bandit 驱动 forced-mask TeaCache，
#       核对 cheap 1-step reward 生效（无昂贵 fallback）+ 等-FLOPs + bandit summary
#   [4] 回归对照：普通 --method teacache（动态阈值，refresh_mask=None）行为不变
#
# 用法：
#   bash scripts/verify_covr_teacache.sh [AUDIT_JSONL]
#   AUDIT_JSONL 缺省用 output/covr_test/covr/ 下最新的 events_*.jsonl
#
# 前置：在项目根目录、已激活能跑 main.py 的 GPU 环境。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # 项目根目录
REPO="$(pwd)"

# ---- 参数 ----
NUM_STEPS=50
TEMPLATE_COUNT=4
N_PROMPTS=80
BATCH_SIZE=32
SEED=42
GUIDANCE=4.5

MANIFEST=/tmp/tc_mask.json
COVR_RUN_DIR=/tmp/covr_tc_bandit         # bandit 运行输出（含 results.json / bandit state）
REGRESS_DIR=/tmp/covr_tc_regression      # 普通 teacache 对照输出

# ---- 定位或生成 audit JSONL ----
AUDIT_GEN_DIR=/tmp/covr_speca_audit
AUDIT="${1:-}"
if [[ -z "${AUDIT}" ]]; then
  AUDIT="$(ls -t output/covr_test/covr/events_*.jsonl 2>/dev/null | head -1 || true)"
  if [[ -z "${AUDIT}" ]]; then
    AUDIT="$(ls -t "${AUDIT_GEN_DIR}"/events_*.jsonl 2>/dev/null | head -1 || true)"
  fi
fi
if [[ -z "${AUDIT}" || ! -f "${AUDIT}" ]]; then
  echo ""
  echo "########## [0/5] 无现成 audit —— 用 SpecA + --covr-shadow 生成 ##########"
  rm -rf "${AUDIT_GEN_DIR}"
  python main.py --model dit --task c2i --dataset imagenet \
      --method speca --metrics fid is latency flops speed \
      --covr-shadow --covr-output-dir "${AUDIT_GEN_DIR}" \
      --seed "${SEED}" --num_steps "${NUM_STEPS}" --n_prompts "${N_PROMPTS}" \
      --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}"
  AUDIT="$(ls -t "${AUDIT_GEN_DIR}"/events_*.jsonl 2>/dev/null | head -1 || true)"
  if [[ -z "${AUDIT}" || ! -f "${AUDIT}" ]]; then
    echo "[FATAL] 生成 audit 失败，检查 SpecA shadow 输出。" >&2
    exit 1
  fi
  echo "  ✓ audit 已生成: ${AUDIT}"
else
  echo "=== 使用已有 audit: ${AUDIT}"
fi

# =============================================================================
# 步骤 1：逻辑单测（秒级，本机即可，放这里作双保险）
# =============================================================================
echo ""
echo "########## [1/5] 单测：forced-schedule + strategy 层 ##########"
python -m pytest tests/test_teacache_forced_schedule.py tests/test_covr_strategy.py -q

# =============================================================================
# 步骤 2：构建 teacache-mask manifest
# =============================================================================
echo ""
echo "########## [2/5] 构建等-FLOPs teacache-mask manifest ##########"
python scripts/build_covr_manifest.py "${AUDIT}" \
    --method teacache-mask --output "${MANIFEST}" \
    --template-count "${TEMPLATE_COUNT}" --num-layers 28 \
    --mandatory-prefix 3 --max-taylor-gap 5 --num-steps "${NUM_STEPS}"

# =============================================================================
# 步骤 3：manifest 结构校验（等-FLOPs 不变量）
# =============================================================================
echo ""
echo "########## [3/5] 校验 manifest：等-FLOPs / mask 结构 ##########"
python - "${MANIFEST}" "${NUM_STEPS}" <<'PY'
import sys
from accelerators.covr_bandit import StrategyManifest
from accelerators.strategy_dispatch import apply_strategy

path, num_steps = sys.argv[1], int(sys.argv[2])
m = StrategyManifest.load(path)
assert m.num_steps == num_steps, (m.num_steps, num_steps)

counts = []
for s in m.strategies:
    mask = s.refresh_mask
    assert s.method == "teacache", s.method
    assert mask is not None and len(mask) == num_steps, s.strategy_id
    assert mask[0] is True, f"{s.strategy_id}: first step must refresh"
    counts.append(sum(mask))
    # dispatch 注入检查
    st = apply_strategy(s, teacache_init_kwargs={
        "num_steps": num_steps, "coefficients": [0, 0, 0, 1, 0]})["teacache_state"]
    assert st["refresh_mask"] == mask, s.strategy_id

assert len(set(counts)) == 1, f"arms NOT equal-FLOPs: {counts}"
print(f"  arms={len(m.strategies)}  calc/arm={counts[0]}  (EQUAL-FLOPS OK)")
print(f"  baseline={m.baseline_strategy_id}  version_key={m.version_key}")
PY

# =============================================================================
# 步骤 4：端到端 —— COVR strategy bandit 驱动 forced-mask TeaCache
# =============================================================================
echo ""
echo "########## [4/5] 端到端：COVR strategy bandit (forced-mask teacache) ##########"
rm -rf "${COVR_RUN_DIR}"
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --covr-strategy-bandit --covr-strategy-manifest "${MANIFEST}" \
    --covr-sentinel-rate 0.05 --covr-sentinel-horizon 0 \
    --covr-profile-stages \
    --output_dir "${COVR_RUN_DIR}" \
    --seed "${SEED}" --num_steps "${NUM_STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}"

# =============================================================================
# 步骤 5：回归对照 —— 普通 teacache（动态阈值，无 COVR）
# =============================================================================
echo ""
echo "########## [5/5] 回归对照：普通 --method teacache（动态阈值） ##########"
rm -rf "${REGRESS_DIR}"
python main.py --model dit --task c2i --dataset imagenet \
    --method teacache --metrics fid is latency flops speed \
    --output_dir "${REGRESS_DIR}" \
    --seed "${SEED}" --num_steps "${NUM_STEPS}" --n_prompts "${N_PROMPTS}" \
    --guidance_scale "${GUIDANCE}" --batch_size "${BATCH_SIZE}"

# =============================================================================
# 自动核对 results.json
# =============================================================================
echo ""
echo "########## 结果核对 ##########"
python - "${COVR_RUN_DIR}/results.json" "${REGRESS_DIR}/results.json" <<'PY'
import json, sys

covr = json.load(open(sys.argv[1]))
regr = json.load(open(sys.argv[2]))
agg = covr["aggregate"]
ok = True

def check(cond, msg):
    global ok
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    ok = ok and cond

# --- 核心验证点：cheap reward 生效，未触发昂贵 fallback ---
prof = agg.get("generation_profile", {}).get("stage_total_s", {})
bandit = agg.get("covr_template_bandit", {})
sentinel_count = bandit.get("sentinel_count", 0)
sentinel_full = bandit.get("sentinel_full_steps", 0)

print(f"\n  sentinel_count={sentinel_count}  sentinel_full_steps={sentinel_full}")
print(f"  profile stages: {sorted(prof)}")

check("terminal_fallback_full" not in prof,
      "未触发昂贵 full-baseline fallback（走了 cheap 1-step reward）")
check("terminal_fidelity_shadow_full" in prof or sentinel_count == 0,
      "cheap terminal-fidelity shadow 生效（或本批无 sentinel）")
# cheap 路径：每条 sentinel 只 +1 步；fallback 会是 +num_steps。
if sentinel_count > 0:
    check(sentinel_full == sentinel_count,
          f"sentinel_full_steps({sentinel_full}) == sentinel_count({sentinel_count})，即每条 sentinel 仅 1 次额外前向")

# --- bandit summary 结构 ---
check("arm_log1p_loss_mean" in bandit,
      "bandit summary 含 arm_log1p_loss_mean")
arm_means = bandit.get("arm_log1p_loss_mean", {})
print(f"  arm_log1p_loss_mean = {arm_means}")
check(bandit.get("completed_trajectories", 0) > 0,
      "有已完成轨迹")

# --- 回归对照：普通 teacache 未启用 bandit ---
check(regr["config"].get("covr_template_bandit") in (False, None),
      "对照组未启用 COVR bandit（动态阈值路径）")
check(regr["config"].get("rel_l1_thresh") is not None,
      "对照组走 teacache 阈值路径（rel_l1_thresh 已记录）")

# --- FID/IS 都产出了（两组都能跑通） ---
for name, r in (("COVR", covr), ("regress", regr)):
    fid = r["aggregate"].get("fid")
    print(f"  {name}: FID={fid}  IS={r['aggregate'].get('is')}")

print()
print("  ====== " + ("全部验证点通过 ✅" if ok else "存在失败项 ❌，见上方 [FAIL]") + " ======")
sys.exit(0 if ok else 1)
PY

echo ""
echo "完整结果："
echo "  bandit run : ${COVR_RUN_DIR}/results.json"
echo "  regression : ${REGRESS_DIR}/results.json"
echo "  manifest   : ${MANIFEST}"
