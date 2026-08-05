#!/usr/bin/env bash
# One-command COVR forced-strategy TeaCache smoke test.
#
# The probe and forced run must use the same runtime identity. This script
# obtains the version key from a one-image observe run, builds an external
# strategy manifest with that key, validates the requested arm, and runs the
# real CLI entry point once.
set -euo pipefail

cd "$(dirname "$0")/.."

STRATEGY_ID="${1:-${COVR_STRATEGY_ID:-uniform}}"
NUM_STEPS="${COVR_NUM_STEPS:-50}"
REFRESH_COUNT="${COVR_REFRESH_COUNT:-8}"
SEED="${COVR_SEED:-42}"
GUIDANCE="${COVR_GUIDANCE_SCALE:-4.0}"
N_PROMPTS="${COVR_N_PROMPTS:-2}"
BATCH_SIZE="${COVR_BATCH_SIZE:-1}"
ROOT="${COVR_SMOKE_ROOT:-/tmp/covr_forced_smoke}"
PROBE_DIR="${ROOT}/version_probe"
MANIFEST="${ROOT}/teacache_k${REFRESH_COUNT}.json"
RUN_DIR="${ROOT}/forced_run"

mkdir -p "${ROOT}"

COMMON_ARGS=(
  --model dit
  --task c2i
  --dataset imagenet
  --method teacache
  --num_steps "${NUM_STEPS}"
  --seed "${SEED}"
  --guidance_scale "${GUIDANCE}"
)

echo "[1/5] Probing COVR runtime version identity"
python main.py "${COMMON_ARGS[@]}" \
  --metrics latency \
  --n_prompts 1 --batch_size 1 \
  --covr-profile-stages \
  --output_dir "${PROBE_DIR}"

VERSION_KEY="$(python - "${PROBE_DIR}/results.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["config"].get("covr_version_key")
if not value:
    raise SystemExit("results.json does not contain config.covr_version_key")
print(value)
PY
)"
echo "      version_key=${VERSION_KEY}"

echo "[2/5] Building external TeaCache strategy manifest"
python scripts/build_budget_manifest.py \
  --output "${MANIFEST}" \
  --method teacache \
  --num-steps "${NUM_STEPS}" \
  --refresh-count "${REFRESH_COUNT}" \
  --mandatory-prefix 3 \
  --baseline-arm "${STRATEGY_ID}" \
  --version-key "${VERSION_KEY}"

echo "[3/5] Validating strategy ID"
python - "${MANIFEST}" "${STRATEGY_ID}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
wanted = sys.argv[2]
strategies = manifest["strategies"]
ids = [item["strategy_id"] for item in strategies]
if wanted not in ids:
    raise SystemExit(
        f"unknown strategy id {wanted!r}; available: {', '.join(ids)}")
selected = next(item for item in strategies if item["strategy_id"] == wanted)
print(f"      strategy_id={wanted}")
print(f"      method={selected['method']}")
print(f"      params={selected['params']}")
print(f"      available={', '.join(ids)}")
PY

echo "[4/5] Running forced COVR strategy"
python main.py "${COMMON_ARGS[@]}" \
  --metrics latency flops speed \
  --n_prompts "${N_PROMPTS}" --batch_size "${BATCH_SIZE}" \
  --covr-profile-stages \
  --covr-strategy-manifest "${MANIFEST}" \
  --covr-force-strategy-id "${STRATEGY_ID}" \
  --output_dir "${RUN_DIR}"

echo "[5/5] Validating forced COVR results"
python scripts/check_covr_forced_smoke.py "${ROOT}" \
  --manifest "${MANIFEST}" \
  --strategy-id "${STRATEGY_ID}"
