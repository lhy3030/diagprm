#!/bin/bash
# ==============================================================================
# Continue SFT collection without touching a previous interrupted run.
#
# This script:
#   1. Reconstructs disease_ids already attempted by a previous sharded run.
#   2. Starts a fresh run_vllm_sft_nanqi4.sh with EXCLUDE_CASE_IDS_FILE set.
#
# Usage on nanqi4:
#   cd /home/ubuntu/liutianshuo/diagprm/codes/recipe/diagprm/scripts
#   PREVIOUS_RUN_ID=20260627_145328 PREVIOUS_TIMESTAMP=20260627_145636 \
#   MAX_TRAIN_CASES=250 MAX_VAL_CASES=0 \
#   nohup bash run_vllm_sft_nanqi4_continue_missing.sh \
#     > /home/ubuntu/liutianshuo/diagprm/diagprm_dataset/nanqi4_sft_continue_missing.log 2>&1 &
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DIAGPRM_ROOT="$(cd "${CODES_DIR}/.." && pwd)"

PYTHON3="${PYTHON3:-/opt/conda/envs/diagprm/bin/python3}"
if [ ! -x "${PYTHON3}" ]; then
  PYTHON3="python3"
fi

PREVIOUS_RUN_ID="${PREVIOUS_RUN_ID:-20260627_145328}"
PREVIOUS_RUN_DIR="${PREVIOUS_RUN_DIR:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2_sft_vllm/${PREVIOUS_RUN_ID}}"
PREVIOUS_TIMESTAMP="${PREVIOUS_TIMESTAMP:-}"
DATASET_DIR="${DATASET_DIR:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2}"
DATASET_JSONL="${DATASET_JSONL:-${DATASET_DIR}/kg_train_dataset.jsonl}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OLD_SEED="${OLD_SEED:-42}"
LEGACY_SEED_PER_WORKER="${LEGACY_SEED_PER_WORKER:-1}"

EXCLUDE_CASE_IDS_FILE="${EXCLUDE_CASE_IDS_FILE:-${PREVIOUS_RUN_DIR}/exclude_attempted_case_ids.txt}"

BUILD_CMD=(
  "${PYTHON3}" "${SCRIPT_DIR}/build_sft_exclude_case_ids.py"
  --dataset_jsonl "${DATASET_JSONL}"
  --run_dir "${PREVIOUS_RUN_DIR}"
  --output "${EXCLUDE_CASE_IDS_FILE}"
  --seed "${OLD_SEED}"
  --num_workers "${NUM_WORKERS}"
)

if [ -n "${PREVIOUS_TIMESTAMP}" ]; then
  BUILD_CMD+=(--timestamp "${PREVIOUS_TIMESTAMP}")
fi
if [ "${LEGACY_SEED_PER_WORKER}" = "1" ]; then
  BUILD_CMD+=(--legacy_seed_per_worker)
fi

echo "[INFO] Building exclude list from previous run"
printf ' %q' "${BUILD_CMD[@]}"
printf '\n'
"${BUILD_CMD[@]}"

EXCLUDE_COUNT="$(wc -l < "${EXCLUDE_CASE_IDS_FILE}" | tr -d ' ')"
export EXCLUDE_CASE_IDS_FILE
export DATASET_DIR
export NUM_WORKERS
export SHARD_SEED="${SHARD_SEED:-42}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "======================================================================"
echo "[INFO] Starting fresh continuation run"
echo "[INFO] Previous run: ${PREVIOUS_RUN_DIR}"
echo "[INFO] Exclude ids:  ${EXCLUDE_CASE_IDS_FILE} (${EXCLUDE_COUNT})"
echo "[INFO] New RUN_ID:   ${RUN_ID}"
echo "[INFO] SHARD_SEED:   ${SHARD_SEED}"
echo "======================================================================"

bash "${SCRIPT_DIR}/run_vllm_sft_nanqi4.sh"
