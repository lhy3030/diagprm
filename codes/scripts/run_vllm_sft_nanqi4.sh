#!/bin/bash
# ==============================================================================
# run_vllm_sft_nanqi4.sh
#
# Run on nanqi4 to:
#   1. Start four local vLLM OpenAI-compatible servers for Qwen3-8B.
#   2. Start four SFT data generation workers, one worker per vLLM server.
#   3. Keep logs and outputs under timestamped directories.
#
# Usage on nanqi4:
#   cd /home/ubuntu/liutianshuo/diagprm/codes/scripts
#   nohup bash run_vllm_sft_nanqi4.sh > ../../diagprm_dataset/nanqi4_sft_master.log 2>&1 &
#
# Small smoke test:
#   MAX_TRAIN_CASES=2 MAX_VAL_CASES=1 CANDIDATES_PER_CASE=2 TOP_K_PER_CASE=1 \
#   nohup bash run_vllm_sft_nanqi4.sh > ../../diagprm_dataset/nanqi4_sft_smoke.log 2>&1 &
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIAGPRM_ROOT="$(cd "${CODES_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/liutianshuo/base_model/Qwen3-8B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
PYTHON3="${PYTHON3:-/opt/conda/envs/diagprm/bin/python3}"
if [ ! -x "${PYTHON3}" ]; then
  PYTHON3="python3"
fi

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
NUM_WORKERS="${NUM_WORKERS:-${#GPU_IDS[@]}}"
BASE_PORT="${BASE_PORT:-8000}"
MAX_WAIT="${MAX_WAIT:-600}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${DIAGPRM_ROOT}/diagprm_dataset/vllm_sft_logs/${RUN_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2_sft_vllm/${RUN_ID}}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

export DATASET_DIR="${DATASET_DIR:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2}"
export MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-500}"
export MAX_VAL_CASES="${MAX_VAL_CASES:-100}"
export MIN_ASK="${MIN_ASK:-2}"
export MAX_ASK="${MAX_ASK:-4}"
export MAX_TURNS="${MAX_TURNS:-5}"
export SEED="${SEED:-42}"
export CANDIDATES_PER_CASE="${CANDIDATES_PER_CASE:-4}"
export TOP_K_PER_CASE="${TOP_K_PER_CASE:-1}"
export MIN_KG_COVERAGE="${MIN_KG_COVERAGE:-0.5}"
export MIN_NEW_FACTS="${MIN_NEW_FACTS:-2}"
export USE_LLM=1
export USE_VLLM=1
export TEACHER_GENERATE_QUESTIONS=1
export LLM_MODEL="${MODEL_NAME}"
export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/diagprm_pycache}"
export HF_HOME="${HF_HOME:-${HOME}/liutianshuo/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
mkdir -p "${HF_HOME}" "${PYTHONPYCACHEPREFIX}"

VLLM_PIDS=()
WORKER_PIDS=()

cleanup() {
  echo "[INFO] Cleaning up workers and vLLM servers..."
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${VLLM_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "======================================================================"
echo "[INFO] nanqi4 vLLM SFT generation"
echo "[INFO] Run id:        ${RUN_ID}"
echo "[INFO] Model path:    ${MODEL_PATH}"
echo "[INFO] GPUs:          ${GPU_IDS_CSV}"
echo "[INFO] Workers:       ${NUM_WORKERS}"
echo "[INFO] Base port:     ${BASE_PORT}"
echo "[INFO] Dataset dir:   ${DATASET_DIR}"
echo "[INFO] Output root:   ${OUTPUT_ROOT}"
echo "[INFO] Log dir:       ${LOG_DIR}"
echo "======================================================================"

for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
  gpu_id="${GPU_IDS[$worker_id]}"
  port=$((BASE_PORT + worker_id))
  log_file="${LOG_DIR}/vllm_gpu${gpu_id}_port${port}.log"

  echo "[INFO] Starting vLLM worker=${worker_id} gpu=${gpu_id} port=${port}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON3}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size 1 \
    --port "${port}" \
    --host 127.0.0.1 \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name "${MODEL_NAME}" \
    > "${log_file}" 2>&1 &
  VLLM_PIDS+=("$!")
done

for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
  port=$((BASE_PORT + worker_id))
  waited=0
  echo "[INFO] Waiting for vLLM worker=${worker_id} on port ${port}"
  while true; do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[INFO] vLLM worker=${worker_id} ready after ${waited}s"
      break
    fi
    if [ "${waited}" -ge "${MAX_WAIT}" ]; then
      echo "[ERROR] vLLM worker=${worker_id} not ready after ${MAX_WAIT}s"
      echo "[ERROR] Check ${LOG_DIR}/vllm_*_port${port}.log"
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
done

for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
  port=$((BASE_PORT + worker_id))
  worker_output="${OUTPUT_ROOT}/worker_${worker_id}"
  worker_log="${LOG_DIR}/sft_worker_${worker_id}.log"
  mkdir -p "${worker_output}"

  echo "[INFO] Starting SFT worker=${worker_id} shard=${worker_id}/${NUM_WORKERS} port=${port}"
  (
    export VLLM_HOST="127.0.0.1"
    export VLLM_PORT="${port}"
    export CASE_STRIDE="${NUM_WORKERS}"
    export CASE_OFFSET="${worker_id}"
    export OUTPUT_BASE_DIR="${worker_output}"
    export SEED="$((SEED + worker_id))"
    bash "${SCRIPT_DIR}/run_generate_diagprm_sft_teacher.sh"
  ) > "${worker_log}" 2>&1 &
  WORKER_PIDS+=("$!")
done

failed=0
for idx in "${!WORKER_PIDS[@]}"; do
  pid="${WORKER_PIDS[$idx]}"
  if wait "${pid}"; then
    echo "[INFO] SFT worker ${idx} finished successfully"
  else
    echo "[ERROR] SFT worker ${idx} failed. Check ${LOG_DIR}/sft_worker_${idx}.log"
    failed=1
  fi
done

if [ "${failed}" != "0" ]; then
  exit 1
fi

echo "======================================================================"
echo "[INFO] All SFT workers complete"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Logs:        ${LOG_DIR}"
echo "======================================================================"
