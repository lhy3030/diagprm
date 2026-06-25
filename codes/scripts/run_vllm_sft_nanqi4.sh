#!/bin/bash
# ==============================================================================
# run_vllm_sft_nanqi4.sh
#
# Run on nanqi4 (2x GPU server) to:
#   1. Start vLLM with Qwen3-8B on both GPUs (tensor-parallel-size=2)
#   2. Wait for vLLM to be ready
#   3. Run SFT generation using vLLM as the LLM backend
#   4. Shut down vLLM after generation is done
#
# Usage (on nanqi4):
#   cd ~/liutianshuo/diagprm/codes/scripts
#   bash run_vllm_sft_nanqi4.sh
#
# Custom settings:
#   MAX_TRAIN_CASES=200 TOP_K_PER_CASE=2 bash run_vllm_sft_nanqi4.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/liutianshuo/base_model/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"

# ── Generation config ─────────────────────────────────────────────────────────
export MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-200}"
export MAX_VAL_CASES="${MAX_VAL_CASES:-0}"
export MIN_ASK="${MIN_ASK:-2}"
export MAX_ASK="${MAX_ASK:-4}"
export MAX_TURNS="${MAX_TURNS:-5}"
export SEED="${SEED:-42}"
export CANDIDATES_PER_CASE="${CANDIDATES_PER_CASE:-4}"
export TOP_K_PER_CASE="${TOP_K_PER_CASE:-2}"
export MIN_KG_COVERAGE="${MIN_KG_COVERAGE:-0.5}"
export MIN_NEW_FACTS="${MIN_NEW_FACTS:-2}"
export USE_LLM=1
export USE_VLLM=1
export VLLM_HOST="localhost"
export LLM_MODEL="Qwen3-8B"
export TEACHER_GENERATE_QUESTIONS=1
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2_sft_vllm}"

echo "======================================================================"
echo "[INFO] DiagPRM SFT generation via local vLLM"
echo "[INFO] Model:           ${MODEL_PATH}"
echo "[INFO] Tensor parallel: ${TENSOR_PARALLEL}"
echo "[INFO] Port:            ${VLLM_PORT}"
echo "[INFO] Train cases:     ${MAX_TRAIN_CASES}"
echo "[INFO] Top-k/case:      ${TOP_K_PER_CASE}"
echo "[INFO] Output dir:      ${OUTPUT_BASE_DIR}"
echo "======================================================================"

# ── Step 1: Start vLLM in background ─────────────────────────────────────────
VLLM_LOG="${DIAGPRM_ROOT}/diagprm_dataset/vllm_server.log"
echo "[INFO] Starting vLLM server (log: ${VLLM_LOG}) ..."

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --port "${VLLM_PORT}" \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code \
    --served-model-name "Qwen3-8B" \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[INFO] vLLM PID: ${VLLM_PID}"

# ── Step 2: Wait for vLLM to be ready ────────────────────────────────────────
echo "[INFO] Waiting for vLLM to be ready on port ${VLLM_PORT} ..."
MAX_WAIT=300
WAITED=0
while true; do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[INFO] vLLM is ready! (waited ${WAITED}s)"
        break
    fi
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[ERROR] vLLM did not start within ${MAX_WAIT}s. Check ${VLLM_LOG}"
        kill "${VLLM_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "[INFO]   still waiting... (${WAITED}s / ${MAX_WAIT}s)"
done

# ── Step 3: Run SFT generation ────────────────────────────────────────────────
echo "[INFO] Starting SFT generation ..."
trap "echo '[INFO] Cleaning up vLLM (PID ${VLLM_PID})...'; kill ${VLLM_PID} 2>/dev/null || true" EXIT

bash "${SCRIPT_DIR}/run_generate_diagprm_sft_teacher.sh"

echo ""
echo "======================================================================"
echo "[INFO] SFT generation complete. Results in: ${OUTPUT_BASE_DIR}"
echo "======================================================================"
