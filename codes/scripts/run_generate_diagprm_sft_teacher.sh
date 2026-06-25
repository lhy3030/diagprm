#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIAGPRM_ROOT="$(cd "${CODES_DIR}/.." && pwd)"

export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp}"

# ============================================================================
# Config — 所有变量均可通过环境变量覆盖，例如：
#   MAX_TRAIN_CASES=1 OUTPUT_BASE_DIR=/tmp/test bash run_generate_diagprm_sft_teacher.sh
# ============================================================================
DATASET_DIR="${DATASET_DIR:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2_sft_teacher_runs}"

MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-200}"
MAX_VAL_CASES="${MAX_VAL_CASES:-50}"
MIN_ASK="${MIN_ASK:-2}"
MAX_ASK="${MAX_ASK:-4}"
MAX_TURNS="${MAX_TURNS:-5}"
SEED="${SEED:-42}"

CANDIDATES_PER_CASE="${CANDIDATES_PER_CASE:-4}"
TOP_K_PER_CASE="${TOP_K_PER_CASE:-1}"
MIN_KG_COVERAGE="${MIN_KG_COVERAGE:-0.5}"
MIN_NEW_FACTS="${MIN_NEW_FACTS:-2}"

USE_LLM="${USE_LLM:-1}"
TEACHER_GENERATE_QUESTIONS="${TEACHER_GENERATE_QUESTIONS:-1}"

# vLLM mode: set USE_VLLM=1 to use local vLLM server instead of remote API
# vLLM server should be started separately before running this script.
# Example: vllm serve /path/to/Qwen3-8B --port 8000 --tensor-parallel-size 2
USE_VLLM="${USE_VLLM:-0}"

if [ "${USE_VLLM}" = "1" ]; then
  VLLM_HOST="${VLLM_HOST:-localhost}"
  VLLM_PORT="${VLLM_PORT:-8000}"
  LLM_API_BASE="${LLM_API_BASE:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
  LLM_API_KEY="${LLM_API_KEY:-EMPTY}"  # vLLM doesn't require a real key
  LLM_MODEL="${LLM_MODEL:-Qwen3-8B}"
else
  LLM_API_BASE="${LLM_API_BASE:-https://aigc.sankuai.com/v1/openai/native}"
  LLM_API_KEY="${LLM_API_KEY:-}"  # 通过环境变量传入，避免硬编码
  LLM_MODEL="${LLM_MODEL:-qwen3-32b-meituan}"
fi

CMD=(
  python3 "${SCRIPT_DIR}/generate_diagprm_sft_from_kg.py"
  --dataset_dir "${DATASET_DIR}"
  --output_dir "${OUTPUT_BASE_DIR}"
  --timestamp_output
  --max_train_cases "${MAX_TRAIN_CASES}"
  --max_val_cases "${MAX_VAL_CASES}"
  --min_ask "${MIN_ASK}"
  --max_ask "${MAX_ASK}"
  --max_turns "${MAX_TURNS}"
  --seed "${SEED}"
  --candidates_per_case "${CANDIDATES_PER_CASE}"
  --top_k_per_case "${TOP_K_PER_CASE}"
  --min_kg_coverage "${MIN_KG_COVERAGE}"
  --min_new_facts "${MIN_NEW_FACTS}"
)

if [ "${USE_LLM}" = "1" ]; then
  CMD+=(
    --use_llm
    --llm_api_base "${LLM_API_BASE}"
    --llm_api_key "${LLM_API_KEY}"
    --llm_model "${LLM_MODEL}"
  )
fi

if [ "${TEACHER_GENERATE_QUESTIONS}" = "1" ]; then
  CMD+=(--teacher_generate_questions)
fi

echo "[INFO] Dataset dir:            ${DATASET_DIR}"
echo "[INFO] Output base dir:        ${OUTPUT_BASE_DIR}"
echo "[INFO] Max train cases:        ${MAX_TRAIN_CASES}"
echo "[INFO] Max val cases:          ${MAX_VAL_CASES}"
echo "[INFO] Candidates per case:    ${CANDIDATES_PER_CASE}"
echo "[INFO] Top-k per case:         ${TOP_K_PER_CASE}"
echo "[INFO] Min KG coverage:        ${MIN_KG_COVERAGE}"
echo "[INFO] Min new facts:          ${MIN_NEW_FACTS}"
echo "[INFO] Use LLM:                ${USE_LLM}"
echo "[INFO] Teacher gen questions:  ${TEACHER_GENERATE_QUESTIONS}"
echo "[INFO] vLLM mode:              ${USE_VLLM}"
echo "[INFO] LLM API base:           ${LLM_API_BASE}"
echo "[INFO] LLM model:              ${LLM_MODEL}"
if [ -z "${LLM_API_KEY}" ] && [ "${USE_LLM}" = "1" ] && [ "${USE_VLLM}" != "1" ]; then
  echo "[WARN] LLM_API_KEY is empty; set it via env var before running."
fi
echo "[INFO] Running command:"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
