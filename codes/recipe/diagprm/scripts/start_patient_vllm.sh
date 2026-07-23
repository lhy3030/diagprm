#!/usr/bin/env bash
set -euo pipefail

# Standalone launcher for the DiagPRM patient/teacher vLLM server.
#
# Example:
#   PATIENT_VLLM_GPU=4 bash start_patient_vllm.sh
#
# Defaults:
#   PATIENT_VLLM_MODEL_PATH=/home/ubuntu/liutianshuo/base_model/Qwen3-8B
#   PATIENT_VLLM_MODEL_NAME=Qwen3-8B
#   PATIENT_VLLM_HOST=127.0.0.1
#   PATIENT_VLLM_PORT=8100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGPRM_ROOT="${DIAGPRM_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
export DIAGPRM_ROOT

if [ -x "/opt/conda/envs/diagprm/bin/python3" ]; then
  export PYTHON3="${PYTHON3:-/opt/conda/envs/diagprm/bin/python3}"
elif [ -x "${HOME}/miniconda3/envs/diagprm/bin/python3" ]; then
  export PYTHON3="${PYTHON3:-${HOME}/miniconda3/envs/diagprm/bin/python3}"
else
  export PYTHON3="${PYTHON3:-python3}"
fi

export START_PATIENT_VLLM=1
source "${SCRIPT_DIR}/patient_vllm_utils.sh"

cleanup() {
  stop_patient_vllm
}
trap cleanup EXIT INT TERM

start_patient_vllm_if_requested

echo "======================================================================"
echo "[INFO] patient vLLM is running"
echo "[INFO] Endpoint: ${PATIENT_API_BASE}"
echo "[INFO] Model:    ${PATIENT_MODEL}"
echo "[INFO] Press Ctrl-C to stop."
echo "======================================================================"

if [ -n "${_DIAGPRM_PATIENT_VLLM_PID}" ]; then
  wait "${_DIAGPRM_PATIENT_VLLM_PID}"
else
  while curl -sf "${PATIENT_API_BASE%/v1}/health" >/dev/null 2>&1; do
    sleep 30
  done
fi
