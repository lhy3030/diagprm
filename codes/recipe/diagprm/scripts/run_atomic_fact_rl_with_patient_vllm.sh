#!/usr/bin/env bash
set -euo pipefail

# Start the Qwen3-8B patient vLLM on GPU 4, wait until it is healthy,
# and then run AtomicFact RL on GPUs 0-3.
#
# Recommended usage on nanqi4:
#   nohup bash recipe/diagprm/scripts/run_atomic_fact_rl_with_patient_vllm.sh \
#     > /home/ubuntu/liutianshuo/diagprm/logs/atomic_fact_rl_launcher.log 2>&1 &

DIAGPRM_ROOT="${DIAGPRM_ROOT:-/home/ubuntu/liutianshuo/diagprm}"
CODES_DIR="${CODES_DIR:-${DIAGPRM_ROOT}/codes}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-diagprm}"

PATIENT_MODEL_PATH="${PATIENT_MODEL_PATH:-/home/ubuntu/liutianshuo/base_model/Qwen3-8B}"
PATIENT_MODEL="${PATIENT_MODEL:-Qwen3-8B}"
PATIENT_GPU="${PATIENT_GPU:-4}"
PATIENT_HOST="${PATIENT_HOST:-0.0.0.0}"
PATIENT_PORT="${PATIENT_PORT:-8100}"
PATIENT_MAX_MODEL_LEN="${PATIENT_MAX_MODEL_LEN:-4096}"
PATIENT_GPU_MEMORY_UTILIZATION="${PATIENT_GPU_MEMORY_UTILIZATION:-0.90}"
PATIENT_STARTUP_TIMEOUT="${PATIENT_STARTUP_TIMEOUT:-300}"
PATIENT_VLLM_ENDPOINT_FILE="${PATIENT_VLLM_ENDPOINT_FILE:-${DIAGPRM_ROOT}/patient_vllm.endpoint}"

ACTOR_LOAD="${ACTOR_LOAD:-${DIAGPRM_ROOT}/checkpoints/Qwen3-1.7B_atomic_fact_sft_prompt_v2/global_step_116/merged_hf}"
RL_GPUS="${RL_GPUS:-0,1,2,3}"
RUN_NAME="${RUN_NAME:-atomic_fact_rl_1b7_sft_prompt_v2_g8_b64}"
N_RESP_PER_PROMPT_OVERRIDE="${N_RESP_PER_PROMPT_OVERRIDE:-8}"
TRAIN_BATCH_SIZE_OVERRIDE="${TRAIN_BATCH_SIZE_OVERRIDE:-64}"
PPO_MINI_BATCH_SIZE_OVERRIDE="${PPO_MINI_BATCH_SIZE_OVERRIDE:-64}"
PPO_MAX_TOKEN_LEN_OVERRIDE="${PPO_MAX_TOKEN_LEN_OVERRIDE:-16384}"
LOG_PROB_MAX_TOKEN_LEN_OVERRIDE="${LOG_PROB_MAX_TOKEN_LEN_OVERRIDE:-16384}"

LOG_DIR="${LOG_DIR:-${DIAGPRM_ROOT}/logs}"
PATIENT_LOG="${PATIENT_LOG:-${LOG_DIR}/patient_vllm_Qwen3-8B_gpu4.log}"
RL_LOG="${RL_LOG:-${LOG_DIR}/${RUN_NAME}.log}"
PATIENT_PID=""
STARTED_PATIENT=0

mkdir -p "${LOG_DIR}" "${DIAGPRM_ROOT}/.cache/huggingface" "${DIAGPRM_ROOT}/.triton"

if [ ! -f "${CONDA_SH}" ]; then
  echo "[ERROR] Conda initialization script not found: ${CONDA_SH}" >&2
  exit 1
fi
if [ ! -d "${CODES_DIR}" ]; then
  echo "[ERROR] Codes directory not found: ${CODES_DIR}" >&2
  exit 1
fi
if [ ! -d "${PATIENT_MODEL_PATH}" ]; then
  echo "[ERROR] Patient model not found: ${PATIENT_MODEL_PATH}" >&2
  exit 1
fi
if [ ! -d "${ACTOR_LOAD}" ]; then
  echo "[ERROR] Actor checkpoint not found: ${ACTOR_LOAD}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${DIAGPRM_ROOT}"

# Ray workers on the same host can reliably access this node IP. Avoid awk
# here because some remote environments inject an incompatible awk program.
if [ -z "${NODE_IP:-}" ]; then
  read -r NODE_IP _ < <(hostname -I 2>/dev/null || true)
fi
NODE_IP="${NODE_IP:-127.0.0.1}"
PATIENT_API_BASE="${PATIENT_API_BASE:-http://${NODE_IP}:${PATIENT_PORT}/v1}"
PATIENT_HEALTH_URL="http://127.0.0.1:${PATIENT_PORT}/health"
printf '%s\n' "${PATIENT_API_BASE}" > "${PATIENT_VLLM_ENDPOINT_FILE}"
export PATIENT_VLLM_ENDPOINT_FILE

cleanup() {
  local exit_code=$?
  if [ "${STARTED_PATIENT}" = "1" ] && [ -n "${PATIENT_PID}" ] && kill -0 "${PATIENT_PID}" 2>/dev/null; then
    echo "[INFO] Stopping patient vLLM (PID ${PATIENT_PID}) ..."
    kill "${PATIENT_PID}" 2>/dev/null || true
    wait "${PATIENT_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

if curl -fsS --max-time 5 "${PATIENT_HEALTH_URL}" >/dev/null 2>&1; then
  echo "[INFO] Reusing healthy patient vLLM at ${PATIENT_API_BASE}"
else
  if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"${PATIENT_PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[ERROR] Port ${PATIENT_PORT} is occupied, but its health endpoint is unavailable." >&2
    exit 1
  fi

  echo "[INFO] Starting patient vLLM on GPU ${PATIENT_GPU} ..."
  (
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    export CUDA_VISIBLE_DEVICES="${PATIENT_GPU}"
    export HF_HOME="${HF_HOME:-${DIAGPRM_ROOT}/.cache/huggingface}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DIAGPRM_ROOT}/.cache}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DIAGPRM_ROOT}/.triton}"

    exec python -m vllm.entrypoints.openai.api_server \
      --model "${PATIENT_MODEL_PATH}" \
      --served-model-name "${PATIENT_MODEL}" \
      --host "${PATIENT_HOST}" \
      --port "${PATIENT_PORT}" \
      --tensor-parallel-size 1 \
      --max-model-len "${PATIENT_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${PATIENT_GPU_MEMORY_UTILIZATION}" \
      --dtype auto \
      --trust-remote-code \
      --disable-log-requests \
      --enforce-eager
  ) >>"${PATIENT_LOG}" 2>&1 &
  PATIENT_PID=$!
  STARTED_PATIENT=1

  echo "[INFO] Patient vLLM PID: ${PATIENT_PID}"
  echo "[INFO] Waiting for patient vLLM health check ..."
  deadline=$((SECONDS + PATIENT_STARTUP_TIMEOUT))
  until curl -fsS --max-time 5 "${PATIENT_HEALTH_URL}" >/dev/null 2>&1; do
    if ! kill -0 "${PATIENT_PID}" 2>/dev/null; then
      echo "[ERROR] Patient vLLM exited during startup. See ${PATIENT_LOG}" >&2
      tail -n 80 "${PATIENT_LOG}" >&2 || true
      exit 1
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      echo "[ERROR] Patient vLLM did not become healthy within ${PATIENT_STARTUP_TIMEOUT}s." >&2
      tail -n 80 "${PATIENT_LOG}" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "[INFO] Patient vLLM is healthy: ${PATIENT_API_BASE}"
fi

# Keep vLLM V1 and use Ray as the distributed executor. The async FSDP worker
# must await execute_method; the corresponding project code contains that fix.
unset PYTORCH_CUDA_ALLOC_CONF
# The patient API is on this same node. atomic_fact_agent_loop.py explicitly
# passes https_proxy to aiohttp for non-loopback URLs, which bypasses no_proxy
# and can cause blank TimeoutError messages in managed task environments.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY PATIENT_PROXY
export CUDA_VISIBLE_DEVICES="${RL_GPUS}"
export VLLM_USE_V1=1
export VERL_VLLM_DISTRIBUTED_BACKEND=ray
export no_proxy="127.0.0.1,localhost,${NODE_IP}${no_proxy:+,${no_proxy}}"
export NO_PROXY="127.0.0.1,localhost,${NODE_IP}${NO_PROXY:+,${NO_PROXY}}"

cd "${CODES_DIR}"
echo "======================================================================"
echo "[INFO] Starting AtomicFact RL"
echo "[INFO] Actor:           ${ACTOR_LOAD}"
echo "[INFO] RL GPUs:         ${RL_GPUS}"
echo "[INFO] Patient API:     ${PATIENT_API_BASE}"
echo "[INFO] Patient model:   ${PATIENT_MODEL}"
echo "[INFO] G:               ${N_RESP_PER_PROMPT_OVERRIDE}"
echo "[INFO] Train batch:     ${TRAIN_BATCH_SIZE_OVERRIDE}"
echo "[INFO] PPO mini-batch:  ${PPO_MINI_BATCH_SIZE_OVERRIDE}"
echo "[INFO] Run name:        ${RUN_NAME}"
echo "[INFO] RL log:          ${RL_LOG}"
echo "======================================================================"

ACTOR_LOAD="${ACTOR_LOAD}" \
MAX_TURNS_OVERRIDE=10 \
ATOMIC_FACT_DOCTOR_THINKING=0 \
ATOMIC_FACT_MIN_TURN_TOKENS=1 \
PATIENT_API_BASE="${PATIENT_API_BASE}" \
PATIENT_MODEL="${PATIENT_MODEL}" \
RUN_NAME="${RUN_NAME}" \
N_RESP_PER_PROMPT_OVERRIDE="${N_RESP_PER_PROMPT_OVERRIDE}" \
TRAIN_BATCH_SIZE_OVERRIDE="${TRAIN_BATCH_SIZE_OVERRIDE}" \
PPO_MINI_BATCH_SIZE_OVERRIDE="${PPO_MINI_BATCH_SIZE_OVERRIDE}" \
PPO_MAX_TOKEN_LEN_OVERRIDE="${PPO_MAX_TOKEN_LEN_OVERRIDE}" \
LOG_PROB_MAX_TOKEN_LEN_OVERRIDE="${LOG_PROB_MAX_TOKEN_LEN_OVERRIDE}" \
bash recipe/diagprm/scripts/run_atomic_fact_rl.sh 2>&1 | tee "${RL_LOG}"
