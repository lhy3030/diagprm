#!/usr/bin/env bash

# Utilities for launching a local OpenAI-compatible vLLM server as the
# DiagPRM patient/teacher LLM. This file is intended to be sourced by training
# and data-generation scripts.

_DIAGPRM_PATIENT_VLLM_PID="${_DIAGPRM_PATIENT_VLLM_PID:-}"

_patient_vllm_count_visible_gpus() {
  local csv="$1"
  if [ -z "${csv}" ]; then
    echo ""
    return
  fi
  local compact="${csv// /}"
  if [ -z "${compact}" ]; then
    echo ""
    return
  fi
  awk -F',' '{print NF}' <<< "${compact}"
}

_patient_vllm_server_alive() {
  local host="$1"
  local port="$2"
  curl -sf "http://${host}:${port}/health" >/dev/null 2>&1
}

_patient_vllm_openai_ready() {
  local host="$1"
  local port="$2"
  curl -sf "http://${host}:${port}/v1/models" >/dev/null 2>&1
}

wait_for_patient_vllm() {
  local host="${PATIENT_VLLM_HOST:-127.0.0.1}"
  local port="${PATIENT_VLLM_PORT:-8100}"
  local max_wait="${PATIENT_VLLM_MAX_WAIT:-900}"
  local waited=0

  echo "[INFO] Waiting for patient vLLM on ${host}:${port}"
  while true; do
    if _patient_vllm_openai_ready "${host}" "${port}"; then
      echo "[INFO] patient vLLM ready after ${waited}s"
      return 0
    fi
    if [ -n "${_DIAGPRM_PATIENT_VLLM_PID}" ] && ! kill -0 "${_DIAGPRM_PATIENT_VLLM_PID}" 2>/dev/null; then
      echo "[ERROR] patient vLLM process exited before becoming ready" >&2
      return 1
    fi
    if [ "${waited}" -ge "${max_wait}" ]; then
      echo "[ERROR] Timed out waiting for patient vLLM after ${max_wait}s" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

_patient_vllm_apply_training_cuda() {
  local gpu_id="$1"
  if [ -n "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
    local n_train_gpus
    n_train_gpus="$(_patient_vllm_count_visible_gpus "${TRAIN_CUDA_VISIBLE_DEVICES}")"
    if [ -n "${n_train_gpus}" ] && [ "${PATIENT_VLLM_KEEP_N_GPUS:-0}" != "1" ]; then
      export N_GPUS="${n_train_gpus}"
    fi
    echo "[INFO] Training CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[INFO] Training N_GPUS=${N_GPUS:-<unset>}"
  else
    echo "[WARN] TRAIN_CUDA_VISIBLE_DEVICES is not set. Make sure the trainer will not use patient GPU ${gpu_id}."
  fi
}

stop_patient_vllm() {
  if [ -n "${_DIAGPRM_PATIENT_VLLM_PID}" ] && kill -0 "${_DIAGPRM_PATIENT_VLLM_PID}" 2>/dev/null; then
    echo "[INFO] Stopping patient vLLM pid=${_DIAGPRM_PATIENT_VLLM_PID}"
    kill "${_DIAGPRM_PATIENT_VLLM_PID}" 2>/dev/null || true
    wait "${_DIAGPRM_PATIENT_VLLM_PID}" 2>/dev/null || true
  fi
}

start_patient_vllm_if_requested() {
  if [ "${START_PATIENT_VLLM:-0}" != "1" ]; then
    return 0
  fi

  local host="${PATIENT_VLLM_HOST:-0.0.0.0}"
  local port="${PATIENT_VLLM_PORT:-8100}"
  local model_path="${PATIENT_VLLM_MODEL_PATH:-/home/ubuntu/liutianshuo/base_model/Qwen3-8B}"
  local model_name="${PATIENT_VLLM_MODEL_NAME:-Qwen3-8B}"
  local gpu_id="${PATIENT_VLLM_GPU:-${PATIENT_GPU:-4}}"
  local py="${PYTHON3:-python3}"
  local log_dir="${PATIENT_VLLM_LOG_DIR:-${DIAGPRM_ROOT:-$(pwd)}/logs/patient_vllm/${RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
  local log_file="${PATIENT_VLLM_LOG_FILE:-${log_dir}/patient_vllm_gpu${gpu_id}_port${port}.log}"
  local max_model_len="${PATIENT_VLLM_MAX_MODEL_LEN:-4096}"
  local gpu_mem="${PATIENT_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  local enforce_eager="${PATIENT_VLLM_ENFORCE_EAGER:-1}"

  # Ray worker 进程运行在独立的网络命名空间（容器环境），无法访问宿主机的 127.0.0.1。
  # 使用宿主机在 pod 网络上的真实 IP（PATIENT_VLLM_BIND_HOST 可覆盖），
  # 同时让 vLLM 监听 0.0.0.0 使 Ray worker 可达。
  local _node_ip="${PATIENT_VLLM_NODE_IP:-}"
  if [ -z "${_node_ip}" ]; then
    read -r _node_ip _ < <(hostname -I 2>/dev/null || true)
  fi
  local _api_host="${_node_ip:-127.0.0.1}"
  export PATIENT_API_BASE="http://${_api_host}:${port}/v1"
  export PATIENT_MODEL="${model_name}"
  export PATIENT_VLLM_ENDPOINT_FILE="${PATIENT_VLLM_ENDPOINT_FILE:-${DIAGPRM_ROOT:-$(pwd)}/patient_vllm.endpoint}"
  printf '%s\n' "${PATIENT_API_BASE}" > "${PATIENT_VLLM_ENDPOINT_FILE}"
  export PATIENT_API_KEY="${PATIENT_API_KEY:-EMPTY}"

  if _patient_vllm_server_alive "${host}" "${port}"; then
    echo "[INFO] Found existing patient vLLM process at ${PATIENT_API_BASE}"
    wait_for_patient_vllm
    echo "[INFO] Reusing existing patient vLLM at ${PATIENT_API_BASE}"
    _patient_vllm_apply_training_cuda "${gpu_id}"
    return 0
  fi

  if [ ! -d "${model_path}" ]; then
    echo "[ERROR] PATIENT_VLLM_MODEL_PATH does not exist: ${model_path}" >&2
    return 1
  fi

  mkdir -p "${log_dir}"

  local extra_args=()
  if [ "${enforce_eager}" = "1" ]; then
    extra_args+=(--enforce-eager)
  fi

  echo "======================================================================"
  echo "[INFO] Starting local patient vLLM"
  echo "[INFO] Model path: ${model_path}"
  echo "[INFO] Model name: ${model_name}"
  echo "[INFO] GPU:        ${gpu_id}"
  echo "[INFO] Endpoint:   ${PATIENT_API_BASE}"
  echo "[INFO] Log file:   ${log_file}"
  echo "======================================================================"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${py}" -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --served-model-name "${model_name}" \
    --host "${host}" \
    --port "${port}" \
    --tensor-parallel-size "${PATIENT_VLLM_TP:-1}" \
    --max-model-len "${max_model_len}" \
    --gpu-memory-utilization "${gpu_mem}" \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1 &
  _DIAGPRM_PATIENT_VLLM_PID="$!"

  wait_for_patient_vllm

  _patient_vllm_apply_training_cuda "${gpu_id}"
}
