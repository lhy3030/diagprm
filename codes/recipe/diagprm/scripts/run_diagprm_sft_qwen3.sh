#!/usr/bin/env bash
set -euo pipefail

# Multi-turn SFT for DiagPRM diagnostic dialogue trajectories.
# Model is selected via MODEL_PATH (default: Qwen3-1.7B).
#
# Default remote usage:
#   cd /home/ubuntu/liutianshuo/diagprm/codes/recipe/diagprm/scripts
#   nohup bash run_diagprm_sft_qwen3.sh \
#     > /home/ubuntu/liutianshuo/diagprm/diagprm_dataset/diagprm_sft.log 2>&1 &
#
# Select a different model:
#   MODEL_PATH=/home/ubuntu/liutianshuo/base_model/Qwen3-8B bash run_diagprm_sft_qwen3.sh
#   MODEL_PATH=/home/ubuntu/liutianshuo/base_model/Qwen3-4B bash run_diagprm_sft_qwen3.sh
#
# Smoke test:
#   TOTAL_TRAINING_STEPS=5 SAVE_FREQ=-1 bash run_diagprm_sft_qwen3.sh
#
# If the stable LoRA defaults work and you want to try full-parameter SFT later:
#   LORA_RANK=0 bash run_diagprm_sft_qwen3.sh
#
# If full-parameter SFT works and you want to try faster kernels later:
#   LORA_RANK=0 FSDP_STRATEGY=fsdp2 USE_REMOVE_PADDING=true bash run_diagprm_sft_qwen3.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGPRM_ROOT="${DIAGPRM_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CODES_DIR="${CODES_DIR:-${DIAGPRM_ROOT}/codes}"

# Mirror the CUDA/NCCL compatibility settings used by recipe/diagprm/run_diagprm.sh.
# This is important on Blackwell/RTX 5090 machines where PyTorch's bundled NCCL
# can hit illegal-memory-access failures during distributed setup.
_NCCL_SO="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)/nvidia/nccl/lib/libnccl.so.2"
if [ -f "${_NCCL_SO}" ]; then
  export LD_PRELOAD="${_NCCL_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "[INFO] Using NCCL: ${_NCCL_SO}"
fi
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"

_RUN_CACHE_BASE="/data/run01/${USER:-$(whoami 2>/dev/null || echo user)}"
if [ -d "${_RUN_CACHE_BASE}" ] || mkdir -p "${_RUN_CACHE_BASE}" 2>/dev/null; then
  _DEFAULT_CACHE_BASE="${_RUN_CACHE_BASE}/.cache"
  _DEFAULT_CONFIG_HOME="${_RUN_CACHE_BASE}/.config"
  _DEFAULT_TRITON_CACHE="${_RUN_CACHE_BASE}/.triton"
else
  _DEFAULT_CACHE_BASE="${DIAGPRM_ROOT}/.cache"
  _DEFAULT_CONFIG_HOME="${DIAGPRM_ROOT}/.config"
  _DEFAULT_TRITON_CACHE="${DIAGPRM_ROOT}/.triton"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_DEFAULT_TRITON_CACHE}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_DEFAULT_CACHE_BASE}}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${_DEFAULT_CONFIG_HOME}}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${TRITON_CACHE_DIR}" "${HF_HOME}" "${XDG_CONFIG_HOME}"

DEFAULT_DATA_DIR="${DIAGPRM_ROOT}/diagprm_dataset/clean_v2_sft_vllm/merged_20260627_20260629"
TRAIN_FILE="${TRAIN_FILE:-${DEFAULT_DATA_DIR}/diagprm_sft_merged_filtered.parquet}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"

# Resolve default model path: prefer remote server path, fallback to relative
if [ -d "/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B" ]; then
  DEFAULT_MODEL_PATH="/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B"
else
  DEFAULT_MODEL_PATH="${DIAGPRM_ROOT}/../base_model/Qwen3-1.7B"
fi

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

# Derive a short model tag from the last component of MODEL_PATH (e.g. "Qwen3-1.7B")
_MODEL_TAG="$(basename "${MODEL_PATH}")"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"

_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${_MODEL_TAG}_diagprm_sft_${_TIMESTAMP}}"
PROJECT_NAME="${PROJECT_NAME:-diagprm-sft}"
SAVE_DIR="${SAVE_DIR:-${DIAGPRM_ROOT}/checkpoints/${EXP_NAME}}"

MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
LR="${LR:-1e-5}"
# Auto-compute SAVE_FREQ as steps-per-epoch so checkpoint is saved once per epoch.
# Falls back to 100 if the parquet file is unreadable (e.g. missing dependency).
if [ -z "${SAVE_FREQ:-}" ]; then
  _N_SAMPLES="$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${TRAIN_FILE}')))" 2>/dev/null || echo "")"
  if [ -n "${_N_SAMPLES}" ] && [ "${_N_SAMPLES}" -gt 0 ] 2>/dev/null; then
    SAVE_FREQ=$(( (_N_SAMPLES + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
    echo "[INFO] Auto SAVE_FREQ=${SAVE_FREQ} (${_N_SAMPLES} samples / batch ${TRAIN_BATCH_SIZE} = 1 ckpt/epoch)"
  else
    SAVE_FREQ=100
    echo "[WARN] Could not read TRAIN_FILE for auto SAVE_FREQ, using default ${SAVE_FREQ}"
  fi
fi
TEST_FREQ="${TEST_FREQ:--1}"
LOGGER="${LOGGER:-console}"
TRUNCATION="${TRUNCATION:-error}"

MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

if [ ! -f "${TRAIN_FILE}" ]; then
  echo "[ERROR] TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
  exit 1
fi

if [ ! -f "${VAL_FILE}" ]; then
  echo "[ERROR] VAL_FILE does not exist: ${VAL_FILE}" >&2
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"
cd "${CODES_DIR}"

export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"
VERL_LOCATION=$(python3 -c "import verl; print(verl.__file__)" 2>/dev/null || true)
if [ -z "${VERL_LOCATION}" ] || [[ "${VERL_LOCATION}" != "${CODES_DIR}"* ]]; then
  echo "[INFO] verl current path: ${VERL_LOCATION:-not installed}"
  echo "[INFO] Installing local verl source with pip install -e ..."
  pip install -e "${CODES_DIR}" --quiet --no-build-isolation
fi

CMD=(
  torchrun
  --standalone
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC_PER_NODE}"
  -m verl.trainer.fsdp_sft_trainer
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.multiturn.enable=true
  data.multiturn.messages_key=messages
  data.multiturn.enable_thinking_key=enable_thinking
  data.max_length="${MAX_LENGTH}"
  data.truncation="${TRUNCATION}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}"
  model.partial_pretrain="${MODEL_PATH}"
  model.trust_remote_code=true
  model.fsdp_config.model_dtype="${MODEL_DTYPE}"
  model.strategy="${FSDP_STRATEGY}"
  model.lora_rank="${LORA_RANK}"
  model.lora_alpha="${LORA_ALPHA}"
  model.target_modules=all-linear
  optim.lr="${LR}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXP_NAME}"
  trainer.default_local_dir="${SAVE_DIR}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.n_gpus_per_node="${NPROC_PER_NODE}"
  trainer.logger="${LOGGER}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.resume_mode=disable
  ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
  use_remove_padding="${USE_REMOVE_PADDING}"
)

if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
  CMD+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

echo "======================================================================"
echo "[INFO] DiagPRM SFT  (model: ${_MODEL_TAG})"
echo "[INFO] DiagPRM root:      ${DIAGPRM_ROOT}"
echo "[INFO] Codes dir:         ${CODES_DIR}"
echo "[INFO] Train file:        ${TRAIN_FILE}"
echo "[INFO] Val file:          ${VAL_FILE}"
echo "[INFO] Model path:        ${MODEL_PATH}"
echo "[INFO] Save dir:          ${SAVE_DIR}"
echo "[INFO] GPUs/processes:    ${NPROC_PER_NODE}"
echo "[INFO] Max length:        ${MAX_LENGTH}"
echo "[INFO] Train batch size:  ${TRAIN_BATCH_SIZE}"
echo "[INFO] Micro batch/GPU:   ${MICRO_BATCH_SIZE_PER_GPU}"
echo "[INFO] Epochs:            ${TOTAL_EPOCHS}"
echo "[INFO] FSDP strategy:     ${FSDP_STRATEGY}"
echo "[INFO] Remove padding:    ${USE_REMOVE_PADDING}"
echo "[INFO] LoRA rank:         ${LORA_RANK}"
echo "[INFO] NCCL NVLS enable:  ${NCCL_NVLS_ENABLE}"
echo "[INFO] NCCL IB disabled:  ${NCCL_IB_DISABLE}"
echo "[INFO] NCCL P2P disabled: ${NCCL_P2P_DISABLE}"
echo "[INFO] Thinking mode:     read enable_thinking column; current merged data is expected to be false"
echo "======================================================================"
printf '[INFO] Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "======================================================================"

"${CMD[@]}"
