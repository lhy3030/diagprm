#!/usr/bin/env bash
set -euo pipefail

# SFT for AtomicFact/DiagPRM with NO think blocks.
#
# Differences from run_atomic_fact_sft_qwen3.sh:
#   1. Uses build_atomic_fact_sft_nothink.py to strip <think>...</think> from
#      every prompt/response and inject /no_think into the system turn.
#   2. Keeps ALL turn types (Question + Final Answer), not final-only.
#   3. Enable thinking is NOT set — model trains in pure no-think mode.
#
# Usage:
#   # All defaults (1.7B, 4 GPUs):
#   bash run_atomic_fact_sft_nothink.sh
#
#   # 4B model, 8 GPUs:
#   MODEL_PATH=/path/to/Qwen3-4B NPROC_PER_NODE=8 bash run_atomic_fact_sft_nothink.sh
#
#   # Final-answer-only rows (like the original script):
#   FINAL_ONLY=1 bash run_atomic_fact_sft_nothink.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGPRM_ROOT="${DIAGPRM_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CODES_DIR="${CODES_DIR:-${DIAGPRM_ROOT}/codes}"

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

# ── Model ──────────────────────────────────────────────────────────────────
if [ -d "/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B" ]; then
  DEFAULT_MODEL_PATH="/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B"
else
  DEFAULT_MODEL_PATH="${DIAGPRM_ROOT}/../base_model/Qwen3-1.7B"
fi
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

# ── Source data ────────────────────────────────────────────────────────────
RAW_JSONL="${RAW_JSONL:-${DIAGPRM_ROOT}/dataset/sft_training_data.jsonl}"
if [ ! -f "${RAW_JSONL}" ] && [ -f "${DIAGPRM_ROOT}/../ATPO-main/dataset/sft_training_data.jsonl" ]; then
  RAW_JSONL="${DIAGPRM_ROOT}/../ATPO-main/dataset/sft_training_data.jsonl"
fi

# ── Output data dir ────────────────────────────────────────────────────────
# FINAL_ONLY=1 → only Final Answer rows (fewer samples, cleaner signal)
# FINAL_ONLY=0 → all rows (Question + Final Answer, more data)
FINAL_ONLY="${FINAL_ONLY:-0}"
if [ "${FINAL_ONLY}" = "1" ]; then
  DEFAULT_SFT_DIR="${DIAGPRM_ROOT}/diagprm_dataset/atomic_fact_sft_nothink_finalonly"
  DEFAULT_TRAIN_NAME="atomic_fact_sft_nothink_finalonly_train.parquet"
  DEFAULT_VAL_NAME="atomic_fact_sft_nothink_finalonly_val.parquet"
  EXTRA_BUILD_ARGS="--final_only"
else
  DEFAULT_SFT_DIR="${DIAGPRM_ROOT}/diagprm_dataset/atomic_fact_sft_nothink"
  DEFAULT_TRAIN_NAME="atomic_fact_sft_nothink_train.parquet"
  DEFAULT_VAL_NAME="atomic_fact_sft_nothink_val.parquet"
  EXTRA_BUILD_ARGS=""
fi
SFT_DIR="${SFT_DIR:-${DEFAULT_SFT_DIR}}"
TRAIN_FILE="${TRAIN_FILE:-${SFT_DIR}/${DEFAULT_TRAIN_NAME}}"
VAL_FILE="${VAL_FILE:-${SFT_DIR}/${DEFAULT_VAL_NAME}}"

# ── Training hyper-params ──────────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
LR="${LR:-1e-5}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
LOGGER="${LOGGER:-console}"
TRUNCATION="${TRUNCATION:-error}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"

_MODEL_TAG="$(basename "${MODEL_PATH}")"
_TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${_MODEL_TAG}_atomic_fact_sft_nothink_${_TS}}"
PROJECT_NAME="${PROJECT_NAME:-atomic-fact-sft-nothink}"
SAVE_DIR="${SAVE_DIR:-${DIAGPRM_ROOT}/checkpoints/${EXP_NAME}}"

# ── Build dataset if needed ────────────────────────────────────────────────
if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VAL_FILE}" ]; then
  if [ ! -f "${RAW_JSONL}" ]; then
    echo "[ERROR] RAW_JSONL does not exist: ${RAW_JSONL}" >&2
    exit 1
  fi
  mkdir -p "${SFT_DIR}"
  echo "[INFO] Building no-think SFT dataset from ${RAW_JSONL} ..."
  python3 "${SCRIPT_DIR}/build_atomic_fact_sft_nothink.py" \
    --input_jsonl "${RAW_JSONL}" \
    --output_dir "${SFT_DIR}" \
    --train_name "$(basename "${TRAIN_FILE}")" \
    --val_name "$(basename "${VAL_FILE}")" \
    ${EXTRA_BUILD_ARGS}
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"
cd "${CODES_DIR}"
export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"

CMD=(
  torchrun
  --standalone
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC_PER_NODE}"
  -m verl.trainer.fsdp_sft_trainer
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.custom_cls.path=recipe/diagprm/atomic_fact_sft_dataset.py
  data.custom_cls.name=AtomicFactRawSFTDataset
  data.prompt_key=prompt
  data.response_key=response
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
  ulysses_sequence_parallel_size=1
  use_remove_padding="${USE_REMOVE_PADDING}"
)

if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
  CMD+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

echo "======================================================================"
echo "[INFO] AtomicFact No-Think SFT"
echo "[INFO] Train file: ${TRAIN_FILE}"
echo "[INFO] Val file:   ${VAL_FILE}"
echo "[INFO] Model:      ${MODEL_PATH}"
echo "[INFO] Save dir:   ${SAVE_DIR}"
echo "[INFO] Epochs:     ${TOTAL_EPOCHS}"
echo "[INFO] Final-only: ${FINAL_ONLY}"
echo "[INFO] Data:       think blocks removed; /no_think in system; all turns"
echo "======================================================================"
printf '[INFO] Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "======================================================================"

"${CMD[@]}"
