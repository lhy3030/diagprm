#!/usr/bin/env bash
set -euo pipefail

# Evaluate an AtomicFact/ATPO checkpoint on held-out MedQA / MedMCQA /
# MedicalExam splits and report final-answer accuracy, matching the ATPO paper
# table style.
#
# Usage:
#   cd /home/ubuntu/liutianshuo/diagprm/codes/recipe/diagprm/scripts
#   CKPT_ROOT=/home/ubuntu/liutianshuo/diagprm/checkpoints/atomic_fact_rl/xxx \
#   BASE_MODEL=/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B \
#   bash run_eval_atomic_fact_checkpoint.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGPRM_ROOT="${DIAGPRM_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CODES_DIR="${CODES_DIR:-${DIAGPRM_ROOT}/codes}"

CKPT_ROOT="${CKPT_ROOT:-}"
BASE_MODEL="${BASE_MODEL:-/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B}"
MODEL_DIR="${MODEL_DIR:-}"
PYTHON3="${PYTHON3:-python3}"

DATASET_DIR="${ATOMIC_FACT_DATASET:-${DIAGPRM_ROOT}/diagprm_dataset/atomic_fact_rl_v1}"
SPLITS="${SPLITS:-medqa medmcqa medicalexam}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PROMPT_STYLE="${PROMPT_STYLE:-dataset}"
PATIENT_MODE="${PATIENT_MODE:-llm}"
DOCTOR_THINKING="${DOCTOR_THINKING:-0}"

RUN_ID="${RUN_ID:-atomic_fact_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${DIAGPRM_ROOT}/codes/outputs/atomic_fact_eval/${RUN_ID}}"

export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"

if [ -z "${MODEL_DIR}" ]; then
  if [ -z "${CKPT_ROOT}" ]; then
    echo "[ERROR] Set CKPT_ROOT=/path/to/checkpoint root or MODEL_DIR=/path/to/merged_hf model" >&2
    exit 1
  fi
  if [ ! -d "${CKPT_ROOT}" ]; then
    echo "[ERROR] CKPT_ROOT does not exist: ${CKPT_ROOT}" >&2
    exit 1
  fi

  if [ -f "${CKPT_ROOT}/latest_checkpointed_iteration.txt" ]; then
    latest_step="$(tr -d '[:space:]' < "${CKPT_ROOT}/latest_checkpointed_iteration.txt")"
    CKPT_DIR="${CKPT_ROOT}/global_step_${latest_step}"
  else
    CKPT_DIR="$(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)"
  fi

  if [ -z "${CKPT_DIR:-}" ] || [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] Could not find global_step_* under ${CKPT_ROOT}" >&2
    exit 1
  fi

  # merge 到对应 step 目录下，便于区分不同 checkpoint
  MODEL_DIR="${CKPT_DIR}/merged_hf"
  if [ ! -f "${MODEL_DIR}/config.json" ]; then
    mkdir -p "${MODEL_DIR}"
    cd "${CODES_DIR}"
    echo "[INFO] Merging FSDP checkpoint to HuggingFace format..."
    # actor weights are under global_step_XXX/actor/
    _ACTOR_DIR="${CKPT_DIR}/actor"
    if [ ! -d "${_ACTOR_DIR}" ]; then
      _ACTOR_DIR="${CKPT_DIR}"
    fi
    echo "[INFO] Using actor dir: ${_ACTOR_DIR}"
    echo "[INFO] Target model dir: ${MODEL_DIR}"
    "${PYTHON3}" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${_ACTOR_DIR}" \
      --target_dir "${MODEL_DIR}" \
      --trust-remote-code
  else
    echo "[INFO] Reusing merged model: ${MODEL_DIR}"
  fi
fi

if [ -f "${MODEL_DIR}/lora_adapter/adapter_config.json" ]; then
  "${PYTHON3}" - <<PY
import json
from pathlib import Path
p = Path("${MODEL_DIR}") / "lora_adapter" / "adapter_config.json"
data = json.loads(p.read_text())
if data.get("lora_alpha", 0) == 0:
    data["lora_alpha"] = int("${LORA_ALPHA:-16}")
    p.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\\n")
    print(f"[INFO] Patched LoRA alpha in {p} to {data['lora_alpha']}")
PY
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "[ERROR] MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi
if [ ! -d "${BASE_MODEL}" ] && [ -d "${MODEL_DIR}/lora_adapter" ]; then
  echo "[ERROR] BASE_MODEL does not exist and MODEL_DIR contains LoRA adapter: ${BASE_MODEL}" >&2
  exit 1
fi
if [ ! -d "${DATASET_DIR}" ]; then
  echo "[ERROR] DATASET_DIR does not exist: ${DATASET_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON3}" "${SCRIPT_DIR}/eval_atomic_fact_checkpoint.py"
  --model_dir "${MODEL_DIR}"
  --base_model "${BASE_MODEL}"
  --dataset_dir "${DATASET_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --max_turns "${MAX_TURNS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --prompt_style "${PROMPT_STYLE}"
  --doctor_enable_thinking "${DOCTOR_THINKING}"
  --patient_mode "${PATIENT_MODE}"
)

CMD+=(--splits)
for split in ${SPLITS}; do
  CMD+=("${split}")
done
if [ -n "${MAX_SAMPLES}" ]; then
  CMD+=(--max_samples "${MAX_SAMPLES}")
fi

echo "======================================================================"
echo "[INFO] AtomicFact paper-style evaluation"
echo "[INFO] Model dir:     ${MODEL_DIR}"
echo "[INFO] Base model:    ${BASE_MODEL}"
echo "[INFO] Dataset dir:   ${DATASET_DIR}"
echo "[INFO] Splits:        ${SPLITS}"
echo "[INFO] Patient mode:  ${PATIENT_MODE}"
echo "[INFO] Prompt style:  ${PROMPT_STYLE}"
echo "[INFO] Doctor think:  ${DOCTOR_THINKING}"
echo "[INFO] Output dir:    ${OUTPUT_DIR}"
echo "======================================================================"
printf '[INFO] Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "======================================================================"

"${CMD[@]}"
