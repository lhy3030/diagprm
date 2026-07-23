#!/usr/bin/env bash
set -euo pipefail

# Merge a verl FSDP AtomicFact/ATPO SFT checkpoint if needed, then run a few
# generation format tests.
#
# Usage:
#   cd /home/ubuntu/liutianshuo/diagprm/codes/recipe/diagprm/scripts
#   CKPT_ROOT=/home/ubuntu/liutianshuo/diagprm/checkpoints/Qwen3-1.7B_atomic_fact_sft_xxx \
#   BASE_MODEL=/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B \
#   bash run_test_atomic_fact_sft_checkpoint.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGPRM_ROOT="${DIAGPRM_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CODES_DIR="${CODES_DIR:-${DIAGPRM_ROOT}/codes}"

CKPT_ROOT="${CKPT_ROOT:-${DIAGPRM_ROOT}/checkpoints/Qwen3-1.7B_atomic_fact_sft}"
BASE_MODEL="${BASE_MODEL:-/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B}"
MERGED_DIR="${MERGED_DIR:-${CKPT_ROOT}/merged_hf_latest}"
PYTHON3="${PYTHON3:-python3}"

SFT_JSONL="${SFT_JSONL:-${DIAGPRM_ROOT}/dataset/sft_training_data.jsonl}"
if [ ! -f "${SFT_JSONL}" ] && [ -f "${DIAGPRM_ROOT}/../ATPO-main/dataset/sft_training_data.jsonl" ]; then
  SFT_JSONL="${DIAGPRM_ROOT}/../ATPO-main/dataset/sft_training_data.jsonl"
fi
NUM_CASES="${NUM_CASES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"

export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"

if [ ! -d "${CKPT_ROOT}" ]; then
  echo "[ERROR] CKPT_ROOT does not exist: ${CKPT_ROOT}" >&2
  exit 1
fi

if [ ! -d "${BASE_MODEL}" ]; then
  echo "[ERROR] BASE_MODEL does not exist: ${BASE_MODEL}" >&2
  exit 1
fi
if [ ! -f "${SFT_JSONL}" ]; then
  echo "[ERROR] SFT_JSONL does not exist: ${SFT_JSONL}" >&2
  exit 1
fi

if [ -f "${CKPT_ROOT}/latest_checkpointed_iteration.txt" ]; then
  latest_step="$(tr -d '[:space:]' < "${CKPT_ROOT}/latest_checkpointed_iteration.txt")"
  CKPT_DIR="${CKPT_ROOT}/global_step_${latest_step}"
else
  CKPT_DIR="$(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)"
fi

if [ -z "${CKPT_DIR:-}" ] || [ ! -d "${CKPT_DIR}" ]; then
  echo "[ERROR] Could not find global_step_* checkpoint under ${CKPT_ROOT}" >&2
  exit 1
fi

echo "======================================================================"
echo "[INFO] AtomicFact SFT checkpoint test"
echo "[INFO] Checkpoint root: ${CKPT_ROOT}"
echo "[INFO] Checkpoint dir:  ${CKPT_DIR}"
echo "[INFO] Base model:      ${BASE_MODEL}"
echo "[INFO] Merged dir:      ${MERGED_DIR}"
echo "[INFO] SFT jsonl:       ${SFT_JSONL}"
echo "======================================================================"

if [ ! -f "${MERGED_DIR}/config.json" ]; then
  mkdir -p "${MERGED_DIR}"
  cd "${CODES_DIR}"
  echo "[INFO] Merging FSDP checkpoint to HuggingFace format..."
  "${PYTHON3}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${MERGED_DIR}" \
    --trust-remote-code
else
  echo "[INFO] Reusing existing merged model: ${MERGED_DIR}"
fi

if [ -f "${MERGED_DIR}/lora_adapter/adapter_config.json" ]; then
  "${PYTHON3}" - <<PY
import json
from pathlib import Path
p = Path("${MERGED_DIR}") / "lora_adapter" / "adapter_config.json"
data = json.loads(p.read_text())
if data.get("lora_alpha", 0) == 0:
    data["lora_alpha"] = int("${LORA_ALPHA:-16}")
    p.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\\n")
    print(f"[INFO] Patched LoRA alpha in {p} to {data['lora_alpha']}")
PY
fi

echo "[INFO] Running AtomicFact generation smoke test..."
"${PYTHON3}" "${SCRIPT_DIR}/test_atomic_fact_sft_checkpoint.py" \
  --model_dir "${MERGED_DIR}" \
  --base_model "${BASE_MODEL}" \
  --sft_jsonl "${SFT_JSONL}" \
  --num_cases "${NUM_CASES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}"
