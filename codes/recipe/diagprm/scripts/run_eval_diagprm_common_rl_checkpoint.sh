#!/usr/bin/env bash
set -euo pipefail

# Evaluate a DiagPRM common-disease RL checkpoint on validation and/or test.
#
# Remote example:
#   cd /home/ubuntu/liutianshuo/diagprm/codes/recipe/diagprm/scripts
#   CKPT_ROOT=/home/ubuntu/liutianshuo/diagprm/checkpoints/diagprm_common_rl/20260703_183633 \
#   CKPT_STEP=210 \
#   BASE_MODEL=/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B \
#   SPLIT=both \
#   N_GPUS=4 \
#   bash run_eval_diagprm_common_rl_checkpoint.sh
#
# Notes:
#   - SPLIT can be val, test, or both.
#   - This script uses trainer.val_only=true, so it performs rollout evaluation
#     and reward computation without actor updates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGPRM_ROOT="${DIAGPRM_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CODES_DIR="${CODES_DIR:-${DIAGPRM_ROOT}/codes}"
PYTHON3="${PYTHON3:-python3}"

_RUN_TIMESTAMP="${_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_ID:-diagprm_common_rl_eval}}"
if [[ "${RUN_NAME}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || [[ "${RUN_NAME}" == *"_"[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] ]]; then
  RUN_ID="${RUN_NAME}"
else
  RUN_ID="${RUN_NAME}_${_RUN_TIMESTAMP}"
fi
export RUN_NAME RUN_ID _RUN_TIMESTAMP
SPLIT="${SPLIT:-both}"

export DIAGPRM_DATASET="${DIAGPRM_DATASET:-${DIAGPRM_ROOT}/diagprm_dataset/common_disease_rl_v1}"
export KG_PATH="${KG_PATH:-${DIAGPRM_DATASET}/clean_master_kg.json}"

CKPT_ROOT="${CKPT_ROOT:-${DIAGPRM_ROOT}/checkpoints/diagprm_common_rl/20260703_183633}"
CKPT_STEP="${CKPT_STEP:-}"
BASE_MODEL="${BASE_MODEL:-${DIAGPRM_ROOT}/../base_model/Qwen3-1.7B}"
MERGED_DIR="${MERGED_DIR:-${CKPT_ROOT}/merged_hf_latest}"

export N_GPUS="${N_GPUS:-4}"
export NNODES="${NNODES:-1}"
export MAX_TURNS_OVERRIDE="${MAX_TURNS_OVERRIDE:-6}"
export FORCE_DIAGNOSE_ON_LAST_TURN="${FORCE_DIAGNOSE_ON_LAST_TURN:-1}"
export PATIENT_MAX_TOKENS="${PATIENT_MAX_TOKENS:-512}"

# Keep reward settings aligned with common-disease RL training.
export ALPHA_OVERRIDE="${ALPHA_OVERRIDE:-0.5}"
export TERMINAL_DIAG_COEF_OVERRIDE="${TERMINAL_DIAG_COEF_OVERRIDE:-1.2}"
export BETA_OVERRIDE="${BETA_OVERRIDE:-1.0}"
export TURN_COEF_OVERRIDE="${TURN_COEF_OVERRIDE:-0.4}"
export WEIGHTED_OVERRIDE="${WEIGHTED_OVERRIDE:-false}"
export R_MAX_OVERRIDE="${R_MAX_OVERRIDE:-1.0}"
export R_WRONG_DIAG_OVERRIDE="${R_WRONG_DIAG_OVERRIDE:-0.0}"
export R_TIMEOUT_OVERRIDE="${R_TIMEOUT_OVERRIDE:--2.0}"
export MIN_NEW_FACTS_FOR_DIAGNOSIS_OVERRIDE="${MIN_NEW_FACTS_FOR_DIAGNOSIS_OVERRIDE:-2}"

# Validation usually uses one rollout per case. Increase this for repeated
# stochastic evaluation, e.g. VAL_N_OVERRIDE=4.
export VAL_N_OVERRIDE="${VAL_N_OVERRIDE:-1}"

export TRAIN_BATCH_SIZE_OVERRIDE="${TRAIN_BATCH_SIZE_OVERRIDE:-16}"
export PPO_MINI_BATCH_SIZE_OVERRIDE="${PPO_MINI_BATCH_SIZE_OVERRIDE:-16}"
export N_RESP_PER_PROMPT_OVERRIDE="${N_RESP_PER_PROMPT_OVERRIDE:-1}"
export AGENT_NUM_WORKERS_OVERRIDE="${AGENT_NUM_WORKERS_OVERRIDE:-${N_GPUS}}"
export INFER_TP_OVERRIDE="${INFER_TP_OVERRIDE:-1}"

export RESUME_MODE=disable
export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"

if [ ! -d "${CKPT_ROOT}" ]; then
  echo "[ERROR] CKPT_ROOT does not exist: ${CKPT_ROOT}" >&2
  exit 1
fi

if [ ! -f "${KG_PATH}" ]; then
  echo "[ERROR] KG_PATH does not exist: ${KG_PATH}" >&2
  exit 1
fi

if [ ! -f "${DIAGPRM_DATASET}/diagprm_train.parquet" ]; then
  echo "[ERROR] Missing train parquet required by trainer init: ${DIAGPRM_DATASET}/diagprm_train.parquet" >&2
  exit 1
fi

if [ -n "${CKPT_STEP}" ]; then
  CKPT_DIR="${CKPT_ROOT}/global_step_${CKPT_STEP}"
elif [ -f "${CKPT_ROOT}/latest_checkpointed_iteration.txt" ]; then
  latest_step="$(tr -d '[:space:]' < "${CKPT_ROOT}/latest_checkpointed_iteration.txt")"
  CKPT_DIR="${CKPT_ROOT}/global_step_${latest_step}"
else
  CKPT_DIR="$(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)"
fi

if [ -z "${CKPT_DIR:-}" ] || [ ! -d "${CKPT_DIR}" ]; then
  echo "[ERROR] Could not find checkpoint dir under ${CKPT_ROOT}" >&2
  exit 1
fi

echo "======================================================================"
echo "[INFO] DiagPRM common RL checkpoint evaluation"
echo "[INFO] Run name:     ${RUN_NAME}"
echo "[INFO] Run id:       ${RUN_ID}"
echo "[INFO] Split:        ${SPLIT}"
echo "[INFO] Dataset:      ${DIAGPRM_DATASET}"
echo "[INFO] KG:           ${KG_PATH}"
echo "[INFO] CKPT root:    ${CKPT_ROOT}"
echo "[INFO] CKPT dir:     ${CKPT_DIR}"
echo "[INFO] Base model:   ${BASE_MODEL}"
echo "[INFO] Merged model: ${MERGED_DIR}"
echo "[INFO] GPUs:         ${N_GPUS}"
echo "======================================================================"

if [ ! -f "${MERGED_DIR}/config.json" ]; then
  mkdir -p "${MERGED_DIR}"
  cd "${CODES_DIR}"
  echo "[INFO] Merging FSDP checkpoint to HuggingFace format..."
  "${PYTHON3}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}/actor" \
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

export ACTOR_LOAD="${MERGED_DIR}"

EVAL_ROOT="${EVAL_ROOT:-${CODES_DIR}/outputs/diagprm_common_rl_eval/${RUN_ID}}"
mkdir -p "${EVAL_ROOT}"

run_one_split() {
  local split_name="$1"
  local eval_file="${DIAGPRM_DATASET}/diagprm_${split_name}.parquet"
  local out_dir="${EVAL_ROOT}/${split_name}"

  if [ ! -f "${eval_file}" ]; then
    echo "[ERROR] Missing ${split_name} parquet: ${eval_file}" >&2
    exit 1
  fi

  mkdir -p "${out_dir}"
  export OUTPUT_DIR="${out_dir}"
  export SAVE_CHECKPOINT_DIR="${out_dir}/no_train_checkpoints"
  export TENSORBOARD_DIR="${out_dir}/tensorboard"

  echo "======================================================================"
  echo "[INFO] Evaluating split=${split_name}"
  echo "[INFO] Eval file: ${eval_file}"
  echo "[INFO] Output:    ${out_dir}"
  echo "======================================================================"

  cd "${CODES_DIR}"
  bash recipe/diagprm/run_diagprm.sh \
    data.val_files="['${eval_file}']" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N_OVERRIDE}" \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.test_freq=0 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.experiment_name="diagprm_common_rl_eval_${RUN_ID}_${split_name}" \
    trainer.rollout_data_dir=null \
    trainer.validation_data_dir="${out_dir}/validation_data"
}

case "${SPLIT}" in
  val)
    run_one_split val
    ;;
  test)
    run_one_split test
    ;;
  both)
    run_one_split val
    run_one_split test
    ;;
  *)
    echo "[ERROR] SPLIT must be val, test, or both; got: ${SPLIT}" >&2
    exit 1
    ;;
esac

echo "======================================================================"
echo "[INFO] Evaluation complete"
echo "[INFO] Eval root: ${EVAL_ROOT}"
echo "[INFO] Validation generations are under:"
echo "[INFO]   ${EVAL_ROOT}/val/validation_data"
echo "[INFO]   ${EVAL_ROOT}/test/validation_data"
echo "======================================================================"
