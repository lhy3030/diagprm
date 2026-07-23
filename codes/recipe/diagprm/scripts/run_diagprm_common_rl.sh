#!/usr/bin/env bash
set -euo pipefail

# Common-disease-only DiagPRM RL run.
# This is a focused setting for debugging KG-as-turn-level-reward without the
# rare-disease diagnosis-name sparsity of the full clean_v2 KG.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGPRM_ROOT="${DIAGPRM_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CODES_DIR="${CODES_DIR:-${DIAGPRM_ROOT}/codes}"
_RUN_TIMESTAMP="${_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_ID:-diagprm_common_rl}}"
if [[ "${RUN_NAME}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || [[ "${RUN_NAME}" == *"_"[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] ]]; then
  RUN_ID="${RUN_NAME}"
else
  RUN_ID="${RUN_NAME}_${_RUN_TIMESTAMP}"
fi
export RUN_NAME RUN_ID _RUN_TIMESTAMP

export DIAGPRM_DATASET="${DIAGPRM_DATASET:-${DIAGPRM_ROOT}/diagprm_dataset/common_disease_rl_v1}"
export KG_PATH="${KG_PATH:-${DIAGPRM_DATASET}/clean_master_kg.json}"

# Common-only RL should force shorter consultations and stronger diagnosis
# pressure than the full rare-disease run.
export MAX_TURNS_OVERRIDE="${MAX_TURNS_OVERRIDE:-6}"
export ALPHA_OVERRIDE="${ALPHA_OVERRIDE:-0.5}"
export TERMINAL_DIAG_COEF_OVERRIDE="${TERMINAL_DIAG_COEF_OVERRIDE:-1.2}"
export WEIGHTED_OVERRIDE="${WEIGHTED_OVERRIDE:-false}"
export R_MAX_OVERRIDE="${R_MAX_OVERRIDE:-1.0}"
export R_WRONG_DIAG_OVERRIDE="${R_WRONG_DIAG_OVERRIDE:-0.0}"
export R_TIMEOUT_OVERRIDE="${R_TIMEOUT_OVERRIDE:--2.0}"
export MIN_NEW_FACTS_FOR_DIAGNOSIS_OVERRIDE="${MIN_NEW_FACTS_FOR_DIAGNOSIS_OVERRIDE:-2}"
export FORCE_DIAGNOSE_ON_LAST_TURN="${FORCE_DIAGNOSE_ON_LAST_TURN:-1}"

export N_RESP_PER_PROMPT_OVERRIDE="${N_RESP_PER_PROMPT_OVERRIDE:-8}"
export N_GPUS="${N_GPUS:-4}"

export SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-${DIAGPRM_ROOT}/checkpoints/diagprm_common_rl/${RUN_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CODES_DIR}/outputs/diagprm_common_rl/${RUN_ID}}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${CODES_DIR}/logs/tensorboard/diagprm_common_rl/${RUN_ID}}"
export RESUME_MODE="${RESUME_MODE:-disable}"

if [ ! -f "${DIAGPRM_DATASET}/diagprm_train.parquet" ]; then
  echo "[ERROR] Missing common RL train file: ${DIAGPRM_DATASET}/diagprm_train.parquet" >&2
  exit 1
fi

if [ ! -f "${KG_PATH}" ]; then
  echo "[ERROR] Missing KG file: ${KG_PATH}" >&2
  exit 1
fi

echo "======================================================================"
echo "[INFO] DiagPRM common-disease RL"
echo "[INFO] Run name:        ${RUN_NAME}"
echo "[INFO] Run id:          ${RUN_ID}"
echo "[INFO] Dataset:         ${DIAGPRM_DATASET}"
echo "[INFO] KG:              ${KG_PATH}"
echo "[INFO] Actor load:      ${ACTOR_LOAD:-default in run_diagprm.sh}"
echo "[INFO] GPUs:            ${N_GPUS}"
echo "[INFO] Max turns:       ${MAX_TURNS_OVERRIDE}"
echo "[INFO] Force final dx:  ${FORCE_DIAGNOSE_ON_LAST_TURN}"
echo "[INFO] Alpha/omega:     ${ALPHA_OVERRIDE}  (A = A_diag + omega * A_turn)"
echo "[INFO] Terminal dx coef: ${TERMINAL_DIAG_COEF_OVERRIDE}"
echo "[INFO] Weighted KG:     ${WEIGHTED_OVERRIDE}"
echo "[INFO] R max/wrong/to:  ${R_MAX_OVERRIDE}/${R_WRONG_DIAG_OVERRIDE}/${R_TIMEOUT_OVERRIDE}"
echo "[INFO] Output dir:      ${OUTPUT_DIR}"
echo "[INFO] Checkpoint dir:  ${SAVE_CHECKPOINT_DIR}"
echo "======================================================================"

cd "${CODES_DIR}"
exec bash recipe/diagprm/run_diagprm.sh "$@"
