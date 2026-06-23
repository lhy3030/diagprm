#!/usr/bin/env bash
set -euo pipefail

# Prepare the naqi4 server for DiagPRM clean_v2 training.
#
# Usage from the local machine:
#   SSHPASS='liutianshuo' bash scripts/prepare_naqi4_clean_v2.sh
#
# To start training after checks pass:
#   SSHPASS='liutianshuo' START_TRAINING=1 bash scripts/prepare_naqi4_clean_v2.sh

REMOTE_HOST="${REMOTE_HOST:-116.172.93.169}"
REMOTE_PORT="${REMOTE_PORT:-63812}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/liutianshuo/diagprm}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-diagprm}"
REMOTE_TMUX_SESSION="${REMOTE_TMUX_SESSION:-diagprm_train}"

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATASET="${LOCAL_DATASET:-${LOCAL_REPO}/diagprm_dataset/clean_v2}"
REMOTE_DATASET="${REMOTE_REPO}/diagprm_dataset/clean_v2"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/diagprm_naqi4_known_hosts -p "${REMOTE_PORT}")

if ! command -v sshpass >/dev/null 2>&1; then
  echo "[ERROR] sshpass is required. Install it or run equivalent ssh/rsync commands manually."
  exit 1
fi

if [ -z "${SSHPASS:-}" ]; then
  echo "[ERROR] Set SSHPASS to the remote SSH password."
  exit 1
fi

required_files=(
  clean_master_kg.json
  diagprm_train.parquet
  diagprm_val.parquet
  diagprm_test.parquet
  split_manifest.json
)

for file in "${required_files[@]}"; do
  if [ ! -f "${LOCAL_DATASET}/${file}" ]; then
    echo "[ERROR] Missing local clean_v2 file: ${LOCAL_DATASET}/${file}"
    exit 1
  fi
done

remote_ssh() {
  sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

echo "[INFO] Checking remote repository..."
remote_ssh "bash -lc '
set -euo pipefail
test -d \"${REMOTE_REPO}\"
cd \"${REMOTE_REPO}\"
git status --short --branch
if [ -n \"\$(git status --porcelain)\" ] && [ \"${ALLOW_DIRTY_REMOTE:-0}\" != \"1\" ]; then
  echo \"[ERROR] Remote repo has uncommitted changes. Set ALLOW_DIRTY_REMOTE=1 to continue.\"
  exit 2
fi
git pull --ff-only || true
'"

echo "[INFO] Syncing clean_v2 dataset to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATASET}/ ..."
sshpass -e ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_DATASET}'"
sshpass -e rsync -av --delete -e "ssh ${SSH_OPTS[*]}" "${LOCAL_DATASET}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATASET}/"

echo "[INFO] Running remote data/code checks..."
remote_ssh "bash -lc '
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate \"${REMOTE_CONDA_ENV}\"
cd \"${REMOTE_REPO}\"
export DIAGPRM_DATASET=\"${REMOTE_DATASET}\"
export KG_PATH=\"${REMOTE_DATASET}/clean_master_kg.json\"
export PYTHONPYCACHEPREFIX=/tmp/diagprm_pycache

python - <<\"PY\"
import json
from pathlib import Path
import pandas as pd

root = Path(\"${REMOTE_DATASET}\")
for name in [\"clean_master_kg.json\", \"diagprm_train.parquet\", \"diagprm_val.parquet\", \"diagprm_test.parquet\"]:
    p = root / name
    assert p.exists(), f\"missing {p}\"

df = pd.read_parquet(root / \"diagprm_train.parquet\")
assert len(df) > 0, \"empty train parquet\"
rm = json.loads(df.iloc[0][\"reward_model\"])[\"ground_truth\"]
assert \"symptom_facts\" in rm and rm[\"symptom_facts\"], \"symptom_facts missing\"
assert rm[\"symptom_facts\"][0][\"fact_id\"].startswith(\"F\"), \"bad fact_id schema\"
print({\"train_rows\": len(df), \"first_disease\": rm[\"disease\"], \"first_fact\": rm[\"symptom_facts\"][0]})
PY

python -m py_compile \
  codes/recipe/diagprm/diagprm_agent_loop.py \
  codes/recipe/diagprm/diagprm_reward_fn.py \
  codes/recipe/diagprm/diagprm_reward_manager.py \
  codes/recipe/diagprm/prompts/patient.py

bash -n codes/recipe/diagprm/run_diagprm.sh
python - <<\"PY\"
from recipe.diagprm.diagprm_reward_fn import compute_episode_rewards_from_history
kg = {\"flu\": {\"fever\": 0.5, \"cough\": 0.5}}
history = [
    {\"doctor_response\": \"{\\\"hypothesis\\\":\\\"flu\\\",\\\"action\\\":\\\"continue\\\",\\\"question\\\":\\\"Do you have fever?\\\"}\", \"patient_answer\": \"Yes\", \"patient_fact_id\": \"F000\"},
    {\"doctor_response\": \"{\\\"hypothesis\\\":\\\"flu\\\",\\\"action\\\":\\\"continue\\\",\\\"question\\\":\\\"Do you have cough?\\\"}\", \"patient_answer\": \"Yes\", \"patient_fact_id\": \"F001\"},
    {\"doctor_response\": \"{\\\"hypothesis\\\":\\\"flu\\\",\\\"action\\\":\\\"diagnose\\\",\\\"diagnosis\\\":\\\"flu\\\"}\", \"patient_answer\": \"\", \"patient_fact_id\": \"unknown\", \"is_final\": True},
]
params = {
    \"beta\": 1.0, \"gamma1\": 0.3, \"turn_coef\": 1.0, \"r_max\": 2.0, \"tau\": 0.5,
    \"format_score\": 0.1, \"weighted\": True, \"evidence_gated_hyp\": True,
    \"wrong_hyp_penalty_scale\": 0.5, \"r_wrong_diag\": -1.0, \"r_timeout\": -1.0,
    \"unknown_penalty\": -0.05, \"duplicate_penalty\": -0.05,
}
rewards, rdiag, details = compute_episode_rewards_from_history(
    history, \"flu\", kg, params, initial_symptoms=[\"fever\"],
    fact_id_to_text={\"F000\": \"fever\", \"F001\": \"cough\"},
)
assert details[0][\"delta_kg\"] == 0.0, details
assert details[1][\"delta_kg\"] > 0.0, details
assert rdiag[-1] == 2.0, rdiag
print({\"reward_sanity\": rewards, \"rdiag\": rdiag})
PY
'"

if [ "${START_TRAINING:-0}" = "1" ]; then
  echo "[INFO] Starting training in tmux session ${REMOTE_TMUX_SESSION} ..."
  remote_ssh "bash -lc '
set -euo pipefail
if tmux has-session -t \"${REMOTE_TMUX_SESSION}\" 2>/dev/null; then
  echo \"[ERROR] tmux session ${REMOTE_TMUX_SESSION} already exists. Attach with: tmux attach -t ${REMOTE_TMUX_SESSION}\"
  exit 3
fi
mkdir -p \"${REMOTE_REPO}/logs\"
tmux new-session -d -s \"${REMOTE_TMUX_SESSION}\" \"bash -lc \\\"source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate ${REMOTE_CONDA_ENV}; cd ${REMOTE_REPO}; export DIAGPRM_DATASET=${REMOTE_DATASET}; export KG_PATH=${REMOTE_DATASET}/clean_master_kg.json; bash codes/recipe/diagprm/run_diagprm.sh 2>&1 | tee logs/diagprm_train_\\\$(date +%Y%m%d_%H%M%S).log\\\"\"
tmux ls
'"
  echo "[INFO] Training launched. Attach with:"
  echo "  ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}"
  echo "  tmux attach -t ${REMOTE_TMUX_SESSION}"
else
  echo "[INFO] Checks passed. To launch training, rerun with START_TRAINING=1."
fi
