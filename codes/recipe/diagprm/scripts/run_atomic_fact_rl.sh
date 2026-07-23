#!/usr/bin/env bash
set -euo pipefail

# Our AtomicFact/DiagPRM RL in the ATPO medical multi-turn setting.
# Task: Question / Final Answer medical QA with hidden atomic facts as
# turn-level reward signal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODES_DIR="$(cd "${RECIPE_DIR}/../.." && pwd)"
DIAGPRM_ROOT="$(cd "${CODES_DIR}/.." && pwd)"

_NCCL_SO="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)/nvidia/nccl/lib/libnccl.so.2"
if [ -f "${_NCCL_SO}" ]; then
  export LD_PRELOAD="${_NCCL_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "[INFO] Using NCCL: ${_NCCL_SO}"
fi
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

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

export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"
cd "${CODES_DIR}"

VERL_LOCATION=$(python3 -c "import verl; print(verl.__file__)" 2>/dev/null || true)
if [ -z "${VERL_LOCATION}" ] || [[ "${VERL_LOCATION}" != "${CODES_DIR}"* ]]; then
  echo "[INFO] verl not found via editable install, using PYTHONPATH=${CODES_DIR}"
  # pip install -e may fail on nodes with different conda paths; PYTHONPATH is enough
fi

# Optional local patient vLLM.
PATIENT_VLLM_UTILS="${SCRIPT_DIR}/patient_vllm_utils.sh"
if [ -f "${PATIENT_VLLM_UTILS}" ]; then
  source "${PATIENT_VLLM_UTILS}"
  start_patient_vllm_if_requested
  trap stop_patient_vllm EXIT
elif [ "${START_PATIENT_VLLM:-0}" = "1" ]; then
  echo "[ERROR] START_PATIENT_VLLM=1 but missing ${PATIENT_VLLM_UTILS}" >&2
  exit 1
fi

SFT_ACTOR="${DIAGPRM_ROOT}/checkpoints/Qwen3-1.7B_atomic_fact_sft_prompt_v2/global_step_116/merged_hf"
if [ -d "${SFT_ACTOR}" ]; then
  DEFAULT_ACTOR="${SFT_ACTOR}"
elif [ -d "/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B" ]; then
  DEFAULT_ACTOR="/home/ubuntu/liutianshuo/base_model/Qwen3-1.7B"
else
  DEFAULT_ACTOR="${DIAGPRM_ROOT}/../base_model/Qwen3-1.7B"
fi
ACTOR_LOAD="${ACTOR_LOAD:-${DEFAULT_ACTOR}}"

DATASET_DIR="${ATOMIC_FACT_DATASET:-${DIAGPRM_ROOT}/diagprm_dataset/atomic_fact_rl_v1}"
TRAIN_FILES="['${DATASET_DIR}/atomic_fact_train.parquet']"
VAL_FILES="['${DATASET_DIR}/atomic_fact_val.parquet']"

if [ ! -f "${DATASET_DIR}/atomic_fact_train.parquet" ]; then
  echo "[ERROR] Missing train parquet: ${DATASET_DIR}/atomic_fact_train.parquet" >&2
  exit 1
fi
if [ ! -d "${ACTOR_LOAD}" ]; then
  echo "[ERROR] ACTOR_LOAD does not exist: ${ACTOR_LOAD}" >&2
  exit 1
fi

NNODES="${NNODES:-1}"
N_GPUS="${N_GPUS:-4}"
NODE_RANK="${NODE_RANK:-0}"

: "${TRAIN_BATCH_SIZE_OVERRIDE:=256}"
: "${PPO_MINI_BATCH_SIZE_OVERRIDE:=256}"
: "${AGENT_NUM_WORKERS_OVERRIDE:=8}"
if [ "${N_GPUS}" -le 1 ]; then
  : "${INFER_TP_OVERRIDE:=1}"
else
  : "${INFER_TP_OVERRIDE:=2}"
fi

RUN_NAME="${RUN_NAME:-atomic_fact_rl}"
_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${RUN_NAME}_${_TS}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-./logs/tensorboard/${RUN_ID}}"
SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-./checkpoints/atomic_fact_rl/${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/atomic_fact_rl/${RUN_ID}}"

adv_estimator="${ADV_ESTIMATOR_OVERRIDE:-diagprm_grpo}"
use_kl_in_reward="${USE_KL_IN_REWARD:-False}"
use_kl_loss="${USE_KL_LOSS:-True}"
kl_loss_coef="${KL_LOSS_COEF:-0.01}"
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.28}"

max_turns="${MAX_TURNS_OVERRIDE:-10}"
max_prompt_length="${MAX_PROMPT_LENGTH_OVERRIDE:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH_OVERRIDE:-4096}"
max_turn_tokens="${ATOMIC_FACT_MAX_TURN_TOKENS:-1024}"
min_turn_tokens="${ATOMIC_FACT_MIN_TURN_TOKENS:-1}"
actor_lr="${ACTOR_LR_OVERRIDE:-1e-6}"
train_batch_size="${TRAIN_BATCH_SIZE_OVERRIDE}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE_OVERRIDE}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
n_resp_per_prompt="${N_RESP_PER_PROMPT_OVERRIDE:-8}"
n_resp_per_prompt_val="${N_RESP_PER_PROMPT_VAL_OVERRIDE:-1}"
infer_tp="${INFER_TP_OVERRIDE}"
train_sp="${TRAIN_SP_OVERRIDE:-1}"
offload="${OFFLOAD_OVERRIDE:-True}"
actor_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_OVERRIDE:-$(( (max_prompt_length + max_response_length) * n_resp_per_prompt ))}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_OVERRIDE:-${actor_max_token_len_per_gpu}}"

beta="${BETA_OVERRIDE:-1.0}"
turn_coef="${TURN_COEF_OVERRIDE:-0.4}"
r_correct="${R_CORRECT_OVERRIDE:-1.0}"
r_wrong="${R_WRONG_OVERRIDE:-0.0}"
r_timeout="${R_TIMEOUT_OVERRIDE:--2.0}"
alpha="${ALPHA_OVERRIDE:-0.5}"
terminal_diag_coef="${TERMINAL_DIAG_COEF_OVERRIDE:-1.2}"
atomic_fact_prompt_style="${ATOMIC_FACT_PROMPT_STYLE:-dataset}"
doctor_thinking="${ATOMIC_FACT_DOCTOR_THINKING:-0}"
export ATOMIC_FACT_PROMPT_STYLE="${atomic_fact_prompt_style}"
export ATOMIC_FACT_DOCTOR_THINKING="${doctor_thinking}"
export ATOMIC_FACT_MAX_TURN_TOKENS="${max_turn_tokens}"
export ATOMIC_FACT_MIN_TURN_TOKENS="${min_turn_tokens}"
export MAX_TURNS_OVERRIDE="${max_turns}"

echo "============================================================"
echo "  AtomicFact-RL Training"
echo "  Model: ${ACTOR_LOAD}"
echo "  Dataset: ${DATASET_DIR}"
echo "  Run id: ${RUN_ID}"
echo "  GPUs: ${N_GPUS}, G=${n_resp_per_prompt}, batch=${train_batch_size}"
echo "  Max turns: ${max_turns}"
echo "  Max doctor tokens/turn: ${max_turn_tokens}"
echo "  Min doctor tokens/turn: ${min_turn_tokens}"
echo "  Max trajectory response tokens: ${max_response_length}"
echo "  Reward: turn_coef=${turn_coef}, r_correct=${r_correct}, r_wrong=${r_wrong}, r_timeout=${r_timeout}"
echo "  Advantage: A_final + ${alpha} * A_turn"
echo "  Prompt style: ${atomic_fact_prompt_style}"
echo "  Doctor thinking: ${doctor_thinking}"
echo "============================================================"

python3 -m recipe.diagprm.diagprm_main \
  algorithm.adv_estimator=${adv_estimator} \
  algorithm.use_kl_in_reward=${use_kl_in_reward} \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.return_raw_chat=True \
  data.train_batch_size=${train_batch_size} \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.seed=42 \
  actor_rollout_ref.model.path=${ACTOR_LOAD} \
  actor_rollout_ref.model_type=qwen3 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr=${actor_lr} \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp} \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
  actor_rollout_ref.rollout.agent.agent_loop_config_path='recipe/diagprm/atomic_fact_agent.yaml' \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS_OVERRIDE} \
  reward_model.reward_manager=atomic_fact \
  reward_model.loop_enable=False \
  reward_coefficients.beta=${beta} \
  reward_coefficients.turn_coef=${turn_coef} \
  +reward_coefficients.r_correct=${r_correct} \
  +reward_coefficients.r_wrong=${r_wrong} \
  reward_coefficients.r_timeout=${r_timeout} \
  algorithm.diagprm_alpha=${alpha} \
  algorithm.terminal_diag_coef=${terminal_diag_coef} \
  critic.enable=False \
  tree_search.M_trajectories=1 \
  tree_search.N_candidates=1 \
  tree_search.call_critic_enabled=False \
  extra_params.use_critic_in_loop=False \
  trainer.logger=['console','tensorboard'] \
  trainer.project_name=atomic_fact_rl \
  trainer.experiment_name=${RUN_ID} \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=${NNODES} \
  trainer.critic_warmup=0 \
  trainer.val_before_train=True \
  trainer.log_val_generations=0 \
  trainer.save_freq=${SAVE_FREQ_OVERRIDE:-30} \
  trainer.test_freq=${TEST_FREQ_OVERRIDE:-10} \
  trainer.total_epochs=${TOTAL_EPOCHS_OVERRIDE:-3} \
  trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
  trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
  trainer.validation_data_dir=${OUTPUT_DIR}/validation_data \
  ${RESUME_MODE:+trainer.resume_mode=${RESUME_MODE}} \
  ${RESUME_PATH:+trainer.resume_from_path=${RESUME_PATH}} \
  "$@"
