#!/usr/bin/env bash
set -euo pipefail

# ATPO baseline on the AtomicFact parquet data.
# This script keeps ATPO's GAE + critic + tree-search + verifier/effective
# reward, while adapting only the interface to the current no-think dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIAGPRM_ROOT="$(cd "${CODES_DIR}/.." && pwd)"

export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export ATPO_DOCTOR_THINKING="${ATPO_DOCTOR_THINKING:-0}"

ACTOR_LOAD="${ACTOR_LOAD:-${DIAGPRM_ROOT}/checkpoints/Qwen3-1.7B_atomic_fact_sft_prompt_v2/global_step_116/merged_hf}"
DATASET_DIR="${ATPO_DATASET_DIR:-${DIAGPRM_ROOT}/diagprm_dataset/atomic_fact_rl_v1}"
PATIENT_API_BASE="${PATIENT_API_BASE:-}"
PATIENT_MODEL="${PATIENT_MODEL:-Qwen3-8B}"

if [ -z "${PATIENT_API_BASE}" ]; then
  echo "[ERROR] Set PATIENT_API_BASE to a live OpenAI-compatible patient/verifier vLLM endpoint." >&2
  exit 1
fi
if [ ! -d "${ACTOR_LOAD}" ]; then
  echo "[ERROR] ACTOR_LOAD does not exist: ${ACTOR_LOAD}" >&2
  exit 1
fi
if [ ! -f "${DATASET_DIR}/atomic_fact_train.parquet" ]; then
  echo "[ERROR] Missing train parquet: ${DATASET_DIR}/atomic_fact_train.parquet" >&2
  exit 1
fi
if [ ! -f "${DATASET_DIR}/atomic_fact_val.parquet" ]; then
  echo "[ERROR] Missing val parquet: ${DATASET_DIR}/atomic_fact_val.parquet" >&2
  exit 1
fi

export PATIENT_API_BASE PATIENT_MODEL
export PATIENT_API_KEY="${PATIENT_API_KEY:-${API_KEY:-EMPTY}}"

NNODES="${NNODES:-1}"
N_GPUS="${N_GPUS:-4}"
INFER_TP="${INFER_TP_OVERRIDE:-2}"
RUN_ID="${RUN_ID:-atomic_fact_atpo_sft_prompt_v2_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODES_DIR}/outputs/atomic_fact_atpo/${RUN_ID}}"
SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-${CODES_DIR}/checkpoints/atomic_fact_atpo/${RUN_ID}}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE_OVERRIDE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE_OVERRIDE:-64}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS_OVERRIDE:-4}"
MAX_TURNS="${MAX_TURNS_OVERRIDE:-10}"
N_RESP="${N_RESP_PER_PROMPT_OVERRIDE:-1}"
N_RESP_VAL="${N_RESP_PER_PROMPT_VAL_OVERRIDE:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS_OVERRIDE:-3}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH_OVERRIDE:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH_OVERRIDE:-4096}"
ACTOR_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN_OVERRIDE:-$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 4 ))}"
LOG_PROB_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN_OVERRIDE:-$(( ACTOR_MAX_TOKEN_LEN * 2 ))}"

export ATPO_DATASET_DIR="${DATASET_DIR}"
export ATPO_RUN_ID="${RUN_ID}"

echo "======================================================================"
echo "[INFO] AtomicFact ATPO baseline"
echo "[INFO] Actor:        ${ACTOR_LOAD}"
echo "[INFO] Dataset:      ${DATASET_DIR}"
echo "[INFO] Patient API:  ${PATIENT_API_BASE}"
echo "[INFO] GPUs:         ${N_GPUS} (doctor), patient is external"
echo "[INFO] Think:        ${ATPO_DOCTOR_THINKING}"
echo "[INFO] Max turns:    ${MAX_TURNS}"
echo "[INFO] Batch / G:    ${TRAIN_BATCH_SIZE} / ${N_RESP}"
echo "[INFO] Output:       ${OUTPUT_DIR}"
echo "======================================================================"

cd "${CODES_DIR}"

python3 -m recipe.atpo_atomic_fact.mt_main \
  algorithm.adv_estimator=gae \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  data.train_files="['${DATASET_DIR}/atomic_fact_train.parquet']" \
  data.val_files="['${DATASET_DIR}/atomic_fact_val.parquet']" \
  data.return_raw_chat=True \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.seed=42 \
  actor_rollout_ref.model.path="${ACTOR_LOAD}" \
  actor_rollout_ref.model_type=qwen3 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR_OVERRIDE:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${INFER_TP}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n="${N_RESP}" \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n="${N_RESP_VAL}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/atpo_atomic_fact/agent.yaml \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  custom_reward_function.path=recipe/atpo_atomic_fact/mt_reward_fn.py \
  custom_reward_function.name=mt_reward_fn \
  reward_model.reward_manager=mt_reward_manager \
  reward_model.loop_enable=False \
  critic.model.path="${ACTOR_LOAD}" \
  critic.model.use_remove_padding=True \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  critic.optim.lr=1e-5 \
  critic.ppo_epochs=1 \
  critic.loss_agg_mode=token-mean \
  tree_search.M_trajectories="${ATPO_TREE_M:-128}" \
  tree_search.N_candidates="${ATPO_TREE_N:-4}" \
  tree_search.variance_threshold="${ATPO_TREE_VARIANCE:-1.2}" \
  tree_search.pruning_enabled=True \
  tree_search.only_use_critic_value=True \
  tree_search.call_critic_enabled=True \
  extra_params.use_critic_in_loop=True \
  reward_coefficients.format_score=0.1 \
  reward_coefficients.verify_score=0.4 \
  reward_coefficients.effective_score=0.5 \
  reward_coefficients.correctness_score=3.0 \
  reward_coefficients.exceed_max_turn_penalty=-1.0 \
  trainer.logger="['console','tensorboard']" \
  trainer.project_name=atomic_fact_atpo \
  trainer.experiment_name="${RUN_ID}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${NNODES}" \
  trainer.critic_warmup=4 \
  trainer.val_before_train=False \
  trainer.log_val_generations=0 \
  trainer.save_freq="${SAVE_FREQ_OVERRIDE:-30}" \
  trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}" \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollout_data" \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation_data" \
  trainer.test_freq="${TEST_FREQ_OVERRIDE:-10}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  "$@"
