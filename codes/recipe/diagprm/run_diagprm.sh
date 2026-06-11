#!/bin/bash
# ============================================================================
# DiagPRM Training Script
# Turn-level GRPO with KG-driven process reward (no Critic)
#
# 使用方法：
#   bash recipe/diagprm/run_diagprm.sh
#
# 环境变量（可覆盖）：
#   ACTOR_LOAD   : 模型路径（必填）
#   KG_PATH      : master_kg.json 路径（必填）
#   NNODES       : 节点数（default: 1）
#   N_GPUS       : 每节点 GPU 数（default: 8）
# ============================================================================

set -e  # 遇到错误立即退出

# ── 数据路径 ──────────────────────────────────────────────────────────────────
# 训练集：由 prepare_mediq_data.py 生成的 parquet 文件
TRAIN_FILES="['./data/diagprm_train.parquet']"
VAL_FILES="['./data/diagprm_val.parquet']"

# ── 目录设置 ──────────────────────────────────────────────────────────────────
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-./logs/tensorboard}"
export SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-./checkpoints/diagprm}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/diagprm}"

# ── 模型路径（必填）──────────────────────────────────────────────────────────
# 推荐：Qwen3-8B-Instruct 或 Qwen3-14B-Instruct
export ACTOR_LOAD="${ACTOR_LOAD:-/path/to/Qwen3-1.7B-Instruct}"
# export ACTOR_LOAD="${ACTOR_LOAD:-/path/to/Qwen3-4B-Instruct}"
# export ACTOR_LOAD="${ACTOR_LOAD:-/path/to/Qwen3-8B-Instruct}"

# ── KG 路径（必填）──────────────────────────────────────────────────────────
# master_kg.json 的绝对路径
export KG_PATH="${KG_PATH:-/path/to/master_kg.json}"

# ── 资源配置 ──────────────────────────────────────────────────────────────────
export NNODES="${NNODES:-1}"
export N_GPUS="${N_GPUS:-8}"
export NODE_RANK="${NODE_RANK:-0}"

# ── Algorithm 超参 ─────────────────────────────────────────────────────────────
# DiagPRM Turn-level GRPO：不使用 Critic，不使用 GAE
adv_estimator=diagprm_grpo
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.01
clip_ratio_low=0.2
clip_ratio_high=0.28

# ── 对话超参 ──────────────────────────────────────────────────────────────────
max_turns=10           # 最大问诊轮数
max_prompt_length=1024
max_response_length=2048
actor_lr=1e-6

# ── 批量大小 ──────────────────────────────────────────────────────────────────
train_batch_size=64
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_resp_per_prompt=4    # 每个 prompt 采 G=4 条轨迹（Turn-level GRPO 需要）
n_resp_per_prompt_val=1

# ── 性能配置 ──────────────────────────────────────────────────────────────────
infer_tp=2
train_sp=1
offload=True
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * n_resp_per_prompt ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

# ── DiagPRM 奖励系数 ──────────────────────────────────────────────────────────
beta=1.0       # KG 覆盖率差分系数
gamma1=0.3     # 假设正确性系数
lam=0.5        # 切换修正系数
r_max=2.0      # 最大确诊奖励
tau=0.5        # 过早确诊阈值
format_score=0.1

echo "============================================================"
echo "  DiagPRM Training"
echo "  Model: ${ACTOR_LOAD}"
echo "  KG: ${KG_PATH}"
echo "  Nodes: ${NNODES} x ${N_GPUS} GPUs"
echo "  adv_estimator: ${adv_estimator}"
echo "  G (rollouts per prompt): ${n_resp_per_prompt}"
echo "  max_turns: ${max_turns}"
echo "============================================================"

python3 -m recipe.diagprm.diagprm_main \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.return_raw_chat=True \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.seed=42 \
    \
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
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path='recipe/diagprm/diagprm_agent.yaml' \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    \
    reward_model.reward_manager=diagprm \
    reward_model.kg_path="${KG_PATH}" \
    reward_model.loop_enable=False \
    \
    reward_coefficients.beta=${beta} \
    reward_coefficients.gamma1=${gamma1} \
    reward_coefficients.lam=${lam} \
    reward_coefficients.r_max=${r_max} \
    reward_coefficients.tau=${tau} \
    reward_coefficients.format_score=${format_score} \
    reward_coefficients.weighted=True \
    \
    critic.enable=False \
    \
    tree_search.M_trajectories=1 \
    tree_search.N_candidates=1 \
    tree_search.call_critic_enabled=False \
    extra_params.use_critic_in_loop=False \
    \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=diagprm \
    trainer.experiment_name=diagprm_qwen3_8b_grpo_g${n_resp_per_prompt} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
    trainer.validation_data_dir=${OUTPUT_DIR}/validation_data \
    "$@"
