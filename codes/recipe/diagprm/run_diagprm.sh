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
#   KG_PATH      : clean_master_kg.json 路径（默认使用 clean_v2，可覆盖）
#   NNODES       : 节点数（default: 1）
#   N_GPUS       : 每节点 GPU 数（default: 8）
# ============================================================================



set -e  # 遇到错误立即退出

# ── NCCL / RTX 5090 (sm_120 Blackwell) 兼容性设置 ──────────────────────────────
# PyTorch 内置 NCCL 2.26.2（cuda12.2编译），对 sm_120 Blackwell 支持不完整。
# nvidia-nccl-cu12==2.30.7 已安装，通过 LD_PRELOAD 强制 torch 使用新版 NCCL。
_NCCL_SO="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)/nvidia/nccl/lib/libnccl.so.2"
if [ -f "${_NCCL_SO}" ]; then
    export LD_PRELOAD="${_NCCL_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
    echo "[INFO] Using NCCL: ${_NCCL_SO}"
fi
# 禁用 NVLS（NVLink SHARP），sm_120 上有 bug
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
# 关闭 NCCL watchdog 的 async error 立即中止行为
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"

# ── vLLM 版本兼容性设置 ────────────────────────────────────────────────────────
# vllm 0.9.x 的 AsyncvLLMServer 使用 V1 引擎（vllm.v1.engine）
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# 阻止 Ray 修改 CUDA_VISIBLE_DEVICES：AsyncvLLMServer 是 @ray.remote(num_cpus=1)
# 不请求 GPU 资源，Ray 会将其 CUDA_VISIBLE_DEVICES 清空，
# 导致 vLLM V1 引擎 spawn 的子进程无法访问 GPU
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── vLLM 调试日志 ──────────────────────────────────────────────────────────────
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export VLLM_TRACE_FUNCTION="${VLLM_TRACE_FUNCTION:-0}"



# ── CUDA 库路径（vllm 需要 libcudart）──────────────────────────────────────────
_CONDA_SITE_PKG="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [ -n "${_CONDA_SITE_PKG}" ]; then
    export LD_LIBRARY_PATH="${_CONDA_SITE_PKG}/nvidia/cu13/lib:${_CONDA_SITE_PKG}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
fi

# ── 工作目录：确保在 codes/ 下运行，使 recipe 包可被 Python 发现 ───────────────
# 本脚本位于 codes/recipe/diagprm/run_diagprm.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIAGPRM_ROOT="$(cd "${CODES_DIR}/.." && pwd)"  # diagprm/ 根目录
export PYTHONPATH="${CODES_DIR}:${PYTHONPATH:-}"
cd "${CODES_DIR}"


# ── 缓存目录（避免 home 磁盘配额不足，同时兼容非 ParaCloud 机器）───────────────
# 若 /data/run01/$USER 可写，则优先使用高速运行盘；否则退回到 repo 内 .cache。
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
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME}/vllm}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${HF_HOME}" "${XDG_CONFIG_HOME}"




# ── 确保使用本地 verl 源码（而非 pip 安装的旧版）────────────────────────────────
# pip install -e 会把旧的 site-packages 指针替换为本地路径，
# 从而让 verl.experimental 等新模块生效
VERL_LOCATION=$(python3 -c "import verl; print(verl.__file__)" 2>/dev/null || true)
if [ -z "${VERL_LOCATION}" ] || [[ "${VERL_LOCATION}" != "${CODES_DIR}"* ]]; then
    echo "[INFO] verl 当前路径: ${VERL_LOCATION:-未安装}"
    echo "[INFO] 正在安装本地 verl 源码（pip install -e .）..."
    pip install -e "${CODES_DIR}" --quiet --no-build-isolation
    echo "[INFO] 本地 verl 安装完成"
fi

# ── 数据路径 ──────────────────────────────────────────────────────────────────
# 训练集：默认使用 clean_v2，可通过同名环境变量覆盖
export DIAGPRM_DATASET="${DIAGPRM_DATASET:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2}"
TRAIN_FILES="['${DIAGPRM_DATASET}/diagprm_train.parquet']"
VAL_FILES="['${DIAGPRM_DATASET}/diagprm_val.parquet']"

# ── 目录设置 ──────────────────────────────────────────────────────────────────
# 每次启动生成时间戳，避免不同 run 的输出互相覆盖
_RUN_TIMESTAMP="${_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-./logs/tensorboard/${_RUN_TIMESTAMP}}"
export SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-./checkpoints/diagprm/${_RUN_TIMESTAMP}}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/diagprm/${_RUN_TIMESTAMP}}"
echo "[INFO] Run timestamp: ${_RUN_TIMESTAMP}"
echo "[INFO] Output dir:    ${OUTPUT_DIR}"

# HF 缓存已在上方统一设置（/data/run01），此处不再覆盖

# ── 模型路径（必填）──────────────────────────────────────────────────────────
# 推荐：Qwen3-8B-Instruct 或 Qwen3-14B-Instruct
export ACTOR_LOAD="${ACTOR_LOAD:-${DIAGPRM_ROOT}/../base_model/Qwen3-1.7B}"
# export ACTOR_LOAD="${ACTOR_LOAD:-${DIAGPRM_ROOT}/../base_model/Qwen3-4B}"
# export ACTOR_LOAD="${ACTOR_LOAD:-${DIAGPRM_ROOT}/../base_model/Qwen3-8B}"

# ── KG 路径（默认 clean_v2，可覆盖）──────────────────────────────────────────
export KG_PATH="${KG_PATH:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2/clean_master_kg.json}"

# ── Patient Agent API Key（内部 aigc.sankuai.com 接口鉴权）──────────────────
# 必须通过环境变量传入，避免硬编码到脚本中：
export PATIENT_API_KEY="${PATIENT_API_KEY:-}"
if [ -z "${PATIENT_API_KEY}" ]; then
    echo "[WARN] PATIENT_API_KEY is not set. Patient API calls will fail without auth."
    echo "[WARN] Run: export PATIENT_API_KEY=<your_token>  before starting training."
fi

# ── HTTP 代理（用于访问美团内网 aigc.sankuai.com）─────────────────────────────
# PATIENT_PROXY 专门用于 Patient API（aigc.sankuai.com），与 https_proxy 解耦。
# 代码层面（diagprm_agent_loop.py）直接读取 PATIENT_PROXY 环境变量作为代理，
# 不再覆盖全局 https_proxy，避免影响其他网络请求（如 HF 下载）。
#
# 选项 A（默认）：使用 job_diagprm.sh 已设置的外网代理转发美团内网请求
#   PATIENT_PROXY 留空 → aiohttp 会走 https_proxy (即外网代理)
# 选项 B：使用 SSH 反向隧道（需在计算节点上建立隧道 → localhost:28888）
#   export PATIENT_PROXY=http://127.0.0.1:28888
#
# 当前配置：不设置 PATIENT_PROXY，让 aiohttp 自动使用已有的 https_proxy 即可
export PATIENT_PROXY="${PATIENT_PROXY:-}"
if [ -n "${PATIENT_PROXY}" ]; then
    echo "[INFO] Using PATIENT_PROXY for Patient API: ${PATIENT_PROXY}"
else
    echo "[INFO] PATIENT_PROXY not set, Patient API will use https_proxy: ${https_proxy:-not set}"
fi

# ── 资源配置 ──────────────────────────────────────────────────────────────────
export NNODES="${NNODES:-1}"
export N_GPUS="${N_GPUS:-8}"
export NODE_RANK="${NODE_RANK:-0}"

# ── GPU 数量自动适配 ────────────────────────────────────────────────────────
# infer_tp (tensor model parallel size) 必须能整除 GPU 数量
# 1 GPU: infer_tp=1; 2+ GPU: infer_tp=2; 8+ GPU: infer_tp=4
if [ "${N_GPUS}" -le 1 ]; then
    : "${TRAIN_BATCH_SIZE_OVERRIDE:=4}"
    : "${PPO_MINI_BATCH_SIZE_OVERRIDE:=4}"
    : "${AGENT_NUM_WORKERS_OVERRIDE:=2}"
    : "${INFER_TP_OVERRIDE:=1}"
elif [ "${N_GPUS}" -le 4 ]; then
    : "${TRAIN_BATCH_SIZE_OVERRIDE:=8}"
    : "${PPO_MINI_BATCH_SIZE_OVERRIDE:=8}"
    : "${AGENT_NUM_WORKERS_OVERRIDE:=4}"
    : "${INFER_TP_OVERRIDE:=2}"
else
    : "${TRAIN_BATCH_SIZE_OVERRIDE:=16}"
    : "${PPO_MINI_BATCH_SIZE_OVERRIDE:=16}"
    : "${AGENT_NUM_WORKERS_OVERRIDE:=8}"
    : "${INFER_TP_OVERRIDE:=2}"
fi

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
max_response_length=4096  # 单轮 response 上限；多轮对话每轮约 300-600 token，4096 避免截断
actor_lr=1e-6

# ── 批量大小 ──────────────────────────────────────────────────────────────────
train_batch_size=${TRAIN_BATCH_SIZE_OVERRIDE}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE_OVERRIDE}
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_resp_per_prompt=${N_RESP_PER_PROMPT_OVERRIDE:-8}  # GRPO group size G；8卡用 8，可通过 N_RESP_PER_PROMPT_OVERRIDE 覆盖
n_resp_per_prompt_val=1

# ── 性能配置 ──────────────────────────────────────────────────────────────────
infer_tp=${INFER_TP_OVERRIDE}
train_sp=1
offload=True
# actor_max_token_len_per_gpu：控制 actor update 时单卡最长 token 数
# = (prompt + response) * G，8卡G=8: (1024+4096)*8 = 40960（开启 offload 可承受）
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * n_resp_per_prompt ))
# log_prob 只需 forward pass，显存更小，保持 1x 即可
log_prob_max_token_len_per_gpu=${actor_max_token_len_per_gpu}

# ── DiagPRM 奖励系数 ──────────────────────────────────────────────────────────
# 奖励结构：r(k) = turn_coef * r_turn(k) + r_diag(k)
#   r_turn(k) = format_reward + Δ_kg + r_hyp   （本轮即时信号，已去掉 r_switch）
#   r_diag(k) = 确诊奖励（仅最后轮）
beta=1.0         # KG 覆盖率差分系数（r_turn 内部）
gamma1=0.3       # 假设正确性系数（r_turn 内部）
turn_coef=1.0    # Turn 奖励总系数（调节 r_turn 相对于 r_diag 的权重）
r_max=2.0        # 最大确诊奖励
tau=0.5          # 过早确诊阈值
format_score=0.1

# ── GiGPO 混合系数 ─────────────────────────────────────────────────────────────
# Â(k) = Â_turn(k) + alpha * Â_diag
#   alpha=0 → 纯 turn 级归一化（无轨迹级信号）
#   alpha=1 → turn 级 + 等权轨迹级混合
alpha=0.5

echo "============================================================"
echo "  DiagPRM Training"
echo "  Model: ${ACTOR_LOAD}"
echo "  KG: ${KG_PATH}"
echo "  Nodes: ${NNODES} x ${N_GPUS} GPUs"
echo "  adv_estimator: ${adv_estimator}"
echo "  G (rollouts per prompt): ${n_resp_per_prompt}"
echo "  prompt batch size: ${train_batch_size}"
echo "  rollout batch size: $(( train_batch_size * n_resp_per_prompt ))"
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path='recipe/diagprm/diagprm_agent.yaml' \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS_OVERRIDE} \
    \
    reward_model.reward_manager=diagprm \
    reward_model.kg_path="${KG_PATH}" \
    reward_model.loop_enable=False \
    \
    reward_coefficients.beta=${beta} \
    reward_coefficients.gamma1=${gamma1} \
    reward_coefficients.turn_coef=${turn_coef} \
    reward_coefficients.r_max=${r_max} \
    reward_coefficients.tau=${tau} \
    reward_coefficients.format_score=${format_score} \
    reward_coefficients.weighted=True \
    \
    algorithm.diagprm_alpha=${alpha} \
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
    ${RESUME_MODE:+trainer.resume_mode=${RESUME_MODE}} \
    ${RESUME_PATH:+trainer.resume_from_path=${RESUME_PATH}} \
    "$@"
