#!/usr/bin/env bash
# ============================================================================
# SLURM Job Script for DiagPRM Training on ParaCloud (NMCC-N46H1)
#
# 提交方法（两步走，分开申请 vLLM 和训练）：
#
#   ★ 步骤一：先提交 Patient vLLM 作业（1 卡，会写入 endpoint 文件）
#   sbatch --partition=gpu_a800 --gpus=1 --time=48:00:00 \
#          --job-name=patient_vllm \
#          --output=/data/home/scwb729/run/diagprm/job_logs/vllm_server_%j.out \
#          /data/home/scwb729/run/diagprm/job_vllm_server.sh
#
#   ★ 步骤二：再提交训练作业（8 卡，会等待 endpoint 文件出现后启动）
#   mkdir -p ~/run/diagprm/job_logs
#   N_GPUS=8 sbatch --partition=gpu_a800 --gpus=8 --time=48:00:00 \
#          --job-name=diagprm_8gpu \
#          --output=/data/home/scwb729/run/diagprm/job_logs/diagprm_8gpu_%j.out \
#          /data/home/scwb729/run/diagprm/codes/recipe/diagprm/job_diagprm.sh
#
# 4 卡训练（步骤二替换为）：
#   N_GPUS=4 sbatch --partition=gpu_a800 --gpus=4 --time=48:00:00 \
#          --job-name=diagprm_4gpu \
#          --output=/data/home/scwb729/run/diagprm/job_logs/diagprm_4gpu_%j.out \
#          /data/home/scwb729/run/diagprm/codes/recipe/diagprm/job_diagprm.sh
#
# 可通过环境变量覆盖：
#   N_GPUS=4 ACTOR_LOAD=/path/to/model sbatch ...
# 注意：两个作业可能在不同节点，通过 /data/run01/scwb729/patient_vllm.endpoint 共享地址
# ============================================================================

set -euo pipefail

echo "============================================================"
echo "  DiagPRM Training"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "============================================================"

# ── 0. 代理设置 ──────────────────────────────────────────────────────────────
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export no_proxy=localhost,127.0.0.1,10.0.0.0/8,10.252.0.0/16

# ── 0.05 缓存重定向（避免 /data/home 磁盘配额不足）────────────────────────────
# ParaCloud 的 /data/home 分区有磁盘配额限制，Triton/HF/vLLM 缓存必须重定向到 /data/run01
export TRITON_CACHE_DIR="/data/run01/scwb729/.triton"
export XDG_CACHE_HOME="/data/run01/scwb729/.cache"
export XDG_CONFIG_HOME="/data/run01/scwb729/.config"
export HF_HOME="/data/run01/scwb729/.cache/huggingface"
export VLLM_CACHE_ROOT="/data/run01/scwb729/.cache/vllm"
export VLLM_NO_USAGE_STATS=1
mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${XDG_CONFIG_HOME}"
echo "[INFO] TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
echo "[INFO] XDG_CACHE_HOME: ${XDG_CACHE_HOME}"
echo "[INFO] XDG_CONFIG_HOME: ${XDG_CONFIG_HOME}"

# ── 0.1 修复 SLURM ROCR_VISIBLE_DEVICES 冲突 ──────────────────────────────────
# SLURM 可能同时设置 CUDA_VISIBLE_DEVICES 和 ROCR_VISIBLE_DEVICES，
# verl 检测到两者同时存在会报错，需要 unset ROCm 相关变量
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

# ── 1. 加载 CUDA 模块 ─────────────────────────────────────────────────────────
# 初始化 module 系统（计算节点环境不同，需要手动 source）
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
elif [ -f /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh
fi

module load gcc/11.3.0 2>/dev/null && echo "gcc-11.3.0 loaded" || echo "[WARN] gcc module not found"
module load cuda/12.8 2>/dev/null && echo "cuda-12.8 loaded" || echo "[WARN] cuda module not found"
module load nccl/2.27_cuda12.8 2>/dev/null && echo "nccl loaded" || echo "[WARN] nccl module not found"

# 如果 nvcc 不在 PATH，尝试直接加 cuda 路径
if ! which nvcc &>/dev/null; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

# ── 2. 激活 conda 环境 ────────────────────────────────────────────────────────
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate diagprm

# 设置 cudnn 库路径（与 basic.sh 保持一致）
export LD_LIBRARY_PATH="$(python - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('nvidia.cudnn')
if spec and spec.origin:
    print(Path(spec.origin).resolve().parent / 'lib')
PY
):${LD_LIBRARY_PATH:-}"

echo "[INFO] Python: $(which python)"
echo "[INFO] Python version: $(python --version)"
echo "[INFO] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4 | tr '\n' ',' | sed 's/,$//')"

# ── 3. 路径设置 ───────────────────────────────────────────────────────────────
DIAGPRM_ROOT="/data/home/scwb729/run/diagprm"
CODES_DIR="${DIAGPRM_ROOT}/codes"
SCRIPT="${CODES_DIR}/recipe/diagprm/run_diagprm.sh"

# ── 4. 作业参数（可通过环境变量覆盖）───────────────────────────────────────────
export NNODES="${NNODES:-1}"
# 自动感知 SLURM 实际分配的 GPU 数量（优先用 SLURM_GPUS_ON_NODE，fallback 到 nvidia-smi）
if [ -z "${N_GPUS:-}" ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        export N_GPUS="${SLURM_GPUS_ON_NODE}"
        echo "[INFO] N_GPUS auto-detected from SLURM_GPUS_ON_NODE: ${N_GPUS}"
    elif [ -n "${SLURM_NTASKS_PER_NODE:-}" ]; then
        export N_GPUS="${SLURM_NTASKS_PER_NODE}"
        echo "[INFO] N_GPUS auto-detected from SLURM_NTASKS_PER_NODE: ${N_GPUS}"
    else
        _detected=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        export N_GPUS="${_detected:-4}"
        echo "[INFO] N_GPUS auto-detected from nvidia-smi: ${N_GPUS}"
    fi
else
    echo "[INFO] N_GPUS set by environment: ${N_GPUS}"
fi
export DIAGPRM_DATASET="${DIAGPRM_DATASET:-${DIAGPRM_ROOT}/diagprm_dataset/clean_v2}"
# 模型默认路径（需提前下载）
export ACTOR_LOAD="${ACTOR_LOAD:-${DIAGPRM_ROOT}/../base_model/Qwen3-1.7B}"

# Patient API Key
export PATIENT_API_KEY="${PATIENT_API_KEY:-21998339899070533708}"

# 日志目录
mkdir -p "${DIAGPRM_ROOT}/job_logs"

echo "[INFO] ACTOR_LOAD: ${ACTOR_LOAD}"
echo "[INFO] DIAGPRM_DATASET: ${DIAGPRM_DATASET}"

# ── 4.5 Patient vLLM 端点配置 ────────────────────────────────────────────────
# 架构：独立申请 1 卡 SLURM 作业运行 job_vllm_server.sh，训练作业通过共享文件
# /data/run01/scwb729/patient_vllm.endpoint 读取 vLLM 的实际节点 IP 和端口。
#
# 使用方法：
#   1. 先提交 vLLM 作业（等待就绪会写入 endpoint 文件）：
#      sbatch --partition=gpu_a800 --gpus=1 --time=48:00:00 \
#             /data/home/scwb729/run/diagprm/job_vllm_server.sh
#   2. 再提交本训练作业（会等待 endpoint 文件出现再启动训练）
#
# 若已有 PATIENT_API_BASE 环境变量则直接使用，跳过 endpoint 文件读取。
ENDPOINT_FILE="${ENDPOINT_FILE:-/data/run01/scwb729/patient_vllm.endpoint}"

if [ -n "${PATIENT_API_BASE:-}" ]; then
    echo "[INFO] PATIENT_API_BASE already set: ${PATIENT_API_BASE}"
elif [ -f "${ENDPOINT_FILE}" ]; then
    _ep=$(cat "${ENDPOINT_FILE}" | tr -d '[:space:]')
    echo "[INFO] Found endpoint file: ${_ep}"
    export PATIENT_API_BASE="${_ep}"
    export PATIENT_MODEL="patient-model"
    export PATIENT_PROXY=""
else
    echo "[INFO] Waiting for Patient vLLM endpoint file: ${ENDPOINT_FILE}"
    _waited=0
    until [ -f "${ENDPOINT_FILE}" ]; do
        sleep 10
        _waited=$(( _waited + 10 ))
        echo "[INFO] Still waiting for endpoint file... ${_waited}s"
        if [ "${_waited}" -ge 600 ]; then
            echo "[WARN] Endpoint file not found after 600s, will use external API as fallback"
            break
        fi
    done
    if [ -f "${ENDPOINT_FILE}" ]; then
        _ep=$(cat "${ENDPOINT_FILE}" | tr -d '[:space:]')
        echo "[INFO] Found endpoint file: ${_ep}"
        export PATIENT_API_BASE="${_ep}"
        export PATIENT_MODEL="patient-model"
        export PATIENT_PROXY=""
    else
        echo "[WARN] Using external Patient API (no endpoint file found)"
    fi
fi

if [ -n "${PATIENT_API_BASE:-}" ]; then
    echo "[INFO] Patient API: ${PATIENT_API_BASE}"
    # 验证连通性
    _health_url="${PATIENT_API_BASE%/v1}/health"
    if curl -sf --max-time 5 "${_health_url}" > /dev/null 2>&1; then
        echo "[INFO] Patient vLLM health check passed"
    else
        echo "[WARN] Patient vLLM health check failed at ${_health_url}, continuing anyway"
    fi
fi

# ── 5. 运行训练脚本 ───────────────────────────────────────────────────────────
cd "${CODES_DIR}"
bash "${SCRIPT}"
