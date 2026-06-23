#!/usr/bin/env bash
#SBATCH --partition=gpu_a800
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=vllm_server
#SBATCH --output=/data/home/scwb729/run/diagprm/job_logs/vllm_server_%j.out

set -euo pipefail
echo "Node: $(hostname)  Job: ${SLURM_JOB_ID}  Time: $(date)"

# ── 代理 ──
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=${https_proxy}
export no_proxy=localhost,127.0.0.1,10.0.0.0/8

# ── 缓存重定向 ──
export TRITON_CACHE_DIR=/data/run01/scwb729/.triton
export XDG_CACHE_HOME=/data/run01/scwb729/.cache
export XDG_CONFIG_HOME=/data/run01/scwb729/.config
export VLLM_NO_USAGE_STATS=1
mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}"

# ── conda ──
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate diagprm

# ── module ──
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.8 2>/dev/null || true

echo "Python: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
VLLM_PORT="${VLLM_PORT:-18001}"
MODEL_PATH="${MODEL_PATH:-/data/home/scwb729/run/base_model/Qwen3-1.7B}"

echo "vLLM port: ${VLLM_PORT}"
echo "Model: ${MODEL_PATH}"
echo "Node IP: $(hostname -I | awk '{print $1}')"

export VLLM_USE_V1=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── 先把 endpoint 写到共享目录，训练作业据此找到 vLLM 地址 ──
ENDPOINT_FILE="/data/run01/scwb729/patient_vllm.endpoint"
rm -f "${ENDPOINT_FILE}"

# 后台启动 vLLM，再轮询健康检查后再写 endpoint
VLLM_NODE_IP=$(hostname -I | awk '{print $1}')

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name patient-model \
    --port "${VLLM_PORT}" \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --disable-log-requests &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# 等待 server 就绪（最多 5 分钟）
_waited=0
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 5; _waited=$(( _waited + 5 ))
    echo "Waiting for vLLM... ${_waited}s"
    if [ "${_waited}" -ge 300 ]; then
        echo "[ERROR] vLLM did not start in 300s, exiting."
        exit 1
    fi
done

# 写入 endpoint 文件供训练作业读取
echo "http://${VLLM_NODE_IP}:${VLLM_PORT}/v1" > "${ENDPOINT_FILE}"
echo "[INFO] vLLM ready! Endpoint: $(cat ${ENDPOINT_FILE})"

# 保持进程存活直到 vLLM 退出
wait ${VLLM_PID}
