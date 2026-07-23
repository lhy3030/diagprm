#!/usr/bin/env bash
#SBATCH --partition=gpu_a800
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=vllm_1b7_sft
#SBATCH --output=/data/home/scwb729/run/diagprm/job_logs/vllm_1b7_sft_%j.out
#SBATCH --qos=gpugpu

set -euo pipefail

echo "============================================================"
echo "  vLLM — Qwen3-1.7B SFT nothink (merged_hf_latest)"
echo "  Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Time: $(date)"
echo "============================================================"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME="/data/run01/scwb729/.cache/huggingface"
export VLLM_NO_USAGE_STATS=1

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load gcc/11.3.0 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate diagprm

NODE_IP=$(hostname -I | awk '{print $1}')
PORT=8201
MODEL_PATH=/data/home/scwb729/run/diagprm/checkpoints/Qwen3-1.7B_atomic_fact_sft_nothink_20260716_075911/merged_hf_latest

echo "[INFO] Node IP: ${NODE_IP}"
echo "[INFO] Serving ${MODEL_PATH} on port ${PORT}"
echo "[INFO] Endpoint: http://${NODE_IP}:${PORT}/v1"

rm -rf /data/run01/scwb729/.cache/vllm/torch_compile_cache 2>/dev/null || true
export TRITON_CACHE_DIR="/data/run01/scwb729/.triton_fresh_$(date +%s)"

python -m vllm.entrypoints.openai.api_server \
    --enforce-eager \
    --model "${MODEL_PATH}" \
    --served-model-name Qwen3-1.7B-sft-nothink \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests
