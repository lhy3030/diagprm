#!/usr/bin/env bash
# ============================================================================
# SLURM Job: Verify DiagPRM environment on compute node
#
# 提交方法：
#   sbatch --partition=gpu_a800 --gres=gpu:1 --time=00:10:00 \
#          --job-name=diagprm_verify \
#          --output=/data/home/scwb729/run/diagprm/job_logs/verify_%j.out \
#          /data/home/scwb729/run/diagprm/codes/recipe/diagprm/job_verify_env.sh
# ============================================================================

set -uo pipefail

echo "============================================================"
echo "  DiagPRM Environment Verification (Compute Node)"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "============================================================"

# ── 代理设置 ──────────────────────────────────────────────────────────────
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128

# ── 加载 CUDA 模块 ─────────────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
elif [ -f /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh
fi

module load gcc/11.3.0 2>/dev/null
module load cuda/12.8 2>/dev/null

# ── 初始化 conda ───────────────────────────────────────────────────────────
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate diagprm

echo "[INFO] Python: $(which python) - $(python --version)"
echo "[INFO] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[INFO] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi not found')"

echo ""
echo "============================================================"
echo "  Core Package Verification"
echo "============================================================"

python /data/home/scwb729/run/diagprm/verify_env.py

echo ""
echo "============================================================"
echo "  CUDA & GPU Verification"
echo "============================================================"

python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    
    # Test CUDA tensor operation
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'CUDA tensor op: OK (result shape: {y.shape})')
    
    # Test flash_attn
    try:
        from flash_attn import flash_attn_func
        print('flash_attn import: OK')
    except Exception as e:
        print(f'flash_attn import: FAIL - {e}')
    
    # Test vllm
    try:
        import vllm
        print(f'vllm import: OK ({vllm.__version__})')
    except Exception as e:
        print(f'vllm import: FAIL - {e}')
else:
    print('CUDA not available on this node!')
"

echo ""
echo "============================================================"
echo "  Dataset & Model Check"
echo "============================================================"

DIAGPRM_ROOT="/data/home/scwb729/run/diagprm"
echo "[INFO] Dataset dir: ${DIAGPRM_ROOT}/diagprm_dataset/clean_v2/"
ls -lh "${DIAGPRM_ROOT}/diagprm_dataset/clean_v2/" 2>/dev/null || echo "[WARN] Dataset not found"
echo ""
echo "[INFO] Base model dir: ${DIAGPRM_ROOT}/../base_model/Qwen3-1.7B/"
ls -lh /data/home/scwb729/run/base_model/Qwen3-1.7B/ 2>/dev/null || echo "[WARN] Base model not found"

echo ""
echo "[DONE] Verification complete at $(date)"
