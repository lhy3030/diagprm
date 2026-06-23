#!/usr/bin/env bash
# ============================================================================
# SLURM Job: Install flash_attn on ParaCloud (from pre-downloaded whl)
#
# 提交方法（在登录节点运行）：
#   mkdir -p ~/run/diagprm/job_logs
#   sbatch --partition=gpu_a800 --gres=gpu:1 --time=00:30:00 \
#          --job-name=flash_attn_install \
#          --output=/data/home/scwb729/run/diagprm/job_logs/flash_attn_%j.out \
#          /data/home/scwb729/run/diagprm/codes/recipe/diagprm/job_install_flash_attn.sh
#
# 查看进度：
#   tail -f ~/run/diagprm/job_logs/flash_attn_*.out
# ============================================================================

set -uo pipefail

echo "============================================================"
echo "  flash_attn Installation (from pre-downloaded whl)"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "============================================================"

# ── 0. 代理设置 ──────────────────────────────────────────────────────────────
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export no_proxy=localhost,127.0.0.1

# ── 1. 设置 TMPDIR 避免 Errno 18 ──────────────────────────────────────────────
export TMPDIR=/data/home/scwb729/tmp_pip
mkdir -p "$TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

echo "[INFO] TMPDIR: $TMPDIR"

# ── 2. 加载 CUDA 模块 ─────────────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
elif [ -f /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh
fi

module load gcc/11.3.0 2>/dev/null && echo "gcc-11.3.0 loaded" || echo "[WARN] gcc module not found"
module load cuda/12.8 2>/dev/null && echo "cuda-12.8 loaded" || echo "[WARN] cuda module not found"

if ! which nvcc &>/dev/null; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
echo "[INFO] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[INFO] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi not found')"

# ── 3. 初始化 conda ───────────────────────────────────────────────────────────
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate diagprm
echo "[INFO] Python: $(which python) - $(python --version)"
echo "[INFO] torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# ── 4. 从本地 whl 文件安装 flash_attn ─────────────────────────────────────────
WHL_FILE="/data/home/scwb729/run/diagprm/wheels/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

echo ""
echo ">>> [1/2] Installing flash_attn from local whl file..."

if [ -f "$WHL_FILE" ]; then
    echo "[INFO] Found whl file: $(ls -lh "$WHL_FILE")"
    
    # 先安装 --no-deps 避免拉取不兼容的依赖
    if pip install "$WHL_FILE" --no-deps 2>&1; then
        echo "[SUCCESS] flash_attn installed from local whl (no-deps)"
    else
        echo "[WARN] --no-deps install failed, trying with deps..."
        if pip install "$WHL_FILE" 2>&1; then
            echo "[SUCCESS] flash_attn installed from local whl"
        else
            echo "[FAIL] flash_attn installation failed"
        fi
    fi
else
    echo "[ERROR] whl file not found: $WHL_FILE"
    echo "[INFO] Trying pip install flash_attn==2.7.4.post1..."
    pip install flash_attn==2.7.4.post1 --no-build-isolation
fi

# ── 5. 验证安装 ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/2] Verification..."
python -c "
import torch
print(f'[OK] torch: {torch.__version__}')
print(f'     CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'     GPU: {torch.cuda.get_device_name(0)}')

try:
    import flash_attn
    print(f'[OK] flash_attn: {flash_attn.__version__}')
except Exception as e:
    print(f'[FAIL] flash_attn: {e}')

try:
    import vllm
    print(f'[OK] vllm: {vllm.__version__}')
except Exception as e:
    print(f'[WARN] vllm: {e}')

try:
    import verl
    print(f'[OK] verl: {verl.__version__}')
except Exception as e:
    print(f'[WARN] verl: {e}')
"

echo ""
echo "[DONE] flash_attn installation job finished at $(date)"
