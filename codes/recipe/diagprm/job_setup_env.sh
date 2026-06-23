#!/usr/bin/env bash
# ============================================================================
# SLURM Job: Setup DiagPRM conda environment on ParaCloud
# 
# 提交方法（在登录节点运行，注意：不要 --mem 或 --cpus-per-task 参数）：
#   mkdir -p ~/run/diagprm/job_logs
#   sbatch --partition=gpu_a800 --gres=gpu:1 --time=04:00:00 \
#          --job-name=diagprm_setup \
#          --output=/data/home/scwb729/run/diagprm/job_logs/setup_env_%j.out \
#          /data/home/scwb729/run/diagprm/codes/recipe/diagprm/job_setup_env.sh
#
# 查看进度：
#   tail -f ~/run/diagprm/job_logs/setup_env_*.out
# ============================================================================

set -euo pipefail

echo "============================================================"
echo "  DiagPRM Environment Setup"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "============================================================"

# ── 0. 代理设置 ──────────────────────────────────────────────────────────────
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export no_proxy=localhost,127.0.0.1

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
module load cmake/4.2.0 2>/dev/null && echo "cmake loaded" || echo "[WARN] cmake module not found"

# 如果 nvcc 不在 PATH，尝试直接加 cuda 路径
if ! which nvcc &>/dev/null; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
echo "[INFO] CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "[INFO] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi not found')"

# ── 2. 初始化 conda ───────────────────────────────────────────────────────────
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate diagprm
echo "[INFO] Python: $(which python) - $(python --version)"

# ── 3. 安装 torch 2.7.1+cu128 ────────────────────────────────────────────────
echo ""
echo ">>> [1/7] Installing torch 2.7.1+cu128..."
pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
echo ">>> torch installed: $(python -c 'import torch; print(torch.__version__)')"

# torchdata（版本较老，不在 cu128 index 里）
echo ">>> Installing torchdata..."
pip install torchdata==0.10.0 2>/dev/null \
    || pip install "torchdata" 2>/dev/null \
    || echo "[WARN] torchdata not available, skipping"

# ── 4. 安装 vllm 0.9.2 ───────────────────────────────────────────────────────
echo ""
echo ">>> [2/7] Installing vllm 0.9.2..."
pip install vllm==0.9.2
echo ">>> vllm installed"

# ── 5. 安装 verl (from codes dir editable) ────────────────────────────────────
CODES_DIR="/data/home/scwb729/run/diagprm/codes"
echo ""
echo ">>> [3/7] Installing verl from: $CODES_DIR"
pip install -e "$CODES_DIR" --no-build-isolation
echo ">>> verl installed: $(python -c 'import verl; print(verl.__version__)' 2>/dev/null || echo unknown)"

# ── 6. 安装其他核心依赖 ───────────────────────────────────────────────────────
echo ""
echo ">>> [4/7] Installing core dependencies..."
pip install \
    accelerate==1.14.0 \
    transformers==4.51.3 \
    datasets==4.0.0 \
    tokenizers==0.21.4 \
    safetensors==0.8.0 \
    sentencepiece==0.2.1 \
    tiktoken==0.13.0 \
    peft==0.19.1

echo ">>> [5/7] Installing ray..."
pip install ray==2.47.1

echo ">>> Installing hydra / omegaconf..."
pip install hydra-core==1.3.2 omegaconf==2.3.0

echo ">>> Installing langchain / langgraph..."
pip install \
    langchain-core==0.3.79 \
    langgraph==0.6.10 \
    langgraph-checkpoint==2.1.2 \
    langgraph-prebuilt==0.6.5

echo ">>> Installing openai / aiohttp..."
pip install openai==1.90.0 aiohttp==3.10.1 aiohttp-cors==0.8.1

echo ">>> Installing tensorboard / wandb..."
pip install tensorboard==2.20.0 wandb==0.27.2

echo ">>> Installing misc packages..."
pip install \
    pydantic==2.12.0 \
    pydantic-settings==2.14.1 \
    fastapi==0.136.3 \
    uvicorn==0.49.0 \
    pandas==2.3.3 \
    pyarrow==24.0.0 \
    numpy==2.2.6 \
    scipy==1.17.1 \
    Pillow==12.2.0 \
    tqdm==4.66.5 \
    psutil==7.2.2 \
    diskcache==5.6.3 \
    einops==0.8.2 \
    codetiming==1.4.0 \
    tensordict==0.9.1 \
    triton==3.3.0 \
    xformers==0.0.30 \
    huggingface_hub \
    GitPython

# ── 7. 安装 flash_attn ────────────────────────────────────────────────────────
echo ""
echo ">>> [6/7] Installing flash_attn..."
# flash_attn 2.7.4 支持 torch 2.7.x + cu12
pip install flash_attn==2.7.4 --no-build-isolation 2>/dev/null || {
    echo "[WARN] flash_attn 2.7.4 failed, trying 2.6.3..."
    pip install flash_attn==2.6.3 --no-build-isolation 2>/dev/null || {
        echo "[WARN] Trying latest flash_attn..."
        pip install flash_attn --no-build-isolation
    }
}
echo ">>> flash_attn installed: $(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'install failed')"

# ── 8. 验证安装 ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  [7/7] Verification"
echo "============================================================"
python -c "
import torch
print(f'[OK] torch: {torch.__version__}')
print(f'     CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'     GPU: {torch.cuda.get_device_name(0)}')
    print(f'     CUDA version: {torch.version.cuda}')

try:
    import flash_attn
    print(f'[OK] flash_attn: {flash_attn.__version__}')
except Exception as e:
    print(f'[WARN] flash_attn: {e}')

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

import ray; print(f'[OK] ray: {ray.__version__}')
import hydra; print(f'[OK] hydra: {hydra.__version__}')
import transformers; print(f'[OK] transformers: {transformers.__version__}')
import langchain_core; print(f'[OK] langchain_core: {langchain_core.__version__}')
import openai; print(f'[OK] openai: {openai.__version__}')
import aiohttp; print(f'[OK] aiohttp: {aiohttp.__version__}')
"

echo ""
echo "[SUCCESS] DiagPRM environment setup complete!"
echo "[INFO] To use: source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh && conda activate diagprm"
