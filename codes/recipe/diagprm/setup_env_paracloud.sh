#!/bin/bash
# ============================================================================
# DiagPRM Environment Setup Script for ParaCloud (NMCC-N46H1)
#
# 用法（在登录节点 ssh.cn-zhongwei-1.paracloud.com 上运行）：
#   bash ~/run/diagprm/codes/recipe/diagprm/setup_env_paracloud.sh
#
# 说明：
#   - conda 位于 /data/apps/miniforge3/25.11.0-1/bin/conda
#   - 环境安装在 ~/.conda/envs/diagprm（即 /data/run01/scwb729/.conda/envs/diagprm）
#   - flash_attn whl 使用 ~/run/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
#   - 注意：此脚本安装 torch 2.7.1+cu128（与旧服务器 naqi4 完全一致）
#     flash_attn whl 对应 torch2.8，若出现不兼容请改用 --no-build-isolation 重新编译
# ============================================================================

set -e

# === 0. 代理设置（联网必须）===
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export no_proxy=localhost,127.0.0.1

# === 1. 初始化 conda ===
CONDA_BIN="/data/apps/miniforge3/25.11.0-1/bin/conda"
if [ ! -f "$CONDA_BIN" ]; then
    echo "[ERROR] conda not found at $CONDA_BIN"
    exit 1
fi
source "$(/data/apps/miniforge3/25.11.0-1/bin/conda shell.bash hook 2>/dev/null || echo /dev/null)"
eval "$($CONDA_BIN shell.bash hook)"

ENV_NAME="diagprm"
PYTHON_VERSION="3.12"

# === 2. 创建 conda 环境 ===
if $CONDA_BIN env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[INFO] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    $CONDA_BIN create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENV_NAME}
PYTHON="$(conda run -n ${ENV_NAME} which python)"
PIP="$(conda run -n ${ENV_NAME} which pip)"

echo "[INFO] Using Python: $PYTHON"
echo "[INFO] Python version: $($PYTHON --version)"

# === 3. 安装 torch 2.7.1+cu128（与 naqi4 完全一致）===
echo "[INFO] Installing torch 2.7.1+cu128..."
conda run -n ${ENV_NAME} pip install \
    torch==2.7.1+cu128 \
    torchaudio==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    torchdata==0.11.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --trusted-host download.pytorch.org

# === 4. 安装 vllm 0.9.2 ===
echo "[INFO] Installing vllm 0.9.2..."
conda run -n ${ENV_NAME} pip install vllm==0.9.2

# === 5. 安装 verl 0.5.0（从代码目录 editable 安装）===
CODES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[INFO] Installing verl from: $CODES_DIR"
conda run -n ${ENV_NAME} pip install -e "$CODES_DIR" --no-build-isolation

# === 6. 安装其他核心依赖 ===
echo "[INFO] Installing core dependencies..."
conda run -n ${ENV_NAME} pip install \
    accelerate==1.14.0 \
    transformers==4.51.3 \
    datasets==4.0.0 \
    tokenizers==0.21.4 \
    safetensors==0.8.0 \
    sentencepiece==0.2.1 \
    tiktoken==0.13.0 \
    peft==0.19.1

# Ray
conda run -n ${ENV_NAME} pip install ray==2.47.1

# Hydra
conda run -n ${ENV_NAME} pip install hydra-core==1.3.2 omegaconf==2.3.0

# LangChain / LangGraph
conda run -n ${ENV_NAME} pip install \
    langchain-core==0.3.79 \
    langgraph==0.6.10 \
    langgraph-checkpoint==2.1.2 \
    langgraph-prebuilt==0.6.5

# OpenAI / aiohttp
conda run -n ${ENV_NAME} pip install \
    openai==1.90.0 \
    aiohttp==3.10.1 \
    aiohttp-cors==0.8.1

# TensorBoard / wandb
conda run -n ${ENV_NAME} pip install \
    tensorboard==2.20.0 \
    wandb==0.27.2

# 其他工具
conda run -n ${ENV_NAME} pip install \
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
    xformers==0.0.30 \
    codetiming==1.4.0 \
    tensordict==0.9.1

# === 7. 安装 flash_attn ===
# 优先用本地 whl（cu12 torch2.8）
FLASH_ATTN_WHL="$HOME/run/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
if [ -f "$FLASH_ATTN_WHL" ]; then
    echo "[INFO] Installing flash_attn from local whl: $FLASH_ATTN_WHL"
    conda run -n ${ENV_NAME} pip install "$FLASH_ATTN_WHL"
else
    echo "[INFO] Local whl not found, installing flash_attn 2.8.3 from PyPI..."
    conda run -n ${ENV_NAME} pip install flash_attn==2.8.3.post1 --no-build-isolation
fi

# === 8. 验证安装 ===
echo ""
echo "[INFO] === 验证安装 ==="
conda run -n ${ENV_NAME} python -c "
import torch
print(f'torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    import flash_attn; print(f'flash_attn version: {flash_attn.__version__}')
except: print('flash_attn: import failed (OK on login node without GPU)')
try:
    import vllm; print(f'vllm version: {vllm.__version__}')
except Exception as e: print(f'vllm: {e}')
try:
    import verl; print(f'verl version: {verl.__version__}')
except Exception as e: print(f'verl: {e}')
import ray; print(f'ray version: {ray.__version__}')
import hydra; print(f'hydra version: {hydra.__version__}')
import transformers; print(f'transformers version: {transformers.__version__}')
"

echo ""
echo "[SUCCESS] DiagPRM environment setup complete!"
echo "Activate with: conda activate diagprm"
