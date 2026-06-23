"""
下载 DiagPRM 训练所需的 Qwen3 基础模型到新服务器
用法（在新服务器 diagprm conda 环境中运行）：

    conda activate diagprm
    export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
    export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
    python ~/run/diagprm/codes/recipe/diagprm/download_diagprm_models.py

或后台运行：
    nohup python ~/run/diagprm/codes/recipe/diagprm/download_diagprm_models.py \
        > ~/run/diagprm/download_models.log 2>&1 &
"""

import os
import sys

# 设置代理（防止忘记设置）
os.environ.setdefault("https_proxy", "http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128")
os.environ.setdefault("http_proxy", "http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128")
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ huggingface_hub 未安装，请运行: pip install huggingface_hub")
    sys.exit(1)

BASE_DIR = "/data/home/scwb729/run/diagprm/model"

# DiagPRM 训练用的 Qwen3 模型
models = [
    (
        "Qwen/Qwen3-1.7B",
        f"{BASE_DIR}/Qwen3-1.7B",
    ),
    (
        "Qwen/Qwen3-4B",
        f"{BASE_DIR}/Qwen3-4B",
    ),
    (
        "Qwen/Qwen3-8B",
        f"{BASE_DIR}/Qwen3-8B",
    ),
]

# 只下载指定模型（默认下载 1.7B 和 4B）
download_targets = os.environ.get("DOWNLOAD_MODELS", "1.7B,4B").split(",")
print(f"[INFO] 将下载: {download_targets}")

for repo_id, local_dir in models:
    model_size = repo_id.split("-")[-1]  # "1.7B" / "4B" / "8B"
    if not any(s in model_size for s in download_targets):
        print(f"[SKIP] {repo_id} (不在下载列表中)")
        continue

    print(f"\n{'='*60}")
    print(f"下载: {repo_id}")
    print(f"路径: {local_dir}")
    print(f"{'='*60}")
    try:
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ {repo_id} 完成")
    except Exception as e:
        print(f"❌ {repo_id} 失败: {e}")
        import traceback
        traceback.print_exc()

print("\n✅ 模型下载任务完成！")
