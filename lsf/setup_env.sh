#!/bin/bash
# ============================================
# 环境初始化脚本 — 首次在新服务器上执行一次即可
# 用法: bash lsf/setup_env.sh
# ============================================

echo "=== Setting up environment ==="

# 1. 加载 module
module load anaconda/3
module load cuda/11.8.0

echo "Loaded modules:"
module list

# 2. 创建 conda 环境
echo ""
echo "Creating conda environment 'wind3d'..."
conda create -n wind3d python=3.10 -y

# 3. 激活环境并安装依赖
conda activate wind3d

# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install numpy pandas pyyaml tqdm matplotlib scipy

echo ""
echo "=== Environment setup complete ==="
echo "To activate: module load anaconda/3 cuda/11.8.0 && conda activate wind3d"
echo "Test GPU: python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
