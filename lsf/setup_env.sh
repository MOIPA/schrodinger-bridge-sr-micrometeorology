#!/bin/bash
# ============================================
# 环境初始化脚本 — 首次在新服务器上执行一次即可
# ============================================
# 注意：必须在 GPU 节点的交互式 session 里执行！
#   bsub -q 723090ib -gpu num=1 -Is /bin/bash
#   bash lsf/setup_env.sh
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

# 3. 激活并安装（用 source activate，脚本里 conda activate 不生效）
source activate wind3d

# PyTorch (CUDA 11.8)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
echo "Installing other dependencies..."
pip install numpy pandas pyyaml tqdm matplotlib scipy

echo ""
echo "=== Verifying ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=== Done! ==="
echo "以后使用只需: module load anaconda/3 cuda/11.8.0 && source activate wind3d"
