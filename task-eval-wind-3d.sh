#!/bin/bash
#SBATCH --job-name=eval_wind3d
#SBATCH --partition=amd_512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=slurm-eval-%j.out

# ====== 3D Wind Field Evaluation ======
# Usage: sbatch task-eval-wind-3d.sh

cd /public3/home/scg4074/tzq/git/schrodinger-bridge-sr-micrometeorology

source activate tzq

echo "=========================================="
echo "3D Wind Field Evaluation"
echo "Start time: $(date)"
echo "=========================================="

python scripts/evaluate_wind_3d.py \
  --config_path configs/config_wind_3d.yml \
  --checkpoint_path data/DL_result/ExperimentSchrodingerBridge3dWind/config_wind_3d/checkpoint.pth \
  --device cpu \
  --split test \
  --num_samples -1 \
  --output_dir results/eval_3d_wind

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
