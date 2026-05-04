#!/bin/bash
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -J final_train
#SBATCH -o slurm-final-train-%j.out
#SBATCH -e slurm-final-train-%j.err
source /public3/soft/modules/module.sh
source /public3/soft/miniforge/24.11/etc/profile.d/conda.sh
export OMP_NUM_THREADS=$SLURM_NPROCS
conda activate tzq

cd $SLURM_SUBMIT_DIR

echo "=== Final: 192x192 + inner_ch=64 + W-weighted + PINN ==="
echo "Start: $(date)"
python scripts/train_schrodinger_bridge_model.py \
  --config_path configs/config_wind_3d_final.yml \
  --experiment_name ExperimentSchrodingerBridge3dWind \
  --device cpu
echo "End: $(date)"
