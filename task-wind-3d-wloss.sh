#!/bin/bash
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -J wloss_train
source /public3/soft/modules/module.sh
source /public3/soft/miniforge/24.11/etc/profile.d/conda.sh
export OMP_NUM_THREADS=$SLURM_NPROCS
conda activate tzq
echo "=== W-weighted Loss Training ==="
echo "Start: $(date)"
python scripts/train_schrodinger_bridge_model.py --config_path configs/config_wind_3d_wloss.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cpu
echo "End: $(date)"
