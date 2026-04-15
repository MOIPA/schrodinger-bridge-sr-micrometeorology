#!/bin/bash
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -J ablation_no_pressure
source /public3/soft/modules/module.sh
source /public3/soft/miniforge/24.11/etc/profile.d/conda.sh
export OMP_NUM_THREADS=$SLURM_NPROCS
conda activate tzq
echo "=== Remove PSFC ==="
echo "Start: $(date)"
python scripts/train_schrodinger_bridge_model.py --config_path configs/config_wind_3d_ablation_no_pressure.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cpu
echo "End: $(date)"
