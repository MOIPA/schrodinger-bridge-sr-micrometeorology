#!/bin/bash                                                           
#SBATCH -p amd_512          
#SBATCH -N 1                           
#SBATCH -n 128                          
#SBATCH -J tzq_train_job                         
source /public3/soft/modules/module.sh                                
source /public3/soft/miniforge/24.11/etc/profile.d/conda.sh
# 关键改动，，设置并行计算线程数量
export OMP_NUM_THREADS=$SLURM_NPROCS
echo "OMP_NUM_THREADS=$SLURM_NPROCS"
conda activate tzq
python scripts/train_schrodinger_bridge_model.py --config_path configs/config_wind.yml  --device cpu
