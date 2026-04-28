#!/bin/bash
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -J eval_all
#SBATCH -o slurm-eval-all-%j.out
#SBATCH -e slurm-eval-all-%j.err
source /public3/soft/modules/module.sh
source /public3/soft/miniforge/24.11/etc/profile.d/conda.sh
export OMP_NUM_THREADS=$SLURM_NPROCS
conda activate tzq

cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "Batch Evaluation — All Experiments"
echo "Start: $(date)"
echo "=========================================="

BASE="data/DL_result/ExperimentSchrodingerBridge3dWind"
EXP="ExperimentSchrodingerBridge3dWind"

# 1. 原始模型
echo ""
echo ">>> [1/7] Original model"
python scripts/evaluate_wind_3d.py \
  --config_path configs/config_wind_3d.yml \
  --checkpoint_path ${BASE}/config_wind_3d/checkpoint.pth \
  --experiment_name ${EXP} --device cpu --split test \
  --output_dir results/eval_original

# 2. W加权loss
echo ""
echo ">>> [2/7] W-weighted loss"
python scripts/evaluate_wind_3d.py \
  --config_path configs/config_wind_3d_wloss.yml \
  --checkpoint_path ${BASE}/config_wind_3d_wloss/checkpoint.pth \
  --experiment_name ${EXP} --device cpu --split test \
  --output_dir results/eval_wloss

# 3. 消融: baseline (all vars)
echo ""
echo ">>> [3/7] Ablation: all vars (baseline)"
python scripts/evaluate_wind_3d.py \
  --config_path configs/config_wind_3d_ablation_all.yml \
  --checkpoint_path ${BASE}/config_wind_3d_ablation_all/checkpoint.pth \
  --experiment_name ${EXP} --device cpu --split test \
  --output_dir results/eval_ablation_all

# 4. 消融: no terrain
echo ""
echo ">>> [4/7] Ablation: no terrain"
python scripts/evaluate_wind_3d.py \
  --config_path configs/config_wind_3d_ablation_no_terrain.yml \
  --checkpoint_path ${BASE}/config_wind_3d_ablation_no_terrain/checkpoint.pth \
  --experiment_name ${EXP} --device cpu --split test \
  --output_dir results/eval_ablation_no_terrain

# 5. 消融: no thermal
echo ""
echo ">>> [5/7] Ablation: no thermal"
python scripts/evaluate_wind_3d.py \
  --config_path configs/config_wind_3d_ablation_no_thermal.yml \
  --checkpoint_path ${BASE}/config_wind_3d_ablation_no_thermal/checkpoint.pth \
  --experiment_name ${EXP} --device cpu --split test \
  --output_dir results/eval_ablation_no_thermal

# 6. 消融: no PBLH
echo ""
echo ">>> [6/7] Ablation: no PBLH"
python scripts/evaluate_wind_3d.py \
  --config_path configs/config_wind_3d_ablation_no_pblh.yml \
  --checkpoint_path ${BASE}/config_wind_3d_ablation_no_pblh/checkpoint.pth \
  --experiment_name ${EXP} --device cpu --split test \
  --output_dir results/eval_ablation_no_pblh

# 7. 消融: no pressure
echo ""
echo ">>> [7/7] Ablation: no pressure"
python scripts/evaluate_wind_3d.py \
  --config_path configs/config_wind_3d_ablation_no_pressure.yml \
  --checkpoint_path ${BASE}/config_wind_3d_ablation_no_pressure/checkpoint.pth \
  --experiment_name ${EXP} --device cpu --split test \
  --output_dir results/eval_ablation_no_pressure

# 8. 汇总对比
echo ""
echo ">>> Generating summary table..."
python scripts/summarize_experiments.py --results_dir results/

# 9. 生成对比图
echo ""
echo ">>> Generating comparison figures..."
python scripts/plot_experiment_results.py --results_dir results/

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results:  results/summary_*.csv"
echo "Figures:  results/figures/*.png"
echo "End: $(date)"
echo "=========================================="
