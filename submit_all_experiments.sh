#!/bin/bash
# ============================================
# 批量提交所有实验
# 包括: W加权loss(1个) + 消融实验(5个) = 6个训练任务
# ============================================
# 用法:
#   chmod +x submit_all_experiments.sh
#   ./submit_all_experiments.sh
# ============================================

echo "=========================================="
echo "Submitting all experiments"
echo "=========================================="

# W-weighted loss experiment
echo "[1/6] W-weighted loss..."
sbatch task-wind-3d-wloss.sh

# Ablation experiments
echo "[2/6] Ablation: baseline (all vars)..."
sbatch task-ablation-all.sh

echo "[3/6] Ablation: no terrain..."
sbatch task-ablation-no-terrain.sh

echo "[4/6] Ablation: no thermal..."
sbatch task-ablation-no-thermal.sh

echo "[5/6] Ablation: no PBLH..."
sbatch task-ablation-no-pblh.sh

echo "[6/6] Ablation: no pressure..."
sbatch task-ablation-no-pressure.sh

echo ""
echo "=========================================="
echo "All 6 experiments submitted!"
echo "Use 'squeue -u \$USER' to check status"
echo "=========================================="
