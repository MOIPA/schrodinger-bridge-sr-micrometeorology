#!/bin/bash
# ============================================
# LSF 脚本使用说明
# ============================================
#
# 目录结构:
#   lsf/
#   ├── setup_env.sh       # 首次环境配置（只跑一次）
#   ├── train_final.lsf    # 最终实验训练
#   ├── eval_final.lsf     # 训练完后评估
#   └── usage.sh           # 本文件（使用说明）
#
# ============================================
# 首次使用
# ============================================
#
# 1. 配置环境（只需做一次）:
#    bash lsf/setup_env.sh
#
# 2. 验证 GPU 可用:
#    bsub -q 723090ib -gpu num=1 -Is /bin/bash
#    module load anaconda/3 cuda/11.8.0
#    source activate wind3d
#    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#
# ============================================
# 提交训练
# ============================================
#
#    bsub < lsf/train_final.lsf
#
# 查看状态:
#    bjobs          # 看所有作业
#    bjobs -l JOB_ID  # 看某个作业详情
#    bpeek JOB_ID   # 看实时输出
#
# 杀作业:
#    bkill JOB_ID
#
# ============================================
# 训练完后评估
# ============================================
#
#    bsub < lsf/eval_final.lsf
#
# ============================================
# LSF vs SLURM 对照
# ============================================
#
#  SLURM              LSF                  说明
#  ─────────────────────────────────────────────────
#  sbatch xxx.sh      bsub < xxx.lsf       提交作业
#  squeue -u $USER    bjobs                 看作业状态
#  scancel JOB_ID     bkill JOB_ID          杀作业
#  srun               bsub -Is             交互模式
#  #SBATCH -p xxx     #BSUB -q xxx         指定队列
#  #SBATCH -N 1       #BSUB -n 4           核数
#  #SBATCH --gres=gpu:1   #BSUB -gpu num=1 GPU
#  #SBATCH -J name    #BSUB -J name        作业名
#  #SBATCH -o file    #BSUB -o file        标准输出
#  $SLURM_SUBMIT_DIR  $LS_SUBCWD           提交目录
#
# ============================================
# 常见问题
# ============================================
#
# Q: 找不到 wind3d 环境?
# A: 确认先 module load anaconda/3，再 source activate wind3d
#
# Q: CUDA out of memory?
# A: 改 config_wind_3d_final.yml 的 batch_size 从 8 改小，比如 4 或 2
#
# Q: 训练中断想续跑?
# A: 直接重新 bsub < lsf/train_final.lsf，脚本会自动加载 checkpoint 续训
#
