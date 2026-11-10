#!/usr/bin/env bash
# 新服务器 multiscale(单卡 RTX PRO 5000 Blackwell)上的 Slurm 提交层。
# 旧集群用 LSF(bsub/bjobs),这里换成 sbatch/squeue/scancel,而且**只有 1 块 GPU**,
# 所以提交前必须先确认卡是空的(见 gpu_free.sh),不能像旧集群那样一次投三臂。
#
# 所有计算都必须在 Slurm 里跑:登录 shell 只有 16 线程 / 40 GB,直接跑训练会被压制。
set -uo pipefail

export REPO="${REPO:-/data/users/tangziqing/sbr}"
export PY_MODULE="${PY_MODULE:-python/ml-2026.10}"
export LOGD="$REPO/logs"
mkdir -p "$LOGD" "$LOGD/sbatch"

# 每个作业固定申请:1 卡、4 核、32 GB、**4 小时**。
# - 4 核够 DataLoader 用(旧集群也是 -n 4),不必抢 16 核;
# - 4 h 是刻意的:这是组里共用的单卡机,不能一个人占一整天。作业到点被 Slurm 杀掉后
#   重新提交即可(训练脚本见到 checkpoint 就续训),等于把卡定期还回调度池;
# - 注意:scontrol 的 CfgTRES/AllocTRES 里**没有 gres/gpu**,说明这块卡没被当作可计量资源,
#   Slurm 不会替我们串行化 GPU 作业。所以"先看 gpu_free.sh 再投、一次只投一个"必须自律;
# - --hint=nomultithread 让 4 个进程落在 4 个物理核上(手册 §5 第 6 条同款建议)。
export SBATCH_RES="${SBATCH_RES:-}"

slurm_defaults() {
    echo "#SBATCH -p main"
    echo "#SBATCH --gres=gpu:1"
    echo "#SBATCH -c 4"
    echo "#SBATCH --mem=32G"
    echo "#SBATCH -t ${GPU_HOURS:-4}:00:00"
    echo "#SBATCH --hint=nomultithread"
}

# 生成并提交一个训练作业。
#   sub_train <cfg_path_relative_to_repo> <tag> <trainer_script> [dependency_jobid]
sub_train() {
    local cfg="$1" tag="$2" trainer="$3" dep="${4:-}"
    local sb="$LOGD/sbatch/${tag}.sbatch"
    {
        echo "#!/bin/bash -l"
        slurm_defaults
        echo "#SBATCH -J ${tag}"
        echo "#SBATCH -o ${LOGD}/${tag}.out"
        echo "#SBATCH -e ${LOGD}/${tag}.err"
        echo "#SBATCH --export=HOME,USER"
        [ -n "$dep" ] && echo "#SBATCH --dependency=afterok:${dep}"
        echo
        echo "module load ${PY_MODULE}"
        echo "cd ${REPO} || exit 1"
        echo "python -u ${trainer} --config_path ${cfg} \\"
        echo "    --experiment_name ExperimentSchrodingerBridgeWindCanvas --device cuda:0"
    } >"$sb"
    sbatch "$sb" | tee -a "$LOGD/submit_history.log"
}

# 生成并提交一个评估作业(84 的 reg 模式),可选依赖训练作业。
#   sub_eval <cfg_root> <out_dir> <tag_csv> [dependency_jobid]
# cfg_root 是"phase 的父目录"(84 按 tag 前缀挑 phase:p4_→phase4,其余→phase3_arch)。
sub_eval() {
    local cfg_root="$1" out_dir="$2" tags="$3" dep="${4:-}"
    local name="eval_$(echo "$tags" | tr ',' '_')"
    local sb="$LOGD/sbatch/${name}.sbatch"
    {
        echo "#!/bin/bash -l"
        slurm_defaults
        echo "#SBATCH -J ${name}"
        echo "#SBATCH -o ${LOGD}/${name}.out"
        echo "#SBATCH -e ${LOGD}/${name}.err"
        echo "#SBATCH --export=HOME,USER"
        [ -n "$dep" ] && echo "#SBATCH --dependency=afterok:${dep}"
        echo
        echo "module load ${PY_MODULE}"
        echo "cd ${REPO} || exit 1"
        echo "python -u scripts/outline/84_ensemble_eval.py --mode reg \\"
        echo "    --cfg_root ${cfg_root} --tags ${tags} \\"
        echo "    --out_dir ${out_dir} --device cuda:0"
    } >"$sb"
    sbatch "$sb" | tee -a "$LOGD/submit_history.log"
}
