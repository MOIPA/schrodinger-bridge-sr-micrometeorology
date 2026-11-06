#!/usr/bin/env bash
# T4.3 三臂在新服务器上的串行执行链。
#
# 为什么是串行:multiscale 只有 1 块 GPU(RTX PRO 5000 Blackwell,48 GB),组里共用,
# 手册明确要求"GPU 作业只能一个接一个跑,用完就让出来"。所以这里用 Slurm 依赖把三个
# 训练按重要性排成一条链,每臂训完自动挂评估,不再像旧集群那样一次投三臂抢卡。
#
# 顺序:aux(假设①) → w(假设②,27→1) → sq3to1_w(假设②,3→1)
# 每个训练都是"有 ckpt 就续训"(从旧集群搬来的 aux 停在 84/150,w 停在 16/150)。
#
# 用法:
#   bash ops/slurm/run_t43.sh            # 真提交
#   DRY_RUN=1 bash ops/slurm/run_t43.sh  # 只打印将要提交的作业
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO="$PWD"
. ops/slurm/slurm_common.sh

DRY_RUN="${DRY_RUN:-0}"
say() { echo "[t43] $*"; }

# 三臂的配置路径与训练脚本(与旧集群保持一致:27→1 走 SI 训练脚本、3→1 走回归脚本)
P4_CFG=configs/深圳/phase4
P4_TRAINER=scripts/train_schrodinger_bridge_model.py
SQ_CFG=sidequest_scale3x/configs/phase3_arch
SQ_TRAINER=scripts/train_regression_model.py

chain_job=""
plan=(
    "p4_27to1_aux|$P4_CFG/config_wind_canvas_p4_27to1_aux.yml|$P4_TRAINER|configs/深圳/phase4"
    "p4_27to1_w|$P4_CFG/config_wind_canvas_p4_27to1_w.yml|$P4_TRAINER|configs/深圳/phase4"
    "sq3to1_w|$SQ_CFG/config_wind_canvas_sq3to1_w.yml|$SQ_TRAINER|sidequest_scale3x/configs/phase3_arch"
)

for row in "${plan[@]}"; do
    IFS='|' read -r tag cfg trainer cfgroot <<<"$row"
    if [ ! -f "$cfg" ]; then say "[跳过] 缺配置 $cfg"; continue; fi
    if grep -qa "Train end" "logs/${tag}.log" 2>/dev/null; then
        say "[跳过] $tag 日志里已有 Train end"; continue
    fi
    # 停在哪一轮:有 ckpt 就是续训,打出来便于核对(可追溯红线的老规矩)
    ck="data/DL_result/ExperimentSchrodingerBridgeWindCanvas/config_wind_canvas_${tag}/checkpoint.pth"
    if [ -f "$ck" ]; then say "$tag: 有 ckpt($(date -r "$ck" '+%m-%d %H:%M'),$(stat -c%s "$ck") B)→ 续训"
    else say "$tag: 无 ckpt → 从零"; fi

    if [ "$DRY_RUN" = "1" ]; then
        say "[DRY] sbatch ${cfg}(${trainer})依赖=${chain_job:-无}"
        say "[DRY] 之后挂评估:${cfgroot} ${tag}"
        continue
    fi

    out=$(sub_train "$cfg" "$tag" "$trainer" "$chain_job")
    echo "$out"
    jid=$(echo "$out" | sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p')
    if [ -z "$jid" ]; then say "[失败] $tag 没拿到 jobid"; exit 1; fi
    say "$tag -> 作业 $jid"
    # 评估紧跟着这一臂(用同一个作业号做 afterok,训完才评)
    sub_eval "$cfgroot" "" "$tag" "$jid" | sed 's/^/    /'
    chain_job="$jid"
done

say "链尾作业 = ${chain_job:-无}"
[ "$DRY_RUN" = "1" ] || squeue -u "$USER" -o "%.8i %.24j %.10T %.10M" | head -10
