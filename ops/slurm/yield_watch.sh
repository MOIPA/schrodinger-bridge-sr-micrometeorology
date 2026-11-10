#!/usr/bin/env bash
# 让卡:组里有人排队等 GPU 时,主动结束我们自己的作业。
#
# 为什么需要它:这台机的 CfgTRES 里没有 gres/gpu,Slurm 并不把 GPU 当可计量资源,
# 所以两个 --gres=gpu:1 的作业可能同时跑;谁先占谁赢。"共用设备不能一个人独用"这条
# 只能靠自觉 + 这个看守来实现。
#
# 用法: nohup bash ops/slurm/yield_watch.sh > logs/yield.log 2>&1 &
#      INTERVAL=300 BATCH="p4_27to1_aux p4_27to1_w sq3to1_w" bash ops/slurm/yield_watch.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
INTERVAL="${INTERVAL:-300}"
BATCH="${BATCH:-}"

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a logs/yield.log; }

log "启动:每 ${INTERVAL}s 检查一次,监控作业名 = ${BATCH:-我校所有作业}"
while :; do
    sleep "$INTERVAL"
    # 别人(非本账号)正在排队且声明需要 GPU 的作业
    waiting=$(squeue -h -o "%u|%T|%b|%j" 2>/dev/null \
        | awk -F'|' -v me="$USER" '$1 != me && $2 == "PENDING" && $3 ~ /gpu/ {print $4"("$1")"}')
    if [ -n "$waiting" ]; then
        log "有人在等 GPU: $(echo "$waiting" | tr '\n' ' ') —— 主动让卡"
        for j in $(squeue -u "$USER" -h -o "%i %j" | awk -v b="$BATCH" '
                    { if (b=="") print $1; else { n=split(b,arr," "); for(i=1;i<=n;i++) if ($2==arr[i]) print $1 } }'); do
            scancel "$j" 2>/dev/null && log "  已 scancel $j"
        done
        log "  让出后退出。重新提交请先跑 ops/slurm/gpu_free.sh 确认卡空了。"
        exit 0
    fi
done
