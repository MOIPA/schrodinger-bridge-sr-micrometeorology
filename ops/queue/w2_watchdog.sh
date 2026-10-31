#!/bin/bash
# 登录节点常驻看守:每 INTERVAL 秒查一次 GPU 队列,一旦出现可用队列,
# 立刻用 64f_w2_min.sh 把 W2 三臂(p3_arch_edm / p3_arch_reg_s2 / p3_arch_unet_small)提交上去,然后退出。
#
# 用法(在 login1 上,脱离 ssh 也能活):
#   setsid nohup bash ops/queue/w2_watchdog.sh >/dev/null 2>&1 &
# 看进展:
#   tail -f logs/w2_watchdog.log
# 停掉:
#   pkill -f w2_watchdog.sh
#
# 注意:
# - 它只查队列、只提交,不碰评估;评估由 64f 挂的 64d 依赖自动触发。
# - 集群换代后队列名会变,q_pick.sh 是唯一可信的判定入口,别在这里写死队列名。
set -u
INTERVAL=${INTERVAL:-600}          # 查队列间隔(秒)
MAX_HOURS=${MAX_HOURS:-72}         # 最多看守多久(小时),防止无限常驻
REPO=${REPO:-$HOME/schrodinger-bridge-sr-micrometeorology}
LOG=${LOG:-$REPO/logs/w2_watchdog.log}

cd "$REPO" || { echo "找不到仓:$REPO"; exit 1; }
mkdir -p logs
# 同时写日志和终端:写到终端的那几行会经过 ssh 信道,正好当保活心跳(网关是"空闲"断连)
log() { echo "$*" | tee -a "$LOG"; }
log "$(date '+%F %T') [start] watchdog 启动,间隔 ${INTERVAL}s,上限 ${MAX_HOURS}h,仓 $REPO"

deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  ts=$(date '+%F %T')
  Q=$(bash ops/queue/q_pick.sh 2>/dev/null)
  if [ -n "$Q" ]; then
    log "$ts [GO] 发现可用队列 $Q,提交 W2 三臂"
    if FORCE_Q="$Q" bash ops/queue/64f_w2_min.sh >> "$LOG" 2>&1; then
      log "$ts [DONE] 提交完成,watchdog 退出"
    else
      log "$ts [WARN] 64f 返回非零,继续看守并重试"
    fi
    exit 0
  fi
  log "$ts [skip] 无可用 GPU 队列"
  sleep "$INTERVAL"
done
log "$(date '+%F %T') [timeout] 看守到期(${MAX_HOURS}h)仍无可用队列,退出"
