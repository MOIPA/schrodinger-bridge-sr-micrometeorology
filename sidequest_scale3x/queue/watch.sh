#!/usr/bin/env bash
# 外传三档重训的"看守":登录节点后台跑,每 10 分钟检查一次
#   bash sidequest_scale3x/queue/watch.sh start   # 起
#   bash sidequest_scale3x/queue/watch.sh stop    # 停
#   ONCE=1 bash ... stop/start 不启动循环,只跑一轮(调试)
#
# 为什么要它:GPU 队列会 TERM_PREEMPT(今天已经杀过一次),而下游评估/级联是挂在这三个
# 训练作业的 ID 上的 —— 一被杀,ID 变了,依赖全落空。看守的逻辑:
#   训练死了(队列里没有同名作业、日志里也没有 "Train end")→ 原队列重投(会 full resume,同一划分下无害);
#   三个都训完 → 若还没有任何下游产物,重新挂一次 chain_after_train.sh;
#   跑完了 → 自己退出。
set -u
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1

TAGS="${TAGS:-sq27to9_reg sq9to3_reg sq3to1_reg}"
Q="${Q:-734090d}"
INTERVAL="${INTERVAL:-600}"
LOG="logs/sq_watch.log"
PIDF="/tmp/sq_watch.pid"
mkdir -p logs

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

running_named() {  # $1=jobname -> 0 若队列里有同名作业
  bjobs -w 2>/dev/null | awk 'NR>1{print $7}' | grep -qx "$1"
}

train_finished() {  # $1=tag
  tail -c 400 "logs/$1.log" 2>/dev/null | tr '\r' '\n' | grep -q "Train end"
}

one_round() {
  local alive=0 need=0
  for t in $TAGS; do
    if running_named "$t"; then alive=$((alive+1)); continue; fi
    if train_finished "$t"; then continue; fi
    log "[重投] $t 不在队列里且没训完,重新提交到 $Q"
    FORCE_Q="$Q" TAGS="$t" bash sidequest_scale3x/queue/train3.sh >/dev/null 2>&1
    need=1
  done

  if [ "$need" = "1" ]; then log "[等] 已重投,下一轮再看"; return 1; fi
  if [ "$alive" -gt 0 ]; then log "[等] $alive 个训练在跑"; return 1; fi

  # 三个都训完了:下游跑过就不管
  if ls sidequest_scale3x/results/*_perhour.npz >/dev/null 2>&1; then
    log "[完成] 下游产物已存在,看守退出"
    return 0
  fi
  if bjobs -w 2>/dev/null | awk 'NR>1{print $7}' | grep -qE "^(eval2_|casc|xfer2)"; then
    log "[等] 下游作业在跑"; return 1
  fi
  local ids="" t
  for t in $TAGS; do
    local jid
    jid=$(bjobs -a -J "$t" -o "jobid stat" 2>/dev/null | awk '$2=="DONE"{print $1}' | tail -1)
    [ -n "$jid" ] && ids="$ids $jid"
  done
  log "[接管] 训练全部结束,重新挂下游(DEPS=$ids)"
  DEPS="$ids" bash sidequest_scale3x/queue/chain_after_train.sh >/dev/null 2>&1
  return 1
}

case "${1:-start}" in
  stop)
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null && rm -f "$PIDF" && log "[停] 看守已停止"
    exit 0;;
  once)
    one_round; exit $?;;
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "看守已在跑 PID=$(cat "$PIDF")"; exit 0
    fi
    nohup bash "$0" loop >>"$LOG" 2>&1 &
    echo $! > "$PIDF"
    log "[起] 看守 PID=$! 间隔 ${INTERVAL}s"
    exit 0;;
  loop)
    while true; do
      one_round && break
      sleep "$INTERVAL"
    done
    rm -f "$PIDF"
    ;;
esac
