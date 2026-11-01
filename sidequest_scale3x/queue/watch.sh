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


count_glob() { ls $1 >/dev/null 2>&1 && ls $1 | wc -l || echo 0; }
have_eval() { [ "$(count_glob 'sidequest_scale3x/results/sq*_reg_perhour.npz')" -ge 3 ]; }
have_xfer() { [ "$(count_glob 'sidequest_scale3x/results/xfer*_perhour.npz')" -ge 2 ]; }
have_casc() { [ "$(count_glob 'sidequest_scale3x/results/*_casc_casc_perhour.npz')" -ge 3 ]; }
downstream_running() { bjobs -w 2>/dev/null | awk 'NR>1{print $7}' | grep -qE "^(eval2_|casc|xfer2)"; }
report_done() { [ -s sidequest_scale3x/results/report4_tables.md ]; }

phase4_state() {  # 回显 wait|done:阶段4 一步 27→1 的自我接管
  local tag=p4_27to1_reg
  if ls results/phase4/*_perhour.npz >/dev/null 2>&1; then echo done; return; fi
  if running_named "$tag"; then echo wait; return; fi
  if ! train_finished "$tag"; then
    log "[重投] $tag 不在队列里且没训完,重新提交到 $Q"
    FORCE_Q="$Q" ARMS="$tag" bash ops/queue/66_phase4_train.sh >/dev/null 2>&1
    echo wait; return
  fi
  if bjobs -w 2>/dev/null | awk 'NR>1{print $7}' | grep -qx "p4eval"; then
    log "[等] p4 评估在跑"; echo wait; return
  fi
  # 队列名可能是空的(全部 closed),此时按候选列表逐个试;并把 bsub 的返回写进日志,
  # 否则"提交失败"和"提交成功"在日志里长得一模一样(2026-11-01 踩过:日志说接管了、队列里没有作业)。
  local cand="${Q:-} 944090d 734090d 813080"
  local q out
  for q in $cand; do
    [ -n "$q" ] || continue
    out=$(FORCE_Q="$q" TAGS="$tag" bash ops/queue/66b_phase4_eval.sh 2>&1)
    if echo "$out" | grep -qa "is submitted"; then
      log "[接管] p4 评估已提交 -> $q: $(echo "$out" | grep -a "Job <" | head -1)"
      echo wait; return
    fi
  done
  log "[失败] p4 评估所有候选队列都没提交成,下一轮再试"
  echo wait
}

one_round() {
  local alive=0 need=0 p4
  p4=$(phase4_state)
  for t in $TAGS; do
    if running_named "$t"; then alive=$((alive+1)); continue; fi
    if train_finished "$t"; then continue; fi
    log "[重投] $t 不在队列里且没训完,重新提交到 $Q"
    FORCE_Q="$Q" TAGS="$t" bash sidequest_scale3x/queue/train3.sh >/dev/null 2>&1
    need=1
  done

  if [ "$need" = "1" ]; then log "[等] 已重投,下一轮再看"; return 1; fi
  if [ "$alive" -gt 0 ]; then log "[等] $alive 个训练在跑"; return 1; fi

  # ---- 下游分块接管:评估 → 迁移 → 级联 → 出表 ----
  local ids="" t
  for t in $TAGS; do
    local jid
    jid=$(bjobs -a -J "$t" -o "jobid stat" 2>/dev/null | awk '$2=="DONE"{print $1}' | tail -1)
    [ -n "$jid" ] && ids="$ids $jid"
  done

  if ! have_eval; then
    if downstream_running; then log "[等] 下游作业在跑(评估未齐)"; return 1; fi
    log "[接管] 训练全部结束,重新挂下游(DEPS=$ids)"
    DEPS="$ids" bash sidequest_scale3x/queue/chain_after_train.sh >/dev/null 2>&1
    return 1
  fi

  if ! have_xfer; then
    if downstream_running; then log "[等] 下游作业在跑(迁移未齐)"; return 1; fi
    log "[接管] 评估已齐,补投交叉迁移"
    TAGS="xfer9to3_by27to9_reg,xfer27to9_by3to1_reg" bash sidequest_scale3x/queue/eval_xfer.sh >/dev/null 2>&1
    return 1
  fi

  if ! have_casc; then
    if downstream_running; then log "[等] 下游作业在跑(级联未齐)"; return 1; fi
    log "[接管] 评估与迁移已齐,补投级联"
    bash sidequest_scale3x/queue/cascade.sh >/dev/null 2>&1
    return 1
  fi

  if ! report_done; then
    log "[出表] 全部产物已齐,生成 report4 三张表"
    bash sidequest_scale3x/queue/report4.sh >>logs/report4.log 2>&1
    return 1
  fi

  if [ "$p4" = "done" ]; then
    log "[完成] 外传下游与阶段4 评估都已产出,看守退出"
    return 0
  fi
  log "[等] 外传全部完成,继续盯阶段4"
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
