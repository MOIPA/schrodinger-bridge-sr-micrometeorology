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
  # 必须扫整份日志:训练脚本在 "Train end" 之后还会打印最佳模型、sigma_d 等若干行,
  # 只看尾部 400 字符会把已训完的档判成"没训完",于是重复重投(2026-11-02 00:35 实际发生)。
  tr '\r' '\n' < "logs/$1.log" 2>/dev/null | grep -qa "Train end"
}


count_glob() { ls $1 >/dev/null 2>&1 && ls $1 | wc -l || echo 0; }
all_exist() { local f; for f in "$@"; do [ -s "$f" ] || return 1; done; return 0; }
# 显式列产物名,不用通配符计数:级联三级产物名不齐整(第 1 级只有一个 _casc,
# 第 2/3 级是两个 _casc),用 '*_casc_casc_*' 计数会永远只数到 2 个,
# 于是看守会以为级联没跑、每 10 分钟重投一次。
have_eval() {
  all_exist sidequest_scale3x/results/sq27to9_reg_perhour.npz \
            sidequest_scale3x/results/sq9to3_reg_perhour.npz \
            sidequest_scale3x/results/sq3to1_reg_perhour.npz
}
have_xfer() {
  all_exist sidequest_scale3x/results/xfer9to3_by27to9_reg_perhour.npz \
            sidequest_scale3x/results/xfer27to9_by3to1_reg_perhour.npz
}
have_casc() {
  all_exist sidequest_scale3x/results/sq27to9_reg_casc_perhour.npz \
            sidequest_scale3x/results/sq9to3_casc_casc_perhour.npz \
            sidequest_scale3x/results/sq3to1_casc_casc_perhour.npz
}
downstream_running() { bjobs -w 2>/dev/null | awk 'NR>1{print $7}' | grep -qE "^(eval2_|casc|xfer2)"; }
# 不只看"文件在不在":部分产物也能生成一份全是 — 的表。只有带 COMPLETE 标记才算完成。
report_done() { grep -qa "STATUS: COMPLETE" sidequest_scale3x/results/report4_tables.md 2>/dev/null; }

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
