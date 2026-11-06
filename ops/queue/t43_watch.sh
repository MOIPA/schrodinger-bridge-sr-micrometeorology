#!/usr/bin/env bash
# T4.3 消融看守:每 INTERVAL 秒查一次可用 GPU 队列,一旦有卡就把三个臂提交出去,然后退出。
# 单独一个脚本而不是塞进 sidequest 的 watch.sh:那是外传的看守(TAGS/产物名都不一样),
# 混在一起会让"哪些作业还没跑"变得说不清。
#
# 用法:
#   bash ops/queue/t43_watch.sh start     # 后台起(nohup),PID 写 /tmp/t43_watch.pid
#   bash ops/queue/t43_watch.sh stop
#   bash ops/queue/t43_watch.sh once      # 只查一次(调试用)
#   INTERVAL=600 bash ops/queue/t43_watch.sh loop
#
# 提交内容(基准臂 p4_27to1_reg / sq3to1_reg 已训好,不重复):
#   p4_27to1_aux  假设①:27→1 补辅助变量组(theta/ph/most/flux)
#   p4_27to1_w    假设②:27→1 加粗端 W
#   sq3to1_w      假设②:3→1 加粗端 W
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || exit 1
LOG="$REPO/logs/t43_watch.log"
PIDF=/tmp/t43_watch.pid
INTERVAL="${INTERVAL:-600}"
ARMS_P4="${ARMS_P4:-p4_27to1_aux p4_27to1_w}"
ARMS_SQ="${ARMS_SQ:-sq3to1_w}"

mkdir -p "$REPO/logs"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

running_named() {  # 有同名作业在跑/排队就为真
    bjobs -w 2>/dev/null | awk 'NR>1{print $7}' | grep -qx "$1"
}

trained() {  # 该臂日志里已经有 Train end 就不再投(避免重复烧卡)
    grep -qa "Train end" "logs/$1.log" 2>/dev/null
}

pending_of() {  # 从一串 tag 里挑出还没训完的
    local t out=""
    for t in $*; do trained "$t" || out="$out $t"; done
    echo $out
}

submit_all() {  # $1 = 队列名
    local q="$1" ok=0
    local out p4 sq
    p4=$(pending_of $ARMS_P4)      # 已训完的臂跳过(中断续训靠 checkpoint 自动 resume)
    sq=$(pending_of $ARMS_SQ)
    if [ -n "$p4" ]; then
        out=$(FORCE_Q="$q" ARMS="$p4" bash ops/queue/66_phase4_train.sh 2>&1)
        echo "$out" >>"$LOG"
        if echo "$out" | grep -qa "is submitted"; then ok=$((ok+1)); else log "[警告] p4 臂提交未见 is submitted"; fi
    fi
    if [ -n "$sq" ]; then
        out=$(FORCE_Q="$q" TAGS="$sq" bash sidequest_scale3x/queue/train3.sh 2>&1)
        echo "$out" >>"$LOG"
        if echo "$out" | grep -qa "is submitted"; then ok=$((ok+1)); else log "[警告] sq 臂提交未见 is submitted"; fi
    fi
    log "[提交完成] 队列 $q;p4=[$p4] sq=[$sq];批次成功 $ok"
    [ "$ok" -ge 1 ]
}

once() {
    local q
    q=$(bash ops/queue/q_pick.sh 2>/dev/null) || true
    if [ -z "$q" ]; then
        log "[skip] 无可用 GPU 队列"
        return 1
    fi
    # 已经在跑就别重复投(抢占后 LSF 会自动重排,不需要我们插手)
    if running_named p4_27to1_aux || running_named sq3to1_w; then
        log "[skip] 已有同名作业在队列里,不重投"
        return 1
    fi
    log "[发现] 队列 $q 有空卡,提交 T4.3 未训完的臂"
    submit_all "$q"
}

case "${1:-loop}" in
    start)
        if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
            echo "已在运行 PID $(cat "$PIDF")"; exit 0
        fi
        nohup bash "$0" loop >>"$LOG" 2>&1 &
        echo $! >"$PIDF"
        echo "已启动 PID $! (日志 $LOG,间隔 ${INTERVAL}s)"
        ;;
    stop)
        if [ -f "$PIDF" ]; then
            kill "$(cat "$PIDF")" 2>/dev/null && echo "已停止 PID $(cat "$PIDF")"
            rm -f "$PIDF"
        fi
        pkill -f "ops/queue/t43_watch.sh loop" 2>/dev/null
        ;;
    once)
        once
        ;;
    loop)
        log "=== T4.3 看守启动 PID $$ 间隔 ${INTERVAL}s ==="
        deadline=$(( $(date +%s) + 72 * 3600 ))
        while [ "$(date +%s)" -lt "$deadline" ]; do
            if once; then
                log "=== 已提交,看守退出 ==="
                exit 0
            fi
            # 三个臂都已训完(有 Train end)也退出,避免空转
            done_n=0
            for t in $ARMS_P4 $ARMS_SQ; do
                grep -qa "Train end" "logs/${t}.log" 2>/dev/null && done_n=$((done_n+1))
            done
            if [ "$done_n" -ge 3 ]; then
                log "=== 三臂均已训完,看守退出 ==="
                exit 0
            fi
            sleep "$INTERVAL"
        done
        log "=== 看守超时(72h)退出 ==="
        ;;
    *)
        echo "用法: $0 {start|stop|once|loop}"; exit 2;;
esac
