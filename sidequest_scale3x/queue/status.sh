#!/usr/bin/env bash
# 一行看全场:外传三档训练 + 阶段4 + 下游作业 + 看守日志
#   bash sidequest_scale3x/queue/status.sh
# 注意:训练日志里 tqdm 用 \r 刷新,必须 tr 成换行才能 grep 到"最后一行"。
set -u
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1

echo "== $(date '+%F %H:%M') =="
bjobs -w 2>/dev/null | awk 'NR==1 || /sq27to9|sq9to3|sq3to1|p4_27|eval2|casc|xfer2/ {print $1, $3, $6, $7}'

echo "-- 训练进度(日志最后一轮)"
for t in sq27to9_reg sq9to3_reg sq3to1_reg p4_27to1_reg; do
  [ -f "logs/$t.log" ] || { echo "  $t: 无日志"; continue; }
  last=$(tr '\r' '\n' < "logs/$t.log" | grep -a "Epoch" | tail -1 | sed 's/.*| //' | cut -c1-40)
  fin=$(tail -c 400 "logs/$t.log" | tr '\r' '\n' | grep -ac "Train end")
  best=$(tr '\r' '\n' < "logs/$t.log" | grep -a "Best epoch" | tail -1 | cut -c1-60)
  if [ "$fin" -gt 0 ]; then
    echo "  $t: 已结束 | $best"
  else
    echo "  $t: $last"
  fi
done

echo "-- 产物"
ls -t sidequest_scale3x/results/*_summary.json 2>/dev/null | head -6 | sed 's/^/  /'
ls -t results/phase4/*_summary.json 2>/dev/null | head -3 | sed 's/^/  /'

echo "-- 看守"
tail -3 logs/sq_watch.log 2>/dev/null | cut -c1-90 | sed 's/^/  /'
