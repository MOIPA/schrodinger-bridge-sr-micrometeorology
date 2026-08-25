#!/bin/bash
# 14_check_queue.sh — 查 GPU 队列状态与任务 PEND 原因
# 手机执行: bash ops/queue/14_check_queue.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/14_check_queue.txt
: > "$OUT"

echo "===== 1. 任务详情(含 PEND 原因) =====" >> "$OUT"
bjobs -w -l 2>/dev/null | grep -a -A3 -B1 "sz_smoke\|PEND\|EXEC_HOST\|Job <" | head -30 >> "$OUT"
bjobs -w 2>/dev/null | head -8 >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 队列资源统计 =====" >> "$OUT"
bqueues 723090ib 2>/dev/null | head -5 >> "$OUT" || echo "(bqueues 不可用)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 队列里有多少任务在排队 =====" >> "$OUT"
bjobs -w -q 723090ib 2>/dev/null | wc -l >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 14_check_queue" && git push
