#!/bin/bash
# 18_list_all_queues.sh — 列出所有队列(不过滤),识别全部可用 GPU 队列
# 手机执行: bash ops/queue/18_list_all_queues.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/18_list_all_queues.txt
: > "$OUT"

echo "===== 1. 所有队列 =====" >> "$OUT"
bqueues 2>/dev/null >> "$OUT" || echo "(bqueues 不可用)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 队列总数与详情列 =====" >> "$OUT"
bqueues -w 2>/dev/null | wc -l >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 18_list_all_queues" && git push
