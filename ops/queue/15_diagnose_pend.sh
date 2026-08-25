#!/bin/bash
# 15_diagnose_pend.sh — 诊断任务 PEND 原因(排队位置/限制/队列占用)
# 手机执行: bash ops/queue/15_diagnose_pend.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/15_diagnose_pend.txt
: > "$OUT"

echo "===== 1. bjobs -p (PEND 原因) =====" >> "$OUT"
bjobs -p 2>/dev/null >> "$OUT" || echo "(bjobs -p 不可用)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. bjobs -l 任务详情 =====" >> "$OUT"
bjobs -l 2>/dev/null | head -60 >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 队列中所有用户任务 =====" >> "$OUT"
bjobs -u all -w -q 723090ib 2>/dev/null | head -25 >> "$OUT" || echo "(无权限查看所有用户)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 队列限制 =====" >> "$OUT"
blimits -q 723090ib 2>/dev/null | head -10 >> "$OUT" || echo "(blimits 不可用)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 5. 所有 GPU 相关队列 =====" >> "$OUT"
bqueues 2>/dev/null | grep -i -E "gpu|723090|QUEUE_NAME" | head -12 >> "$OUT"

echo "" >> "$OUT"
echo "===== 6. 我的任务提交命令确认 =====" >> "$OUT"
bjobs -w 2>/dev/null | head -5 >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 15_diagnose_pend" && git push
