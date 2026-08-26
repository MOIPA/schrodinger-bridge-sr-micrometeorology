#!/bin/bash
# 38_probe_queue_idle.sh — 探测候选 GPU 队列空闲度,选最闲的提交
# 手机执行: bash ops/queue/38_probe_queue_idle.sh (约 1 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/38_probe_queue_idle.txt
: > "$OUT"

echo "===== 1. 候选 GPU 队列状态 (NJOBS PEND RUN SUSP) =====" >> "$OUT"
echo "队列名: NJOBS PEND RUN SUSP  (PEND 越少越空闲)" >> "$OUT"
echo "-----------------------------------------------------" >> "$OUT"
for q in 83a100ib 62v100ib 72rtxib e5v4p100ib 9654p6000ib 6148v100ib 7552v100 734090ib 7k83; do
  LINE=$(bqueues -w "$q" 2>/dev/null | tail -1)
  echo "$LINE" | awk -v q="$q" '{print q": " $8" "$9" "$10" "$11}' >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 2. 我的任务排队情况 =====" >> "$OUT"
bjobs -w 2>/dev/null | head -12 >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 推荐排序(按 PEND 数升序) =====" >> "$OUT"
for q in 83a100ib 62v100ib 72rtxib e5v4p100ib 9654p6000ib 6148v100ib 7552v100 734090ib 7k83; do
  PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
  RUN=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $10}')
  echo "$PEND $RUN $q" >> "$OUT"
done | sort -n -k1,1 | awk '{print "PEND="$1" RUN="$2"  "$3}' >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 38_probe_queue_idle" && git push
