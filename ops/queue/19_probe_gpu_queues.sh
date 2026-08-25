#!/bin/bash
# 19_probe_gpu_queues.sh — 探测候选 GPU 队列可用性(逐个提交测试任务)
# 手机执行: bash ops/queue/19_probe_gpu_queues.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/19_probe_gpu_queues.txt
: > "$OUT"

echo "===== 1. 队列描述(找 GPU 信息) =====" >> "$OUT"
bqueues -l 2>/dev/null | grep -i -B2 -A6 "gpu" | head -60 >> "$OUT" || echo "(bqueues -l 无 GPU 描述)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 逐个测试提交(候选 GPU 队列) =====" >> "$OUT"
for q in 72rtxib 83a100ib 62v100ib 945090ib 9654p6000ib 734090ib 7552v100 6148v100ib e5v4p100ib 7k83; do
  echo "--- $q ---" >> "$OUT"
  bsub -q "$q" -n 1 -J probe_test -o /dev/null -e /dev/null "sleep 3" 2>&1 | head -2 >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 3. 等 15 秒看测试任务状态 =====" >> "$OUT"
sleep 15
bjobs -w 2>/dev/null | head -12 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 当前 723090ib 任务 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep -a "sz_smoke" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 19_probe_gpu_queues" && git push
