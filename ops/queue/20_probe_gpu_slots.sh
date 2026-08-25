#!/bin/bash
# 20_probe_gpu_slots.sh — 带 GPU 请求探测候选队列是否有空闲 GPU 槽位
# 手机执行: bash ops/queue/20_probe_gpu_slots.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/20_probe_gpu_slots.txt
: > "$OUT"

echo "===== 1. 清理遗留 probe 任务 =====" >> "$OUT"
for j in 14001779 14001783 14001786; do
  bkill "$j" 2>&1 | head -1 >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 2. 带 GPU 请求的探测任务 =====" >> "$OUT"
for q in 72rtxib 83a100ib 945090ib 6148v100ib e5v4p100ib; do
  echo "--- $q ---" >> "$OUT"
  bsub -q "$q" -gpu "num=1:mode=exclusive_process" -n 1 -J gpu_probe -o /dev/null -e /dev/null "sleep 10" 2>&1 | head -1 >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 3. 等 40 秒后状态 =====" >> "$OUT"
sleep 40
bjobs -w 2>/dev/null | head -15 >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 20_probe_gpu_slots" && git push
