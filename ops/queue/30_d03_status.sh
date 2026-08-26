#!/bin/bash
# 30_d03_status.sh — 查 d03 全量预处理进度 + 评估任务进度
# 手机执行: bash ops/queue/30_d03_status.sh (随时可跑,完成时自动回传统计量)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/30_d03_status.txt
: > "$OUT"

echo "===== 1. d03 预处理进度 =====" >> "$OUT"
if [ -f logs/prep_d03_full.pid ] && kill -0 "$(cat logs/prep_d03_full.pid)" 2>/dev/null; then
  echo "状态: 运行中 (PID $(cat logs/prep_d03_full.pid))" >> "$OUT"
else
  echo "状态: 已结束(看下面是否完成)" >> "$OUT"
fi
echo "已生成 npz: $(ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | wc -l) / 预期 1488" >> "$OUT"
tail -6 logs/prep_d03_full.log 2>/dev/null >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 评估任务 sz_eval 进度 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep sz_eval >> "$OUT" || echo "(不在队列,可能已结束)" >> "$OUT"
LATEST=$(ls -t logs/sz_eval_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  grep -a "Evaluating\|Test samples\|Saved\|complete" "$LATEST" 2>/dev/null | tail -10 >> "$OUT"
fi
ls results/sz_eval/*.json 2>/dev/null | wc -l | xargs -I{} echo "已产出评估 JSON: {} 个" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 完成判断 =====" >> "$OUT"
N_NPZ=$(ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | wc -l)
if [ "$N_NPZ" -ge 1488 ]; then
  echo ">>> d03 预处理完成!统计量在 logs/prep_d03_full.log 尾部" >> "$OUT"
  echo "--- 统计结果 ---" >> "$OUT"
  grep -a "biases:\|scales:\|day=\|night=\|day_night\|    [a-z_0-9]*: " logs/prep_d03_full.log 2>/dev/null | tail -60 >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传(含评估 JSON,如有新产出)
cd ops && git add . && git add ../results/sz_eval 2>/dev/null; git commit -m "result 30_d03_status" && git push
