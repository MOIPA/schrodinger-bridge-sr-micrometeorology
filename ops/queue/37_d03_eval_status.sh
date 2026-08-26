#!/bin/bash
# 37_d03_eval_status.sh — 查跨域评估进度,完成时回传 results/sz_eval_d03/
# 手机执行: bash ops/queue/37_d03_eval_status.sh (随时可跑,完成时自动回传结果)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/37_d03_eval_status.txt
: > "$OUT"

echo "===== 1. 跨域评估任务状态 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep sz_d03_eval >> "$OUT" || echo "(不在队列,可能已结束)" >> "$OUT"

LATEST=$(ls -t logs/sz_d03_eval_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST ($(stat -c %y "$LATEST" | cut -d. -f1))" >> "$OUT"
  echo "--- 评估进度 ---" >> "$OUT"
  grep -a "Evaluating\|Test samples\|Saved\|SKIP\|Traceback\|Error" "$LATEST" 2>/dev/null | tail -12 >> "$OUT"
  echo "--- 日志尾部(含跨域对比表) ---" >> "$OUT"
  tail -25 "$LATEST" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 2. 已产出结果 =====" >> "$OUT"
if ls results/sz_eval_d03/*.json >/dev/null 2>&1; then
  ls -la results/sz_eval_d03/ >> "$OUT"
  echo "(若含 all_summaries.json 且任务不在队列 = 已全部完成)" >> "$OUT"
else
  echo "(尚无结果)" >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传(含跨域评估结果,如有)
cd ops && git add . && git add ../results/sz_eval_d03 2>/dev/null; git commit -m "result 37_d03_eval_status" && git push
