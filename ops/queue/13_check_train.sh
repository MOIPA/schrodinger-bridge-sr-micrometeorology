#!/bin/bash
# 13_check_train.sh — 查深圳 smoke 训练状态与日志(loss 是否下降)
# 手机执行: bash ops/queue/13_check_train.sh  (可重复跑)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/13_check_train.txt
: > "$OUT"

echo "===== 1. bjobs 状态 =====" >> "$OUT"
bjobs -w 2>/dev/null | head -8 >> "$OUT" || echo "(bjobs 不可用)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 训练日志 =====" >> "$OUT"
LATEST=$(ls -t logs/sz_smoke_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST ($(stat -c %y "$LATEST" 2>/dev/null | cut -d. -f1))" >> "$OUT"
  echo "--- 最后 25 行 ---" >> "$OUT"
  tail -25 "$LATEST" >> "$OUT"
  echo "--- loss 行摘要 ---" >> "$OUT"
  grep -a "avg loss\|error" "$LATEST" 2>/dev/null | tail -8 >> "$OUT" || echo "(暂无 loss 行)" >> "$OUT"
else
  echo "(尚无日志文件)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 3. 错误日志 =====" >> "$OUT"
LERR=$(ls -t logs/sz_smoke_*.err 2>/dev/null | head -1)
if [ -n "$LERR" ]; then
  echo "err: $LERR" >> "$OUT"
  tail -15 "$LERR" >> "$OUT"
else
  echo "(无 err 文件)" >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 13_check_train" && git push
