#!/bin/bash
# 22_cleanup_and_check.sh — 杀 723090ib 旧任务 + 等 GPU 训练出首个 loss + 回传
# 手机执行: bash ops/queue/22_cleanup_and_check.sh (约 3-5 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/22_cleanup_and_check.txt
: > "$OUT"

echo "===== 1. 杀 723090ib 旧任务 14001288 =====" >> "$OUT"
bkill 14001288 2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 等 3 分钟让 GPU 任务出首个 loss =====" >> "$OUT"
sleep 180

echo "" >> "$OUT"
echo "===== 3. 当前任务状态 =====" >> "$OUT"
bjobs -w 2>/dev/null | head -8 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. GPU 训练日志 (14002792) =====" >> "$OUT"
GFILE=logs/sz_smoke_14002792.out
if [ -f "$GFILE" ]; then
  echo "--- 尾部 15 行 ---" >> "$OUT"
  tail -15 "$GFILE" >> "$OUT"
  echo "--- avg loss 摘要 ---" >> "$OUT"
  grep -a "avg loss" "$GFILE" 2>/dev/null | head -5 >> "$OUT"
else
  echo "(GPU 日志尚未生成)" >> "$OUT"
  ls -t logs/sz_smoke_*.out 2>/dev/null | head -3 >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 22_cleanup_and_check" && git push
