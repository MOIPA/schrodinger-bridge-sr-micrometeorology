#!/bin/bash
# 16_check_login_gpu.sh — 检查登录节点是否有可用 GPU(绕过队列直接训练)
# 手机执行: bash ops/queue/16_check_login_gpu.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/16_check_login_gpu.txt
: > "$OUT"

echo "===== 1. 登录节点 GPU =====" >> "$OUT"
nvidia-smi -L 2>&1 | head -10 >> "$OUT"
echo "---" >> "$OUT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head -10 >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 我的排队任务 =====" >> "$OUT"
bjobs -w 2>/dev/null | head -5 >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 排队时间 =====" >> "$OUT"
bjobs -l 2>/dev/null | grep -a "Eligible pending" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 16_check_login_gpu" && git push
