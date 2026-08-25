#!/bin/bash
# 23_check_gpu_exit.sh — 查 GPU 任务 14002792 退出原因(err/out 全文关键段)
# 手机执行: bash ops/queue/23_check_gpu_exit.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/23_check_gpu_exit.txt
: > "$OUT"

echo "===== 1. err 文件全文 =====" >> "$OUT"
if [ -f logs/sz_smoke_14002792.err ]; then
  cat -v logs/sz_smoke_14002792.err >> "$OUT"
else
  echo "(无 err 文件)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 2. out 文件关键段(Traceback/Error/退出码) =====" >> "$OUT"
if [ -f logs/sz_smoke_14002792.out ]; then
  grep -a -i -B2 -A8 "traceback\|error\|terminated\|exit" logs/sz_smoke_14002792.out | head -60 >> "$OUT"
  echo "--- 文件头 30 行 ---" >> "$OUT"
  head -30 logs/sz_smoke_14002792.out >> "$OUT"
  echo "--- 文件尾 30 行 ---" >> "$OUT"
  tail -30 logs/sz_smoke_14002792.out >> "$OUT"
else
  echo "(无 out 文件)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 3. 当前 bjobs =====" >> "$OUT"
bjobs -w 2>/dev/null | head -8 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 历史任务退出码 =====" >> "$OUT"
bhist -l 14002792 2>/dev/null | grep -a "Done successfully\|Exited\|exit code" | head -5 >> "$OUT" || echo "(bhist 不可用)" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 23_check_gpu_exit" && git push
