#!/bin/bash
# 08_check_stats.sh — 拉取全量统计结果(若进程未结束,过几分钟重跑本脚本,幂等)
# 手机执行: bash ops/queue/08_check_stats.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/08_check_stats.txt
: > "$OUT"

echo "===== 1. 统计进程(未结束则稍后再跑本脚本) =====" >> "$OUT"
pgrep -af "compute_stats_sz" >> "$OUT" || echo "(统计已结束)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 统计日志全文 =====" >> "$OUT"
if [ -f logs/stats_sz_full.out ]; then
  cat -v logs/stats_sz_full.out >> "$OUT"
else
  echo "(日志不存在)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 3. 日志最后 3 行(确认是否完成) =====" >> "$OUT"
tail -3 logs/stats_sz_full.out 2>/dev/null >> "$OUT" || echo "(无日志)" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 08_check_stats" && git push
