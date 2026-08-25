#!/bin/bash
# 06_cat_resume_log.sh — 完整拉回 ysu 续跑日志(检查统计是否完成/是否有 traceback)
# 手机执行: bash ops/queue/06_cat_resume_log.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/06_cat_resume_log.txt
: > "$OUT"

echo "===== 1. 续跑日志文件信息 =====" >> "$OUT"
ls -la logs/prep_sz_ysu_resume.out 2>/dev/null >> "$OUT" || echo "(日志不存在)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 日志全文 =====" >> "$OUT"
if [ -f logs/prep_sz_ysu_resume.out ]; then
  cat -v logs/prep_sz_ysu_resume.out >> "$OUT"
  echo "" >> "$OUT"
  echo "--- 日志行数 ---" >> "$OUT"
  wc -l logs/prep_sz_ysu_resume.out >> "$OUT"
else
  echo "(日志不存在)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 3. 当前预处理进程 =====" >> "$OUT"
pgrep -af "prepare_wind_data_3d_sz" >> "$OUT" || echo "(无进程)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 输出目录现状 =====" >> "$OUT"
ls prepare_npz_wind_3d_sz/ 2>/dev/null | wc -l >> "$OUT"
ls -lt prepare_npz_wind_3d_sz/ 2>/dev/null | head -3 >> "$OUT"

echo "" >> "$OUT"
echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 06_cat_resume_log" && git push
