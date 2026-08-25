#!/bin/bash
# 05_check_final.sh — 验证 ysu 续跑结果:数量/统计量(抄进配置用)/日志尾部
# 若 ysu 尚未到 4464,等 15-20 分钟后再跑一次本脚本(幂等,可重复)
# 手机执行: bash ops/queue/05_check_final.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/05_check_final.txt
: > "$OUT"

echo "===== 1. 服务器时间 =====" >> "$OUT"
date >> "$OUT"

echo "===== 2. 预处理进程(应为空/已结束) =====" >> "$OUT"
pgrep -af "prepare_wind_data_3d_sz" >> "$OUT" || echo "(进程已结束)" >> "$OUT"

echo "===== 3. npz 总数与 scheme 分布(期望 8928 = 4464+4464) =====" >> "$OUT"
ls prepare_npz_wind_3d_sz/ 2>/dev/null | wc -l >> "$OUT"
ls prepare_npz_wind_3d_sz/ 2>/dev/null | sed 's/_.*//' | sort | uniq -c >> "$OUT"

echo "===== 4. ysu 最新文件(应到 7-31 23:50) =====" >> "$OUT"
ls prepare_npz_wind_3d_sz/ 2>/dev/null | grep '^ysu' | sort | tail -2 >> "$OUT"

echo "===== 5. 续跑日志尾部(biases/scales/昼夜统计,抄进深圳配置) =====" >> "$OUT"
tail -120 logs/prep_sz_ysu_resume.out 2>/dev/null || echo "(日志不存在)" >> "$OUT"

echo "===== 6. 最新 npz 写入时间 =====" >> "$OUT"
ls -lt prepare_npz_wind_3d_sz/ 2>/dev/null | head -3 >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 05_check_final" && git push
