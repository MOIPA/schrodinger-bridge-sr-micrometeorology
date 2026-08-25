#!/bin/bash
# 03_check_logs.sh — 查看预处理日志全文,确认中断原因;核对 ysu 缺失范围与原始数据完整性
# 手机在服务器项目根目录执行: bash ops/queue/03_check_logs.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/03_check_logs.txt
: > "$OUT"

echo "===== 1. prep_sz_nohup.out 全文 =====" >> "$OUT"
wc -c logs/prep_sz_nohup.out 2>/dev/null >> "$OUT"
cat -v logs/prep_sz_nohup.out 2>/dev/null >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. prep_sz_13999610.out (8-24 21:51) =====" >> "$OUT"
cat -v logs/prep_sz_13999610.out 2>/dev/null >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. prep_sz_13999655.out (8-24 22:06) =====" >> "$OUT"
cat -v logs/prep_sz_13999655.out 2>/dev/null >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. ysu 缺失范围确认 =====" >> "$OUT"
echo "--- 7-10 的 ysu(应到 1750,查 1800 以后) ---" >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep '^ysu_20200710T18' >> "$OUT" || echo "(无 7-10 18:00+)" >> "$OUT"
echo "--- 7-11 起 ysu 数量 ---" >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep -c '^ysu_2020071[1-9]' >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep -c '^ysu_2020072' >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep -c '^ysu_2020073' >> "$OUT"

echo "" >> "$OUT"
echo "===== 5. ysu 原始 wrfout 文件(应 744 个) =====" >> "$OUT"
ls /fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_ysu/ 2>/dev/null | wc -l >> "$OUT"
ls /fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_ysu/ 2>/dev/null | head -2 >> "$OUT"
ls /fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_ysu/ 2>/dev/null | tail -2 >> "$OUT"

echo "" >> "$OUT"
echo "===== 完成 =====" >> "$OUT"
cat "$OUT"
