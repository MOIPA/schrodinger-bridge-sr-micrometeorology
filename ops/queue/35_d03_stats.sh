#!/bin/bash
# 35_d03_stats.sh — d03 数据集全量统计(遍历全部 1488 个 npz,修复 created 列表不完整)
# 手机执行: bash ops/queue/35_d03_stats.sh (约 2-3 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/35_d03_stats.txt
: > "$OUT"

echo "===== 1. 确认数据完整 =====" >> "$OUT"
echo "npz 数量: $(ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | wc -l) / 预期 1488" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 全量统计(1488 个文件) =====" >> "$OUT"
/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/compute_stats_d03.py 2>&1 | tail -130 >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 35_d03_stats" && git push
