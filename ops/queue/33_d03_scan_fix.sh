#!/bin/bash
# 33_d03_scan_fix.sh — 扫描 d03 npz 完整性,删除破损文件(断点续跑会自动重建)
# 手机执行: bash ops/queue/33_d03_scan_fix.sh (约 1-2 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/33_d03_scan_fix.txt
: > "$OUT"

PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python

echo "===== 1. 进程状态(重启后是否在跑) =====" >> "$OUT"
ps aux 2>/dev/null | grep -a "prepare_wind_data_3d_sz_d03" | grep -a -v grep | head -5 >> "$OUT"
echo "已生成 npz: $(ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | wc -l) / 1488" >> "$OUT"
tail -3 logs/prep_d03_full.log 2>/dev/null >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 完整性扫描(逐个 np.load 验证) =====" >> "$OUT"
$PY - <<'PYEOF' >> "$OUT" 2>&1
import glob
import numpy as np
import os

files = sorted(glob.glob("prepare_npz_wind_3d_sz_d03/*.npz"))
bad_files = []
for i, p in enumerate(files):
    try:
        with np.load(p) as d:
            # 抽查关键变量存在
            for k in ['hr_u_ml0', 'lr_u_ml0', 'swdown']:
                _ = d[k]
    except Exception as e:
        bad_files.append((p, str(e)[:80]))
    if (i + 1) % 200 == 0:
        print("  已检查 {}/{}".format(i + 1, len(files)))

print("检查完成: 共 {} 个文件".format(len(files)))
if bad_files:
    print("发现破损文件 {} 个,删除后重新生成:".format(len(bad_files)))
    for p, err in bad_files:
        print("  {}  ({})".format(p, err))
        os.remove(p)
else:
    print("全部完整,无破损文件")
PYEOF

echo "" >> "$OUT"
echo "===== 3. 删除后数量 =====" >> "$OUT"
echo "当前 npz: $(ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | wc -l) / 1488 (被删的会由后台任务自动重建)" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 33_d03_scan_fix" && git push
