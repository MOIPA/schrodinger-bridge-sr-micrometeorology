#!/bin/bash
# 09_find_corrupt.sh — 扫描输出目录中损坏的 npz(非有效 zip 文件)
# 手机执行: bash ops/queue/09_find_corrupt.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/09_find_corrupt.txt
: > "$OUT"

echo "===== 1. 扫描损坏 npz =====" >> "$OUT"
/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python - "$OUT" <<'PYEOF'
import sys, glob, zipfile
out = sys.argv[1]
files = sorted(glob.glob('prepare_npz_wind_3d_sz/*.npz'))
bad = []
for f in files:
    try:
        if not zipfile.is_zipfile(f):
            bad.append(f)
    except Exception as e:
        bad.append(f + '  EXC: ' + str(e))
with open(out, 'a') as fo:
    fo.write('--- 检查文件数: %d\n' % len(files))
    fo.write('--- 损坏文件数: %d\n' % len(bad))
    for b in bad:
        fo.write('    ' + b + '\n')
PYEOF

echo "" >> "$OUT"
echo "===== 2. 异常小的 npz (< 1MB,正常约 1.7MB) =====" >> "$OUT"
ls -la prepare_npz_wind_3d_sz/ | grep -v "^total" | awk '$5 < 1000000 {print $5, $9}' >> "$OUT"
echo "(空 = 无异常小文件)" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 09_find_corrupt" && git push
