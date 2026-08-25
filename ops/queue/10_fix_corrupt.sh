#!/bin/bash
# 10_fix_corrupt.sh — 删除损坏 npz,重跑预处理自动补生成,重扫验证(约 1-2 分钟)
# 手机执行: bash ops/queue/10_fix_corrupt.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/10_fix_corrupt.txt
: > "$OUT"

echo "===== 1. 删除损坏文件 =====" >> "$OUT"
rm -f prepare_npz_wind_3d_sz/ysu_20200719T061000.npz prepare_npz_wind_3d_sz/ysu_20200719T070000.npz
echo "已删除。当前 npz 数:" >> "$OUT"
ls prepare_npz_wind_3d_sz/ 2>/dev/null | wc -l >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 重生成缺失 npz (skip-existing,只补 2 个) =====" >> "$OUT"
/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/prepare_wind_data_3d_sz.py --scheme ysu --workers 4 --skip_stats >> "$OUT" 2>&1

echo "" >> "$OUT"
echo "===== 3. 重扫损坏 =====" >> "$OUT"
/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python - "$OUT" <<'PYEOF'
import sys, glob, zipfile
out = sys.argv[1]
files = sorted(glob.glob('prepare_npz_wind_3d_sz/*.npz'))
bad = [f for f in files if not zipfile.is_zipfile(f)]
with open(out, 'a') as fo:
    fo.write('--- 检查文件数: %d\n' % len(files))
    fo.write('--- 损坏文件数: %d\n' % len(bad))
    for b in bad:
        fo.write('    ' + b + '\n')
PYEOF

echo "" >> "$OUT"
echo "===== 4. 修复的 2 个文件 =====" >> "$OUT"
ls -la prepare_npz_wind_3d_sz/ysu_20200719T061000.npz prepare_npz_wind_3d_sz/ysu_20200719T070000.npz 2>/dev/null >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 10_fix_corrupt" && git push
