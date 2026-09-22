#!/bin/bash
# 53 · 抽取冒烟:fine/coarse 各 2 个文件,检查 npz keys/shapes
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
OUT=ops/result/53_extract_smoke.txt
{
$PY scripts/outline/20_extract_fine.py --scheme both --limit 2 --workers 1
echo "----"
$PY scripts/outline/21_extract_coarse.py --scheme both --limit 2 --workers 1
echo
echo "=== inspect npz ==="
$PY - <<'PYEOF'
import glob
import numpy as np
for pat in ["/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology/prepare_npz_outline_fine/*.npz",
            "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology/prepare_npz_outline_coarse/*.npz"]:
    files = sorted(glob.glob(pat))
    print(pat)
    print("  count", len(files))
    for f in files[:4]:
        with np.load(f) as d:
            print("  ", f.split("/")[-1])
            for k in sorted(d.keys()):
                print("      {} shape={} dtype={} min={:.4g} max={:.4g}".format(
                    k, d[k].shape, d[k].dtype, float(np.nanmin(d[k])), float(np.nanmax(d[k]))))
PYEOF
} > "$OUT" 2>&1
echo "exit=$?" >> "$OUT"
cd ops && git add . && git commit -m "result 53 extract smoke" && git push
