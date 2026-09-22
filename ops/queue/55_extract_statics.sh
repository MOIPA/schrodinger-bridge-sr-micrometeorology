#!/bin/bash
# 55 · 阶段0 T0.2 静态场与几何表(z_agl / logz0 / landuse 分数 / 重网格表 / AGL 表)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
STATIC_DIR=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology/prepare_npz_outline_static
{
$PY scripts/outline/10_extract_statics.py
echo "exit=$?"
ls -la "$STATIC_DIR"
} > ops/result/55_statics.txt 2>&1
mkdir -p results/outline
cp "$STATIC_DIR/meta.json" results/outline/statics_meta.json 2>/dev/null
git add ops/result results/outline
git commit -m "result 55 statics (log + meta)" && git push
