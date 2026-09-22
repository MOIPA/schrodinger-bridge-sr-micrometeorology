#!/bin/bash
# 57 · 阶段0 canvas 管线冒烟:挂 datalink -> 分块划分(若缺)-> 训练侧 Dataset + UNet 前向 + SI 损失
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
mkdir -p data/DL_data
ln -sfn "$ROOT/prepare_npz_outline_fine" data/DL_data/wrf_3d_v2_canvas
OUT=ops/result/57_smoke_canvas.txt
{
echo "=== datalink ==="
ls -la data/DL_data/ | grep wrf_3d
echo
echo "=== 30 块划分(如未生成)==="
if [ ! -f "$ROOT/prepare_npz_outline_static/split.json" ]; then
  $PY scripts/outline/30_blocks_and_split.py
else
  echo "split.json 已存在,跳过"
fi
echo
echo "=== smoke ==="
$PY scripts/smoke_wind_canvas.py --config_path configs/深圳/config_wind_canvas_smoke.yml --device cpu --backward
echo "smoke exit=$?"
} > "$OUT" 2>&1
cp "$ROOT/prepare_npz_outline_static/split.json" results/outline/split.json 2>/dev/null
git add ops/result results/outline
git commit -m "result 57 smoke canvas" && git push
