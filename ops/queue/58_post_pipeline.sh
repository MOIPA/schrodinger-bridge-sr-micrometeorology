#!/bin/bash
# 58 · 抽取完成后:块划分 -> 标准化统计 -> AGL 往返 -> 校验 -> canvas 冒烟 -> 真值诊断
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/58_post_pipeline.txt
mkdir -p results/outline data/DL_data logs
ln -sfn "$ROOT/prepare_npz_outline_fine" data/DL_data/wrf_3d_v2_canvas
{
echo "=== 30 块划分 ==="
$PY scripts/outline/30_blocks_and_split.py
echo "=== 40 标准化统计 ==="
$PY scripts/outline/40_stats.py
echo "=== 60 AGL 往返(T0.4)==="
$PY scripts/outline/60_agl_operator.py --json_out results/outline/agl_roundtrip.json
echo "=== 50 校验 ==="
$PY scripts/outline/50_verify_dataset.py --json_out results/outline/verify_report.json
echo "=== canvas 冒烟 ==="
$PY scripts/smoke_wind_canvas.py --config_path "configs/深圳/config_wind_canvas_smoke.yml" --device cpu --backward
echo "smoke exit=$?"
echo "=== 70 真值诊断(T0.5)==="
$PY scripts/outline/70_truth_diagnostics.py --json_out results/outline/truth_diagnostics.json
} > "$OUT" 2>&1
cp "$ROOT/prepare_npz_outline_static/split.json" results/outline/split.json 2>/dev/null
cp "$ROOT/prepare_npz_outline_static/normalize_config.json" results/outline/normalize_config.json 2>/dev/null
cp "$ROOT/prepare_npz_outline_static/meta.json" results/outline/statics_meta.json 2>/dev/null
git add ops/result results/outline
git commit -m "result 58 post-extraction pipeline" && git push
