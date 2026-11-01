#!/usr/bin/env bash
# 在登录节点跑级联组装预检(CPU,不需要 GPU、不占队列)。
set -u
REPO="${REPO:-$HOME/schrodinger-bridge-sr-micrometeorology}"
PY3D="${PY3D:-/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python}"
cd "$REPO" || exit 1
echo "=== cascade smoke: stage2 ==="
"$PY3D" ops/debug/cascade_smoke.py --stage 2 --n_lev 12 2>&1 | tail -25
echo "=== cascade smoke: stage3 ==="
"$PY3D" ops/debug/cascade_smoke.py --stage 3 --n_lev 12 2>&1 | tail -25
echo "=== done ==="
