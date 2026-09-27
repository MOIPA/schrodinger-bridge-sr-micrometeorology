#!/bin/bash
# 59e_rank_phase1.sh — 阶段 1 排序重算(92):主指标 10–500 m 矢量 RMSE + 配对移动块 bootstrap
# 手机执行: bash ops/queue/59e_rank_phase1.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
OUT=ops/result/59e_rank_phase1.txt
mkdir -p ops/result results/phase1
{
echo "===== 92 排序(块长 24 h) ====="
$PY3D -u scripts/outline/92_rank_phase1.py --results_dir results/phase1 --base_tag base --block 24
echo ""
echo "===== 结果文件 ====="
ls -l results/phase1
} > "$OUT" 2>&1
cat "$OUT"
git add ops/result results/phase1
git commit -m "result 59e phase1 ranking"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
