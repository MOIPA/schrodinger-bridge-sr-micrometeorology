#!/bin/bash
# 56 · 阶段0 全量抽取:d04 全 10 分钟帧(2 方案,744 文件)+ d02 逐时(31 文件)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
{
$PY scripts/outline/20_extract_fine.py --scheme both --workers 6
echo "fine exit=$?"
$PY scripts/outline/21_extract_coarse.py --scheme both --workers 4
echo "coarse exit=$?"
du -sh "$ROOT/prepare_npz_outline_fine" 2>/dev/null
du -sh "$ROOT/prepare_npz_outline_coarse" 2>/dev/null
ls "$ROOT/prepare_npz_outline_fine" | wc -l
ls "$ROOT/prepare_npz_outline_coarse" | wc -l
} > ops/result/56_extract_full.txt 2>&1
cd ops && git add . && git commit -m "result 56 extract full" && git push
