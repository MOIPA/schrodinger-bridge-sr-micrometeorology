#!/bin/bash
# 51 · 阶段0(T0.1)数据探查:case 目录结构 / wrfinput / 层高 z_agl / eta 一致性 / 变量清单
# 输出: results/outline/00_probe_wrf_data.json + ops/result/51_probe_outline.txt
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
mkdir -p results/outline logs ops/result
PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
$PY scripts/outline/00_probe_wrf_data.py --json_out results/outline/00_probe_wrf_data.json > ops/result/51_probe_outline.txt 2>&1
echo "exit=$?" >> ops/result/51_probe_outline.txt
cd ops && git add . && git commit -m "result 51 probe outline" && git push
