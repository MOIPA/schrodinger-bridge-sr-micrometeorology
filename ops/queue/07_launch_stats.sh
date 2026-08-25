#!/bin/bash
# 07_launch_stats.sh — 登录节点 nohup 启动全量统计(8928 npz,约 10-15 分钟)
# 手机执行: bash ops/queue/07_launch_stats.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result

PYBIN=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
LOG=logs/stats_sz_full.out

if pgrep -f "compute_stats_sz.py" > /dev/null 2>&1; then
  echo "!!! 统计已在运行,不重复启动 !!!"
else
  nohup "$PYBIN" scripts/compute_stats_sz.py --data_dir prepare_npz_wind_3d_sz > "$LOG" 2>&1 &
  echo "已启动统计 nohup, PID=$!"
fi

OUT=ops/result/07_launch_stats.txt
{
  echo "===== 07 启动状态 ====="
  date
  pgrep -af "compute_stats_sz" || echo "(无进程)"
} > "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 07_launch_stats" && git push
