#!/bin/bash
# 34_d03_force_restart.sh — 强制重启 d03 预处理:杀旧进程(重算浪费)+ 清破损 + 新代码重启
# 新逻辑:先跳过已生成的时次再计算,重启不再重算已完成部分
# 手机执行: bash ops/queue/34_d03_force_restart.sh (约 2-3 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/34_d03_force_restart.txt
: > "$OUT"

PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
LOG=logs/prep_d03_full.log

echo "===== 1. 杀掉旧预处理进程 =====" >> "$OUT"
PIDS=$(ps aux 2>/dev/null | grep -a "prepare_wind_data_3d_sz_d03" | grep -a -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "旧进程: $PIDS" >> "$OUT"
  kill $PIDS 2>&1 | head -5 >> "$OUT"
  sleep 3
  # 杀不死就强杀
  ps aux 2>/dev/null | grep -a "prepare_wind_data_3d_sz_d03" | grep -a -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
  echo "已清理" >> "$OUT"
else
  echo "(无旧进程)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 2. 完整性扫描 + 清除破损文件 =====" >> "$OUT"
$PY - <<'PYEOF' >> "$OUT" 2>&1
import glob
import numpy as np
import os

files = sorted(glob.glob("prepare_npz_wind_3d_sz_d03/*.npz"))
bad = []
for i, p in enumerate(files):
    try:
        with np.load(p) as d:
            for k in ['hr_u_ml0', 'lr_u_ml0', 'swdown']:
                _ = d[k]
    except Exception as e:
        bad.append(p)
print("扫描 {} 个文件,破损 {} 个".format(len(files), len(bad)))
for p in bad:
    print("  删除: {}".format(p))
    os.remove(p)
PYEOF

echo "" >> "$OUT"
echo "===== 3. 用新逻辑重启(先跳过再计算) =====" >> "$OUT"
N_NPZ=$(ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | wc -l)
echo "已有 npz: $N_NPZ (新逻辑启动后直接跳过这些,不再重算)" >> "$OUT"
setsid nohup $PY scripts/prepare_wind_data_3d_sz_d03.py --scheme both --workers 4 \
  > "$LOG" 2>&1 < /dev/null &
echo $! > logs/prep_d03_full.pid
echo "已重启 (PID $!)" >> "$OUT"

sleep 25
echo "" >> "$OUT"
echo "===== 4. 25 秒后日志(确认进入计算而非重算) =====" >> "$OUT"
tail -6 "$LOG" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 34_d03_force_restart" && git push
