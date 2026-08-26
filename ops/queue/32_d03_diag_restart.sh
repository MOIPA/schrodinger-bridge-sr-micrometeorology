#!/bin/bash
# 32_d03_diag_restart.sh — 诊断 d03 预处理进程 + 若被杀则降核重启(断点续跑)
# 手机执行: bash ops/queue/32_d03_diag_restart.sh (约 1 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/32_d03_diag_restart.txt
: > "$OUT"

PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
LOG=logs/prep_d03_full.log

echo "===== 1. 进程诊断 =====" >> "$OUT"
echo "--- prep 相关进程 ---" >> "$OUT"
ps aux 2>/dev/null | grep -a "prepare_wind_data_3d_sz_d03" | grep -a -v grep | head -12 >> "$OUT"
echo "(无输出 = 进程已死)" >> "$OUT"
echo "--- PID 文件记录 ---" >> "$OUT"
cat logs/prep_d03_full.pid 2>/dev/null >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 日志尾部 15 行(找死因) =====" >> "$OUT"
tail -15 "$LOG" 2>/dev/null >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 产物状态 =====" >> "$OUT"
N_NPZ=$(ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | wc -l)
echo "已生成 npz: $N_NPZ / 预期 1488" >> "$OUT"
LATEST_NPZ=$(ls -t prepare_npz_wind_3d_sz_d03/ 2>/dev/null | head -1)
echo "最新文件: $LATEST_NPZ ($(stat -c %y prepare_npz_wind_3d_sz_d03/$LATEST_NPZ 2>/dev/null | cut -d. -f1))" >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 重启判断 =====" >> "$OUT"
if ps aux 2>/dev/null | grep -a "prepare_wind_data_3d_sz_d03" | grep -a -v grep | grep -a -v "bash" >/dev/null; then
  echo ">>> 进程还活着,不动。若进度不涨再找我" >> "$OUT"
elif [ "$N_NPZ" -ge 1488 ]; then
  echo ">>> 已完成 1488,无需重启" >> "$OUT"
else
  echo ">>> 进程已死,重启全量(workers 4,断点续跑,跳过已生成的 $N_NPZ 个)" >> "$OUT"
  echo ">>> 用 setsid 启动,彻底脱离终端会话(防手机断开时被清理)" >> "$OUT"
  setsid nohup $PY scripts/prepare_wind_data_3d_sz_d03.py --scheme both --workers 4 \
    > "$LOG" 2>&1 < /dev/null &
  echo $! > logs/prep_d03_full.pid
  echo "已重启 (PID $!), 日志 $LOG" >> "$OUT"
  sleep 20
  echo "--- 20 秒后日志头部 ---" >> "$OUT"
  head -8 "$LOG" >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 32_d03_diag_restart" && git push
