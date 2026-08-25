#!/bin/bash
# 17_launch_cpu_smoke.sh — 登录节点 CPU 跑微型冒烟训练(5% 训练集,5 epoch,约 10-20 分钟)
# 用于在 GPU 队列排队期间验证训练循环 + loss 下降;checkpoint 目录独立(smoke_cpu)
# 手机执行: bash ops/queue/17_launch_cpu_smoke.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/17_launch_cpu_smoke.txt
: > "$OUT"

echo "===== 1. 检查是否有同配置训练在跑 =====" >> "$OUT"
pgrep -af "config_wind_3d_sz_smoke_cpu" >> "$OUT" || echo "(无)" >> "$OUT"

module load anaconda/3 2>/dev/null || source /fs00/software/anaconda/3/etc/profile.d/conda.sh 2>/dev/null
source activate wind3d 2>/dev/null || conda activate wind3d 2>/dev/null

echo "" >> "$OUT"
echo "===== 2. 启动 CPU 冒烟训练 (nohup) =====" >> "$OUT"
LOG=logs/sz_smoke_cpu.out
nohup python scripts/train_schrodinger_bridge_model.py \
  --config_path configs/深圳/config_wind_3d_sz_smoke_cpu.yml \
  --experiment_name ExperimentSchrodingerBridge3dWind \
  --device cpu > "$LOG" 2>&1 &
echo "已启动, PID=$! 日志=$LOG" >> "$OUT"

sleep 60
echo "" >> "$OUT"
echo "===== 3. 60 秒后日志首段 =====" >> "$OUT"
tail -25 "$LOG" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 17_launch_cpu_smoke" && git push
