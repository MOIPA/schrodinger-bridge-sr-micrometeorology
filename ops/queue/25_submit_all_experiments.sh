#!/bin/bash
# 25_submit_all_experiments.sh — 一条命令提交深圳第一轮全部 5 个正式训练(3 队列并行)
# 83a100ib: baseline+lrcond | 62v100ib: day+pinn | 72rtxib: night
# 手机执行: bash ops/queue/25_submit_all_experiments.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/25_submit_all_experiments.txt
: > "$OUT"

echo "===== 1. 提交 5 个训练任务 =====" >> "$OUT"
submit_one() {
  local name=$1 queue=$2 yml=$3
  bsub -q "$queue" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J "sz_$name" \
    -o "logs/sz_${name}_%J.out" -e "logs/sz_${name}_%J.err" \
    "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/train_schrodinger_bridge_model.py --config_path configs/深圳/$yml --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0" \
    2>&1 | head -1 >> "$OUT"
}

echo "--- baseline -> 83a100ib ---" >> "$OUT"
submit_one baseline 83a100ib config_wind_3d_sz_baseline.yml
echo "--- lrcond -> 83a100ib ---" >> "$OUT"
submit_one lrcond 83a100ib config_wind_3d_sz_lrcond.yml
echo "--- day -> 62v100ib ---" >> "$OUT"
submit_one day 62v100ib config_wind_3d_sz_day.yml
echo "--- pinn -> 62v100ib ---" >> "$OUT"
submit_one pinn 62v100ib config_wind_3d_sz_pinn.yml
echo "--- night -> 72rtxib ---" >> "$OUT"
submit_one night 72rtxib config_wind_3d_sz_night.yml

echo "" >> "$OUT"
echo "===== 2. 等 60 秒后的任务列表 =====" >> "$OUT"
sleep 60
bjobs -w 2>/dev/null | head -12 >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 提交摘要 =====" >> "$OUT"
echo "任务名 -> 配置 -> 队列:" >> "$OUT"
echo "  sz_baseline -> baseline -> 83a100ib (A100)" >> "$OUT"
echo "  sz_lrcond   -> lrcond   -> 83a100ib (A100)" >> "$OUT"
echo "  sz_day      -> day      -> 62v100ib (V100)" >> "$OUT"
echo "  sz_pinn     -> pinn     -> 62v100ib (V100)" >> "$OUT"
echo "  sz_night    -> night    -> 72rtxib  (TITAN RTX)" >> "$OUT"
echo "(另有 sz_smoke 在 83a100ib 跑冒烟)" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 25_submit_all_experiments" && git push
