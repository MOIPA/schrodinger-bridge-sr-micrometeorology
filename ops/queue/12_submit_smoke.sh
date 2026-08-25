#!/bin/bash
# 12_submit_smoke.sh — bsub 提交深圳 smoke 冒烟训练(64ch/50ep),等 90s 回传状态与首段日志
# 手机执行: bash ops/queue/12_submit_smoke.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/12_submit_smoke.txt
: > "$OUT"

echo "===== 1. 提交 bsub =====" >> "$OUT"
bsub -q 723090ib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_smoke \
  -o logs/sz_smoke_%J.out -e logs/sz_smoke_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/train_schrodinger_bridge_model.py --config_path configs/深圳/config_wind_3d_sz_smoke.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0" >> "$OUT" 2>&1

echo "" >> "$OUT"
echo "===== 2. 等待 90 秒后查看任务状态 =====" >> "$OUT"
sleep 90
bjobs -w 2>/dev/null | head -10 >> "$OUT" || echo "(bjobs 不可用)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 训练日志首段 =====" >> "$OUT"
LATEST=$(ls -t logs/sz_smoke_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST" >> "$OUT"
  tail -30 "$LATEST" >> "$OUT"
else
  echo "(尚无日志文件,稍后跑 13 号检查)" >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 12_submit_smoke" && git push
