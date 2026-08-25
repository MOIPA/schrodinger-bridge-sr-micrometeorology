#!/bin/bash
# 21_resubmit_gpu.sh — 一步到位: 杀旧任务 + 切到 945090ib 队列提交 smoke 训练 + 轮询等待 + 回传
# 手机执行: bash ops/queue/21_resubmit_gpu.sh (约 3-10 分钟,自动完成)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/21_resubmit_gpu.txt
: > "$OUT"

echo "===== 1. 杀掉 723090ib 的旧 sz_smoke (避免同配置双任务) =====" >> "$OUT"
OLD=$(bjobs -w 2>/dev/null | awk '$NF=="sz_smoke" {print $1}')
if [ -n "$OLD" ]; then
  for j in $OLD; do bkill "$j" 2>&1 | head -1 >> "$OUT"; done
else
  echo "(无旧任务)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 2. 提交 smoke 训练到 945090ib =====" >> "$OUT"
bsub -q 945090ib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_smoke \
  -o logs/sz_smoke_%J.out -e logs/sz_smoke_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/train_schrodinger_bridge_model.py --config_path configs/深圳/config_wind_3d_sz_smoke.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0" 2>&1 | head -2 >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 轮询等待任务启动(最长 10 分钟) =====" >> "$OUT"
for i in $(seq 1 20); do
  sleep 30
  STAT=$(bjobs -w 2>/dev/null | grep -a sz_smoke | awk '{print $3}')
  LATEST=$(ls -t logs/sz_smoke_*.out 2>/dev/null | head -1)
  LOSS=$(grep -a "avg loss" "$LATEST" 2>/dev/null | tail -1)
  echo "[$((i*30))s] STAT=$STAT LOSS=${LOSS:-无}" >> "$OUT"
  if [ -n "$LOSS" ]; then
    echo ">>> 已出现 avg loss,退出等待" >> "$OUT"
    break
  fi
  if [ -z "$STAT" ]; then
    echo ">>> 任务不在 bjobs 中(可能已结束或失败),退出等待" >> "$OUT"
    break
  fi
done

echo "" >> "$OUT"
echo "===== 4. 最终状态与日志尾部 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep -a sz_smoke >> "$OUT" || echo "(任务已不在队列)" >> "$OUT"
LATEST=$(ls -t logs/sz_smoke_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST" >> "$OUT"
  echo "--- 最后 20 行 ---" >> "$OUT"
  tail -20 "$LATEST" >> "$OUT"
  echo "--- loss 摘要 ---" >> "$OUT"
  grep -a "avg loss" "$LATEST" 2>/dev/null | head -3 >> "$OUT"
  grep -a "avg loss" "$LATEST" 2>/dev/null | tail -3 >> "$OUT"
fi
LERR=$(ls -t logs/sz_smoke_*.err 2>/dev/null | head -1)
if [ -n "$LERR" ]; then
  echo "--- err 尾部 ---" >> "$OUT"
  tail -10 "$LERR" >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 21_resubmit_gpu" && git push
