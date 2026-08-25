#!/bin/bash
# 24_resubmit_a100.sh — 换 83a100ib(A100)重提交 smoke;先查环境兼容性;轮询到 avg loss
# 手机执行: bash ops/queue/24_resubmit_a100.sh (约 5-10 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/24_resubmit_a100.txt
: > "$OUT"

echo "===== 1. wind3d env torch/CUDA 版本 =====" >> "$OUT"
module load anaconda/3 2>/dev/null || source /fs00/software/anaconda/3/etc/profile.d/conda.sh 2>/dev/null
source activate wind3d 2>/dev/null || conda activate wind3d 2>/dev/null
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('archs', torch.cuda.get_arch_list())" 2>&1 | head -5 >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 候选队列 GPU 类型(bqueues -l 摘要) =====" >> "$OUT"
for q in 83a100ib 945090ib 62v100ib e5v4p100ib 72rtxib; do
  echo "--- $q ---" >> "$OUT"
  bqueues -l "$q" 2>/dev/null | grep -a -i -A3 "GPU\|gpu" | head -6 >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 3. 提交 smoke 训练到 83a100ib =====" >> "$OUT"
bsub -q 83a100ib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_smoke \
  -o logs/sz_smoke_a100_%J.out -e logs/sz_smoke_a100_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/train_schrodinger_bridge_model.py --config_path configs/深圳/config_wind_3d_sz_smoke.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0" 2>&1 | head -2 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 轮询等待(最长 9 分钟) =====" >> "$OUT"
for i in $(seq 1 18); do
  sleep 30
  LATEST=$(ls -t logs/sz_smoke_a100_*.out 2>/dev/null | head -1)
  LOSS=$(grep -a "avg loss" "$LATEST" 2>/dev/null | tail -1)
  ERR=$(grep -a "CUDA error\|Traceback" "$LATEST" 2>/dev/null | tail -1)
  STAT=$(bjobs -w 2>/dev/null | grep -a sz_smoke | awk '{print $3}' | tr '\n' ' ')
  echo "[$((i*30))s] STAT=$STAT LOSS=${LOSS:-无} ERR=${ERR:-无}" >> "$OUT"
  if [ -n "$LOSS" ]; then
    echo ">>> 出现 avg loss,退出等待" >> "$OUT"
    break
  fi
  if [ -n "$ERR" ]; then
    echo ">>> 出现 CUDA/Traceback 错误,退出等待" >> "$OUT"
    break
  fi
  if [ -z "$STAT" ]; then
    echo ">>> 任务不在 bjobs 中(可能结束/失败),退出等待" >> "$OUT"
    break
  fi
done

echo "" >> "$OUT"
echo "===== 5. 最终状态与日志尾部 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep -a sz_smoke >> "$OUT" || echo "(任务不在队列)" >> "$OUT"
LATEST=$(ls -t logs/sz_smoke_a100_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST" >> "$OUT"
  tail -12 "$LATEST" >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 24_resubmit_a100" && git push
