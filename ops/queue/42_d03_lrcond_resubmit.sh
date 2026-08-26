#!/bin/bash
# 42_d03_lrcond_resubmit.sh — lrcond-d03 重提交:P6000 不兼容,换兼容队列
# 已实测兼容:72rtxib(sm75) 83a100ib(sm80) 62v100ib(sm70) e5v4p100ib(sm60) 6148v100ib(sm70) 7552v100(sm70)
# 已实测不兼容:9654p6000ib(sm61,无 cu118 kernel)
# 手机执行: bash ops/queue/42_d03_lrcond_resubmit.sh (约 4 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/42_d03_lrcond_resubmit.txt
: > "$OUT"

echo "===== 1. 清理失败任务 =====" >> "$OUT"
FAILED=$(bjobs -w 2>/dev/null | grep sz_d03lrcond | awk '{print $1}')
if [ -n "$FAILED" ]; then
  bkill $FAILED 2>&1 | head -2 >> "$OUT"
  echo "已杀: $FAILED" >> "$OUT"
else
  echo "(任务已不在队列,无需清理)" >> "$OUT"
fi
# 顺带清掉报错的 err 文件避免混淆
rm -f logs/sz_d03lrcond_*.err logs/sz_d03lrcond_*.out 2>/dev/null
echo "已清理旧日志" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 选兼容队列(P6000 已排除) =====" >> "$OUT"
pick_queue() {
  local best="" best_run=999999
  for q in 72rtxib e5v4p100ib 6148v100ib 7552v100 7k83 83a100ib; do
    local PEND RUN
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    RUN=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $10}')
    echo "  $q: PEND=$PEND RUN=$RUN" >> "$OUT"
    if [ "$PEND" = "0" ] 2>/dev/null && [ "$RUN" -lt "$best_run" ] 2>/dev/null; then
      best="$q"; best_run="$RUN"
    fi
  done
  if [ -z "$best" ]; then echo "83a100ib"; else echo "$best"; fi
}
QUEUE=$(pick_queue)
echo ">>> 选中队列: $QUEUE" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 重提交 lrcond-d03 训练 =====" >> "$OUT"
bsub -q "$QUEUE" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_d03lrcond \
  -o logs/sz_d03lrcond_%J.out -e logs/sz_d03lrcond_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/train_schrodinger_bridge_model.py --config_path configs/深圳/config_wind_3d_sz_d03_lrcond.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0" \
  2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 等 3 分钟确认启动 =====" >> "$OUT"
sleep 180
bjobs -w 2>/dev/null | grep -a "sz_d03lrcond\|sz_d03base" >> "$OUT" || echo "(无 d03 训练任务?)" >> "$OUT"
LATEST=$(ls -t logs/sz_d03lrcond_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST" >> "$OUT"
  tail -8 "$LATEST" >> "$OUT"
  grep -a "avg loss\|Traceback\|CUDA error" "$LATEST" 2>/dev/null | tail -3 >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 42_d03_lrcond_resubmit" && git push
