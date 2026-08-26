#!/bin/bash
# 39_d03_eval_resubmit.sh — 杀掉 83a100ib 排队任务,自动选空闲队列重提交
# 选择策略:按验证优先级选 PEND=0 的队列(72rtxib 优先,已实测兼容)
# 手机执行: bash ops/queue/39_d03_eval_resubmit.sh (约 1-2 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/39_d03_eval_resubmit.txt
: > "$OUT"

echo "===== 1. 杀掉 83a100ib 上排队的评估任务 =====" >> "$OUT"
OLD=$(bjobs -w 2>/dev/null | grep sz_d03_eval | grep PEND | awk '{print $1}')
if [ -n "$OLD" ]; then
  bkill $OLD 2>&1 | head -3 >> "$OUT"
  echo "已杀: $OLD" >> "$OUT"
else
  echo "(无 PEND 的评估任务)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 2. 自动选空闲队列(按验证优先级,PEND=0 才选) =====" >> "$OUT"
pick_queue() {
  for q in 72rtxib e5v4p100ib 9654p6000ib 6148v100ib 7552v100 7k83 83a100ib; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    RUN=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $10}')
    echo "  $q: PEND=$PEND RUN=$RUN" >> "$OUT"
    if [ "$PEND" = "0" ] 2>/dev/null; then
      echo "$q"
      return
    fi
  done
  echo "83a100ib"
}
QUEUE=$(pick_queue)
echo ">>> 选中队列: $QUEUE" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 重提交跨域评估到 $QUEUE =====" >> "$OUT"
bsub -q "$QUEUE" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_d03_eval \
  -o logs/sz_d03_eval_%J.out -e logs/sz_d03_eval_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/evaluate_sz_experiments.py --d03 --device cuda:0 --results_dir results/sz_eval_d03" \
  2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 任务状态 + pinn 训练进度 =====" >> "$OUT"
sleep 10
bjobs -w 2>/dev/null | head -8 >> "$OUT"
PLATEST=$(ls -t logs/sz_pinn_*.out 2>/dev/null | head -1)
if [ -n "$PLATEST" ]; then
  echo "--- pinn 训练进度 ---" >> "$OUT"
  grep -a "Epoch [0-9]* /\|avg loss" "$PLATEST" 2>/dev/null | tail -3 >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 39_d03_eval_resubmit" && git push
