#!/bin/bash
# 48_A_group_eval.sh — A 组纯评估实验(A1-A6, sr-exp-design skill)
#   A1 功率谱 + A2 逐层 + A3 强风分箱 + A5 跨方案: baseline/allLR/phys (diag)
#   A4 条件变量打乱消融: baseline (shuffle)
#   A6 跨域矩匹配: baseline d04 模型 + d03 数据 + d04 输入统计 (moments)
# 运行: bash ops/queue/48_A_group_eval.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p logs results/A_group ops/result
OUT=ops/result/48_A_group_eval.txt
: > "$OUT"
echo "===== A 组评估提交 $(date) =====" >> "$OUT"

bsub -q 72rtxib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
  -J sz_A_group -o logs/sz_A_group_%J.out \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
   python scripts/evaluate_sz_A_group.py --mode diag --model baseline && \
   python scripts/evaluate_sz_A_group.py --mode diag --model allLR && \
   python scripts/evaluate_sz_A_group.py --mode diag --model phys && \
   python scripts/evaluate_sz_A_group.py --mode shuffle --model baseline && \
   python scripts/evaluate_sz_A_group.py --mode moments --model baseline" 2>&1 | tee -a "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

cd ops && git add . && git commit -m "result 48_A_group_eval" && git push
