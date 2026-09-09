#!/bin/bash
# 50_A_gpu.sh — A 组剩余全部改走 GPU(CPU 推理太慢, 每模型 1h+)
#   单个 bsub 任务串行跑: phys diag -> allLR diag -> moments -> shuffle
#   每步 python -u + 内部日志(规避 LSF 输出丢失), 完成后自动回传 results/
# 运行: bash ops/queue/50_A_gpu.sh   (总耗时约 40-60 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p logs results/A_group ops/result
OUT=ops/result/50_A_gpu.txt
: > "$OUT"

# 先杀掉还在跑的 CPU 版(phys/allLR/moments),避免占 CPU
pkill -f "evaluate_sz_A_group.py --mode diag" 2>/dev/null
pkill -f "evaluate_sz_A_group.py --mode moments" 2>/dev/null
sleep 2

bsub -q 6148v100ib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J sz_A_gpu -o logs/sz_A_gpu_%J.out \
    "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
     python -u scripts/evaluate_sz_A_group.py --mode diag --model phys > logs/sz_A_phys.log 2>&1 && \
     python -u scripts/evaluate_sz_A_group.py --mode diag --model allLR > logs/sz_A_allLR.log 2>&1 && \
     python -u scripts/evaluate_sz_A_group.py --mode moments --model baseline > logs/sz_A_moments.log 2>&1 && \
     python -u scripts/evaluate_sz_A_group.py --mode shuffle --model baseline > logs/sz_A_shuffle.log 2>&1" \
    >> "$OUT" 2>&1

echo "=== 已提交 $(date) ===" >> "$OUT"
cat "$OUT"
cd ops && git add . && git commit -m "result 50_A_gpu" && git push
