#!/bin/bash
# 49_A_restart.sh — A 组剩余任务一键重启(修复显示名 bug 后)
#   1. phys diag + allLR diag + moments(全部 CPU, 登录节点, 内部日志)
#   2. shuffle 单独 bsub GPU(6148v100ib)
#   3. 完成/提交后自动 git 回传
# 运行: bash ops/queue/49_A_restart.sh   (约 30-40 分钟, 建议 setsid 或让终端挂着)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p logs results/A_group ops/result
module load anaconda/3 2>/dev/null; source activate wind3d 2>/dev/null
OUT=ops/result/49_A_restart.txt
: > "$OUT"

echo "=== phys diag (CPU) $(date) ===" >> "$OUT"
python -u scripts/evaluate_sz_A_group.py --mode diag --model phys --device cpu \
    > logs/sz_A_phys_cpu.log 2>&1
echo "phys exit=$?" >> "$OUT"

echo "=== allLR diag (CPU) $(date) ===" >> "$OUT"
python -u scripts/evaluate_sz_A_group.py --mode diag --model allLR --device cpu \
    > logs/sz_A_allLR_cpu.log 2>&1
echo "allLR exit=$?" >> "$OUT"

echo "=== moments (CPU) $(date) ===" >> "$OUT"
python -u scripts/evaluate_sz_A_group.py --mode moments --model baseline --device cpu \
    > logs/sz_A_moments_cpu.log 2>&1
echo "moments exit=$?" >> "$OUT"

echo "=== shuffle (GPU bsub) $(date) ===" >> "$OUT"
bsub -q 6148v100ib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J sz_A_shuffle -o logs/sz_A_shuffle_%J.out \
    "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python -u scripts/evaluate_sz_A_group.py --mode shuffle --model baseline > logs/sz_A_shuffle.log 2>&1" \
    >> "$OUT" 2>&1

echo "=== 完成 $(date) ===" >> "$OUT"
ls results/A_group/ >> "$OUT"
cat "$OUT"
cd ops && git add . && git commit -m "result 49_A_restart" && git push
