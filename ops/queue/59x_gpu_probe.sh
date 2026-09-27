#!/bin/bash
# 59x_gpu_probe.sh — 逐队列探测 wind3d torch 与队列 GPU 的兼容性(1 分钟出结果)
# 手机执行: bash ops/queue/59x_gpu_probe.sh ; 之后 cat ops/result/59x_gpu_probe.txt
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/59x_gpu_probe.txt
mkdir -p ops/result logs
: > "$OUT"
for q in e5v4p100ib 6148v100ib 7552v100 62v100ib 72rtxib 9654p6000ib 7k83 83a100ib; do
  if ! bqueues -w "$q" > /dev/null 2>&1; then
    echo "QUEUE=$q 不存在,跳过" >> "$OUT"
    continue
  fi
  bsub -q "$q" -gpu "num=1:mode=exclusive_process" -n 2 -J "gpuprobe" \
    -o "logs/gpuprobe_${q}_%J.out" -e "logs/gpuprobe_${q}_%J.err" \
    "cd $ROOT && echo \"QUEUE=$q\" && /fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python -u scripts/outline/93_gpu_probe.py || echo \"QUEUE=$q FAIL\"" \
    > /dev/null 2>&1
  echo "已提交 $q" >> "$OUT"
done
echo "探测作业已提交,约 1-2 分钟后用下面命令看结果:" >> "$OUT"
echo "  cat ops/result/59x_gpu_probe.txt ; tail -5 logs/gpuprobe_*_*.out" >> "$OUT"
cat "$OUT"
cd ops && git add result && git commit -m "result 59x gpu probe submitted" && git push || true
