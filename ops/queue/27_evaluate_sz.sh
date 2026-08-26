#!/bin/bash
# 27_evaluate_sz.sh — 一条命令评估深圳全部实验(baseline/lrcond/day/night/pinn)
# 在 A100 上跑 GPU 推理,全量 test + day/night 子集对比
# 手机执行: bash ops/queue/27_evaluate_sz.sh (约 30-60 分钟,跑完自动回传)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/27_evaluate_sz.txt
: > "$OUT"

echo "===== 1. 检查各 checkpoint 是否存在 =====" >> "$OUT"
for name in baseline lrcond day night pinn; do
  CKPT=data/DL_result/ExperimentSchrodingerBridge3dWind/config_wind_3d_sz_${name}/checkpoint.pth
  if [ -f "$CKPT" ]; then
    echo "  $name: $(du -h "$CKPT" | cut -f1) OK" >> "$OUT"
  else
    echo "  $name: (缺失,评估时跳过)" >> "$OUT"
  fi
done

echo "" >> "$OUT"
echo "===== 2. 提交评估任务到 83a100ib (A100) =====" >> "$OUT"
bsub -q 83a100ib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_eval \
  -o logs/sz_eval_%J.out -e logs/sz_eval_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/evaluate_sz_experiments.py --device cuda:0 --results_dir results/sz_eval" \
  2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 任务状态 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep sz_eval | head -3 >> "$OUT"
echo "(任务跑完后结果在 results/sz_eval/,评估日志在 logs/sz_eval_*.out)" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 27_evaluate_sz" && git push
