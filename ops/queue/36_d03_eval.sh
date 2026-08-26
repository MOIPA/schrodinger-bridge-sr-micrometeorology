#!/bin/bash
# 36_d03_eval.sh — 跨域评估:现有模型(baseline/lrcond/pinn)在原生 d03 数据上评估
# 实验#2:训练在模拟 LR、评估在原生 d03,量化分布偏移
# 手机执行: bash ops/queue/36_d03_eval.sh (提交后约 10-20 分钟,结果用 37 查)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/36_d03_eval.txt
: > "$OUT"

echo "===== 1. 建数据软链 wrf_3d_v1_sz_d03 =====" >> "$OUT"
ln -sfn ~/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d_sz_d03 \
  data/DL_data/wrf_3d_v1_sz_d03
ls -la data/DL_data/wrf_3d_v1_sz_d03 | head -3 >> "$OUT"
echo "软链数据 npz 数: $(ls data/DL_data/wrf_3d_v1_sz_d03/*.npz 2>/dev/null | wc -l)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 检查配置与 checkpoint =====" >> "$OUT"
for cfg in config_wind_3d_sz_d03_baseline.yml config_wind_3d_sz_d03_lrcond.yml; do
  if [ -f "configs/深圳/$cfg" ]; then echo "  配置 OK: $cfg"; else echo "  配置缺失: $cfg"; fi
done
for name in baseline lrcond pinn; do
  CKPT=data/DL_result/ExperimentSchrodingerBridge3dWind/config_wind_3d_sz_${name}/checkpoint.pth
  if [ -f "$CKPT" ]; then echo "  $name checkpoint OK ($(du -h "$CKPT" | cut -f1))"; else echo "  $name checkpoint 缺失"; fi
done

echo "" >> "$OUT"
echo "===== 3. 提交跨域评估任务到 83a100ib =====" >> "$OUT"
bsub -q 83a100ib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_d03_eval \
  -o logs/sz_d03_eval_%J.out -e logs/sz_d03_eval_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/evaluate_sz_experiments.py --d03 --device cuda:0 --results_dir results/sz_eval_d03" \
  2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 任务状态 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep sz_d03_eval | head -2 >> "$OUT"
echo "(任务跑完结果在 results/sz_eval_d03/,评估日志在 logs/sz_d03_eval_*.out)" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 36_d03_eval" && git push
