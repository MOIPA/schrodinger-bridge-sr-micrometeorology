#!/bin/bash
# 44_d03_pinn_train.sh — 提交 pinn-d03 训练(真实退化 + 物理约束)
# 配置:config_wind_3d_sz_d03_pinn.yml(d03 输入 + HR 条件 + div=0.1 vort=0.05)
# 补全交叉矩阵:d03 训练 {baseline, lrcond, pinn} × 评估 d03,与 d04 侧 pinn 同口径
# 选队列:避开 P6000(sm61 无 cu118 kernel)、选 PEND=0 且 RUN 最少的兼容队列
# 手机执行: bash ops/queue/44_d03_pinn_train.sh (约 4 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/44_d03_pinn_train.txt
: > "$OUT"

echo "===== 1. 检查是否已有训练(防重复提交) =====" >> "$OUT"
CKPT=data/DL_result/ExperimentSchrodingerBridge3dWind/config_wind_3d_sz_d03_pinn/checkpoint.pth
if [ -f "$CKPT" ]; then
  echo "已有 checkpoint: $(stat -c %y "$CKPT" | cut -d. -f1) $(du -h "$CKPT" | cut -f1)" >> "$OUT"
  echo ">>> 已训练过,中止提交(如需重训先手动删除该 checkpoint)" >> "$OUT"
  echo "===== 完成(中止) =====" >> "$OUT"
  cat "$OUT"
  cd ops && git add . && git commit -m "result 44_d03_pinn_train" && git push
  exit 0
fi
echo "无既有 checkpoint,继续" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 验证配置可加载 =====" >> "$OUT"
module load anaconda/3 2>/dev/null; source activate wind3d 2>/dev/null || conda activate wind3d 2>/dev/null
python - <<'PYEOF' >> "$OUT" 2>&1
import sys
sys.path.insert(0, ".")
from src.dl_config.config_loader import load_config
c = load_config("ExperimentSchrodingerBridge3dWind",
                "configs/深圳/config_wind_3d_sz_d03_pinn.yml")
print("配置加载 OK")
print("  in_channel:", c.model.in_channel, "(预期 44)")
print("  dl_data_ver:", c.loader.dl_data_ver)
print("  div/vort:", c.si.divergence_weight, c.si.vorticity_weight, "(预期 0.1/0.05)")
print("  条件前4个:", c.data.input_variable_names[18:22], "(预期 HR 条件)")
PYEOF

echo "" >> "$OUT"
echo "===== 3. 选兼容队列(PEND=0 且 RUN 最少,排除 P6000) =====" >> "$OUT"
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
echo "===== 4. 提交 pinn-d03 训练 =====" >> "$OUT"
bsub -q "$QUEUE" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_d03pinn \
  -o logs/sz_d03pinn_%J.out -e logs/sz_d03pinn_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/train_schrodinger_bridge_model.py --config_path configs/深圳/config_wind_3d_sz_d03_pinn.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0" \
  2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 5. 等 3 分钟确认启动 =====" >> "$OUT"
sleep 180
bjobs -w 2>/dev/null | grep -a "sz_d03pinn\|sz_d03base\|sz_d03lrcond" >> "$OUT" || echo "(无 d03 训练任务?)" >> "$OUT"
LATEST=$(ls -t logs/sz_d03pinn_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST" >> "$OUT"
  tail -8 "$LATEST" >> "$OUT"
  grep -a "avg loss\|Traceback\|CUDA error" "$LATEST" 2>/dev/null | tail -3 >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 44_d03_pinn_train" && git push
