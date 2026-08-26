#!/bin/bash
# 41_d03_lrcond_train.sh — 提交 lrcond-d03 训练(真实部署全低精度场景)
# 配置:config_wind_3d_sz_d03_lrcond.yml(输入 18 lr wind + 8 lr 条件,全部 d03 统计)
# 选队列升级:避开已有 RUN 任务的队列,两个 d03 训练分到不同队列并行
# 手机执行: bash ops/queue/41_d03_lrcond_train.sh (约 4 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/41_d03_lrcond_train.txt
: > "$OUT"

echo "===== 1. 验证配置可加载 =====" >> "$OUT"
module load anaconda/3 2>/dev/null; source activate wind3d 2>/dev/null || conda activate wind3d 2>/dev/null
python - <<'PYEOF' >> "$OUT" 2>&1
import sys
sys.path.insert(0, ".")
from src.dl_config.config_loader import load_config
c = load_config("ExperimentSchrodingerBridge3dWind",
                "configs/深圳/config_wind_3d_sz_d03_lrcond.yml")
print("配置加载 OK")
print("  in_channel:", c.model.in_channel, "(预期 44)")
print("  dl_data_ver:", c.loader.dl_data_ver)
print("  input 前4个:", c.data.input_variable_names[:4], "...")
PYEOF

echo "" >> "$OUT"
echo "===== 2. 自动选空闲队列(PEND=0 且 RUN 最少,避开已有任务) =====" >> "$OUT"
pick_queue() {
  local best="83a100ib" best_run=999999
  for q in 72rtxib e5v4p100ib 9654p6000ib 6148v100ib 7552v100 7k83 83a100ib; do
    local PEND RUN
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    RUN=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $10}')
    echo "  $q: PEND=$PEND RUN=$RUN" >> "$OUT"
    if [ "$PEND" = "0" ] 2>/dev/null && [ "$RUN" -lt "$best_run" ] 2>/dev/null; then
      best="$q"; best_run="$RUN"
    fi
  done
  echo "$best"
}
QUEUE=$(pick_queue)
echo ">>> 选中队列: $QUEUE" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 提交 lrcond-d03 训练 =====" >> "$OUT"
bsub -q "$QUEUE" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_d03lrcond \
  -o logs/sz_d03lrcond_%J.out -e logs/sz_d03lrcond_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/train_schrodinger_bridge_model.py --config_path configs/深圳/config_wind_3d_sz_d03_lrcond.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0" \
  2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 等 3 分钟看启动情况 =====" >> "$OUT"
sleep 180
bjobs -w 2>/dev/null | grep -a "sz_d03lrcond\|sz_d03base" >> "$OUT" || echo "(无 d03 训练任务?)" >> "$OUT"
LATEST=$(ls -t logs/sz_d03lrcond_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST" >> "$OUT"
  tail -6 "$LATEST" >> "$OUT"
  grep -a "avg loss\|Traceback\|CUDA error" "$LATEST" 2>/dev/null | tail -3 >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 41_d03_lrcond_train" && git push
