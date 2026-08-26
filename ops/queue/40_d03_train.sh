#!/bin/bash
# 40_d03_train.sh — 实验#3:在原生 d03 数据上训练真实场景基准模型
# 配置:config_wind_3d_sz_d03_baseline.yml(输入 lr_* d03 统计,目标 hr_* d04 统计)
# 自动选空闲队列提交,约 3-5 分钟确认启动
# 手机执行: bash ops/queue/40_d03_train.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/40_d03_train.txt
: > "$OUT"

echo "===== 1. 检查数据/配置 =====" >> "$OUT"
echo "d03 npz: $(ls data/DL_data/wrf_3d_v1_sz_d03/*.npz 2>/dev/null | wc -l) (预期 1488)" >> "$OUT"
if [ -f configs/深圳/config_wind_3d_sz_d03_baseline.yml ]; then
  echo "配置 OK" >> "$OUT"
else
  echo "配置缺失!" >> "$OUT"
  echo "===== 完成(中止) =====" >> "$OUT"
  cat "$OUT"
  cd ops && git add . && git commit -m "result 40_d03_train" && git push
  exit 1
fi

echo "" >> "$OUT"
echo "===== 2. 验证配置可加载(in_channel=44 等) =====" >> "$OUT"
module load anaconda/3 2>/dev/null; source activate wind3d 2>/dev/null || conda activate wind3d 2>/dev/null
python - <<'PYEOF' >> "$OUT" 2>&1
import sys
sys.path.insert(0, ".")
from src.dl_config.config_loader import load_config
c = load_config("ExperimentSchrodingerBridge3dWind",
                "configs/深圳/config_wind_3d_sz_d03_baseline.yml")
print("配置加载 OK")
print("  in_channel:", c.model.in_channel, "(预期 44)")
print("  inner_channel:", c.model.inner_channel)
print("  dl_data_ver:", c.loader.dl_data_ver)
print("  epochs:", c.train.epochs if hasattr(c.train, 'epochs') else '?')
print("  input_variables:", len(c.data.input_variable_names), "个")
PYEOF

echo "" >> "$OUT"
echo "===== 3. 自动选空闲队列 =====" >> "$OUT"
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
echo "===== 4. 提交 d03 训练 =====" >> "$OUT"
bsub -q "$QUEUE" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_d03base \
  -o logs/sz_d03base_%J.out -e logs/sz_d03base_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/train_schrodinger_bridge_model.py --config_path configs/深圳/config_wind_3d_sz_d03_baseline.yml --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0" \
  2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 5. 等 3 分钟看启动情况 =====" >> "$OUT"
sleep 180
bjobs -w 2>/dev/null | grep sz_d03base >> "$OUT" || echo "(不在队列?)" >> "$OUT"
LATEST=$(ls -t logs/sz_d03base_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "日志: $LATEST" >> "$OUT"
  tail -6 "$LATEST" >> "$OUT"
  grep -a "avg loss\|Traceback\|CUDA error\|Error" "$LATEST" 2>/dev/null | tail -3 >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 40_d03_train" && git push
