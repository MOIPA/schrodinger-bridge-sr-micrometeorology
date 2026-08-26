#!/bin/bash
# 43_training_status.sh — 训练任务总进度(pinn / d03base / d03lrcond / d03pinn)
# 手机执行: bash ops/queue/43_training_status.sh (随时可跑)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/43_training_status.txt
: > "$OUT"

echo "===== 1. 任务状态 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep -a -E "sz_pinn|sz_d03base|sz_d03lrcond|sz_d03pinn|sz_d03lrpin" | head -10 >> "$OUT"
if ! bjobs -w 2>/dev/null | grep -a -E "sz_pinn|sz_d03base|sz_d03lrcond|sz_d03pinn|sz_d03lrpin" >/dev/null; then
  echo "(训练任务都不在队列,可能全部结束)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 2. 各任务训练进度 =====" >> "$OUT"
for name in sz_pinn sz_d03base sz_d03lrcond sz_d03pinn sz_d03lrpin; do
  echo "--- $name ---" >> "$OUT"
  LATEST=$(ls -t logs/${name}_*.out 2>/dev/null | grep -v cpu | head -1)
  if [ -z "$LATEST" ]; then
    echo "(无日志)" >> "$OUT"
    continue
  fi
  echo "日志: $LATEST ($(stat -c %y "$LATEST" 2>/dev/null | cut -d. -f1))" >> "$OUT"
  grep -a "Epoch [0-9]* /\|avg loss" "$LATEST" 2>/dev/null | tail -4 >> "$OUT"
  # 异常检查
  grep -a "Traceback\|CUDA error\|Error" "$LATEST" 2>/dev/null | tail -1 >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 3. checkpoint 情况 =====" >> "$OUT"
for d in config_wind_3d_sz_pinn config_wind_3d_sz_d03_baseline config_wind_3d_sz_d03_lrcond config_wind_3d_sz_d03_pinn config_wind_3d_sz_d03_lrcond_pinn; do
  CKPT=data/DL_result/ExperimentSchrodingerBridge3dWind/$d/checkpoint.pth
  if [ -f "$CKPT" ]; then
    echo "$d: $(stat -c %y "$CKPT" | cut -d. -f1) $(du -h "$CKPT" | cut -f1)" >> "$OUT"
  else
    echo "$d: (尚无 checkpoint)" >> "$OUT"
  fi
done

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 43_training_status" && git push
