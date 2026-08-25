#!/bin/bash
# 26_status_all.sh — 总状态: 所有训练任务状态 + 各日志 loss 进度 + 异常检查
# 手机执行: bash ops/queue/26_status_all.sh (可随时重复跑)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/26_status_all.txt
: > "$OUT"

echo "===== 1. 所有任务 =====" >> "$OUT"
bjobs -w 2>/dev/null | head -12 >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 各任务 loss 进度 =====" >> "$OUT"
for name in sz_smoke sz_baseline sz_lrcond sz_day sz_pinn sz_night; do
  echo "--- $name ---" >> "$OUT"
  LATEST=$(ls -t logs/${name}_*.out 2>/dev/null | grep -v cpu | head -1)
  if [ -z "$LATEST" ]; then
    echo "(无日志)" >> "$OUT"
    continue
  fi
  echo "日志: $LATEST ($(stat -c %y "$LATEST" 2>/dev/null | cut -d. -f1))" >> "$OUT"
  grep -a "Epoch [0-9]* /\|avg loss" "$LATEST" 2>/dev/null | tail -4 >> "$OUT"
  echo "异常检查:" >> "$OUT"
  grep -a -i "error\|traceback\|cuda" "$LATEST" 2>/dev/null | tail -2 >> "$OUT" || echo "(无异常)" >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 3. 完成情况(各任务 checkpoint) =====" >> "$OUT"
for d in config_wind_3d_sz_baseline config_wind_3d_sz_lrcond config_wind_3d_sz_day config_wind_3d_sz_pinn config_wind_3d_sz_night; do
  CKPT=data/DL_result/ExperimentSchrodingerBridge3dWind/$d/checkpoint.pth
  if [ -f "$CKPT" ]; then
    echo "$d: $(stat -c %y "$CKPT" 2>/dev/null | cut -d. -f1) $(du -h "$CKPT" | cut -f1)" >> "$OUT"
  else
    echo "$d: (尚无 checkpoint)" >> "$OUT"
  fi
done

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 26_status_all" && git push
