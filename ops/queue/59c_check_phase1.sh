#!/bin/bash
# 59c_check_phase1.sh — 阶段 1 训练状态:队列/日志尾/checkpoint/波次完成度
# 手机执行: bash ops/queue/59c_check_phase1.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
OUT=ops/result/59c_check_phase1.txt
mkdir -p ops/result
: > "$OUT"

EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP

ALL="base t13_most t13_mostflux t13_w t13_theta t13_ph t12_zagldiff t12_hgtdiff t15_z0 t15_z0_urban t15_z0_urban_wv t14_cos t14_noenc_cos t16_residual t17_coords"

echo "===== 1. 队列里的 p1_ 作业 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep "p1_" >> "$OUT" || echo "(无)" >> "$OUT"
echo "" >> "$OUT"

echo "===== 2. 逐配置:checkpoint / 最新日志尾 =====" >> "$OUT"
DONE=0
for tag in $ALL; do
  DIR="$RESULT_BASE/config_wind_canvas_p1_$tag"
  if [ -f "$DIR/checkpoint.pth" ]; then
    DONE=$((DONE+1))
    MT=$(date -r "$DIR/checkpoint.pth" "+%m-%d %H:%M" 2>/dev/null)
    LINE=$(grep -a "Epoch " "$DIR/log.txt" 2>/dev/null | tail -1)
    LOSS=$(grep -a "avg loss" "$DIR/log.txt" 2>/dev/null | tail -1)
    echo "[OK] $tag  ckpt=$MT  $LINE  $LOSS" >> "$OUT"
  else
    LOG="logs/p1_${tag}.log"
    if [ -f "$LOG" ]; then
      echo "[..] $tag  $(grep -ac 'Epoch ' "$LOG" 2>/dev/null) epoch 行; $(tail -1 "$LOG" 2>/dev/null | cut -c1-120)" >> "$OUT"
    else
      echo "[--] $tag  无日志" >> "$OUT"
    fi
  fi
done
echo "" >> "$OUT"
echo "checkpoint 完成: $DONE / 15" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 错误扫描 =====" >> "$OUT"
grep -al "Traceback\|CUDA error\|RuntimeError" logs/p1_*.log 2>/dev/null >> "$OUT" || echo "(无)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. GPU 队列现状 =====" >> "$OUT"
for q in 72rtxib e5v4p100ib 9654p6000ib 6148v100ib 7552v100 7k83 83a100ib; do
  L=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9, $10}')
  echo "  $q: PEND RUN = $L" >> "$OUT"
done

cat "$OUT"
cd ops && git add result && git commit -m "result 59c phase1 check" && git push || true
