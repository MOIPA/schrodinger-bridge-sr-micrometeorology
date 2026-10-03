#!/bin/bash
# 62b_check_diag.sh — 训练状态快查(诊断批 + W4 补跑):逐 run 打当前轮次与最近损失行。
# 判读:train/valid 停在 ~0.27~0.30 平台 = 已冻结;~0.08~0.13 = 健康;l2 的损失口径不同(≈0.02~0.03)。
# 用法: bash ops/queue/62b_check_diag.sh [标签列表,默认全部在训 run]
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
TAGS="${*:-p2_spec_warm p2_ext_warm p2_vort_warm p2_l1_r3 p2_l2_r2 p2_l1r2_cos p2_l1r2_lr2e4 p2_l1_r2_long}"
RES=data/DL_result/ExperimentSchrodingerBridgeWindCanvas
echo "== 队列 =="
bjobs -w 2>/dev/null | grep p2 | awk '{print $1, $3, $4, $7}'
echo "== 逐 run(末轮 + 最近 3 行 train,valid)=="
for t in $TAGS; do
  echo "-- $t $(grep -oE 'Epoch [0-9]+' logs/$t.log 2>/dev/null | tail -1)"
  tail -3 "$RES/config_wind_canvas_$t/model_loss_history.csv" 2>/dev/null
done
