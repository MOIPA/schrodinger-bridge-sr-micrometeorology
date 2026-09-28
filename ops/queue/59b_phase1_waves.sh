#!/bin/bash
# 59b_phase1_waves.sh — 阶段 1 训练波次提交(4 波 4/4/4/3)
# 逻辑:队列里有 p1_ 作业就只报告不提交;否则提交"第一个未完成波次"中缺 checkpoint 的配置。
# 队列入队即 PEND 的规则:提交后 90 秒检查,凡 PEND 立即 bkill 并换队列重投。
# 手机执行: bash ops/queue/59b_phase1_waves.sh   (每波结束后重跑即可)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/59b_phase1_waves.txt
mkdir -p ops/result logs
: > "$OUT"

EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
# 队列白名单(59x_gpu_probe 实测):9654p6000ib = RTX PRO 6000 Blackwell(sm_120),
# wind3d 的 torch 2.6+cu118 只支持到 sm_90 -> "no kernel image" 直接崩,禁用;
# 72rtxib / 7k83 未验证(可能同为 Ada/Blackwell),暂不列入。
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"

WAVE1="base t13_most t13_mostflux t13_w"
WAVE2="t13_theta t13_ph t12_zagldiff t12_hgtdiff"
WAVE3="t15_z0 t15_z0_urban t15_z0_urban_wv t14_cos"
WAVE4="t14_noenc_cos t16_residual t17_coords"
WAVES=("$WAVE1" "$WAVE2" "$WAVE3" "$WAVE4")

cfg_path() { echo "configs/深圳/phase1/config_wind_canvas_p1_$1.yml"; }
ck_path()  { echo "$RESULT_BASE/config_wind_canvas_p1_$1/checkpoint.pth"; }

pick_queue() {
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then echo "$q"; return; fi
  done
  echo "83a100ib"
}

submit_one() {  # $1=tag $2=queue
  local tag=$1 q=$2
  bsub -q "$q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "p1_$tag" -o "logs/p1_${tag}_%J.out" -e "logs/p1_${tag}_%J.err" \
    "cd $ROOT && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
python -u scripts/train_schrodinger_bridge_model.py \
--config_path configs/深圳/phase1/config_wind_canvas_p1_${tag}.yml \
--experiment_name $EXP --device cuda:0 > logs/p1_${tag}.log 2>&1" \
    2>&1 | head -1
}

echo "===== 0. 队列里的 p1_ 作业 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep "p1_" >> "$OUT" || echo "(无)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 1. 找第一个有缺口的波次(RUN 中算在跑;PEND 算缺口,先清再换队列) =====" >> "$OUT"
RUN_NAMES=$(bjobs -o "job_name stat" -noheader 2>/dev/null | awk '$2=="RUN"{print $1}')
TOSUBMIT=""
WAVE_NO=0
for w in 1 2 3 4; do
  MISS=""
  CAND=""
  for tag in ${WAVES[$((w-1))]}; do
    ck=$(ck_path "$tag")
    if [ ! -f "$ck" ]; then
      MISS="$MISS $tag"
      if ! echo "$RUN_NAMES" | grep -qx "p1_$tag"; then CAND="$CAND $tag"; fi
    fi
  done
  echo "波次 $w: 缺 checkpoint:${MISS:- 无};可补投:${CAND:- 无}" >> "$OUT"
  if [ -n "$CAND" ] && [ -z "$TOSUBMIT" ]; then
    TOSUBMIT="$CAND"; WAVE_NO=$w
  fi
done

# 只清理"本轮要补投"配置中仍 PEND 的旧作业(不误伤其它波次)
for tag in $TOSUBMIT; do
  for JID in $(bjobs -o "jobid job_name stat" -noheader 2>/dev/null \
      | awk -v n="p1_$tag" '$2==n && $3=="PEND"{print $1}'); do
    echo ">>> 清理 PEND 作业 $JID (p1_$tag),换队列重投" >> "$OUT"
    bkill "$JID" >> "$OUT" 2>&1
  done
done
[ -n "$TOSUBMIT" ] && sleep 3

if [ -z "$TOSUBMIT" ]; then
  echo "" >> "$OUT"
  echo "没有可补投的配置(要么在跑,要么已完成)。若 15 个 checkpoint 齐全,可跑 59d 评估。" >> "$OUT"
  cat "$OUT"; exit 0
fi
echo "" >> "$OUT"
echo "===== 2. 提交(波次 $WAVE_NO): $TOSUBMIT =====" >> "$OUT"
for tag in $TOSUBMIT; do
  Q=$(pick_queue)
  echo ">>> $tag -> 队列 $Q" >> "$OUT"
  submit_one "$tag" "$Q" >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 3. 90 秒后检查是否 PEND(PEND 立即换队列重投) =====" >> "$OUT"
sleep 90
bjobs -w 2>/dev/null | grep "p1_" >> "$OUT" || echo "(bjobs 无记录?)" >> "$OUT"
for tag in $TOSUBMIT; do
  LINE=$(bjobs -w 2>/dev/null | grep " p1_$tag ")
  STAT=$(echo "$LINE" | awk '{print $3}')
  if [ "$STAT" = "PEND" ]; then
    JID=$(echo "$LINE" | awk '{print $1}')
    echo ">>> $tag($JID) PEND,换队列重投" >> "$OUT"
    bkill "$JID" >> "$OUT" 2>&1
    Q=$(pick_queue)
    echo ">>> $tag -> 新队列 $Q" >> "$OUT"
    submit_one "$tag" "$Q" >> "$OUT"
  fi
done
echo "===== 完成(每波结束后重跑本脚本提交下一波) =====" >> "$OUT"
cat "$OUT"

cd ops && git add result && git commit -m "result 59b phase1 waves" && git push || true
