#!/bin/bash
# 60c_check_phase2.sh — 阶段 2 训练状态:队列概览 / 逐 tag 日志尾 / checkpoint / 错误扫描
#   + PEND 即换队列(复用 59c 的检查 + 59b 的换队列重投做法)
# 手机执行: bash ops/queue/60c_check_phase2.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/60c_check_phase2.txt
mkdir -p ops/result logs results/phase2
: > "$OUT"

EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
CFG_DIR="configs/深圳/phase2"
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"

ALL="p2_l2 p2_div_mid p2_spec p2_ext p2_vort p2_div_lo p2_div_hi p2_combo"

cfg_of() { echo "$CFG_DIR/config_wind_canvas_p2_${1#p2_}.yml"; }
ck_of()  { echo "$RESULT_BASE/config_wind_canvas_p2_${1#p2_}/checkpoint.pth"; }

pick_queue() {
  # 只在"没有排队积压(PEND=0)"的队列里挑,并选 RUN 最少的(近似空闲 GPU 最多)
  local BEST="" BESTRUN=1000000 P R
  for q in $QUEUES; do
    read -r P R <<< "$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9, $10}')"
    [ "$P" = "0" ] || continue
    [ -z "$R" ] && continue
    if [ "$R" -lt "$BESTRUN" ] 2>/dev/null; then BEST="$q"; BESTRUN="$R"; fi
  done
  echo "${BEST:-83a100ib}"
}

submit_one() {  # $1=tag $2=queue
  local tag=$1 q=$2
  bsub -q "$q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "$tag" -o "logs/${tag}_%J.out" -e "logs/${tag}_%J.err" \
    "cd $ROOT && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
python -u scripts/train_schrodinger_bridge_model.py \
--config_path $CFG_DIR/config_wind_canvas_p2_${tag#p2_}.yml \
--experiment_name $EXP --device cuda:0 > logs/${tag}.log 2>&1" \
    2>&1 | head -1
}

echo "===== 1. 队列里的 p2_ 作业 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep "p2_" >> "$OUT" || echo "(无)" >> "$OUT"
echo "" >> "$OUT"

echo "===== 2. 逐配置:checkpoint / 最新日志尾 =====" >> "$OUT"
DONE=0
for tag in $ALL; do
  CK=$(ck_of "$tag")
  CFG=$(cfg_of "$tag")
  if [ -f "$CK" ]; then
    DONE=$((DONE+1))
    MT=$(date -r "$CK" "+%m-%d %H:%M" 2>/dev/null)
    DIR=$(dirname "$CK")
    LINE=$(grep -a "Epoch " "$DIR/log.txt" 2>/dev/null | tail -1)
    LOSS=$(grep -a "avg loss" "$DIR/log.txt" 2>/dev/null | tail -1)
    echo "[OK] $tag  ckpt=$MT  $LINE  $LOSS" >> "$OUT"
  else
    LOG="logs/${tag}.log"
    if [ -f "$LOG" ]; then
      echo "[..] $tag  $(grep -ac 'Epoch ' "$LOG" 2>/dev/null) epoch 行; $(tail -1 "$LOG" 2>/dev/null | cut -c1-120)" >> "$OUT"
    elif [ -f "$CFG" ]; then
      echo "[--] $tag  配置在但无日志(未提交?)" >> "$OUT"
    else
      echo "[--] $tag  无配置无日志(未生成/未提交?)" >> "$OUT"
    fi
  fi
done
echo "" >> "$OUT"
echo "checkpoint 完成: $DONE / 8" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 错误扫描 =====" >> "$OUT"
grep -al "Traceback\|CUDA error\|RuntimeError" logs/p2_*.log 2>/dev/null >> "$OUT" || echo "(无)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. PEND 检查与换队列重投 =====" >> "$OUT"
RESUB=0
for tag in $ALL; do
  LINE=$(bjobs -w 2>/dev/null | grep " $tag ")
  STAT=$(echo "$LINE" | awk '{print $3}')
  if [ "$STAT" = "PEND" ]; then
    JID=$(echo "$LINE" | awk '{print $1}')
    CFG=$(cfg_of "$tag")
    if [ ! -f "$CFG" ]; then
      echo ">>> $tag($JID) PEND 但缺配置 $CFG,跳过" >> "$OUT"
      continue
    fi
    echo ">>> $tag($JID) PEND,换队列重投" >> "$OUT"
    bkill "$JID" >> "$OUT" 2>&1
    Q=$(pick_queue)
    echo ">>> $tag -> 队列 $Q" >> "$OUT"
    submit_one "$tag" "$Q" >> "$OUT"
    RESUB=$((RESUB+1))
  fi
done
[ "$RESUB" = "0" ] && echo "(无 PEND 作业)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 5. GPU 队列现状 =====" >> "$OUT"
for q in $QUEUES 9654p6000ib; do
  L=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9, $10}')
  echo "  $q: PEND RUN = $L" >> "$OUT"
done

cat "$OUT"
git add ops/result
git add results/phase2 2>/dev/null
git commit -m "result 60c phase2 check"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
