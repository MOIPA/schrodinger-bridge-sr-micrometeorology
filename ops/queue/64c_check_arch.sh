#!/bin/bash
# 64c_check_arch.sh — 阶段 3 架构训练状态:队列概览 / 逐 tag 日志尾 / checkpoint / 错误扫描
#   + PEND 即换队列重投(复刻 63c;按 tag 分派训练脚本,与 64b 同规则)
# 前置依赖:64b 已提交过训练(否则显示"配置在但无日志")。
# 手机执行: bash ops/queue/64c_check_arch.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/64c_check_arch.txt
mkdir -p ops/result logs results/phase3_arch
: > "$OUT"

EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
CFG_DIR="configs/深圳/phase3_arch"
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"

ALL="p3_arch_unet_s2 p3_arch_swin p3_arch_swin_s2 p3_arch_reg p3_arch_reg_s2 p3_arch_edm p3_arch_edm_s2"

cfg_of() { echo "$CFG_DIR/config_wind_canvas_${1}.yml"; }
ck_of()  { echo "$RESULT_BASE/config_wind_canvas_${1}/checkpoint.pth"; }

reg_of() {
  case "$1" in
    p3_arch_edm_s2) echo p3_arch_reg_s2;;
    p3_arch_edm)    echo p3_arch_reg;;
    *) echo "$1";;
  esac
}

pick_queue() {
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
  local tag=$1 q=$2 script extra=""
  case "$tag" in
    p3_arch_reg*) script=scripts/train_regression_model.py;;
    p3_arch_edm*)
      script=scripts/train_edm_correction.py
      local rt; rt=$(reg_of "$tag")
      extra="--reg_tag $rt --reg_config_path $CFG_DIR/config_wind_canvas_${rt}.yml";;
    *) script=scripts/train_schrodinger_bridge_model.py;;
  esac
  bsub -q "$q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "$tag" -o "logs/${tag}_%J.out" -e "logs/${tag}_%J.err" \
    "cd $ROOT && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
python -u $script \
--config_path $CFG_DIR/config_wind_canvas_${tag}.yml \
--experiment_name $EXP --device cuda:0 $extra > logs/${tag}.log 2>&1" \
    2>&1 | head -1
}

echo "===== 1. 队列里的 p3_arch 作业 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep "p3_arch" >> "$OUT" || echo "(无)" >> "$OUT"
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
    END=$(grep -ac "Train end:" "logs/${tag}.log" 2>/dev/null)
    echo "[OK] $tag  ckpt=$MT  Train end 标记=$END  $LINE  $LOSS" >> "$OUT"
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
echo "checkpoint 完成: $DONE / 7" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 错误扫描 =====" >> "$OUT"
grep -al "Traceback\|CUDA error\|RuntimeError" logs/p3_arch_*.log 2>/dev/null >> "$OUT" || echo "(无)" >> "$OUT"

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
    case "$tag" in
      p3_arch_edm*)
        RT=$(reg_of "$tag")
        if [ ! -f "$(ck_of "$RT")" ]; then
          echo ">>> $tag($JID) PEND 且配对回归 $RT 无 checkpoint,跳过(等 W1)" >> "$OUT"
          continue
        fi
        ;;
    esac
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
git add results/phase3_arch 2>/dev/null
git commit -m "result 64c arch check"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
