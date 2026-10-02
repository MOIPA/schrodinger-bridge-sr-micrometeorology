#!/bin/bash
# 60b_phase2_waves.sh — 阶段 2 训练波次提交。参数: 1|2|3
#   W1 = p2_l2 p2_div_mid
#   W2 = p2_spec p2_ext p2_vort p2_div_lo p2_div_hi   (需人工判定 W1 后产生的 results/phase2/GATE_W1_PASS)
#   W3 = p2_combo                                     (需 GATE_W2_PASS;86 变体表若不含 combo 则跳过)
# 闸门文件只由人工创建并推送,服务器 git pull 到之后本脚本才放行;例如:
#   touch results/phase2/GATE_W1_PASS && git add results/phase2 && \
#     git commit -m "gate: W1 pass" && git push
# 规则同 59b:提交后 90 秒检查,凡 PEND 立即 bkill 并换队列重投;作业名 = run tag(如 p2_l2),
# 内部日志 logs/p2_<tag>.log,LSF 日志 logs/p2_<tag>_%J.out。
# 手机执行: bash ops/queue/60b_phase2_waves.sh 1
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
git --no-pager log --oneline -1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/60b_phase2_waves.txt
mkdir -p ops/result logs
: > "$OUT"

EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
CFG_DIR="configs/深圳/phase2"
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"  # 9654p6000ib 为 Blackwell,与 wind3d torch 不兼容
QUEUES="${FORCE_Q:-$QUEUES}"  # 队列异常时覆盖:FORCE_Q=6148v100ib bash ops/queue/60b_phase2_waves.sh 2

W1="p2_l2 p2_div_mid"
W2="p2_spec p2_ext p2_vort p2_div_lo p2_div_hi p2_div_mid_warm"
W3="p2_combo"

WAVE="${1:-}"
case "$WAVE" in
  1) TAGS="$W1";;
  2) TAGS="$W2";;
  3) TAGS="$W3";;
  *) echo "用法: bash ops/queue/60b_phase2_waves.sh 1|2|3"; exit 1;;
esac

# 配置/checkpoint 路径规则(与 86 生成器/训练脚本一致):config_wind_canvas_p2_<短名>.yml
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

echo "===== 0. 队列里的 p2_ 作业 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep "p2_" >> "$OUT" || echo "(无)" >> "$OUT"

# 硬闸门:W2/W3 必须先有人工判定文件(不自动创建)
if [ "$WAVE" != "1" ]; then
  GATE="results/phase2/GATE_W$((WAVE-1))_PASS"
  if [ ! -f "$GATE" ]; then
    {
      echo ""
      echo "===== 闸门:$GATE 不存在,拒绝提交 W$WAVE ====="
      echo "先跑 60e: bash ops/queue/60e_phase2_gate.sh -> results/phase2/ranking_phase2_gate.md"
      echo "人工判定通过后创建闸门文件并推送(服务器需 git pull 到):"
      echo "  touch $GATE && git add results/phase2 && git commit -m \"gate: W$((WAVE-1)) pass\" && git push"
    } >> "$OUT"
    cat "$OUT"
    git add ops/result
    git commit -m "result 60b phase2 waves: W$WAVE blocked by gate"
    git push || { git fetch origin && git merge --no-edit origin/main && git push; }
    exit 0
  fi
fi

echo "" >> "$OUT"
echo "===== 1. W$WAVE 提交清单: $TAGS =====" >> "$OUT"
SUBMITTED=""
for tag in $TAGS; do
  CK=$(ck_of "$tag")
  CFG=$(cfg_of "$tag")
  if [ -f "$CK" ]; then
    # checkpoint 存在 ≠ 已完成:训练中断(节点争抢 SIGINT 等)也会留下 best checkpoint,
    # 以日志里的 "Train end:" 作为正常结束标记,否则重投续训(checkpoint 自动 resume)
    if grep -q "Train end:" "logs/${tag}.log" 2>/dev/null; then
      echo "[跳过] $tag 已完成(checkpoint + Train end 标记)" >> "$OUT"
      continue
    fi
    echo "[续训] $tag 有 checkpoint 但未见 Train end(疑似中断),重投续训" >> "$OUT"
  fi
  if [ ! -f "$CFG" ]; then
    echo "[警告] 缺配置 $CFG(先跑 60_phase2_setup.sh 生成;W3 的 p2_combo 需 86 变体表含 combo),跳过 $tag" >> "$OUT"
    continue
  fi
  if bjobs -o "job_name stat" -noheader 2>/dev/null | awk '$2=="RUN"{print $1}' | grep -qx "$tag"; then
    echo "[跳过] $tag 正在 RUN" >> "$OUT"
    continue
  fi
  for JID in $(bjobs -o "jobid job_name stat" -noheader 2>/dev/null \
      | awk -v n="$tag" '$2==n && $3=="PEND"{print $1}'); do
    echo ">>> 清理 PEND 作业 $JID ($tag),换队列重投" >> "$OUT"
    bkill "$JID" >> "$OUT" 2>&1
  done
  Q=$(pick_queue)
  echo ">>> $tag -> 队列 $Q" >> "$OUT"
  submit_one "$tag" "$Q" >> "$OUT"
  SUBMITTED="$SUBMITTED $tag"
done

if [ -z "$SUBMITTED" ]; then
  echo "" >> "$OUT"
  echo "没有可补投的配置(要么在跑,要么已完成,要么缺配置)。" >> "$OUT"
  cat "$OUT"
  git add ops/result
  git commit -m "result 60b phase2 waves W$WAVE: nothing to submit"
  git push || { git fetch origin && git merge --no-edit origin/main && git push; }
  exit 0
fi

echo "" >> "$OUT"
echo "===== 2. 90 秒后 PEND 检查(PEND 立即换队列重投) =====" >> "$OUT"
sleep 90
bjobs -w 2>/dev/null | grep "p2_" >> "$OUT" || echo "(bjobs 无 p2_ 记录?)" >> "$OUT"
for tag in $SUBMITTED; do
  LINE=$(bjobs -w 2>/dev/null | grep " $tag ")
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

echo "" >> "$OUT"
echo "===== 3. 后续 =====" >> "$OUT"
echo "查状态: bash ops/queue/60c_check_phase2.sh" >> "$OUT"
echo "闸门评估: bash ops/queue/60e_phase2_gate.sh(人工判定后创建 results/phase2/GATE_W${WAVE}_PASS 并推送)" >> "$OUT"
if [ "$WAVE" != "3" ]; then
  echo "下一波: bash ops/queue/60b_phase2_waves.sh $((WAVE+1))" >> "$OUT"
fi
cat "$OUT"
git add ops/result
git commit -m "result 60b phase2 waves W$WAVE"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
