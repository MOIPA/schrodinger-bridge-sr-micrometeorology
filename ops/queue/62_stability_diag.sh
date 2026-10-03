#!/bin/bash
# 62_stability_diag.sh — 训练稳定性诊断批(2026-10-03 冻结现象:配置无关的 ~50% 轨迹崩塌)
# 4 个短实验(快速节点 ~2.5h/个,慢节点 ~6h):
#   p2_l1_r3       基线重复(seed 78270)          —— 崩塌率样本(第 2 个独立 seed)
#   p2_l2_r2       L2 重复(seed 78270)           —— 检验"L2 更稳"线索(l2 原臂 best 132 未冻)
#   p2_l1r2_cos    l1_r2 配置 + cosine lr 衰减   —— 候选修复 A(seed 78269 已知冻结)
#   p2_l1r2_lr2e4  l1_r2 配置 + lr 2e-4          —— 候选修复 B(seed 78269 已知冻结)
# 判读:修复臂存活到 150 轮 → 候选有效;仍冻 → 该候选无效。l1_r3/l2_r2 补崩塌率数据。
# 手机执行: bash ops/queue/62_stability_diag.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
git --no-pager log --oneline -1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/62_stability_diag.txt
mkdir -p ops/result logs
: > "$OUT"

EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
CFG_DIR="configs/深圳/phase2"
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"
QUEUES="${FORCE_Q:-$QUEUES}"
# 可选参数:只投指定 tag;默认投全部诊断臂
TAGS="${*:-p2_l1_r3 p2_l2_r2 p2_l1r2_cos p2_l1r2_lr2e4}"

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

submit_one() {
  local tag=$1 q=$2
  bsub -q "$q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "$tag" -o "logs/${tag}_%J.out" -e "logs/${tag}_%J.err" \
    "cd $ROOT && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
python -u scripts/train_schrodinger_bridge_model.py \
--config_path $CFG_DIR/config_wind_canvas_${tag}.yml \
--experiment_name $EXP --device cuda:0 > logs/${tag}.log 2>&1" \
    2>&1 | head -1
}

echo "===== 0. 诊断批提交: $TAGS =====" >> "$OUT"
SUBMITTED=""
for tag in $TAGS; do
  CK="$RESULT_BASE/config_wind_canvas_${tag}/checkpoint.pth"
  CFG="$CFG_DIR/config_wind_canvas_${tag}.yml"
  if bjobs -w 2>/dev/null | grep -q " $tag "; then
    echo "[跳过] $tag 已在队列" >> "$OUT"; continue
  fi
  if [ ! -f "$CFG" ]; then echo "[警告] 缺配置 $CFG,跳过 $tag" >> "$OUT"; continue; fi
  Q=$(pick_queue)
  echo ">>> $tag -> 队列 $Q" >> "$OUT"
  submit_one "$tag" "$Q" >> "$OUT"
  SUBMITTED="$SUBMITTED $tag"
done

echo "" >> "$OUT"
echo "===== 90 秒后 PEND 检查 =====" >> "$OUT"
sleep 90
for tag in $SUBMITTED; do
  LINE=$(bjobs -w 2>/dev/null | grep " $tag ")
  STAT=$(echo "$LINE" | awk '{print $3}')
  if [ "$STAT" = "PEND" ]; then
    JID=$(echo "$LINE" | awk '{print $1}')
    echo ">>> $tag($JID) PEND,换队列重投" >> "$OUT"
    bkill "$JID" >> "$OUT" 2>&1
    submit_one "$tag" "$(pick_queue)" >> "$OUT"
  fi
done
bjobs -w 2>/dev/null | grep p2 | awk '{print $1, $3, $4, $7}' >> "$OUT"

cat "$OUT"
git add ops/result
git commit -m "result 62 stability diag batch"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
