#!/bin/bash
# 60e_phase2_gate.sh — W1 试点闸门评估(快速):默认 p2_div_mid + 基线 r_t14_noenc_cos,可传参覆盖
#   ① 90 AGL 评估(全量 test split)② 94 三类诊断 ③ 92 排序 -> results/phase2/ranking_phase2_gate.md
#   ④ 97 净效应表 -> results/phase2/phase2_net_effects.md(基于现有 *_diag.json)
#   本脚本不创建 GATE 文件;人工判定通过后自行创建并推送(60b 的 W2/W3 硬闸门):
#     touch results/phase2/GATE_W1_PASS && git add results/phase2 && \
#       git commit -m "gate: W1 pass" && git push
# 手机执行: bash ops/queue/60e_phase2_gate.sh [tag1,tag2]   (登录节点提交,GPU 节点跑 --inner)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
CFG_DIR="configs/深圳/phase2"
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"

cfg_of() { echo "$CFG_DIR/config_wind_canvas_p2_${1#p2_}.yml"; }
ck_of()  { echo "$RESULT_BASE/config_wind_canvas_p2_${1#p2_}/checkpoint.pth"; }

if [ "$1" != "--inner" ]; then
  git pull --no-rebase
  git --no-pager log --oneline -1
  mkdir -p ops/result logs results/phase2
  TAGS="${1:-}"
  [ -z "$TAGS" ] && TAGS="p2_div_mid"
  TAGS_CSV=$(echo "$TAGS" | tr ' ' ',')
  QUEUES="${FORCE_Q:-$QUEUES}"   # 队列异常时覆盖:FORCE_Q=6148v100ib bash ops/queue/60e_phase2_gate.sh ...
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  echo "W1 闸门评估($TAGS_CSV + 基线 r_t14_noenc_cos)-> 队列 $Q"
  SUB=$(bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "p2gate" -o "logs/p2gate_%J.out" -e "logs/p2gate_%J.err" \
    "cd $ROOT && GATE_TAGS=$TAGS_CSV bash ops/queue/60e_phase2_gate.sh --inner" 2>&1)
  echo "$SUB"
  JID=$(echo "$SUB" | grep -oE '[0-9]+' | head -1)
  sleep 90
  ST=$(bjobs -o "jobid stat" -noheader 2>/dev/null | awk -v j="$JID" '$1==j{print $2}')
  if [ "$ST" = "PEND" ]; then
    echo ">>> p2gate($JID) PEND: bkill;换队列重跑本脚本(必要时 FORCE_Q=<队列>)"
    bkill "$JID"
  else
    echo ">>> p2gate($JID) 状态: ${ST:-已不在队列(可能已开始或极快失败,查 logs/)}"
  fi
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/60e_phase2_gate.txt
: > "$OUT"
mkdir -p results/phase2

TAGS="${GATE_TAGS:-p2_div_mid}"
TAGS=$(echo "$TAGS" | tr ',' ' ')
TAGS_CSV=$(echo "$TAGS" | tr ' ' ',')
BASE_TAG="r_t14_noenc_cos"
BASE_CFG="configs/深圳/phase1r/config_wind_canvas_p1r_t14_noenc_cos.yml"
BASE_CK="data/DL_result/$EXP/config_wind_canvas_p1r_t14_noenc_cos/checkpoint.pth"
PHYS_CFG="configs/深圳/phase2/config_wind_canvas_p2_div_mid.yml"
[ -f "$PHYS_CFG" ] || PHYS_CFG=$(cfg_of "$(echo $TAGS | awk '{print $1}')")

{
echo "===== W1 试点闸门评估: $TAGS_CSV + 基线 ====="

echo ""
echo "===== 1. 90 AGL 评估 ====="
$PY3D -u scripts/outline/90_agl_eval_phase1.py --config_path "$BASE_CFG" --checkpoint "$BASE_CK" \
  --split test --tag "$BASE_TAG" --out_dir results/phase2 || echo "FAIL 90 $BASE_TAG"
for tag in $TAGS; do
  cfg=$(cfg_of "$tag")
  ck=$(ck_of "$tag")
  if [ ! -f "$cfg" ] || [ ! -f "$ck" ]; then
    echo "跳过 $tag(缺配置或 checkpoint)"
    continue
  fi
  echo "--- $tag ---"
  $PY3D -u scripts/outline/90_agl_eval_phase1.py --config_path "$cfg" --checkpoint "$ck" \
    --split test --tag "$tag" --out_dir results/phase2 || echo "FAIL 90 $tag"
done

echo ""
echo "===== 2. 94 三类诊断 ====="
$PY3D -u scripts/outline/94_phase2_diagnostics.py --config_path "$BASE_CFG" --checkpoint "$BASE_CK" \
  --split test --tag "$BASE_TAG" --out_dir results/phase2 --phys_from_config "$PHYS_CFG" \
  || echo "FAIL 94 $BASE_TAG"
for tag in $TAGS; do
  cfg=$(cfg_of "$tag")
  ck=$(ck_of "$tag")
  if [ ! -f "$cfg" ] || [ ! -f "$ck" ]; then
    continue
  fi
  echo "--- $tag ---"
  $PY3D -u scripts/outline/94_phase2_diagnostics.py --config_path "$cfg" --checkpoint "$ck" \
    --split test --tag "$tag" --out_dir results/phase2 || echo "FAIL 94 $tag"
done

echo ""
echo "===== 3. 闸门排序(92) ====="
$PY3D -u scripts/outline/92_rank_phase1.py --results_dir results/phase2 --base_tag "$BASE_TAG" \
  --tags "$TAGS_CSV,$BASE_TAG" --out_prefix results/phase2/ranking_phase2_gate || echo "FAIL rank"
cat results/phase2/ranking_phase2_gate.md

echo ""
echo "===== 4. 净效应表(97) ====="
$PY3D -u scripts/outline/97_phase2_tables.py --results_dir results/phase2 --base_tag "$BASE_TAG" \
  --rank_csv results/phase2/ranking_phase2_gate.csv --diag_tags "$TAGS_CSV" || echo "FAIL tables"

echo ""
echo "===== 5. 人工判定 ====="
echo "看 results/phase2/ranking_phase2_gate.md 与 phase2_net_effects.md;通过后:"
echo "  touch results/phase2/GATE_W1_PASS && git add results/phase2 && git commit -m \"gate: W1 pass\" && git push"
echo "然后服务器上: bash ops/queue/60b_phase2_waves.sh 2"
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase2
git commit -m "result 60e phase2 W1 gate eval"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
