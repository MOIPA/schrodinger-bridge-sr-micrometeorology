#!/bin/bash
# 60d_eval_phase2.sh — 阶段 2 终评(一个 GPU 作业内串行):
#   ① 90 AGL 评估:--tags 缺省 = 全部有 checkpoint 的 p2_*,外加基线 r_t14_noenc_cos 重评(回归检查,应复现 0.9601)
#   ② 94 三类诊断(散度/能谱/极值):全部 run + 基线(基线用 --phys_from_config 读阶段 2 的物理参数)
#   ③ 91 双三次基线 -> results/phase2
#   ④ 92 排序(base=r_t14_noenc_cos)-> results/phase2/ranking_phase2.csv/_perlevel/_strata/.md
#   ⑤ 97 净效应表 -> results/phase2/phase2_net_effects.md
#   ⑥ 回传 ops/result + results/phase2 + configs/深圳/phase2(不静默,失败会打印)
# 手机执行: bash ops/queue/60d_eval_phase2.sh [--tags p2_l2,p2_div_mid] [--after-training p2_l2,p2_div_mid]
#   --after-training 只对"仍在 bjobs 里"的作业名挂 LSF 依赖(已结束的作业名会报 No matching job found)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
CFG_DIR="configs/深圳/phase2"
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"
ALL="p2_l2 p2_div_mid p2_spec p2_ext p2_vort p2_div_lo p2_div_hi p2_combo"

cfg_of() { echo "$CFG_DIR/config_wind_canvas_p2_${1#p2_}.yml"; }
ck_of()  { echo "$RESULT_BASE/config_wind_canvas_p2_${1#p2_}/checkpoint.pth"; }

INNER=""
TAGS=""
AFTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --inner) INNER=1; shift;;
    --tags) TAGS="$2"; shift 2;;
    --tags=*) TAGS="${1#--tags=}"; shift;;
    --after-training) AFTER="$2"; shift 2;;
    --after-training=*) AFTER="${1#--after-training=}"; shift;;
    *) echo "未知参数: $1(用法: [--inner] [--tags a,b] [--after-training a,b])"; shift;;
  esac
done

if [ -z "$INNER" ]; then
  git pull --no-rebase
  git --no-pager log --oneline -1
  mkdir -p ops/result logs results/phase2
  # 默认 tags:全部有 checkpoint 的 p2_*
  if [ -z "$TAGS" ]; then
    T=""
    for t in $ALL; do [ -f "$(ck_of $t)" ] && T="$T,$t"; done
    TAGS="${T#,}"
  fi
  TAGS_CSV=$(echo "$TAGS" | tr ' ' ',')
  if [ -z "$TAGS_CSV" ]; then
    echo "没有任何 p2 checkpoint,评估清单为空;先训练(bash ops/queue/60b_phase2_waves.sh 1)"
    exit 0
  fi
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  # GPU 计算节点的 PATH 可能没有 git(2026-10-02 实测 6148v100ib 报 git: command not found,
  # 导致作业末尾的结果回传静默失败);提交时取登录节点的 git 目录注入作业 PATH
  GITDIR=$(dirname "$(command -v git)")

  DEP=""
  if [ -n "$AFTER" ]; then
    ACTIVE=$(bjobs -o "job_name" -noheader 2>/dev/null)
    for t in $(echo "$AFTER" | tr ',' ' '); do
      if echo "$ACTIVE" | grep -qx "$t"; then
        if [ -z "$DEP" ]; then DEP="ended($t)"; else DEP="$DEP && ended($t)"; fi
        echo "依赖: ended($t)"
      else
        echo "依赖: $t 已不在 bjobs 里,不挂(否则 LSF 报 No matching job found)"
      fi
    done
  fi
  echo "评估清单: $TAGS_CSV + 基线 r_t14_noenc_cos(回归检查)"
  if [ -n "$DEP" ]; then
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -w "$DEP" -J "p2eval" -o "logs/p2eval_%J.out" -e "logs/p2eval_%J.err" \
      "cd $ROOT && export PATH=$GITDIR:\$PATH && EVAL_TAGS=$TAGS_CSV bash ops/queue/60d_eval_phase2.sh --inner"
  else
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -J "p2eval" -o "logs/p2eval_%J.out" -e "logs/p2eval_%J.err" \
      "cd $ROOT && export PATH=$GITDIR:\$PATH && EVAL_TAGS=$TAGS_CSV bash ops/queue/60d_eval_phase2.sh --inner"
  fi
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/60d_eval_phase2.txt
: > "$OUT"
mkdir -p results/phase2

TAGS="${TAGS:-${EVAL_TAGS:-}}"
TAGS=$(echo "$TAGS" | tr ',' ' ')
if [ -z "$TAGS" ]; then
  T=""
  for t in $ALL; do [ -f "$(ck_of $t)" ] && T="$T $t"; done
  TAGS="$T"
fi
TAGS_CSV=$(echo "$TAGS" | tr ' ' ',')
BASE_TAG="r_t14_noenc_cos"
BASE_CFG="configs/深圳/phase1r/config_wind_canvas_p1r_t14_noenc_cos.yml"
BASE_CK="data/DL_result/$EXP/config_wind_canvas_p1r_t14_noenc_cos/checkpoint.pth"
PHYS_CFG="configs/深圳/phase2/config_wind_canvas_p2_div_mid.yml"
[ -f "$PHYS_CFG" ] || PHYS_CFG=$(cfg_of "$(echo $TAGS | awk '{print $1}')")

{
echo "===== 0. 评估清单 ====="
echo "p2 runs: ${TAGS_CSV:-(无)}"
echo "baseline: $BASE_TAG(阶段 1 主指标 0.9601,回归检查)"
echo "94 的物理参数来源(基线配置没有 si.phys_*): $PHYS_CFG"

echo ""
echo "===== 1. 90 AGL 评估(全画布 test split) ====="
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
echo "--- $BASE_TAG(基线重评) ---"
$PY3D -u scripts/outline/90_agl_eval_phase1.py --config_path "$BASE_CFG" --checkpoint "$BASE_CK" \
  --split test --tag "$BASE_TAG" --out_dir results/phase2 || echo "FAIL 90 $BASE_TAG"
$PY3D -c "
import json, os
p = 'results/phase2/${BASE_TAG}_summary.json'
if os.path.exists(p):
    m = json.load(open(p))['main_rmse_vec']
    ok = abs(m - 0.9601) < 0.01
    print('BASE REGRESSION: main_rmse_vec=%.4f (phase1=0.9601, diff=%+.4f) -> %s' % (m, m - 0.9601, 'PASS' if ok else 'WARN'))
else:
    print('BASE REGRESSION: 缺 ${BASE_TAG}_summary.json')
"

echo ""
echo "===== 2. 94 三类诊断(散度/能谱/极值) ====="
for tag in $TAGS; do
  cfg=$(cfg_of "$tag")
  ck=$(ck_of "$tag")
  if [ ! -f "$cfg" ] || [ ! -f "$ck" ]; then
    echo "跳过 $tag(缺配置或 checkpoint)"
    continue
  fi
  echo "--- $tag ---"
  $PY3D -u scripts/outline/94_phase2_diagnostics.py --config_path "$cfg" --checkpoint "$ck" \
    --split test --tag "$tag" --out_dir results/phase2 || echo "FAIL 94 $tag"
done
echo "--- $BASE_TAG(基线诊断) ---"
$PY3D -u scripts/outline/94_phase2_diagnostics.py --config_path "$BASE_CFG" --checkpoint "$BASE_CK" \
  --split test --tag "$BASE_TAG" --out_dir results/phase2 --phys_from_config "$PHYS_CFG" \
  || echo "FAIL 94 $BASE_TAG"

echo ""
echo "===== 3. 双三次基线(91) ====="
$PY3D -u scripts/outline/91_bicubic_baseline.py --split test --out_dir results/phase2 \
  || echo "FAIL bicubic"

echo ""
echo "===== 4. 排序(92) ====="
if [ -n "$TAGS_CSV" ]; then RANK_TAGS="$TAGS_CSV,$BASE_TAG,baseline_bicubic"; else RANK_TAGS="$BASE_TAG,baseline_bicubic"; fi
$PY3D -u scripts/outline/92_rank_phase1.py --results_dir results/phase2 --base_tag "$BASE_TAG" \
  --tags "$RANK_TAGS" --out_prefix results/phase2/ranking_phase2 || echo "FAIL rank"
cat results/phase2/ranking_phase2.md

echo ""
echo "===== 5. 净效应表(97) ====="
$PY3D -u scripts/outline/97_phase2_tables.py --results_dir results/phase2 --base_tag "$BASE_TAG" \
  --diag_tags "$TAGS_CSV" || echo "FAIL tables"

echo ""
echo "===== 6. 产出 ====="
ls -l results/phase2 | tail -40
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase2
git add "configs/深圳/phase2" 2>/dev/null
git commit -m "result 60d phase2 eval + ranking + net effects"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
