#!/bin/bash
# 63d_eval_phase3.sh — 阶段 3 T3.5 评估(一个 GPU 作业内串行,复刻 60d 模式):
#   ① 90 AGL 评估:默认 tags = p2_l1r2_lr2e4(重评)+ 全部有 checkpoint 的 p3_*;
#      p2_ 用 configs/深圳/phase2/、p3_ 用 configs/深圳/phase3/ 的配置;产物全部落 results/phase3/。
#   ② p2_l1r2_lr2e4 重评的特殊地位:
#      - 复现性回归:主指标应复现 results/phase2 的 0.9574(确定性采样,期望几乎逐位一致);
#      - 切变补齐:该参考臂由此在 results/phase3 下拿到与 p3 各臂同口径的最新逐小时累加产物
#        (含评估链任务扩展的切变类累加量),避免跨目录口径差。
#   ③ txt 记录每个 tag 主指标 + 与 0.9574 的显式对照;回传 ops/result + results/phase3。
# 前置依赖:checkpoint 已在 data/DL_result/ExperimentSchrodingerBridgeWindCanvas/ 下;
#   90 评估脚本(需要 torch,GPU 节点跑)。63b 训练完可直接 --after-training。
# 手机执行: bash ops/queue/63d_eval_phase3.sh [--tags p3_agl,p3_joint] [--after-training p3_agl,p3_joint]
#   --after-training 只对"仍在 bjobs 里"的作业名挂 LSF 依赖(已结束的作业名会报 No matching job found)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"
QUEUES="${FORCE_Q:-$QUEUES}"  # 队列异常时覆盖:FORCE_Q=62v100ib bash ops/queue/63d_eval_phase3.sh ...
# 阶段 3 全部 run(tag 自带 p2_/p3_ 前缀;p2 仅重评参考臂)
ALL="p2_l1r2_lr2e4 p3_agl p3_joint p3_agllw"
BASE_TAG="p2_l1r2_lr2e4"       # 参考臂(phase2 主指标 0.9574)
BASE_REF=0.9574                # results/phase2 记录值;重评对照用

cfg_of() {  # $1=tag;按前缀选阶段目录
  local t=$1 d=phase3
  case "$t" in p2_*) d=phase2;; esac
  echo "configs/深圳/$d/config_wind_canvas_${t}.yml"
}
ck_of()  { echo "$RESULT_BASE/config_wind_canvas_${1}/checkpoint.pth"; }

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
  mkdir -p ops/result logs results/phase3
  # 默认 tags:p2_l1r2_lr2e4(必评)+ 全部有 checkpoint 的 p3_*
  if [ -z "$TAGS" ]; then
    T=""
    for t in $ALL; do [ -f "$(ck_of $t)" ] && T="$T,$t"; done
    TAGS="${T#,}"
  fi
  TAGS_CSV=$(echo "$TAGS" | tr ' ' ',')
  if [ -z "$TAGS_CSV" ]; then
    echo "没有任何 checkpoint,评估清单为空;先训练(bash ops/queue/63b_phase3_waves.sh)"
    exit 0
  fi
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  # GPU 计算节点的 PATH 可能没有 git,提交时取登录节点的 git 目录注入作业 PATH
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
  echo "评估清单: $TAGS_CSV(含参考臂 $BASE_TAG 重评,对照主指标 $BASE_REF)"
  if [ -n "$DEP" ]; then
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -w "$DEP" -J "p3eval" -o "logs/p3eval_%J.out" -e "logs/p3eval_%J.err" \
      "cd $ROOT && export PATH=$GITDIR:\$PATH && EVAL_TAGS=$TAGS_CSV bash ops/queue/63d_eval_phase3.sh --inner"
  else
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -J "p3eval" -o "logs/p3eval_%J.out" -e "logs/p3eval_%J.err" \
      "cd $ROOT && export PATH=$GITDIR:\$PATH && EVAL_TAGS=$TAGS_CSV bash ops/queue/63d_eval_phase3.sh --inner"
  fi
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/63d_eval_phase3.txt
: > "$OUT"
mkdir -p results/phase3

TAGS="${TAGS:-${EVAL_TAGS:-}}"
TAGS=$(echo "$TAGS" | tr ',' ' ')
if [ -z "$TAGS" ]; then
  T=""
  for t in $ALL; do [ -f "$(ck_of $t)" ] && T="$T $t"; done
  TAGS="$T"
fi
TAGS_CSV=$(echo "$TAGS" | tr ' ' ',')
echo "$TAGS" | grep -qw "$BASE_TAG" || TAGS="$BASE_TAG $TAGS"
TAGS_CSV=$(echo "$TAGS" | tr ' ' ',')

{
echo "===== 0. 评估清单 ====="
echo "tags: $TAGS_CSV"
echo "参考臂: $BASE_TAG(phase2 主指标 $BASE_REF;重评 = 复现性回归 + 切变类累加补齐)"
echo "产物目录: results/phase3/(全部 tag 统一落这里,便于 92/97/99 同目录比较)"

echo ""
echo "===== 1. 90 AGL 评估(全画布 test split) ====="
for tag in $TAGS; do
  cfg=$(cfg_of "$tag")
  ck=$(ck_of "$tag")
  if [ ! -f "$cfg" ] || [ ! -f "$ck" ]; then
    echo "跳过 $tag(缺配置 $cfg 或 checkpoint $ck)"
    continue
  fi
  echo "--- $tag ---"
  $PY3D -u scripts/outline/90_agl_eval_phase1.py --config_path "$cfg" --checkpoint "$ck" \
    --split test --tag "$tag" --out_dir results/phase3 || echo "FAIL 90 $tag"
done

echo ""
echo "===== 2. 主指标汇总与 $BASE_REF 对照 ====="
$PY3D -c "
import json, os
base_ref = $BASE_REF
base_tag = '$BASE_TAG'
rows = []
for t in '$TAGS'.split():
    p = 'results/phase3/%s_summary.json' % t
    if not os.path.exists(p):
        rows.append((t, None, None))
        continue
    m = json.load(open(p))
    rows.append((t, m['main_rmse_vec'], m.get('n_hours')))
print('| tag | main_rmse_vec | n_hours | 与 %.4f 之差 | 判定 |' % base_ref)
print('|---|---|---|---|---|')
for t, v, n in rows:
    if v is None:
        print('| %s | (缺 summary) | - | - | - |' % t)
        continue
    d = v - base_ref
    if t == base_tag:
        ok = abs(d) < 0.005
        print('| %s | %.4f | %s | %+.4f | %s |' % (t, v, n, d, 'PASS(回归)' if ok else 'WARN(回归偏离>0.005)'))
    else:
        print('| %s | %.4f | %s | %+.4f | 对照 |' % (t, v, n, d))
print()
print('口径:main_rmse_vec = 90 的 10-500 m 全分层矢量 RMSE(标准化空间,确定性采样);')
print('      参考值 %.4f 来自 results/phase2/ 的同一 run(2026-10-03 记录)。' % base_ref)
"

echo ""
echo "===== 3. 产出 ====="
ls -l results/phase3 | tail -40
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase3
git commit -m "result 63d phase3 eval (90;incl p2_l1r2_lr2e4 re-eval regression vs 0.9574)"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
