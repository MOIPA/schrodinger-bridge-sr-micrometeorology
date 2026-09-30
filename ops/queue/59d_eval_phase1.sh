#!/bin/bash
# 59d_eval_phase1.sh — 阶段 1 评估(一个 GPU 作业内串行):双三次基线 91 + 各 run 的 AGL 评估 90 + 排序 92
# 手机执行: bash ops/queue/59d_eval_phase1.sh   (登录节点提交自己,GPU 节点内跑 --inner)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"  # 9654p6000ib 为 Blackwell,与 wind3d torch 不兼容
# PHASE=phase1(直接输出) | phase1r(残差输出,基准 = 已训好的 t16_residual)
if [ "${PHASE:-phase1}" = "phase1r" ]; then
  CFG_DIR="configs/深圳/phase1r"; CFG_PREFIX="config_wind_canvas_p1r_"
  TAG_PREFIX="r_"; BASE_TAG="t16_residual"
  ALL="t13_most t13_mostflux t13_w t13_theta t13_ph t12_zagldiff t12_hgtdiff t14_cos t14_noenc_cos t15_z0 t15_z0_urban t15_z0_urban_wv t17_coords"
else
  CFG_DIR="configs/深圳/phase1";  CFG_PREFIX="config_wind_canvas_p1_"
  TAG_PREFIX=""; BASE_TAG="base"
  ALL="base t13_most t13_mostflux t13_w t13_theta t13_ph t12_zagldiff t12_hgtdiff t15_z0 t15_z0_urban t15_z0_urban_wv t14_cos t14_noenc_cos t16_residual t17_coords"
fi

if [ "$1" != "--inner" ]; then
  git pull --no-rebase
  mkdir -p ops/result logs results/phase1
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  if [ "$1" = "--after-training" ]; then
    # 依赖"当前仍在系统里的"训练作业全部 ended(已结束的作业名 LSF 解析不到,不能写进依赖)
    DEP=""
    ACTIVE=$(bjobs -o "job_name" -noheader 2>/dev/null | sed -n 's/^p1_//p')
    for t in $ALL; do
      echo "$ACTIVE" | grep -qx "$t" || continue
      if [ -z "$DEP" ]; then DEP="ended(p1_$t)"; else DEP="$DEP && ended(p1_$t)"; fi
    done
    if [ -z "$DEP" ]; then
      echo "没有在跑的训练作业,直接提交评估 -> 队列 $Q"
      bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
        -J "p1eval" -o "logs/p1eval_%J.out" -e "logs/p1eval_%J.err" \
        "cd $ROOT && PHASE=${PHASE:-phase1} bash ops/queue/59d_eval_phase1.sh --inner"
      # 注意:bsub 不继承父 shell 的环境变量,PHASE 必须写进 payload
    else
      echo "提交阶段 1 评估作业(依赖在跑的训练作业结束)-> 队列 $Q"
      bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
        -w "$DEP" -J "p1eval" -o "logs/p1eval_%J.out" -e "logs/p1eval_%J.err" \
        "cd $ROOT && PHASE=${PHASE:-phase1} bash ops/queue/59d_eval_phase1.sh --inner"
    fi
  else
    echo "提交阶段 1 评估作业 -> 队列 $Q"
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -J "p1eval" -o "logs/p1eval_%J.out" -e "logs/p1eval_%J.err" \
      "cd $ROOT && PHASE=${PHASE:-phase1} bash ops/queue/59d_eval_phase1.sh --inner"
  fi
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/59d_eval_phase1.txt
: > "$OUT"
mkdir -p results/phase1
echo "===== 1. 双三次插值基线(T1.1①,91) =====" >> "$OUT"
$PY3D -u scripts/outline/91_bicubic_baseline.py --split test --out_dir results/phase1 >> "$OUT" 2>&1 \
  || echo "FAIL bicubic" >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 各 run 的 AGL 评估(90) =====" >> "$OUT"
for tag in $ALL; do
  CK="data/DL_result/$EXP/$CFG_PREFIX$tag/checkpoint.pth"
  if [ -f "$CK" ]; then
    echo "--- $tag ---" >> "$OUT"
    $PY3D -u scripts/outline/90_agl_eval_phase1.py \
      --config_path "$CFG_DIR/$CFG_PREFIX$tag.yml" \
      --checkpoint "$CK" --split test --tag "$TAG_PREFIX$tag" --out_dir results/phase1 >> "$OUT" 2>&1 \
      || echo "FAIL $tag" >> "$OUT"
  else
    echo "跳过 $tag(无 checkpoint)" >> "$OUT"
  fi
done

# phase1r:基准 t16_residual 的配置在 phase1/,单独补评一次(补上逐小时累加量,才能出置信区间)
if [ "$BASE_TAG" != "base" ]; then
  CK="data/DL_result/$EXP/config_wind_canvas_p1_${BASE_TAG}/checkpoint.pth"
  if [ -f "$CK" ]; then
    echo "--- 补评基准 $BASE_TAG ---" >> "$OUT"
    $PY3D -u scripts/outline/90_agl_eval_phase1.py \
      --config_path "configs/深圳/phase1/config_wind_canvas_p1_${BASE_TAG}.yml" \
      --checkpoint "$CK" --split test --tag "$BASE_TAG" --out_dir results/phase1 >> "$OUT" 2>&1 \
      || echo "FAIL $BASE_TAG" >> "$OUT"
  fi
fi

echo "" >> "$OUT"
echo "===== 3. 排序(92) =====" >> "$OUT"
TAGS=""
for tag in $ALL; do TAGS="$TAGS,$TAG_PREFIX$tag"; done
TAGS="${TAGS#,},$BASE_TAG,baseline_bicubic"
$PY3D -u scripts/outline/92_rank_phase1.py --results_dir results/phase1 --base_tag "$BASE_TAG" \
  --tags "$TAGS" --out_prefix "results/phase1/ranking_${PHASE:-phase1}" >> "$OUT" 2>&1 \
  || echo "FAIL rank" >> "$OUT"

git add ops/result results/phase1
git commit -m "result 59d phase1 eval + ranking"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
