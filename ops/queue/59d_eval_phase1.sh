#!/bin/bash
# 59d_eval_phase1.sh — 阶段 1 评估(一个 GPU 作业内串行):双三次基线 91 + 各 run 的 AGL 评估 90 + 排序 92
# 手机执行: bash ops/queue/59d_eval_phase1.sh   (登录节点提交自己,GPU 节点内跑 --inner)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"  # 9654p6000ib 为 Blackwell,与 wind3d torch 不兼容
ALL="base t13_most t13_mostflux t13_w t13_theta t13_ph t12_zagldiff t12_hgtdiff t15_z0 t15_z0_urban t15_z0_urban_wv t14_cos t14_noenc_cos t16_residual t17_coords"

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
    # 依赖 15 个训练作业全部 ended(含 DONE/EXIT):训练跑完自动评估,不依赖本地会话
    DEP=""
    for t in $ALL; do
      if [ -z "$DEP" ]; then DEP="ended(p1_$t)"; else DEP="$DEP && ended(p1_$t)"; fi
    done
    echo "提交阶段 1 评估作业(依赖 15 个训练作业结束)-> 队列 $Q"
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -w "$DEP" -J "p1eval" -o "logs/p1eval_%J.out" -e "logs/p1eval_%J.err" \
      "cd $ROOT && bash ops/queue/59d_eval_phase1.sh --inner"
  else
    echo "提交阶段 1 评估作业 -> 队列 $Q"
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -J "p1eval" -o "logs/p1eval_%J.out" -e "logs/p1eval_%J.err" \
      "cd $ROOT && bash ops/queue/59d_eval_phase1.sh --inner"
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
  CK="data/DL_result/$EXP/config_wind_canvas_p1_$tag/checkpoint.pth"
  if [ -f "$CK" ]; then
    echo "--- $tag ---" >> "$OUT"
    $PY3D -u scripts/outline/90_agl_eval_phase1.py \
      --config_path "configs/深圳/phase1/config_wind_canvas_p1_$tag.yml" \
      --checkpoint "$CK" --split test --tag "$tag" --out_dir results/phase1 >> "$OUT" 2>&1 \
      || echo "FAIL $tag" >> "$OUT"
  else
    echo "跳过 $tag(无 checkpoint)" >> "$OUT"
  fi
done

echo "" >> "$OUT"
echo "===== 3. 排序(92) =====" >> "$OUT"
$PY3D -u scripts/outline/92_rank_phase1.py --results_dir results/phase1 --base_tag base >> "$OUT" 2>&1 \
  || echo "FAIL rank" >> "$OUT"

git add ops/result results/phase1
git commit -m "result 59d phase1 eval + ranking"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
