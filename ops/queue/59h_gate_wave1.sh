#!/bin/bash
# 59h_gate_wave1.sh — 第一波闸门评估(快速):只评 phase1r 的 4 个 wave-1 run,与基线排序
# 手机执行: bash ops/queue/59h_gate_wave1.sh   (登录节点提交,GPU 节点跑 --inner)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas

if [ "$1" != "--inner" ]; then
  git pull --no-rebase
  mkdir -p ops/result logs results/phase1
  Q=""
  for q in e5v4p100ib 6148v100ib 62v100ib 7552v100 83a100ib; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  echo "提交第一波闸门评估 -> 队列 $Q"
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "p1gate" -o "logs/p1gate_%J.out" -e "logs/p1gate_%J.err" \
    "cd $ROOT && bash ops/queue/59h_gate_wave1.sh --inner"
  exit 0
fi

cd "$ROOT" || exit 1
OUT=ops/result/59h_gate_wave1.txt
: > "$OUT"
mkdir -p results/phase1
echo "===== wave-1 残差 run 评估 =====" >> "$OUT"
for t in t13_most t13_mostflux t13_w t13_theta; do
  CK="data/DL_result/$EXP/config_wind_canvas_p1r_$t/checkpoint.pth"
  if [ -f "$CK" ]; then
    echo "--- $t ---" >> "$OUT"
    $PY3D -u scripts/outline/90_agl_eval_phase1.py \
      --config_path "configs/深圳/phase1r/config_wind_canvas_p1r_$t.yml" \
      --checkpoint "$CK" --split test --tag "r_$t" --out_dir results/phase1 >> "$OUT" 2>&1 \
      || echo "FAIL $t" >> "$OUT"
  else
    echo "缺 $t" >> "$OUT"
  fi
done

echo "" >> "$OUT"
echo "===== 排序(与残差基准/基线)=====" >> "$OUT"
$PY3D -u scripts/outline/92_rank_phase1.py --results_dir results/phase1 --base_tag t16_residual \
  --tags "r_t13_most,r_t13_mostflux,r_t13_w,r_t13_theta,t16_residual,baseline_bicubic" \
  --out_prefix results/phase1/ranking_phase1r_gate >> "$OUT" 2>&1 || echo "FAIL rank" >> "$OUT"
cat results/phase1/ranking_phase1r_gate.md >> "$OUT"

git add ops/result results/phase1
git commit -m "result 59h wave1 gate eval"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
