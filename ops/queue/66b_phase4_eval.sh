#!/usr/bin/env bash
# 阶段 4(跨分辨率直连)评估入口:p4_* 臂
#
#   bash ops/queue/66b_phase4_eval.sh                       # q_pick 选队列提交,默认评 p4_27to1_reg
#   FORCE_Q=944090d bash ops/queue/66b_phase4_eval.sh
#   TAGS="p4_27to1_reg p4_3to1_reg" bash ops/queue/66b_phase4_eval.sh
#   AFTER=14833575 bash ops/queue/66b_phase4_eval.sh        # 挂"训练结束后才跑"的依赖
#   INNER=1 bash ...                                        # 在计算节点内部直接跑
#
# 口径与阶段 3 / 外传完全一致(84_ensemble_eval.py --mode reg,主指标 AGL 10-500 m),
# 数据划分已核对与阶段 3 相同(test = 2020-07-16T00..07-31T23,144 h),
# 因此结果可与级联端点、外传三档直接对比。
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

ROOT="${ROOT:-$PWD}"
PY3D="${PY3D:-/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python}"
CFG_ROOT="configs/深圳"
OUT_DIR="results/phase4"
TAGS="${TAGS:-p4_27to1_reg}"
QUEUE="${FORCE_Q:-}"

mkdir -p logs "$OUT_DIR"

if [ -z "${INNER:-}" ]; then
  if [ -z "$QUEUE" ]; then
    QUEUE=$(bash ops/queue/q_pick.sh 2>/dev/null || true)
  fi
  if [ -z "$QUEUE" ]; then
    echo ">>> 无可用 GPU 队列;稍后重试或 FORCE_Q=<队列>"
    exit 2
  fi
  DEP_OPTS=()
  [ -n "${AFTER:-}" ] && DEP_OPTS=(-w "ended(${AFTER})")
  bsub -q "$QUEUE" -gpu "num=1:mode=exclusive_process" -n 4 \
    -R "rusage[mem=32000]" ${DEP_OPTS:+"${DEP_OPTS[@]}"} \
    -J p4eval -o "logs/p4eval_%J.out" -e "logs/p4eval_%J.err" \
    bash -lc "cd '$ROOT' && bash ops/queue/gpu_keepalive.sh start; \
INNER=1 TAGS='$TAGS' PY3D='$PY3D' bash ops/queue/66b_phase4_eval.sh; \
bash ops/queue/gpu_keepalive.sh stop"
  echo "submitted p4 eval -> $QUEUE (TAGS=$TAGS)"
  exit 0
fi

for t in $TAGS; do
  echo "===== eval $t ====="
  "$PY3D" -u scripts/outline/84_ensemble_eval.py \
    --mode reg --tags "$t" --split test --device cuda:0 \
    --cfg_root "$CFG_ROOT" --out_dir "$OUT_DIR" || { echo "FAIL $t"; continue; }
done

echo "===== 汇总 ====="
"$PY3D" sidequest_scale3x/scripts/summarize3.py --dir "$OUT_DIR" || true
