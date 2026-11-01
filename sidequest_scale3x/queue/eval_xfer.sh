#!/usr/bin/env bash
# 交叉迁移评估(零样本):把 A 尺度的权重搬去评 B 尺度的任务
# 用法:bash sidequest_scale3x/queue/eval_xfer.sh            # 本机直接跑(需 GPU)
#       INNER=0 bash ...                                      # 外层用 q_pick 提交
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

PY="${PY3:-python3}"
TAGS="${TAGS:-xfer9to3_by27to9,xfer27to9_by3to1}"
OUT="${OUT:-sidequest_scale3x/results}"
CK_ROOT="${CK_ROOT:-data/DL_result/ExperimentSchrodingerBridgeWindCanvas}"

if [ "${INNER:-1}" != "1" ]; then
  Q=$(bash ops/queue/q_pick.sh 2>/dev/null || true)
  [ -z "$Q" ] && { echo ">>> 无可用 GPU 队列"; exit 2; }
  echo ">>> 提交到 $Q"
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J sqxfer -o "logs/sqxfer_%J.out" -e "logs/sqxfer_%J.err" \
    "cd $PWD && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
bash ops/queue/gpu_keepalive.sh start; \
INNER=1 TAGS=$TAGS bash sidequest_scale3x/queue/eval_xfer.sh > logs/sqxfer.log 2>&1; \
bash ops/queue/gpu_keepalive.sh stop"
  exit 0
fi

IFS=',' read -ra LIST <<< "$TAGS"
for t in "${LIST[@]}"; do
  echo "=== [$t] ==="
  "$PY" -u scripts/outline/84_ensemble_eval.py \
    --mode reg --tags "$t" --n_ensemble 1 \
    --cfg_root sidequest_scale3x/configs --ck_root "$CK_ROOT" \
    --out_dir "$OUT" --device "${DEV:-cuda:0}" | tail -4
done
"$PY" sidequest_scale3x/scripts/summarize3.py --dir "$OUT" || true
