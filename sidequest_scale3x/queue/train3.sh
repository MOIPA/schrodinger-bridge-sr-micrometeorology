#!/bin/bash
# 外传 · 三组训练提交(27→9 / 9→3 / 3→1)
# 与阶段 3 共用同一个回归训练脚本与超参,只换配置。
# 用法:
#   bash sidequest_scale3x/queue/train3.sh                 # 自动选队列
#   FORCE_Q=734090d bash sidequest_scale3x/queue/train3.sh # 指定队列
#   DRY_RUN=1 bash sidequest_scale3x/queue/train3.sh       # 只打印
#   TAGS="sq3to1_reg" bash ...                             # 只提一组
set -u
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
ROOT="${ROOT:-/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology}"
CFG_DIR="sidequest_scale3x/configs"
EXP="ExperimentSchrodingerBridgeWindCanvas"
OUT="sidequest_scale3x/results/train3_submit.txt"
mkdir -p logs sidequest_scale3x/results
TAGS="${TAGS:-sq27to9_reg sq9to3_reg sq3to1_reg}"

say() { echo "$*" | tee -a "$OUT"; }

if [ -n "${FORCE_Q:-}" ]; then
  Q="$FORCE_Q"
else
  Q=$(bash ops/queue/q_pick.sh 2>/dev/null)
fi
if [ -z "${Q:-}" ]; then
  say ">>> 无可用 GPU 队列(全部 closed/满载),稍后重试或 FORCE_Q=<队列>"
  exit 2
fi
say "===== 外传三组提交 queue=$Q $(date '+%F %T') ====="

for tag in $TAGS; do
  cfg="$CFG_DIR/config_wind_canvas_${tag}.yml"
  if [ ! -f "$cfg" ]; then say "!! 缺配置 $cfg(先跑 prep_stats.sh)"; continue; fi
  if bjobs -w 2>/dev/null | grep -q " $tag "; then say "[skip] $tag 已在队列"; continue; fi
  if [ "${DRY_RUN:-0}" = "1" ]; then
    say "[DRY] bsub -q $Q -gpu num=1 -n 4 -J $tag ... python scripts/train_regression_model.py --config_path $cfg"
    continue
  fi
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "$tag" -o "logs/${tag}_%J.out" -e "logs/${tag}_%J.err" \
    "cd $ROOT && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
bash ops/queue/gpu_keepalive.sh start; \
python -u scripts/train_regression_model.py \
--config_path $CFG_DIR/config_wind_canvas_${tag}.yml \
--experiment_name $EXP --device cuda:0 > logs/${tag}.log 2>&1; \
bash ops/queue/gpu_keepalive.sh stop" 2>&1 | head -1 | tee -a "$OUT"
done

say "===== 之后:bjobs -w | grep sq_ 看状态;完成后跑评估 ====="
