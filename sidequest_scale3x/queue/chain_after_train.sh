#!/usr/bin/env bash
# 统一口径重训完成后,一次挂上全部下游作业(按作业号挂依赖,不靠轮询)。
#
#   bash sidequest_scale3x/queue/chain_after_train.sh            # 用默认三作业号
#   DEPS="14833682 14833683 14833684" Q=944090d bash ...
#   DRY_RUN=1 bash ...                                          # 只打印 bsub
#
# 挂哪些:
#   1) 三档重评(eval3 INNER):sq27to9_reg / sq9to3_reg / sq3to1_reg
#   2) 交叉迁移重算:eval_xfer(两个方向)
#   3) 级联 27→9→3→1:cascade.sh(三级串跑,末尾自己 summarize3)
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

DEPS="${DEPS:-14833682 14833683 14833684}"
Q="${Q:-944090d}"
PY3D="${PY3D:-/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python}"
ROOT="${ROOT:-$PWD}"

COND=""
for j in $DEPS; do
  if [ -z "$COND" ]; then COND="ended($j)"; else COND="$COND && ended($j)"; fi
done
echo ">>> 依赖条件: $COND"

sub() {  # sub <jobname> <inner-command>
  local name="$1"; shift
  local cmd="$1"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY] bsub -q $Q -J $name -w \"$COND\" bash -lc \"$cmd\""
    return
  fi
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 \
    -R "rusage[mem=32000]" -w "$COND" \
    -J "$name" -o "logs/${name}_%J.out" -e "logs/${name}_%J.err" \
    bash -lc "cd '$ROOT' && bash ops/queue/gpu_keepalive.sh start; $cmd; \
bash ops/queue/gpu_keepalive.sh stop"
}

for t in sq27to9_reg sq9to3_reg sq3to1_reg; do
  sub "eval2_${t}" "INNER=1 TAGS='$t' PY3D='$PY3D' bash sidequest_scale3x/queue/eval3.sh"
done

sub "xfer2" "TAGS='xfer9to3_by27to9_reg,xfer27to9_by3to1_reg' PY3='$PY3D' bash sidequest_scale3x/queue/eval_xfer.sh"

sub "casc" "PY3='$PY3D' bash sidequest_scale3x/queue/cascade.sh"
