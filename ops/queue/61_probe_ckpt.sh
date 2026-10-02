#!/bin/bash
# 61_probe_ckpt.sh — 通用"对任意 run 的 checkpoint 跑 96 权重探针"(训练中途诊断/复核用)。
# 输出各损失分项原始值(data/div/vort/spectral/extreme)+ hinge 激活率,用于回答
# "训练损失变化来自数据项还是物理项"这类问题,也是重定标权重的依据。
#
# 用法(登录节点,短命令;默认 CPU 直接跑,20 batch 约几分钟):
#   bash ops/queue/61_probe_ckpt.sh <tag> [split] [max_batches] [device]
#   tag 例:p2_div_mid(配置 configs/深圳/phase2/config_wind_canvas_<tag>.yml,
#   checkpoint data/DL_result/ExperimentSchrodingerBridgeWindCanvas/config_wind_canvas_<tag>/checkpoint.pth)
# 输出:results/phase2/probe_<tag>_ckpt.json + 屏幕表格;GPU 提交:GPU=1 前缀
set -u
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
TAG="${1:?用法: bash ops/queue/61_probe_ckpt.sh <tag> [split] [max_batches] [device]}"
SPLIT="${2:-valid}"
NB="${3:-20}"
DEV="${4:-cpu}"
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
CFG="configs/深圳/phase2/config_wind_canvas_${TAG}.yml"
CK="data/DL_result/$EXP/config_wind_canvas_${TAG}/checkpoint.pth"
OUTJ="results/phase2/probe_${TAG}_ckpt.json"
LOG="logs/probe_${TAG}_ckpt.log"
mkdir -p logs results/phase2

if [ "${GPU:-0}" = "1" ]; then
  QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  # GPU 计算节点 PATH 可能没有 git(实测报 git: command not found,结果回传会静默失败)
  GITDIR=$(dirname "$(command -v git)")
  echo "提交 $TAG 探针 -> $Q"
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "probe_${TAG}" -o "logs/probe_${TAG}_%J.out" \
    "cd $ROOT && export PATH=$GITDIR:\$PATH && bash ops/queue/61_probe_ckpt.sh $TAG $SPLIT $NB cuda:0 > logs/probe_${TAG}_job.log 2>&1"
  exit 0
fi

echo "== $TAG: cfg=$CFG ckpt=$CK split=$SPLIT n=$NB device=$DEV =="
$PY3D -u scripts/outline/96_weight_probe.py \
  --config_path "$CFG" --checkpoint "$CK" --split "$SPLIT" \
  --max_batches "$NB" --device "$DEV" --out_json "$OUTJ" 2>&1 | tee "$LOG"
echo "exit=$?"
ls -l "$OUTJ" 2>/dev/null
git add ops/result results/phase2 2>/dev/null
git commit -m "result 61 probe $TAG" 2>/dev/null
git push 2>/dev/null || { git fetch origin && git merge --no-edit origin/main && git push; }
