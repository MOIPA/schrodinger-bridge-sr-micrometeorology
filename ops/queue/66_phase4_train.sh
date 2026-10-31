#!/bin/bash
# 66_phase4_train.sh — 阶段 4 T4.1 跨分辨率训练提交
#
# 前置: ops/queue/65_phase4_prep.sh 已为对应域(d01/d03)生成
#       prepare_npz_outline_coarse_<dom> / prepare_npz_outline_static_<dom>
#
# 用法:
#   bash ops/queue/66_phase4_train.sh                        # 默认两档都投
#   bash ops/queue/66_phase4_train.sh --arms p4_27to1_reg    # 只投 27->1
#   FORCE_Q=734090d bash ops/queue/66_phase4_train.sh        # 指定队列
#   bash ops/queue/66_phase4_train.sh --after 14831000       # 等某个 prep/作业结束再开跑
#   DRY_RUN=1 bash ops/queue/66_phase4_train.sh              # 只打印 bsub
set -uo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
export REPO="$PWD"

ROOT="${ROOT:-/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology}"
CFG_DIR="configs/深圳/phase4"
ARMS="${ARMS:-p4_27to1_reg p4_3to1_reg}"
AFTER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2;;
    --arms=*) ARMS="${1#--arms=}"; shift;;
    --after) AFTER="$2"; shift 2;;
    --after=*) AFTER="${1#--after=}"; shift;;
    *) echo "未知参数: $1"; shift;;
  esac
done

OUT=ops/result/66_phase4_train.txt
mkdir -p ops/result logs
: > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }

say "===== 阶段4 T4.1 训练提交;臂: $ARMS ====="
if [ -z "${FORCE_Q:-}" ]; then
  Q=$(bash ops/queue/q_pick.sh 2>/dev/null)
  if [ -z "$Q" ]; then
    say ">>> 当前无可用 GPU 队列(全部 closed/满载),不提交。稍后重跑或 FORCE_Q=<队列>。"
    exit 2
  fi
  say ">>> 队列: $Q(q_pick)"
else
  Q="$FORCE_Q"
  say ">>> 队列: $Q(强制)"
fi

DEP=()
[ -n "$AFTER" ] && DEP=(-w "ended($AFTER)")

FAIL=0
for tag in $ARMS; do
  CFG="$CFG_DIR/config_wind_canvas_${tag}.yml"
  if [ ! -f "$CFG" ]; then
    say "[跳过] 缺配置 $CFG(先跑 41_gen_phase4_configs.py)"
    FAIL=$((FAIL+1)); continue
  fi
  DOM=$(echo "$tag" | sed -E 's/^p4_([0-9]+)to1.*/\1/')
  [ "$DOM" = "27" ] && DOM=d01
  [ "$DOM" = "3" ] && DOM=d03
  COARSE="$ROOT/prepare_npz_outline_coarse_${DOM}"
  if [ ! -d "$COARSE" ] || [ -z "$(ls "$COARSE"/c_*_*.npz 2>/dev/null | head -1)" ]; then
    say "[跳过] $tag 缺粗端数据 $COARSE(先跑 65_phase4_prep.sh DOMAIN=$DOM)"
    FAIL=$((FAIL+1)); continue
  fi
  case "$tag" in
    *_reg*) SCRIPT=scripts/train_regression_model.py;;
    *)      SCRIPT=scripts/train_schrodinger_bridge_model.py;;
  esac

  if [ "${DRY_RUN:-0}" = "1" ]; then
    say "[DRY_RUN] bsub -q $Q -gpu num=1:mode=exclusive_process -n 4 -R rusage[mem=32000] ${DEP[*]:-} -J $tag $SCRIPT --config_path $CFG"
    continue
  fi

  say ">>> $tag -> 队列 $Q($(basename "$CFG"))"
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    "${DEP[@]}" \
    -J "$tag" -o "logs/${tag}_%J.out" -e "logs/${tag}_%J.err" \
    "cd $ROOT && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
bash ops/queue/gpu_keepalive.sh start; \
python -u $SCRIPT --config_path $CFG --experiment_name $tag --device cuda:0 \
  > logs/${tag}.log 2>&1; \
bash ops/queue/gpu_keepalive.sh stop" 2>&1 | tee -a "$OUT" | tail -1
done

say ""
say "查状态: bjobs -w | grep p4_ ;tail -n 20 logs/p4_27to1_reg.log"
say "训完后评估: 复用 84_ensemble_eval.py(T4.1 评估脚本待写:按 tag 前缀 p4_ 取 ckpt 与粗端域)"
[ "$FAIL" -gt 0 ] && exit 3
exit 0
