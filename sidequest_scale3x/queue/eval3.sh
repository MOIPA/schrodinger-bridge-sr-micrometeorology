#!/bin/bash
# 外传·三尺度 3×:训练完之后的评估入口
#
#   bash sidequest_scale3x/queue/eval3.sh            # 用 q_pick 选队列提交
#   FORCE_Q=734090d bash sidequest_scale3x/queue/eval3.sh
#   TAGS="sq27to9_reg" bash sidequest_scale3x/queue/eval3.sh   # 只评一组
#
# 评估口径与阶段 3 完全一致(84_ensemble_eval.py --mode reg):
#   主指标 = AGL 10-500 m 层平均风矢量 RMSE(池化)
#   y0    = 粗端重网格到细网格(即"普通插值基线"),增益 = 1 - RMSE/RMSE_y0
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
ROOT="${ROOT:-/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology}"
PY3D="${PY3D:-/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python}"
CFG_ROOT="sidequest_scale3x/configs"
OUT_DIR="sidequest_scale3x/results"
EXP="ExperimentSchrodingerBridgeWindCanvas"
TAGS="${TAGS:-sq27to9_reg sq9to3_reg sq3to1_reg}"
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
  for t in $TAGS; do
    bsub -q "$QUEUE" -gpu "num=1:mode=exclusive_process" -n 4 \
      -R "rusage[mem=32000]" \
      -J "sqeval_${t}" -o "logs/sqeval_${t}.out" -e "logs/sqeval_${t}.err" \
      bash -lc "cd '$ROOT' && export PATH=/usr/bin:\$PATH && \
        INNER=1 TAGS='$t' bash sidequest_scale3x/queue/eval3.sh"
    echo "submitted eval ${t} -> ${QUEUE}"
  done
  exit 0
fi

for t in $TAGS; do
  echo "===== eval $t ====="
  "$PY3D" - <<PYEOF || true
import sys, importlib.util as iu, yaml
sys.path.insert(0, "scripts/outline")
spec = iu.spec_from_file_location("m84", "scripts/outline/84_ensemble_eval.py")
m = iu.module_from_spec(spec); spec.loader.exec_module(m)
cfg = m.load_dataset_config("$CFG_ROOT/phase3_arch/config_wind_canvas_${t}.yml")
ctx = m.build_static_ctx(cfg, "cpu")
print("[shapecheck]", cfg.data.statics_dir.split("/")[-1],
      ctx["tables"]["idx_m"].shape, ctx["tables"]["zagl_mass"].shape)
PYEOF
  "$PY3D" scripts/outline/84_ensemble_eval.py \
    --mode reg --tags "$t" --split test --device cuda:0 \
    --cfg_root "$CFG_ROOT" --out_dir "$OUT_DIR" || { echo "FAIL $t"; continue; }
done

"$PY3D" sidequest_scale3x/scripts/summarize3.py --dir "$OUT_DIR" \
  --json_out "$OUT_DIR/summary3.json" --md_out "$OUT_DIR/summary3.md" || true
