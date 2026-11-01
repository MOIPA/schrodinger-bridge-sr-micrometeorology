#!/usr/bin/env bash
# 级联超分 27→9→3→1:三级串起来,与"一步 27→1"对照。
# 关键点:中间级的粗端用上一级预测的风场 + 上一级粗端的地面场(重网格过来),
#         所以整条级联只用 27 km 的信息,和一步 27→1 信息量对等。
# 用法:bash sidequest_scale3x/queue/cascade.sh          # 需 GPU
#       STAGES=1 bash ...                               # 只跑到第 1 级(调试)
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

PY="${PY3:-/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python}"
WK="${WK:-/tmp/casc}"
ROOT_DATA="${ROOT_DATA:-$PWD}"
CFG_DIR=sidequest_scale3x/configs/phase3_arch
OUT=sidequest_scale3x/results
CK="data/DL_result/ExperimentSchrodingerBridgeWindCanvas"
STAGES="${STAGES:-3}"
MAXFR="${MAXFR:-0}"   # >0 时每级只跑前 N 个时次(冒烟测试用)

mkdir -p "$WK"
FR_OPT=""
if [ "$MAXFR" -gt 0 ]; then FR_OPT="--max_frames $MAXFR"; fi

link_ckpt() {  # $1=新建的 tag 目录名, $2=真实权重目录名(同级)
  local dst="$CK/$1" src="$2"
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    echo ">>> 拒绝:"$dst" 是真实目录,不是软链。若在这里 ln -sfn,软链会建到目录【内部】,"         "上一轮就是这样把冒烟作业指到了正在训练的 checkpoint 上。请先人工确认再删。" >&2
    exit 1
  fi
  ln -sfn "$src" "$dst"
}

fresh_dir() {  # $1=目录;冒烟测试时清掉上一轮的残留,避免 synth 读到旧 dump
  if [ -n "$FR_OPT" ]; then rm -rf "$1"; fi
  mkdir -p "$1"
}

fresh_dir "$WK/s1"
echo "=== [1/3] 27→9 (dump 到 $WK/s1) ==="
"$PY" -u scripts/outline/84_ensemble_eval.py --mode reg --tags sq27to9_reg \
  --cfg_root sidequest_scale3x/configs --out_dir "$OUT" --out_suffix _casc \
  --dump_fields "$WK/s1" $FR_OPT --device "${DEV:-cuda:0}" | tail -3
[ -f "$OUT/sq27to9_reg_casc_perhour.npz" ] || { echo ">>> 27→9 未产出,中止"; exit 1; }
[ "$STAGES" -lt 2 ] && { echo "STAGES=$STAGES,停在第 1 级"; exit 0; }

fresh_dir "$WK/s2"
echo "=== [2/3] 组装 9→3 的粗端 ==="
"$PY" sidequest_scale3x/scripts/synth_coarse.py \
  --pred_dir "$WK/s1" --coarse_dir "$ROOT_DATA/prepare_npz_outline_coarse" \
  --td_from "$ROOT_DATA/prepare_npz_outline_coarse_d01" \
  --td_statics "$ROOT_DATA/prepare_npz_outline_static_sq27to9" \
  --out_dir "$ROOT_DATA/prepare_npz_outline_coarse_casc_s2"
"$PY" sidequest_scale3x/scripts/gen_cascade_cfg.py \
  --src "$CFG_DIR/config_wind_canvas_sq9to3_reg.yml" --tag sq9to3_casc \
  --coarse_dir "$ROOT_DATA/prepare_npz_outline_coarse_casc_s2" >/dev/null
link_ckpt config_wind_canvas_sq9to3_casc config_wind_canvas_sq9to3_reg

echo "=== [2/3] 9→3 推理 ==="
"$PY" -u scripts/outline/84_ensemble_eval.py --mode reg --tags sq9to3_casc \
  --cfg_root sidequest_scale3x/configs --out_dir "$OUT" --out_suffix _casc \
  --dump_fields "$WK/s2" $FR_OPT --device "${DEV:-cuda:0}" | tail -3
[ -f "$OUT/sq9to3_casc_casc_perhour.npz" ] || { echo ">>> 9→3 未产出,中止"; exit 1; }
[ "$STAGES" -lt 3 ] && { echo "STAGES=$STAGES,停在第 2 级"; exit 0; }

echo "=== [3/3] 组装 3→1 的粗端 ==="
"$PY" sidequest_scale3x/scripts/synth_coarse.py \
  --pred_dir "$WK/s2" --coarse_dir "$ROOT_DATA/prepare_npz_outline_coarse_d03" \
  --td_from "$ROOT_DATA/prepare_npz_outline_coarse_casc_s2" \
  --td_statics "$ROOT_DATA/prepare_npz_outline_static_sq9to3" \
  --out_dir "$ROOT_DATA/prepare_npz_outline_coarse_casc_s3"
"$PY" sidequest_scale3x/scripts/gen_cascade_cfg.py \
  --src "$CFG_DIR/config_wind_canvas_sq3to1_reg.yml" --tag sq3to1_casc \
  --coarse_dir "$ROOT_DATA/prepare_npz_outline_coarse_casc_s3" >/dev/null
link_ckpt config_wind_canvas_sq3to1_casc config_wind_canvas_sq3to1_reg

echo "=== [3/3] 3→1 推理(级联终点) ==="
"$PY" -u scripts/outline/84_ensemble_eval.py --mode reg --tags sq3to1_casc \
  --cfg_root sidequest_scale3x/configs --out_dir "$OUT" --out_suffix _casc \
  $FR_OPT --device "${DEV:-cuda:0}" | tail -3
[ -f "$OUT/sq3to1_casc_casc_perhour.npz" ] || { echo ">>> 3→1 未产出,中止"; exit 1; }

echo "=== 汇总 ==="
"$PY" sidequest_scale3x/scripts/summarize3.py --dir "$OUT" || true
