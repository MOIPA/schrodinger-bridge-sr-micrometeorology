#!/usr/bin/env bash
# cascade_cpu_smoke.sh — 级联(27→9→3)链路的**纯 CPU 冒烟测试**。
#
# 目的:在真跑 144 时次之前,用 1 个时次 + 旧 checkpoint 把"dump → synth → 生成配置 → 二级推理"
#       这条链走通。真跑会在 GPU 队列上排几小时,若这里有一个路径/键名 bug,代价太大。
#
# 隔离:用**独立 tag**(*_smoke_reg)与独立输出目录,不写 `sidequest_scale3x/results/`,
#       不碰正在训练的 checkpoint 目录(只读软链到 `_old_split/` 下的旧权重)。
#
# 用法(登录节点,不需要 GPU):
#   PY3=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python \
#     bash ops/debug/cascade_cpu_smoke.sh
#   KEEP=1 ...   # 保留中间产物便于排查
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

PY="${PY3:-python3}"
WK="${WK:-/tmp/casc_cpu_smoke}"
OUT="${OUT:-$WK/out}"
CK="data/DL_result/ExperimentSchrodingerBridgeWindCanvas"
CFGS=sidequest_scale3x/configs/phase3_arch
OLDCK=data/DL_result/_old_split
NFR="${NFR:-1}"

say() { echo "[cpu-smoke] $*"; }

[ -f "$OLDCK/config_wind_canvas_sq27to9_reg_preunify/checkpoint.pth" ] || {
  echo ">>> 找不到旧 checkpoint: $OLDCK/config_wind_canvas_sq27to9_reg_preunify/checkpoint.pth"; exit 2; }
[ -f "$OLDCK/config_wind_canvas_sq9to3_reg_preunify/checkpoint.pth" ] || {
  echo ">>> 找不到旧 checkpoint: $OLDCK/config_wind_canvas_sq9to3_reg_preunify/checkpoint.pth"; exit 2; }

rm -rf "$WK"; mkdir -p "$WK/s1" "$OUT"

say "① 用独立 tag 建软链(只读旧权重,不碰训练目录)"
ln -sfn ../_old_split/config_wind_canvas_sq27to9_reg_preunify "$CK/config_wind_canvas_sq27to9_smoke_reg"
ln -sfn ../_old_split/config_wind_canvas_sq9to3_reg_preunify  "$CK/config_wind_canvas_sq9to3_smoke_reg"

say "② 生成两个冒烟 tag 的配置(84 只认 cfg_root/phase3_arch 下的同名 yml)"
"$PY" sidequest_scale3x/scripts/gen_cascade_cfg.py \
  --src "$CFGS/config_wind_canvas_sq27to9_reg.yml" --tag sq27to9_smoke_reg \
  --coarse_dir "$PWD/prepare_npz_outline_coarse_d01" >/dev/null

say "③ 一级 27→9 推理(CPU,$NFR 帧)+ dump"
"$PY" -u scripts/outline/84_ensemble_eval.py --mode reg --tags sq27to9_smoke_reg \
  --cfg_root sidequest_scale3x/configs --out_dir "$OUT" --out_suffix _cs \
  --dump_fields "$WK/s1" --max_frames "$NFR" --device cpu 2>&1 | tail -5
[ -f "$OUT/sq27to9_smoke_reg_cs_perhour.npz" ] || { echo ">>> 一级未产出 perhour"; exit 1; }
n_dump=$(ls "$WK/s1"/d_*.npz 2>/dev/null | wc -l)
say "   dump 帧数 = $n_dump"
[ "$n_dump" -ge 1 ] || { echo ">>> dump 为空"; exit 1; }

say "④ 组装二级粗端(synth_coarse)"
"$PY" sidequest_scale3x/scripts/synth_coarse.py \
  --pred_dir "$WK/s1" --coarse_dir "$PWD/prepare_npz_outline_coarse" \
  --td_from "$PWD/prepare_npz_outline_coarse_d01" \
  --td_statics "$PWD/prepare_npz_outline_static_sq27to9" \
  --out_dir "$WK/coarse_s2" 2>&1 | tail -5
n_co=$(ls "$WK/coarse_s2"/c_*.npz 2>/dev/null | wc -l)
say "   组装出的粗端文件数 = $n_co"
[ "$n_co" -ge 1 ] || { echo ">>> synth 没有产出;看上面的 [跳过] 原因"; exit 1; }

say "⑤ 生成二级配置(coarse_dir 指向刚组装出来的目录)"
"$PY" sidequest_scale3x/scripts/gen_cascade_cfg.py \
  --src "$CFGS/config_wind_canvas_sq9to3_reg.yml" --tag sq9to3_smoke_reg \
  --coarse_dir "$WK/coarse_s2" >/dev/null
grep -n "coarse_dir" "$CFGS/config_wind_canvas_sq9to3_smoke_reg.yml" | tail -1

say "⑥ 二级 9→3 推理(CPU)"
"$PY" -u scripts/outline/84_ensemble_eval.py --mode reg --tags sq9to3_smoke_reg \
  --cfg_root sidequest_scale3x/configs --out_dir "$OUT" --out_suffix _cs \
  --max_frames "$NFR" --device cpu 2>&1 | tail -5
[ -f "$OUT/sq9to3_smoke_reg_cs_perhour.npz" ] || { echo ">>> 二级未产出 perhour"; exit 1; }

say "⑦ 清理临时 tag 配置与软链"
rm -f "$CFGS/config_wind_canvas_sq9to3_smoke_reg.yml" "$CFGS/config_wind_canvas_sq27to9_smoke_reg.yml"
rm -f "$CK/config_wind_canvas_sq27to9_smoke_reg" "$CK/config_wind_canvas_sq9to3_smoke_reg"
[ "${KEEP:-0}" = "1" ] || rm -rf "$WK"

say "通过:dump → synth → cfg → 二级推理 全链路无异常"
