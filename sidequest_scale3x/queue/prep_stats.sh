#!/bin/bash
# 外传 · 三组的划分/标准化/配置(纯 CPU,可本地跑也可 LSF)
# 前置:prep_data.sh 已产出 prepare_npz_outline_fine_d02/d03 与三套 static_sq*
set -u
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
PY3D="${PY3D:-/fs00/software/anaconda/3/bin/python}"
SCH="${SCH:-myj}"

if [ "${RUN_LSF:-0}" = "1" ]; then
  bsub -q 6458q -n 4 -R "rusage[mem=16000]" \
       -o "$HOME/schrodinger-bridge-sr-micrometeorology/logs/sq_stats.%J.out" \
       -J sq_stats bash sidequest_scale3x/queue/prep_stats.sh
  exit 0
fi

run() { echo ">>> $*"; "$@" || { echo "!! 失败: $*"; exit 1; }; }

# 组: tag 细端目录 粗端目录
while read -r tag fine coarse; do
  # ONLY=sq3to1 只处理指定组(便于失败后单独补跑)
  if [ -n "${ONLY:-}" ] && [ "$tag" != "$ONLY" ]; then continue; fi
  [ -n "$tag" ] || continue
  static="prepare_npz_outline_static_${tag}"
  ver="sq${tag#sq}"; ver="${ver%_reg}"

  if [ ! -d "$fine" ]; then echo "!! 缺细端目录 $fine(先跑 prep_data.sh)"; exit 1; fi
  if [ ! -d "$coarse" ]; then echo "!! 缺粗端目录 $coarse"; exit 1; fi
  if [ ! -d "$static" ]; then echo "!! 缺静态目录 $static(先跑 prep_data.sh)"; exit 1; fi

  # 1) 数据目录:数据集只认 data/DL_data/<ver>/
  mkdir -p "data/DL_data/$ver"
  if ! ls "data/DL_data/$ver"/*.npz >/dev/null 2>&1; then
    ln -sf "$PWD/$fine"/f_${SCH}_*.npz "data/DL_data/$ver/" 2>/dev/null
    echo ">>> 链接 $(ls data/DL_data/$ver/*.npz | wc -l) 个细端文件 -> $ver"
  fi

  # 2) 时间划分(不存在才生成)
  if [ ! -f "$static/split.json" ]; then
    run "$PY3D" scripts/outline/30_blocks_and_split.py \
        --coarse_dir "$coarse" --fine_dir "$fine" --scheme "$SCH" \
        --json_out "$static/split.json"
  else
    echo ">>> [skip] $static/split.json 已存在"
  fi

  # 3) 标准化统计
  if [ ! -f "$static/normalize_config.json" ]; then
    run "$PY3D" scripts/outline/40_stats.py \
        --fine_dir "$fine" --coarse_dir "$coarse" --static_dir "$static" --schemes "$SCH"
  else
    echo ">>> [skip] $static/normalize_config.json 已存在"
  fi
done <<EOF
sq27to9 prepare_npz_outline_fine_d02 prepare_npz_outline_coarse_d01
sq9to3  prepare_npz_outline_fine_d03 prepare_npz_outline_coarse
sq3to1  prepare_npz_outline_fine prepare_npz_outline_coarse_d03
EOF

echo "===== 生成配置 ====="
run "$PY3D" sidequest_scale3x/scripts/gen_configs.py
echo "===== 完成 ====="
