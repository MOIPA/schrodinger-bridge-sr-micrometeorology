#!/bin/bash
# 外传 · 三尺度 3× 数据准备(纯 CPU,不占 GPU)
#   1) 抽 d02 / d03 的细端 npz(744 时次 × 2 scheme)
#   2) 为三组算静态场与重网格权重:
#        A 27→9 : fine=d02  coarse=d01
#        B  9→3 : fine=d03  coarse=d02
#        C  3→1 : fine=d04  coarse=d03
# 用法:
#   bash sidequest_scale3x/queue/prep_data.sh              # 前台跑
#   RUN_LSF=1 bash sidequest_scale3x/queue/prep_data.sh    # 提交到 CPU 队列
set -u
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
PY3D="${PY3D:-/fs00/software/anaconda/3/bin/python}"
SCH="${SCH:-myj}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" sidequest_scale3x/results

run() {
  echo ">>> $*"
  "$@" || { echo "!! 失败: $*"; exit 1; }
}

first_file() {
  "$PY3D" - "$1" <<'PY'
import glob, os, sys
d = sys.argv[1]
f = sorted(glob.glob("/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_%s/wrfout_%s_*" % ("myj", d)))
print(f[0] if f else "")
PY
}

prep_pair() {
  local tag="$1" fine="$2" coarse="$3"
  local ff cf
  ff=$(first_file "$fine"); cf=$(first_file "$coarse")
  [ -n "$ff" ] && [ -n "$cf" ] || { echo "!! 找不到 $fine/$coarse 文件"; exit 1; }
  if [ -f "prepare_npz_outline_static_${tag}/statics.npz" ]; then
    echo ">>> [skip] statics_${tag} 已存在"
  else
    run "$PY3D" scripts/outline/10_extract_statics.py \
        --fine_file "$ff" --coarse_file "$cf" \
        --out_dir "prepare_npz_outline_static_${tag}"
  fi
}

if [ "${RUN_LSF:-0}" = "1" ]; then
  bsub -q 6458q -n 6 -R "rusage[mem=16000]" \
       -o "$HOME/schrodinger-bridge-sr-micrometeorology/logs/sq_prep.%J.out" \
       -J sq_prep bash sidequest_scale3x/queue/prep_data.sh
  exit 0
fi

echo "===== 1. 细端抽取 ====="
for dom in d02 d03; do
  out="prepare_npz_outline_fine_${dom}"
  if ls "$out"/f_${SCH}_*.npz >/dev/null 2>&1; then
    n=$(ls "$out"/f_${SCH}_*.npz | wc -l)
    echo ">>> [skip] $out 已有 $n 个文件"
  else
    run "$PY3D" sidequest_scale3x/scripts/extract_fine.py \
        --domain "$dom" --scheme "$SCH" --workers 6 --out_dir "$out"
  fi
done

echo "===== 2. 静态场与重网格权重 ====="
prep_pair sq27to9 d02 d01
prep_pair sq9to3  d03 d02
prep_pair sq3to1  d04 d03

echo "===== 完成 ====="
du -sh prepare_npz_outline_fine_d02 prepare_npz_outline_fine_d03 2>/dev/null
ls -l prepare_npz_outline_static_sq*/ 2>/dev/null | head -20
