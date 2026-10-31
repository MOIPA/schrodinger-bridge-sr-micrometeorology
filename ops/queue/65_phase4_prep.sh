#!/bin/bash
# 65_phase4_prep.sh — 阶段 4 T4.1 跨分辨率:为某个粗端域(默认 d01/27 km)准备数据集与统计量
#
# 只在服务器跑(需要 WRF wrfout + netCDF4 的 pytorch-gpu 环境)。纯 CPU,不占 GPU 队列。
#
# 做四件事,全部幂等(已存在的文件自动跳过):
#   1) 10_extract_statics.py  --coarse_file <d01> --out_dir prepare_npz_outline_static_d01
#      -> 重算 "d01 -> d04" 的双线性重网格权重(mass/U/V 三类交错位置)
#   2) 把开发档的 split.json 复制进新 static 目录(划分按时间切块,与粗端域无关,可复用)
#   3) 21_extract_coarse.py --domain d01 --out_dir prepare_npz_outline_coarse_d01
#   4) 40_stats.py 用新粗端 + 新 static 重算标准化统计量(只在训练块上算)
#
# 用法:
#   bash ops/queue/65_phase4_prep.sh              # d01 / myj
#   DOMAIN=d03 bash ops/queue/65_phase4_prep.sh   # 3 km 档
#   PY3D=/path/to/python bash ops/queue/65_phase4_prep.sh   # 指定带 netCDF4 的解释器
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
export REPO="$PWD"

# 可选:作为 CPU 作业提交(抽取 744 个时次的重网格很吃 CPU,别在登录节点硬跑)
#   RUN_LSF=1 bash ops/queue/65_phase4_prep.sh
if [ "${RUN_LSF:-0}" = "1" ] && [ -z "${LSB_JOBID:-}" ]; then
  mkdir -p logs
  bsub -q "${CPU_Q:-6458q}" -n 4 -R "rusage[mem=16000]" \
    -J "p4prep_${DOMAIN:-d01}" -o "logs/p4prep_${DOMAIN:-d01}_%J.out" \
    "cd $PWD && DOMAIN=${DOMAIN:-d01} SCHEME=${SCHEME:-myj} WORKERS=${WORKERS:-8} RUN_LSF=2 bash ops/queue/65_phase4_prep.sh"
  exit $?
fi

DOMAIN="${DOMAIN:-d01}"
SCHEME="${SCHEME:-myj}"
WORKERS="${WORKERS:-8}"
ROOT="${ROOT:-/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology}"
PY3D="${PY3D:-/fsb/home/yutingwang/ytw_tangzq/.conda/envs/pytorch-gpu/bin/python}"
WRF_BASE="${WRF_BASE:-/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen}"
WRF_DIR="${WRF_DIR:-meso_202007_myj}"

COARSE_OUT="$ROOT/prepare_npz_outline_coarse_${DOMAIN}"
STATIC_OUT="$ROOT/prepare_npz_outline_static_${DOMAIN}"
SPLIT_SRC="$ROOT/prepare_npz_outline_static/split.json"
OUT=ops/result/65_phase4_prep_${DOMAIN}.txt
mkdir -p "$(dirname "$OUT")" logs

say() { echo "$@" | tee -a "$OUT"; }

say "===== 0. 阶段4 数据准备: 粗端域 ${DOMAIN} / scheme ${SCHEME} ====="
say "repo    : $PWD"
say "python  : $PY3D"
say "wrfout  : $WRF_BASE/$WRF_DIR/wrfout_${DOMAIN}_*"

FIRST=$(ls "$WRF_BASE/$WRF_DIR"/wrfout_${DOMAIN}_* 2>/dev/null | head -1 || true)
if [ -z "$FIRST" ]; then
  say ">>> 找不到 $DOMAIN 的 wrfout,停。检查 WRF_BASE/WRF_DIR。"
  exit 2
fi
N_FILES=$(ls "$WRF_BASE/$WRF_DIR"/wrfout_${DOMAIN}_* 2>/dev/null | wc -l | tr -d ' ')
say "样例文件: $(basename "$FIRST");共 $N_FILES 个"
[ -f "$SPLIT_SRC" ] || { say ">>> 缺 $SPLIT_SRC(先跑阶段 0 的 30_blocks_and_split.py)"; exit 2; }

say ""
say "===== 1. 静态场 + 重网格权重(${DOMAIN} -> d04)====="
if [ -f "$STATIC_OUT/statics.npz" ]; then
  say "[跳过] 已存在 $STATIC_OUT/statics.npz"
else
  "$PY3D" scripts/outline/10_extract_statics.py --coarse_file "$FIRST" --out_dir "$STATIC_OUT" 2>&1 | tail -8 | tee -a "$OUT"
fi

say ""
say "===== 2. 复用开发档划分 split.json ====="
mkdir -p "$STATIC_OUT"
if [ -f "$STATIC_OUT/split.json" ]; then
  say "[跳过] split.json 已在"
else
  cp "$SPLIT_SRC" "$STATIC_OUT/split.json"
  say "已复制: $(basename "$SPLIT_SRC") -> $STATIC_OUT/"
fi

say ""
say "===== 3. 粗端逐时 npz 抽取(${DOMAIN},原生 C 网格)====="
"$PY3D" scripts/outline/21_extract_coarse.py --scheme "$SCHEME" --domain "$DOMAIN" \
  --out_dir "$COARSE_OUT" --workers "$WORKERS" 2>&1 | tail -12 | tee -a "$OUT"

N_COARSE=$(ls "$COARSE_OUT"/c_${SCHEME}_*.npz 2>/dev/null | wc -l | tr -d ' ')
say "粗端 npz 数: $N_COARSE(开发档 d02 应为 $N_FILES 左右)"

say ""
say "===== 4. 标准化统计量(只用训练块)====="
"$PY3D" scripts/outline/40_stats.py --coarse_dir "$COARSE_OUT" --static_dir "$STATIC_OUT" \
  --schemes "$SCHEME" 2>&1 | tail -10 | tee -a "$OUT"

say ""
say "===== 5. 生成阶段 4 配置 ====="
"$PY3D" scripts/outline/41_gen_phase4_configs.py --data_root "$ROOT" 2>&1 | tee -a "$OUT"

say ""
say "===== 完成。下一步 ====="
say "  核对: ls $COARSE_OUT | wc -l ; ls $STATIC_OUT"
say "  提交训练: bash ops/queue/66_phase4_train.sh --arms p4_27to1_reg"
say "  注意: 27->1 km 的粗端只有 1/9 的格点数,通道数不变(23 层 U/V),先在 1 个 epoch 上看 loss 是否正常。"
