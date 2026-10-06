#!/bin/bash
# 64d_arch_eval.sh — 阶段 3 架构对比评估(一个 GPU 作业内串行,复刻 63d 模式):
#   ① 84 si 模式:unet_s2 / swin / swin_s2 各臂,N=16 集合均值(add_noise=True 随机采样),
#      产物 results/phase3_arch/<tag>_perhour.npz + _summary.json + _ensemble.json;
#   ② 参考臂 p2_l1r2_lr2e4 ens16 重评:同一集合协议,out_suffix=_ens16
#      (与 results/phase3 下既有的确定性 ODE 产物 0.9574 区分;两者口径不同,不做回归判定);
#   ③ 84 reg 模式:回归臂(N=1 确定性单次前向);
#   ④ 84 two_step 模式:EDM 臂(N=16 集合均值;回归网从 checkpoint 的 reg 记录定位);
#   ⑤ 主指标汇总:读 *_summary.json 打印表(与 phase2 参考值 0.9574 并列展示,注明口径)。
# 前置依赖:64b 训完;84_ensemble_eval.py 与 83 生成的 phase3_arch 配置。
# 手机执行: bash ops/queue/64d_arch_eval.sh [--tags a,b] [--after-training a,b]
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
OUT_DIR=results/phase3_arch
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"
QUEUES="${FORCE_Q:-$QUEUES}"
N_ENS=16
BASE_REF=0.9574            # results/phase2 确定性 ODE 记录值(仅并列展示,口径不同)

ALL_SI="p3_arch_unet_s2 p3_arch_unet_small p3_arch_unet_small_s2 p3_arch_swin p3_arch_swin_s2"
ALL_REG="p3_arch_reg p3_arch_reg_s2"
ALL_EDM="p3_arch_edm p3_arch_edm_s2"
REF_TAG="p2_l1r2_lr2e4"

cfg_of() {  # $1=tag;按前缀选阶段目录
  local t=$1 d=phase3_arch
  case "$t" in p2_*) d=phase2;; esac
  echo "configs/深圳/$d/config_wind_canvas_${t}.yml"
}
ck_of()  { echo "$RESULT_BASE/config_wind_canvas_${1}/checkpoint.pth"; }

INNER=""
TAGS=""
AFTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --inner) INNER=1; shift;;
    --tags) TAGS="$2"; shift 2;;
    --tags=*) TAGS="${1#--tags=}"; shift;;
    --after-training) AFTER="$2"; shift 2;;
    --after-training=*) AFTER="${1#--after-training=}"; shift;;
    *) echo "未知参数: $1(用法: [--inner] [--tags a,b] [--after-training a,b])"; shift;;
  esac
done

# 只保留有 checkpoint 的 tag(避免 84 因缺 ckpt 跳过;two_step 要求 tags/edm_tags 并行)
filter_ck() {
  local out=""
  for t in $1; do [ -f "$(ck_of "$t")" ] && out="$out $t"; done
  echo "$out"
}

if [ -z "$INNER" ]; then
  git pull --no-rebase
  git --no-pager log --oneline -1
  mkdir -p ops/result logs results/phase3_arch
  SI_T=$(filter_ck "$ALL_SI")
  REG_T=$(filter_ck "$ALL_REG")
  EDM_T=$(filter_ck "$ALL_EDM")
  REF_T2=""
  [ -f "$(ck_of "$REF_TAG")" ] && REF_T2="$REF_TAG"
  if [ -z "$TAGS" ]; then
    TAGS_CSV=$(echo "$SI_T $REG_T $EDM_T $REF_T2" | tr ' ' ',' | sed 's/^,//;s/,,*/,/g')
  else
    TAGS_CSV=$(echo "$TAGS" | tr ' ' ',')
  fi
  if [ -z "${SI_T}${REG_T}${EDM_T}${REF_T2}${TAGS}" ]; then
    echo "没有任何 checkpoint,评估清单为空;先训练(bash ops/queue/64b_arch_waves.sh)"
    exit 0
  fi
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  # GPU 计算节点的 PATH 可能没有 git,提交时取登录节点的 git 目录注入作业 PATH
  GITDIR=$(dirname "$(command -v git)")

  DEP=""
  if [ -n "$AFTER" ]; then
    ACTIVE=$(bjobs -o "job_name" -noheader 2>/dev/null)
    for t in $(echo "$AFTER" | tr ',' ' '); do
      if echo "$ACTIVE" | grep -qx "$t"; then
        if [ -z "$DEP" ]; then DEP="ended($t)"; else DEP="$DEP && ended($t)"; fi
        echo "依赖: ended($t)"
      else
        echo "依赖: $t 已不在 bjobs 里,不挂(否则 LSF 报 No matching job found)"
      fi
    done
  fi
  echo "评估清单: $TAGS_CSV"
  echo "  si 臂: $SI_T(N=$N_ENS 集合均值);reg 臂: $REG_T(确定性);" \
       "edm 臂: $EDM_T(N=$N_ENS);参考臂: ${REF_T2:-(缺 ckpt)}(_ens16 重评)"
  if [ -n "$DEP" ]; then
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -w "$DEP" -J "p3eval_arch" -o "logs/p3eval_arch_%J.out" -e "logs/p3eval_arch_%J.err" \
      "cd $ROOT && export PATH=$GITDIR:\$PATH && EVAL_TAGS=$TAGS_CSV SI_TAGS='$SI_T' REG_TAGS='$REG_T' EDM_TAGS='$EDM_T' bash ops/queue/64d_arch_eval.sh --inner"
  else
    bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
      -J "p3eval_arch" -o "logs/p3eval_arch_%J.out" -e "logs/p3eval_arch_%J.err" \
      "cd $ROOT && export PATH=$GITDIR:\$PATH && EVAL_TAGS=$TAGS_CSV SI_TAGS='$SI_T' REG_TAGS='$REG_T' EDM_TAGS='$EDM_T' bash ops/queue/64d_arch_eval.sh --inner"
  fi
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/64d_arch_eval.txt
: > "$OUT"
mkdir -p results/phase3_arch

SI_T="${SI_TAGS:-$(filter_ck "$ALL_SI")}"
REG_T="${REG_TAGS:-$(filter_ck "$ALL_REG")}"
EDM_T="${EDM_TAGS:-$(filter_ck "$ALL_EDM")}"
SI_T=$(echo "$SI_T" | tr ',' ' ')
REG_T=$(echo "$REG_T" | tr ',' ' ')
EDM_T=$(echo "$EDM_T" | tr ',' ' ')
REF_T2=""
[ -f "$(ck_of $REF_TAG)" ] && REF_T2="$REF_TAG"

# --tags 过滤(外层的清单里再取交集;参考臂要显式列出才评)
EV_LIST=$(echo "${EVAL_TAGS:-}" | tr ',' ' ')
if [ -n "$EV_LIST" ]; then
  keep() { local o=""; for t in $1; do for k in $EV_LIST; do [ "$k" = "$t" ] && { o="$o $t"; break; }; done; done; echo "$o"; }
  SI_T=$(keep "$SI_T"); REG_T=$(keep "$REG_T"); EDM_T=$(keep "$EDM_T")
  echo "$EV_LIST" | grep -qw "$REF_TAG" || REF_T2=""
fi

{
echo "===== 0. 评估清单 ====="
echo "si 臂(N=$N_ENS 集合均值,默认 suffix):$SI_T"
echo "参考臂(N=$N_ENS,out_suffix=_ens16):$REF_T2"
echo "reg 臂(N=1 确定性):$REG_T"
echo "edm 臂(N=$N_ENS 集合均值):$EDM_T"
echo "产物目录:$OUT_DIR(全部 84 产物;绝不覆盖 results/phase3 的既有文件名)"

echo ""
echo "===== 1. 84 si 模式(各臂 N=$N_ENS 集合均值) ====="
if [ -n "$SI_T" ]; then
  CSV=$(echo "$SI_T" | tr ' ' ',')
  $PY3D -u scripts/outline/84_ensemble_eval.py --mode si --tags "$CSV" \
    --n_ensemble $N_ENS --split test --out_dir $OUT_DIR --device cuda:0 \
    || echo "FAIL 84 si ($CSV)"
else
  echo "(无 si 臂 checkpoint,跳过)"
fi

echo ""
echo "===== 2. 参考臂 ens16 重评($REF_TAG;_ens16) ====="
if [ -n "$REF_T2" ]; then
  $PY3D -u scripts/outline/84_ensemble_eval.py --mode si --tags "$REF_TAG" \
    --n_ensemble $N_ENS --out_suffix _ens16 --split test --out_dir $OUT_DIR --device cuda:0 \
    || echo "FAIL 84 si 参考臂"
  echo "对照:results/phase2 的确定性 ODE 主指标 = $BASE_REF;本产物为 N=$N_ENS 随机采样均值,"
  echo "      两者口径不同(90 号 add_noise=False);仅并列展示,不判回归。"
else
  echo "(缺参考臂 checkpoint,跳过;$BASE_REF 仅作历史对照)"
fi

echo ""
echo "===== 3. 84 reg 模式(确定性单次前向) ====="
if [ -n "$REG_T" ]; then
  CSV=$(echo "$REG_T" | tr ' ' ',')
  $PY3D -u scripts/outline/84_ensemble_eval.py --mode reg --tags "$CSV" \
    --n_ensemble 1 --split test --out_dir $OUT_DIR --device cuda:0 \
    || echo "FAIL 84 reg ($CSV)"
else
  echo "(无 reg 臂 checkpoint,跳过)"
fi

echo ""
echo "===== 4. 84 two_step 模式(EDM,N=$N_ENS 集合均值) ====="
if [ -n "$EDM_T" ]; then
  CSV=$(echo "$EDM_T" | tr ' ' ',')
  $PY3D -u scripts/outline/84_ensemble_eval.py --mode two_step \
    --tags "$CSV" --edm_tags "$CSV" --n_ensemble $N_ENS --split test \
    --out_dir $OUT_DIR --device cuda:0 || echo "FAIL 84 two_step ($CSV)"
else
  echo "(无 edm 臂 checkpoint,跳过)"
fi

echo ""
echo "===== 5. 主指标汇总(summary json;phase2 参考 $BASE_REF 仅并列) ====="
$PY3D -c "
import glob, json, os
rows = []
for p in sorted(glob.glob('$OUT_DIR/*_summary.json')):
    d = json.load(open(p))
    rows.append((d.get('mode', '?'), d.get('tag', os.path.basename(p)),
                 d.get('n_ensemble'), d.get('n_hours'), d.get('main_rmse_vec')))
print('| mode | tag | N | n_hours | main_rmse_vec (m/s) | 与 %.4f 之差 |' % $BASE_REF)
print('|---|---|---|---|---|---|')
for m, t, n, h, v in rows:
    if v is None:
        print('| %s | %s | %s | %s | (缺) | - |' % (m, t, n, h)); continue
    print('| %s | %s | %s | %s | %.4f | %+.4f |' % (m, t, n, h, v, v - $BASE_REF))
print()
print('口径:主指标 = AGL 10-500 m 池化矢量 RMSE;si/edm 为 N=$N_ENS 集合均值(随机协议),')
print('      reg 为确定性单次前向;与 phase2 的 %.4f(90 号确定性 ODE)口径不同。' % $BASE_REF)
"

echo ""
echo "===== 6. 产出 ====="
ls -l $OUT_DIR | tail -40
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase3_arch
git commit -m "result 64d arch eval (84;si/reg/two_step + p2_l1r2_lr2e4 ens16 re-eval)"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
