#!/bin/bash
# 64e_arch_report.sh — 阶段 3 架构对比终评报告(一个 GPU 作业内串行,复刻 63e 模式):
#   ① 99_phase3_tables.py:主指标(AGL 10-500 m 矢量 RMSE)+ 切变 + 配对 bootstrap,
#      --perhour_dir results/phase3_arch --base_tag 默认 p3_arch_unet_s2
#      (同种子协议的 UNet 对照臂;若缺其产物回退 p2_l1r2_lr2e4_ens16);
#   ② ensemble json 汇总:各集合臂(N>1)的主层 spread / CRPS / rank 均匀性 /
#      spread-skill 四分位 RMSE,表格打印(数据来自 84 的 *_ensemble.json);
#   ③ 全部 stdout/表格写 ops/result/64_arch_report.txt,回传 ops/result + results/phase3_arch。
# 前置依赖:64d 已评估(否则 99 缺输入;缺文件的行标 —,不抛异常)。
# 手机执行: bash ops/queue/64e_arch_report.sh   (登录节点提交,GPU 节点跑 --inner)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"

if [ "$1" != "--inner" ]; then
  git pull --no-rebase
  git --no-pager log --oneline -1
  mkdir -p ops/result logs results/phase3_arch
  QUEUES="${FORCE_Q:-$QUEUES}"
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  # GPU 计算节点 PATH 可能没有 git,提交时注入登录节点 git 目录
  GITDIR=$(dirname "$(command -v git)")
  echo "阶段 3 架构对比终评报告(99 表 + ensemble json 汇总)-> 队列 $Q"
  SUB=$(bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "p3report_arch" -o "logs/p3report_arch_%J.out" -e "logs/p3report_arch_%J.err" \
    "cd $ROOT && export PATH=$GITDIR:\$PATH && bash ops/queue/64e_arch_report.sh --inner" 2>&1)
  echo "$SUB"
  JID=$(echo "$SUB" | grep -oE '[0-9]+' | head -1)
  sleep 90
  ST=$(bjobs -o "jobid stat" -noheader 2>/dev/null | awk -v j="$JID" '$1==j{print $2}')
  if [ "$ST" = "PEND" ]; then
    echo ">>> p3report_arch($JID) PEND: bkill;换队列重跑本脚本(必要时 FORCE_Q=<队列>)"
    bkill "$JID"
  else
    echo ">>> p3report_arch($JID) 状态: ${ST:-已不在队列(可能已开始或极快失败,查 logs/)}"
  fi
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/64_arch_report.txt
: > "$OUT"
mkdir -p results/phase3_arch

REPORT_TAGS="p3_arch_swin p3_arch_swin_s2 p3_arch_unet_s2 p3_arch_reg p3_arch_reg_s2 \
p3_arch_edm p3_arch_edm_s2 p2_l1r2_lr2e4_ens16"
TAGS_CSV=$(echo "$REPORT_TAGS" | tr -s ' \n' ' ' | tr ' ' ',')
BASE_TAG="p3_arch_unet_s2"
[ -f "results/phase3_arch/p3_arch_unet_s2_perhour.npz" ] \
  || BASE_TAG="p2_l1r2_lr2e4_ens16"

{
echo "===== 阶段 3 架构对比终评报告 ====="
echo "tags: $TAGS_CSV;Δ 基准 = $BASE_TAG"
echo "口径:si/edm 为 N=16 集合均值(随机采样),reg 为确定性单次前向;"
echo "      参考臂 _ens16 = p2_l1r2_lr2e4 的集合重评(与 phase2 确定性 0.9574 口径不同)。"

echo ""
echo "===== 1. 99 阶段 3 汇总表(主指标 + 切变 + 配对 bootstrap) ====="
if [ -f scripts/outline/99_phase3_tables.py ]; then
  $PY3D -u scripts/outline/99_phase3_tables.py --perhour_dir results/phase3_arch \
    --tags "$TAGS_CSV" --base_tag "$BASE_TAG" \
    || { echo "99 带参调用失败(以 99 的 argparse 为准),退回无参调用"; \
         $PY3D -u scripts/outline/99_phase3_tables.py || echo "FAIL 99"; }
  for f in $(ls -1 results/phase3_arch/*.md 2>/dev/null | sort -u); do
    echo "--- $f ---"
    cat "$f"
  done
else
  echo "FAIL: 缺 scripts/outline/99_phase3_tables.py"
fi

echo ""
echo "===== 2. 集合分布指标汇总(*_ensemble.json;主指标层 10-500 m) ====="
$PY3D -c "
import glob, json
import numpy as np
rows = []
for p in sorted(glob.glob('results/phase3_arch/*_ensemble.json')):
    d = json.load(open(p))
    ml = [i for i, h in enumerate(d['agl_targets']) if 10 <= h <= 500]
    sv = float(np.mean([d['spread_vec_per_level'][i] for i in ml]))
    sw = float(np.mean([d['spread_w_per_level'][i] for i in ml]))
    hist = np.asarray(d['rank_hist_main_pooled'], dtype=float)
    exp = hist.sum() / (d['n_ensemble'] + 1.0)
    rel = float(np.abs(hist - exp).max() / max(exp, 1e-12))
    rmse = [b['rmse'] for b in d['spread_skill']['bins']]
    rows.append((d['tag'], d['mode'], d['n_ensemble'], sv, sw, d['crps_speed_main'], rel, rmse))
print('| tag | mode | N | spread_vec | spread_w | CRPS(风速) | rank 最大相对偏差 | spread-skill RMSE 四分位 |')
print('|---|---|---|---|---|---|---|---|')
for t, m, n, sv, sw, c, r, rmse in rows:
    print('| %s | %s | %d | %.4f | %.4f | %.4f | %.1f%% | %s |' % (
        t, m, n, sv, sw, c, 100.0 * r, ' / '.join('%.4f' % x for x in rmse)))
if not rows:
    print('(无 *_ensemble.json:64d 的 si/edm 臂未评估?)')
print()
print('口径:spread/spread_w = 主指标层逐像素样本标准差(ddof=1)的池化均值;')
print('      CRPS = 样本式 fair 估计于风速(逐层像素 x 小时池化均值);')
print('      rank 最大相对偏差 = 主层池化秩直方图对理想均匀 counts/(N+1) 的最大相对偏差;')
print('      spread-skill = 每小时主层像素按 spread 四分位分组的集合均值 RMSE(0-25/25-50/50-75/75-100%)。')
"

echo ""
echo "===== 3. 产出 ====="
ls -l results/phase3_arch | tail -60
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase3_arch
git commit -m "result 64e arch report: 99 tables + ensemble summary"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
