#!/bin/bash
# 63e_phase3_report.sh — 阶段 3 T3.5 终评报告(一个 GPU 作业内串行,复刻 60d/60e 模式):
#   ① 99_phase3_tables.py:主指标(AGL 10–500 m 矢量 RMSE)+ 切变误差 + 配对 bootstrap;
#      调用 --perhour_dir results/phase3 --tags <全部 tag> --base_tag p2_l1r2_lr2e4,
#      失败则退回无参调用(默认同参)并把两次结果都记进日志;
#   ② 98_phase3_spatial.py:逐像素 MAE 图 + 陡/平 x 城/郊 池化 RMSE
#      (tags = p2_l1r2_lr2e4 p3_agl p3_joint p3_agllw,缺 checkpoint 的会自动跳过);
#   ③ 全部 stdout/表格写 ops/result/63_phase3_report.txt,回传 ops/result + results/phase3。
# 前置依赖:63d 已评估过(否则 98 会跳过缺 checkpoint/配置的 tag,99 缺输入);
#   98/99 脚本来自本阶段任务(98 由评估分析任务产出、99 由表格任务产出)。
# 手机执行: bash ops/queue/63e_phase3_report.sh   (登录节点提交,GPU 节点跑 --inner)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"

if [ "$1" != "--inner" ]; then
  git pull --no-rebase
  git --no-pager log --oneline -1
  mkdir -p ops/result logs results/phase3
  QUEUES="${FORCE_Q:-$QUEUES}"
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  # GPU 计算节点 PATH 可能没有 git,提交时注入登录节点 git 目录
  GITDIR=$(dirname "$(command -v git)")
  echo "阶段 3 终评报告(99 表 + 98 空间)-> 队列 $Q"
  SUB=$(bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "p3report" -o "logs/p3report_%J.out" -e "logs/p3report_%J.err" \
    "cd $ROOT && export PATH=$GITDIR:\$PATH && bash ops/queue/63e_phase3_report.sh --inner" 2>&1)
  echo "$SUB"
  JID=$(echo "$SUB" | grep -oE '[0-9]+' | head -1)
  sleep 90
  ST=$(bjobs -o "jobid stat" -noheader 2>/dev/null | awk -v j="$JID" '$1==j{print $2}')
  if [ "$ST" = "PEND" ]; then
    echo ">>> p3report($JID) PEND: bkill;换队列重跑本脚本(必要时 FORCE_Q=<队列>)"
    bkill "$JID"
  else
    echo ">>> p3report($JID) 状态: ${ST:-已不在队列(可能已开始或极快失败,查 logs/)}"
  fi
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/63_phase3_report.txt
: > "$OUT"
mkdir -p results/phase3

REPORT_TAGS="p2_l1r2_lr2e4 p3_agl p3_joint p3_agllw"
TAGS_CSV=$(echo "$REPORT_TAGS" | tr ' ' ',')

{
echo "===== 阶段 3 T3.5 终评报告 ====="
echo "tags: $TAGS_CSV;参考臂主指标(phase2)= 0.9574"

echo ""
echo "===== 1. 99 阶段 3 汇总表(主指标 + 切变 + 配对 bootstrap) ====="
if [ -f scripts/outline/99_phase3_tables.py ]; then
  $PY3D -u scripts/outline/99_phase3_tables.py --perhour_dir results/phase3 \
    --tags "$TAGS_CSV" --base_tag p2_l1r2_lr2e4 \
    || { echo "99 带参调用失败(以 99 的 argparse 为准),退回无参调用(默认同参)"; \
         $PY3D -u scripts/outline/99_phase3_tables.py || echo "FAIL 99"; }
  for f in $(ls -1 results/phase3/*.md 2>/dev/null | sort -u); do
    echo "--- $f ---"
    cat "$f"
  done
else
  echo "FAIL: 缺 scripts/outline/99_phase3_tables.py(表格任务产物,先合入再跑)"
fi

echo ""
echo "===== 2. 98 空间误差分析(逐像素 MAE 图 + 陡/平 x 城/郊 池化 RMSE) ====="
if [ -f scripts/outline/98_phase3_spatial.py ]; then
  $PY3D -u scripts/outline/98_phase3_spatial.py --tags $REPORT_TAGS \
    --split test --out_dir results/phase3 --device cuda:0 || echo "FAIL 98"
else
  echo "FAIL: 缺 scripts/outline/98_phase3_spatial.py"
fi

echo ""
echo "===== 3. 产出 ====="
ls -l results/phase3 | tail -40
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase3
git commit -m "result 63e phase3 report: tables + spatial"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
