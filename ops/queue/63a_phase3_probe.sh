#!/bin/bash
# 63a_phase3_probe.sh — 阶段 3 AGL 探针(GPU 作业;登录节点无 GPU,必须上队列):
#   跑 88_phase3_probe.py(约 10 分钟 GPU;valid split,20 个 batch),
#   产出 results/phase3/probe.json(lambda_joint + channel_weights_72),供 87 生成配置。
# 前置:p2_l1r2_lr2e4 训练已完成(checkpoint 存在);63_phase3_setup.sh 的探针步骤会提示先跑本脚本。
# 手机/登录节点执行: bash ops/queue/63a_phase3_probe.sh   (提交后立即返回,GPU 节点跑 --inner)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"  # 9654p6000ib 为 Blackwell,与 wind3d torch 不兼容

if [ "$1" != "--inner" ]; then
  git pull --no-rebase
  git --no-pager log --oneline -1
  mkdir -p ops/result logs results/phase3
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  # GPU 计算节点 PATH 可能没有 git(实测报 git: command not found,结果回传会静默失败)
  GITDIR=$(dirname "$(command -v git)")
  echo "提交阶段 3 AGL 探针(约 10 分钟)-> 队列 $Q"
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "p3probe" -o "logs/p3probe_%J.out" -e "logs/p3probe_%J.err" \
    "cd $ROOT && export PATH=$GITDIR:\$PATH && bash ops/queue/63a_phase3_probe.sh --inner"
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/63a_phase3_probe.txt
: > "$OUT"
CFG="configs/深圳/phase2/config_wind_canvas_p2_l1r2_lr2e4.yml"
CK="data/DL_result/$EXP/config_wind_canvas_p2_l1r2_lr2e4/checkpoint.pth"
echo "===== 88 AGL 探针(基座 p2_l1r2_lr2e4) =====" >> "$OUT"
for f in "$CFG" "$CK"; do
  if [ -f "$f" ]; then echo "[OK] $f" >> "$OUT"; else echo "[缺] $f" >> "$OUT"; fi
done
$PY3D -u scripts/outline/88_phase3_probe.py \
  --config_path "$CFG" --checkpoint "$CK" --split valid --max_batches 20 \
  --device cuda:0 --out_json results/phase3/probe.json >> "$OUT" 2>&1
echo "88 exit=$?" >> "$OUT"

echo "" >> "$OUT"
echo "===== 产出摘要与下一步 =====" >> "$OUT"
ls -l results/phase3/probe.json >> "$OUT" 2>&1
$PY3D -c "
import json, os
p = 'results/phase3/probe.json'
if os.path.exists(p):
    d = json.load(open(p))
    print('lambda_joint = %.6g' % d.get('lambda_joint', float('nan')))
    print('data_mean = %.6g  agl_mean = %.6g' % (d.get('data_mean', float('nan')), d.get('agl_mean', float('nan'))))
    cw = d.get('channel_weights_72') or []
    print('channel_weights_72: n=%d mean=%.4f' % (len(cw), (sum(cw)/len(cw)) if cw else float('nan')))
    print('tables_match_load_tables = %s' % d.get('meta', {}).get('tables_match_load_tables'))
    print('下一步(登录节点): bash ops/queue/63_phase3_setup.sh  # 读 probe.json 生成配置并自检')
else:
    print('缺 probe.json,看上面 88 的报错')
" >> "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase3
git commit -m "result 63a phase3 AGL probe"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
