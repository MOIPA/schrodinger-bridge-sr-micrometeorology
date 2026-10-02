#!/bin/bash
# 60a_phase2_probe.sh — 阶段 2 权重探针(V* 基线 checkpoint + 任一 p2 配置的 phys_* 参数):
#   跑 96_weight_probe.py(约 10 分钟 GPU;valid split,20 个 batch),产出
#   results/phase2/probe_weights.json(suggested_weights),并打印可直接执行的 86 回填命令。
# 前置:60_phase2_setup.sh 已生成 configs/深圳/phase2(config_wind_canvas_p2_div_mid.yml)。
# 手机执行: bash ops/queue/60a_phase2_probe.sh   (登录节点提交,GPU 节点跑 --inner)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
EXP=ExperimentSchrodingerBridgeWindCanvas
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"  # 9654p6000ib 为 Blackwell,与 wind3d torch 不兼容

if [ "$1" != "--inner" ]; then
  git pull --no-rebase
  git --no-pager log --oneline -1
  mkdir -p ops/result logs results/phase2
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  echo "提交阶段 2 权重探针(约 10 分钟)-> 队列 $Q"
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "p2probe" -o "logs/p2probe_%J.out" -e "logs/p2probe_%J.err" \
    "cd $ROOT && bash ops/queue/60a_phase2_probe.sh --inner"
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/60a_phase2_probe.txt
: > "$OUT"
CFG="configs/深圳/phase2/config_wind_canvas_p2_div_mid.yml"
VSTAR="data/DL_result/$EXP/config_wind_canvas_p1r_t14_noenc_cos/checkpoint.pth"
echo "===== 96 权重探针(V* 基线 r_t14_noenc_cos) =====" >> "$OUT"
for f in "$CFG" "$VSTAR"; do
  if [ -f "$f" ]; then echo "[OK] $f" >> "$OUT"; else echo "[缺] $f" >> "$OUT"; fi
done
$PY3D -u scripts/outline/96_weight_probe.py \
  --config_path "$CFG" --checkpoint "$VSTAR" --split valid --max_batches 20 \
  --device cuda:0 --out_json results/phase2/probe_weights.json >> "$OUT" 2>&1
echo "96 exit=$?" >> "$OUT"

echo "" >> "$OUT"
echo "===== 产出与 86 回填命令 =====" >> "$OUT"
ls -l results/phase2/probe_weights.json >> "$OUT" 2>&1
$PY3D -c "
import json, os
p = 'results/phase2/probe_weights.json'
if os.path.exists(p):
    w = json.load(open(p)).get('suggested_weights', {})
    ks = [('div_lo', 'div-lo'), ('div_mid', 'div-mid'), ('div_hi', 'div-hi'),
          ('spectral', 'spectral'), ('extreme', 'extreme'), ('vorticity', 'vorticity')]
    parts = []
    for k, flag in ks:
        v = w.get(k)
        if v is not None:
            parts.append('--%s %.6g' % (flag, v))
    print('86 回填命令(服务器仓库根目录执行):')
    print('  python scripts/outline/86_gen_phase2_configs.py ' + ' '.join(parts))
    print('然后重跑 81(--config_dir \"configs/深圳/phase2\")与 95 复核')
else:
    print('缺 probe_weights.json,看上面 96 的报错')
" >> "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase2
git commit -m "result 60a phase2 weight probe"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
