#!/bin/bash
# 65_case_snapshots.sh — 导出个例风场快照(供 10-09 组会汇报配图,本地绘图用)
#
# 任务:scripts/outline/case_snapshots.py
#   - 测试集(myj)按近地风速选 top-3 强风个例
#   - 对每个个例推理 p2_l1r2_lr2e4(基线)与 p3_joint(联合监督)两个模型
#   - 存 AGL 100/300 m 的 u/v/w 场:truth / y0 / pred_base / pred_joint 四个来源
#   - 产物 results/report_10_09/cases.npz(约数 MB),回传后本地运行
#     scripts/plot_report_10_09.py 出个例对比图
#
# 预计耗时:读 144 帧数据(几分钟)+ 6 次推理(分钟级)
# 手机执行: bash ops/queue/65_case_snapshots.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/65_case_snapshots.txt
: > "$OUT"

echo "===== 1. 检查前置产物(checkpoint 是否在) =====" >> "$OUT"
for tag in p2_l1r2_lr2e4 p3_joint; do
  CK=data/DL_result/ExperimentSchrodingerBridgeWindCanvas/config_wind_canvas_${tag}/checkpoint.pth
  if [ -f "$CK" ]; then
    echo "$tag: OK ($(stat -c %y "$CK" | cut -d. -f1))" >> "$OUT"
  else
    echo "$tag: 缺 checkpoint,中止" >> "$OUT"
    cat "$OUT"; git add ops/result/65_case_snapshots.txt 2>/dev/null
    git commit -m "result 65_case_snapshots" 2>/dev/null; git push 2>/dev/null; exit 1
  fi
done

echo "" >> "$OUT"
echo "===== 2. 选兼容队列(PEND=0 且 RUN 最少,排除 P6000) =====" >> "$OUT"
pick_queue() {
  local best="" best_run=999999
  for q in 72rtxib 7552v100 7k83 e5v4p100ib 6148v100ib 83a100ib; do
    local PEND RUN
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    RUN=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $10}')
    echo "  $q: PEND=$PEND RUN=$RUN" >> "$OUT"
    if [ "$PEND" = "0" ] 2>/dev/null && [ "$RUN" -lt "$best_run" ] 2>/dev/null; then
      best="$q"; best_run="$RUN"
    fi
  done
  if [ -z "$best" ]; then echo "7552v100"; else echo "$best"; fi
}
QUEUE=$(pick_queue)
echo ">>> 选中队列: $QUEUE" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 提交快照任务 =====" >> "$OUT"
bsub -q "$QUEUE" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J sz_snapshot \
  -o logs/sz_snapshot_%J.out -e logs/sz_snapshot_%J.err \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python scripts/outline/case_snapshots.py --device cuda:0" \
  2>&1 | head -1 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 等待完成(最多两轮,每轮 4 分钟) =====" >> "$OUT"
sleep 240
LATEST=$(ls -t logs/sz_snapshot_*.out 2>/dev/null | head -1)
[ -n "$LATEST" ] && { echo "日志: $LATEST" >> "$OUT"; tail -15 "$LATEST" >> "$OUT"; }
if [ ! -f results/report_10_09/cases.npz ]; then
  echo "(第一轮未见 npz,再等 4 分钟)" >> "$OUT"
  sleep 240
  [ -n "$LATEST" ] && tail -5 "$LATEST" >> "$OUT"
fi
if [ -f results/report_10_09/cases.npz ]; then
  echo ">>> cases.npz 已生成: $(du -h results/report_10_09/cases.npz | cut -f1)" >> "$OUT"
else
  echo ">>> 警告:尚未见 cases.npz,检查日志" >> "$OUT"
fi

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传(主仓库内提交 ops/result 与 cases.npz;ops 非独立 git 仓库)
git add ops/result/65_case_snapshots.txt results/report_10_09 2>/dev/null
git commit -m "result 65_case_snapshots" 2>/dev/null
git push 2>/dev/null
echo done
