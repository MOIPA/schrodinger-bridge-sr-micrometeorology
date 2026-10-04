#!/bin/bash
# 63b2_requeue_v100.sh — 把正在跑/排队的 p3_ 训练作业换到指定队列(默认:p3_agl->6148v100ib,
#   p3_joint->7552v100,用满两个空闲 V100 队列并避免同节点争用)。
# 逻辑:bkill 同名作业 -> 以新队列重投(有 checkpoint 时训练脚本自动 resume,不丢进度,
#   只损失当前未落盘的轮内进度)-> 90 秒后复查队列状态。
# 用法(登录节点):
#   bash ops/queue/63b2_requeue_v100.sh                         # 默认表
#   bash ops/queue/63b2_requeue_v100.sh p3_agl 6148v100ib       # 自定义 tag/队列对
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
EXP=ExperimentSchrodingerBridgeWindCanvas
OUT=ops/result/63b2_requeue_v100.txt
mkdir -p ops/result logs
: > "$OUT"

args=("$@")
if [ ${#args[@]} -eq 0 ]; then
  args=(p3_agl 6148v100ib p3_joint 7552v100)
fi

submit_one() {  # $1=tag $2=queue
  local tag=$1 q=$2
  bsub -q "$q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "$tag" -o "logs/${tag}_%J.out" -e "logs/${tag}_%J.err" \
    "cd $ROOT && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
python -u scripts/train_schrodinger_bridge_model.py \
--config_path configs/深圳/phase3/config_wind_canvas_${tag}.yml \
--experiment_name $EXP --device cuda:0 > logs/${tag}.log 2>&1" 2>&1 | head -1
}

echo "===== 换队列(先杀后投,checkpoint 自动 resume)=====" >> "$OUT"
while [ ${#args[@]} -ge 2 ]; do
  tag="${args[0]}"; q="${args[1]}"; args=("${args[@]:2}")
  for JID in $(bjobs -o "jobid job_name" -noheader 2>/dev/null | awk -v n="$tag" '$2==n{print $1}'); do
    echo ">>> bkill $JID ($tag)" >> "$OUT"
    bkill "$JID" >> "$OUT" 2>&1
  done
  sleep 5
  echo ">>> $tag -> 队列 $q" >> "$OUT"
  submit_one "$tag" "$q" >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 90 秒后复查 =====" >> "$OUT"
sleep 90
bjobs -w 2>/dev/null | grep "p3_" >> "$OUT" || echo "(bjobs 无 p3_ 记录?)" >> "$OUT"

echo "" >> "$OUT"
echo "后续: tail -f logs/p3_agl.log logs/p3_joint.log 看轮速(resume 后应连续推进)" >> "$OUT"
cat "$OUT"
git add ops/result
git commit -m "result 63b2 requeue v100"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
