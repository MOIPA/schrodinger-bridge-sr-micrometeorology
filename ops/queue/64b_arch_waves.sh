#!/bin/bash
# 64b_arch_waves.sh — 阶段 3 架构对比训练波次提交(波次用环境变量 WAVE 选择,默认 1)
#   W1 = p3_arch_swin(SwinIR 训练) p3_arch_reg(回归步训练)
#   W2 = p3_arch_swin_s2 p3_arch_reg_s2 p3_arch_unet_s2(种子对照/重跑)
#        p3_arch_unet_small p3_arch_unet_small_s2(容量对齐臂 ≈20M,与 SwinIR inner=536 同量级)
#        p3_arch_edm p3_arch_edm_s2(两步法;需配对 reg 的 checkpoint 先存在)
# 闸门:W2 需 W1 两臂的前波评估产物(64d 产生):
#   results/phase3_arch/p3_arch_swin_summary.json 且 p3_arch_reg_summary.json;
# 不自动创建闸门文件,缺产物直接拒绝提交并打印补救命令。
# 按 tag 分派训练脚本:
#   p3_arch_reg* -> scripts/train_regression_model.py
#   p3_arch_edm* -> scripts/train_edm_correction.py(+ --reg_tag/--reg_config_path)
#   其余(swin/unet) -> scripts/train_schrodinger_bridge_model.py
# 规则同 63b:提交后 90 秒检查,凡 PEND 立即 bkill 并换队列重投;作业名 = tag,
# 内部日志 logs/${tag}.log,LSF 日志 logs/${tag}_%J.out;GPU 计算节点 PATH 可能
# 没有 git,提交时注入登录节点 git 目录;训练中断以日志 "Train end:" 为准,
# 有 checkpoint 无标记则重投续训。
# 前置依赖:configs/深圳/phase3_arch/*.yml(由 64_arch_setup.sh 的 83 生成)。
# 手机执行: bash ops/queue/64b_arch_waves.sh          # W1
#           WAVE=2 bash ops/queue/64b_arch_waves.sh    # W2(需前波评估产物)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
git --no-pager log --oneline -1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/64b_arch_waves.txt
mkdir -p ops/result logs results/phase3_arch
: > "$OUT"

EXP=ExperimentSchrodingerBridgeWindCanvas
RESULT_BASE=data/DL_result/$EXP
CFG_DIR="configs/深圳/phase3_arch"
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"  # 9654p6000ib 为 Blackwell,与 wind3d torch 不兼容
QUEUES="${FORCE_Q:-$QUEUES}"  # 队列异常时覆盖:FORCE_Q=6148v100ib WAVE=2 bash ops/queue/64b_arch_waves.sh

W1="p3_arch_swin p3_arch_reg"
W2="p3_arch_swin_s2 p3_arch_reg_s2 p3_arch_unet_s2 p3_arch_unet_small p3_arch_unet_small_s2 p3_arch_edm p3_arch_edm_s2"

WAVE="${WAVE:-${1:-1}}"
case "$WAVE" in
  1) TAGS="$W1";;
  2) TAGS="$W2";;
  *) echo "用法: WAVE=1|2 bash ops/queue/64b_arch_waves.sh(默认 WAVE=1)"; exit 1;;
esac

cfg_of() { echo "$CFG_DIR/config_wind_canvas_${1}.yml"; }
ck_of()  { echo "$RESULT_BASE/config_wind_canvas_${1}/checkpoint.pth"; }

# edm tag -> 配对的 reg tag(p3_arch_edm -> p3_arch_reg;p3_arch_edm_s2 -> p3_arch_reg_s2)
reg_of() {
  case "$1" in
    p3_arch_edm_s2) echo p3_arch_reg_s2;;
    p3_arch_edm)    echo p3_arch_reg;;
    *) echo "$1";;
  esac
}

pick_queue() {
  # 只在"没有排队积压(PEND=0)"的队列里挑,并选 RUN 最少的(近似空闲 GPU 最多)
  local BEST="" BESTRUN=1000000 P R
  for q in $QUEUES; do
    read -r P R <<< "$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9, $10}')"
    [ "$P" = "0" ] || continue
    [ -z "$R" ] && continue
    if [ "$R" -lt "$BESTRUN" ] 2>/dev/null; then BEST="$q"; BESTRUN="$R"; fi
  done
  echo "${BEST:-83a100ib}"
}

submit_one() {  # $1=tag $2=queue
  local tag=$1 q=$2 script extra=""
  case "$tag" in
    p3_arch_reg*) script=scripts/train_regression_model.py;;
    p3_arch_edm*)
      script=scripts/train_edm_correction.py
      local rt; rt=$(reg_of "$tag")
      extra="--reg_tag $rt --reg_config_path $CFG_DIR/config_wind_canvas_${rt}.yml";;
    *) script=scripts/train_schrodinger_bridge_model.py;;
  esac
  bsub -q "$q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "$tag" -o "logs/${tag}_%J.out" -e "logs/${tag}_%J.err" \
    "cd $ROOT && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && \
python -u $script \
--config_path $CFG_DIR/config_wind_canvas_${tag}.yml \
--experiment_name $EXP --device cuda:0 $extra > logs/${tag}.log 2>&1" \
    2>&1 | head -1
}

echo "===== 0. WAVE=$WAVE;队列里的 p3_arch 作业 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep "p3_arch" >> "$OUT" || echo "(无)" >> "$OUT"

# 闸门:W2 需 W1 两臂的前波评估产物(64d 产生;不自动创建)
if [ "$WAVE" = "2" ]; then
  GATE_FAIL=""
  for t in $W1; do
    [ -f "results/phase3_arch/${t}_summary.json" ] \
      || GATE_FAIL="$GATE_FAIL results/phase3_arch/${t}_summary.json"
  done
  if [ -n "$GATE_FAIL" ]; then
    {
      echo ""
      echo "===== 闸门:缺前波评估产物,拒绝提交 W$WAVE ====="
      echo "缺失:$GATE_FAIL"
      echo "先训练完 W1 两臂,再评估(si 臂 N=16 集合均值):"
      echo "  bash ops/queue/64c_check_arch.sh     # 确认训练完(日志见 Train end:)"
      echo "  bash ops/queue/64d_arch_eval.sh      # 84 评估 -> results/phase3_arch/*_summary.json"
      echo "产物齐了再: WAVE=2 bash ops/queue/64b_arch_waves.sh"
    } >> "$OUT"
    cat "$OUT"
    git add ops/result
    git commit -m "result 64b arch waves: W$WAVE blocked by gate"
    git push || { git fetch origin && git merge --no-edit origin/main && git push; }
    exit 0
  fi
fi

echo "" >> "$OUT"
echo "===== 1. W$WAVE 提交清单: $TAGS =====" >> "$OUT"
SUBMITTED=""
for tag in $TAGS; do
  CK=$(ck_of "$tag")
  CFG=$(cfg_of "$tag")
  if [ -f "$CK" ]; then
    # checkpoint 存在 ≠ 已完成(训练中断也会留下 best checkpoint);
    # 以日志里的 "Train end:" 作为正常结束标记,否则重投续训(checkpoint 自动 resume)
    if grep -q "Train end:" "logs/${tag}.log" 2>/dev/null; then
      echo "[跳过] $tag 已完成(checkpoint + Train end 标记)" >> "$OUT"
      continue
    fi
    echo "[续训] $tag 有 checkpoint 但未见 Train end(疑似中断),重投续训" >> "$OUT"
  fi
  if [ ! -f "$CFG" ]; then
    echo "[警告] 缺配置 $CFG(先跑 64_arch_setup.sh 生成),跳过 $tag" >> "$OUT"
    continue
  fi
  case "$tag" in
    p3_arch_edm*)
      RT=$(reg_of "$tag")
      if [ ! -f "$(ck_of "$RT")" ]; then
        echo "[等] $tag 需配对回归 $RT 的 checkpoint(尚不存在),跳过;" \
             "先跑 W1/64b 把 $RT 训完(或 --after-training 依赖)" >> "$OUT"
        continue
      fi
      ;;
  esac
  if bjobs -o "job_name stat" -noheader 2>/dev/null | awk '$2=="RUN"{print $1}' | grep -qx "$tag"; then
    echo "[跳过] $tag 正在 RUN" >> "$OUT"
    continue
  fi
  for JID in $(bjobs -o "jobid job_name stat" -noheader 2>/dev/null \
      | awk -v n="$tag" '$2==n && $3=="PEND"{print $1}'); do
    echo ">>> 清理 PEND 作业 $JID ($tag),换队列重投" >> "$OUT"
    bkill "$JID" >> "$OUT" 2>&1
  done
  Q=$(pick_queue)
  echo ">>> $tag -> 队列 $Q($(basename "$(cfg_of $tag)"))" >> "$OUT"
  submit_one "$tag" "$Q" >> "$OUT"
  SUBMITTED="$SUBMITTED $tag"
done

if [ -z "$SUBMITTED" ]; then
  echo "" >> "$OUT"
  echo "没有可补投的配置(要么在跑,要么已完成,要么缺配置/等配对 checkpoint)。" >> "$OUT"
  cat "$OUT"
  git add ops/result
  git commit -m "result 64b arch waves W$WAVE: nothing to submit"
  git push || { git fetch origin && git merge --no-edit origin/main && git push; }
  exit 0
fi

echo "" >> "$OUT"
echo "===== 2. 90 秒后 PEND 检查(PEND 立即换队列重投) =====" >> "$OUT"
sleep 90
bjobs -w 2>/dev/null | grep "p3_arch" >> "$OUT" || echo "(bjobs 无 p3_arch 记录?)" >> "$OUT"
for tag in $SUBMITTED; do
  LINE=$(bjobs -w 2>/dev/null | grep " $tag ")
  STAT=$(echo "$LINE" | awk '{print $3}')
  if [ "$STAT" = "PEND" ]; then
    JID=$(echo "$LINE" | awk '{print $1}')
    echo ">>> $tag($JID) PEND,换队列重投" >> "$OUT"
    bkill "$JID" >> "$OUT" 2>&1
    Q=$(pick_queue)
    echo ">>> $tag -> 新队列 $Q" >> "$OUT"
    submit_one "$tag" "$Q" >> "$OUT"
  fi
done

echo "" >> "$OUT"
echo "===== 3. 后续 =====" >> "$OUT"
echo "查状态: bash ops/queue/64c_check_arch.sh" >> "$OUT"
if [ "$WAVE" = "1" ]; then
  echo "W1 两臂训练完 -> 评估: bash ops/queue/64d_arch_eval.sh" >> "$OUT"
  echo "评估产物存在后 W2: WAVE=2 bash ops/queue/64b_arch_waves.sh" >> "$OUT"
else
  echo "全臂训练完 -> 评估: bash ops/queue/64d_arch_eval.sh" >> "$OUT"
  echo "评估完 -> 终评报告: bash ops/queue/64e_arch_report.sh" >> "$OUT"
fi
cat "$OUT"
git add ops/result
git commit -m "result 64b arch waves W$WAVE"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
