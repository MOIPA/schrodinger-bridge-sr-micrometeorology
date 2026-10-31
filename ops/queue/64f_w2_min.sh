#!/bin/bash
# 64f_w2_min.sh — W2 精简版"一条命令":只跑证据支持的三个臂,训完自动接评估。
#   臂集(默认,可用 ARMS 覆盖):
#     p3_arch_edm        两步法(回归均值 + 扩散残差),检验集合离散度能否校准
#     p3_arch_reg_s2     回归臂第二种子,复核 −12.1% 是否稳
#     p3_arch_unet_small 19.87 M 容量对照,给 SwinIR(19.2 M)一个公平对手
#   流程:①64b 提交训练(WAVE=2 + ARMS 子集);②64d 以 ended(训练作业) 依赖挂评估;
#         ③打印后续两行(64e 可在登录节点本地跑,无需 GPU)。
# 依据:docs/阶段报告/阶段3c_架构对比_终评.md 第 5 节。
# 用法: bash ops/queue/64f_w2_min.sh            # 有卡就跑
#        ARMS="p3_arch_edm" bash ops/queue/64f_w2_min.sh   # 只跑一臂
#        FORCE_Q=<队列> bash ops/queue/64f_w2_min.sh       # 指定队列
cd "${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export REPO="$PWD"
OUT=ops/result/64f_w2_min.txt
mkdir -p ops/result logs
: > "$OUT"

ARMS="${ARMS:-p3_arch_edm p3_arch_reg_s2 p3_arch_unet_small}"
CSV=$(echo "$ARMS" | tr ' ' ',')

say() { echo "$@" | tee -a "$OUT"; }

say "===== 0. W2 精简版;臂: $ARMS ====="
if [ -z "${FORCE_Q:-}" ]; then
  Q=$(bash ops/queue/q_pick.sh 2>/dev/null)
  if [ -z "$Q" ]; then
    say ">>> 当前无可用 GPU 队列(全部 closed/满载),不提交。稍后重跑,或 FORCE_Q=<队列> 强制。"
    exit 2
  fi
  say ">>> 队列: $Q(q_pick 选出)"
  export FORCE_Q="$Q"
else
  say ">>> 队列: $FORCE_Q(强制指定)"
fi

say ""
say "===== 1. 提交训练(64b WAVE=2 + ARMS) ====="
WAVE=2 ARMS="$ARMS" bash ops/queue/64b_arch_waves.sh 2>&1 | tail -40 | tee -a "$OUT"

say ""
if [ -n "${DRY_RUN:-}" ]; then
  say "===== 2. [DRY_RUN] 跳过评估挂载(64d) ====="
else
say "===== 2. 挂评估(64d,--after-training 依赖训练作业 ended) ====="
bash ops/queue/64d_arch_eval.sh --tags "$CSV" --after-training "$CSV" 2>&1 | tail -20 | tee -a "$OUT"
fi

say ""
say "===== 3. 训练+评估完成后(64e 纯 numpy,登录节点本地跑即可) ====="
say "  LOCAL=1 bash ops/queue/64e_arch_report.sh"
say "  LOCAL=1 ROOT=\$PWD REPO=\$PWD PY3D=python3 bash ops/queue/64e_arch_report.sh   # 或在笔记本上"
say ""
say "查状态: bjobs -w | grep p3_arch ; tail -n 20 logs/p3_arch_edm.log"

if [ -z "${DRY_RUN:-}" ]; then
  git add ops/result docs 2>/dev/null
  git commit -qm "result 64f w2-min: submit arms = $ARMS" 2>/dev/null
  git push 2>&1 | tail -1
else
  say "[DRY_RUN] 不提交、不推送"
fi
