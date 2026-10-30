#!/bin/bash
# q_pick.sh — 新集群(2026-10 换代后)GPU 队列选择器
#   逐个候选队列检查三件事,全满足才返回:
#     ① 队列状态 Open
#     ② 队列内有状态 ok 的主机
#     ③ 该主机还有空闲 GPU(ngpus_physical 空闲数 > 0)
#   旧逻辑只看队列级 PEND 数,换代后失效(很多队列 PEND=0 但主机 closed / 无 GPU)。
# 用法:
#   Q=$(bash ops/queue/q_pick.sh) || echo "无可用队列"
#   Q_CAND="7552v100 945090ib" bash ops/queue/q_pick.sh
# 候选顺序 = 偏好顺序(先用熟环境:V100 与新 4090 系;5090 为 Blackwell,暂排除)
CAND="${Q_CAND:-7552v100 734090d 944090d 75434090ib 734090ib 814080 813080}"

for q in $CAND; do
  st=$(bqueues -w "$q" 2>/dev/null | awk 'NR==2{print $2}')
  [ -z "$st" ] && continue
  case "$st" in
    Open*) ;;
    *) continue ;;
  esac
  hosts=$(bqueues -l "$q" 2>/dev/null | sed -n 's/^HOSTS:[[:space:]]*//p' | tr ',' ' ')
  [ -z "$hosts" ] && continue
  for h in $hosts; do
    hs=$(bhosts -w "$h" 2>/dev/null | awk 'NR==2{print $2}')
    [ "$hs" = "ok" ] || continue
    free=$(bhosts -l "$h" 2>/dev/null | sed -n '/ngpus_physical/{n;p;}' | head -1 | awk '{print $2}')
    [ -z "$free" ] && continue
    if awk -v f="$free" 'BEGIN{exit !(f+0>0)}'; then
      echo "$q"
      exit 0
    fi
  done
done
exit 1
