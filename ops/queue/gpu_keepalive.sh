#!/bin/bash
# gpu_keepalive.sh — 训训练开始前先"占住"GPU:
#   背景每 60 s 摸一次 cuda,避免作业被"4 分钟不用 GPU 就杀"的策略干掉(见 10-31 18:25 TERM_PREEMPT)。
#   用法: bash ops/queue/gpu_keepalive.sh start | stop
#   由 64b 的 bsub 命令行调用;失败静默(不影响训练)。
MODE="${1:-start}"
PIDF="/tmp/gpu_keepalive_${LSB_JOBID:-$$}.pid"
if [ "$MODE" = "start" ]; then
  ( while true; do
      python -c "import torch;torch.zeros(256,256,device='cuda')" >/dev/null 2>&1 || true
      sleep 60
    done ) &
  echo $! > "$PIDF"
else
  [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null; rm -f "$PIDF"
fi
