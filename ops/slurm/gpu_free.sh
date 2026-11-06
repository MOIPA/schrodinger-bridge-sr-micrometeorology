#!/usr/bin/env bash
# 新服务器只有 1 块 GPU,提交前先看它是否空着(以及是不是我们自己占着)。
# 退出码:0 = 可以用;1 = 别人在用(等);2 = 我们自己有作业在跑(别重复投)。
set -uo pipefail

mine=$(squeue -u "$USER" -h -o "%i %j %b" 2>/dev/null | grep -c "gpu" || true)
if [ "${mine:-0}" -gt 0 ]; then
    echo "ours"; squeue -u "$USER" -h -o "  %i %j %T %b" 2>/dev/null; exit 2
fi

others=$(squeue -h -o "%u %b" 2>/dev/null | awk -v me="$USER" '$1 != me && $0 ~ /gpu/ {n++} END{print n+0}')
if [ "$others" -gt 0 ]; then
    echo "busy"; squeue -h -o "  %i %u %T %b %M" 2>/dev/null | awk '$0 ~ /gpu/'; exit 1
fi

echo "free"; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null; exit 0
