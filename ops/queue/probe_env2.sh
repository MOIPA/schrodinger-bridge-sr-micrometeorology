#!/bin/bash
# probe_env2.sh — 逐步探针:每步带 timeout,输出实时追加到共享盘日志(作业运行中即可读)
# 用法: bsub -q 7552v100 -gpu "num=1" -n 12 -R "rusage[mem=8000]" -o /dev/null \
#         "bash ~/schrodinger-bridge-sr-micrometeorology/ops/queue/probe_env2.sh"
L="$HOME/schrodinger-bridge-sr-micrometeorology/logs/env_probe_${LSB_JOBID}.log"
mkdir -p "$(dirname "$L")"
exec >>"$L" 2>&1
echo "=== start $(date) host=$(hostname) queue=$LSB_QUEUE"

echo "--- 1 nvidia-smi"; timeout 20 nvidia-smi -L; echo "rc=$?"

echo "--- 2 module func"; type module 2>&1 | head -2; echo "rc=$?"

echo "--- 3 module avail cuda"; timeout 25 bash -lc 'module avail cuda 2>&1' | head -12; echo "rc=$?"

echo "--- 4 conda"; which conda 2>&1; echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV rc=$?"

echo "--- 5 torch via login shell"; timeout 90 bash -lc 'source activate wind3d 2>/dev/null; python -c "import torch,sys;print(sys.executable);print(torch.__version__, torch.version.cuda);print(\"avail\",torch.cuda.is_available());print(\"arch\",torch.cuda.get_arch_list())"' 2>&1 | tail -8; echo "rc=$?"

echo "--- 6 matmul"; timeout 90 bash -lc 'source activate wind3d 2>/dev/null; python -c "
import torch
if torch.cuda.is_available():
    x=torch.randn(256,256,device=\"cuda\"); print(\"matmul\", float((x@x).sum())!=0.0)
    print(\"device\", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
else:
    print(\"no cuda\")
"' 2>&1 | tail -5; echo "rc=$?"

echo "=== done $(date)"
