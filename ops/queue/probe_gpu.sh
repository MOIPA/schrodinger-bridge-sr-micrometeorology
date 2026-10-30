#!/bin/bash
# probe_gpu.sh — 集群换代后的一次性探针:确认 GPU 型号 / CUDA 环境 / torch 兼容性
# 用法(登录节点): bsub -q 75434090ib -gpu "num=1" -n 2 -R "rusage[mem=8000]" \
#                        -o logs/probe_gpu_%J.out -e logs/probe_gpu_%J.err \
#                        "bash ~/schrodinger-bridge-sr-micrometeorology/ops/queue/probe_gpu.sh"
echo "host=$(hostname)"
echo "LSB_QUEUE=$LSB_QUEUE  LSB_HOSTS=$LSB_HOSTS"
command -v nvidia-smi >/dev/null && nvidia-smi -L
echo "--- module avail cuda ---"
module avail cuda 2>&1 | head -20
echo "--- try cuda/11.8.0 ---"
module load anaconda/3 2>&1 | head -3
module load cuda/11.8.0 2>&1 | head -3
echo "NVCC=$(command -v nvcc)"
echo "--- python/torch ---"
source activate wind3d 2>&1 | head -3
python - <<'PY'
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "built_cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
        print("arch_list", torch.cuda.get_arch_list())
        x = torch.randn(512, 512, device="cuda")
        print("matmul_ok", float((x @ x).sum()) != 0.0)
except Exception as e:
    import traceback; traceback.print_exc()
try:
    import numpy, scipy
    print("numpy", numpy.__version__, "scipy", scipy.__version__)
except Exception as e:
    print("numpy/scipy err", e)
PY
