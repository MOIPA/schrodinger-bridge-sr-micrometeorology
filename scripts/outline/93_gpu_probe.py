# -*- coding: utf-8 -*-
"""GPU 探测:确认 wind3d 环境的 torch 能在当前节点 GPU 上执行(队列兼容性自检)。

用于阶段 1 波次提交前的队列筛查:9654p6000ib 曾报
"CUDA error: no kernel image is available for execution on the device"。
"""
import torch

print("DEV=" + torch.cuda.get_device_name(0))
print("CAP=" + str(torch.cuda.get_device_capability(0)))
x = torch.zeros(3).cuda()
print("SUM=" + str(float(x.sum().item())))
print("PROBE OK")
