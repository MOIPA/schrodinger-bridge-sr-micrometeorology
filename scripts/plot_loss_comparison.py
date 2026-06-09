#!/usr/bin/env python3
"""
训练 Loss 曲线对比：PINN 大模型 vs NS 模型
从两个训练日志中提取 loss，绘制在同一张图上。
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# ---- 文件路径 ----
pinn_log = "logs-big-pinn-model-1000-train/final_train_7574348.out"
ns_log   = "logs-ns-model-1000-train/final_train_13104236.out"

# ---- 解析 ----
def parse_losses(path):
    epochs, losses = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r"train error: avg loss = ([\d.]+)", line)
            if m:
                epochs.append(len(epochs))  # 0, 1, 2, ...
                losses.append(float(m.group(1)))
    return np.array(epochs), np.array(losses)

ep_pinn, lo_pinn = parse_losses(pinn_log)
ep_ns,   lo_ns   = parse_losses(ns_log)

print(f"PINN 模型: {len(lo_pinn)} epochs, loss {lo_pinn[0]:.4f} → {lo_pinn[-1]:.4f}")
print(f"NS   模型: {len(lo_ns)} epochs, loss {lo_ns[0]:.4f} → {lo_ns[-1]:.4f}")

# ---- 绘图 ----
plt.figure(figsize=(10, 5))

# Linear scale
plt.subplot(1, 2, 1)
plt.plot(ep_pinn, lo_pinn, linewidth=0.6, alpha=0.8, label=f'PINN Big (final={lo_pinn[-1]:.4f})')
plt.plot(ep_ns,   lo_ns,   linewidth=0.6, alpha=0.8, label=f'NS Kinematic (final={lo_ns[-1]:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (linear scale)')
plt.legend()
plt.grid(True, alpha=0.3)

# Log scale
plt.subplot(1, 2, 2)
plt.plot(ep_pinn, lo_pinn, linewidth=0.6, alpha=0.8, label='PINN Big')
plt.plot(ep_ns,   lo_ns,   linewidth=0.6, alpha=0.8, label='NS Kinematic')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/loss_curve_pinn_vs_ns.png", dpi=150, bbox_inches='tight')
plt.savefig("results/loss_curve_pinn_vs_ns.pdf", bbox_inches='tight')
print(f"\nSaved:")
print(f"  results/loss_curve_pinn_vs_ns.png")
print(f"  results/loss_curve_pinn_vs_ns.pdf")

# ---- 终端输出关键数值 ----
print(f"\n=== Loss Comparison ===")
print(f"{'Epoch':<10} {'PINN':>10} {'NS':>10} {'Δ(NS-PINN)':>12}")
for ep in [1, 10, 50, 100, 200, 400, 600, 800, 1000]:
    if ep < len(lo_pinn) and ep < len(lo_ns):
        d = lo_ns[ep] - lo_pinn[ep]
        print(f"{ep:<10} {lo_pinn[ep]:>10.4f} {lo_ns[ep]:>10.4f} {d:>+12.4f}")
