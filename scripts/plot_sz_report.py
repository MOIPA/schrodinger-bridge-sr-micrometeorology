# -*- coding: utf-8 -*-
"""
生成组会汇报(深圳跨域实验)的全部配图。

数据来源:
  - results/sz_eval_final/      d04 最终评估(300ep checkpoint)
  - results/sz_eval_d03_final/  d03 跨域评估(最终 checkpoint)
  - results/sz_eval_d03train/   d03 同域评估
  - results/sz_eval/            中途 checkpoint 评估(用于 pinn 171ep vs 300ep 对比)
  - results/sz_train_losses.json 训练 loss 曲线(服务器从日志解析)

用法:
  python scripts/plot_sz_report.py --out_dir 组会汇报/2026-08-28
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Heiti TC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_COLORS = {
    "baseline": "#1f77b4",
    "lrcond": "#ff7f0e",
    "pinn": "#d62728",
    "lrcond_pinn": "#9467bd",
    "day": "#2ca02c",
    "night": "#8c564b",
}
MODEL_LABELS = {
    "baseline": "baseline",
    "lrcond": "allLR",
    "pinn": "phys",
    "lrcond_pinn": "allLR+phys",
    "day": "day",
    "night": "night",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def overall(metrics_path):
    d = load_json(metrics_path)
    return d["results"]["by_component"]["Overall"]


def fig_train_losses(out_dir):
    """训练 loss 曲线:d04 5 模型 + d03 4 模型。"""
    loss_path = os.path.join(ROOT, "results", "sz_train_losses.json")
    if not os.path.exists(loss_path):
        print("[skip] {} 不存在,跳过 loss 曲线".format(loss_path))
        return
    data = load_json(loss_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    groups = [("d04 模拟低精度训练", [m for m in data if not m.startswith("d03")]),
              ("d03 真实低精度训练", [m for m in data if m.startswith("d03")])]
    for ax, (title, models) in zip(axes, groups):
        for m in models:
            label = MODEL_LABELS.get(m, m)
            color = MODEL_COLORS.get(m, "#555555")
            losses = data[m]
            ep = range(1, len(losses) + 1)
            ax.plot(ep, losses, label=label, color=color, linewidth=1.2, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_train_losses.png"), dpi=150)
    plt.close(fig)
    print("saved fig_train_losses.png")


def fig_cross_matrix(out_dir):
    """四格交叉矩阵热图(RMSE)。"""
    matrix = np.array([
        [0.1494, 0.2410],   # baseline: 模拟训练 -> d04/d03 评估
        [0.1827, 0.2674],   # lrcond
        [0.2053, 0.2510],   # pinn
        [np.nan, 0.3473],   # lrcond+pinn(仅 d03 同域)
    ])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["d04 (模拟)", "d03 (真实)"], fontsize=11)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["baseline", "allLR", "phys", "allLR+phys"], fontsize=11)
    ax.set_xlabel("评估数据", fontsize=12)
    ax.set_ylabel("训练数据均为 d04 (模拟低精度)", fontsize=11)
    for i in range(4):
        for j in range(2):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, "{:.4f}".format(v), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if v > 0.22 else "black")
    ax.set_title("d03 评估交叉矩阵 (RMSE, 越小越好)", fontsize=12, fontweight="bold")
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cross_matrix.png"), dpi=150)
    plt.close(fig)
    print("saved fig_cross_matrix.png")


def fig_cross_domain(out_dir):
    """跨域退化:d04 vs d03 RMSE 柱状对比 + 增幅标签。"""
    models = ["baseline", "lrcond", "pinn"]
    d04 = [0.1494, 0.1827, 0.2053]
    d03 = [0.2410, 0.2674, 0.2510]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w / 2, d04, w, label="d04 (域内)", color="#4C9BE8", alpha=0.9)
    b2 = ax.bar(x + w / 2, d03, w, label="d03 (跨域, 真实3km)", color="#E84C4C", alpha=0.9)
    for bars, vals in [(b1, d04), (b2, d03)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    "{:.4f}".format(v), ha="center", fontsize=9)
    for i, (a, b) in enumerate(zip(d04, d03)):
        ax.text(i, max(a, b) + 0.030, "+{:.0f}%".format(100 * (b - a) / a),
                ha="center", fontsize=10, fontweight="bold", color="#8B0000")
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "allLR", "phys"], fontsize=12)
    ax.set_ylabel("RMSE (标准化)", fontsize=11)
    ax.set_title("跨域分布偏移:模拟训练模型在真实 3km 输入上的退化", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cross_domain.png"), dpi=150)
    plt.close(fig)
    print("saved fig_cross_domain.png")


def fig_d03_samedomain(out_dir):
    """d03 同域:4 模型 RMSE + W 分量。"""
    models = ["baseline", "lrcond", "pinn", "lrcond_pinn"]
    rmse = [0.2857, 0.2842, 0.2870, 0.3473]
    w_rmse = [0.4474, 0.4543, 0.4397, 0.5507]
    colors = [MODEL_COLORS[m] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    b1 = ax.bar(x - 0.2, rmse, 0.4, label="Overall RMSE", color=colors, alpha=0.85)
    b2 = ax.bar(x + 0.2, w_rmse, 0.4, label="W 分量 RMSE", color=colors,
                alpha=0.45, edgecolor=colors)
    for bars, vals in [(b1, rmse), (b2, w_rmse)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    "{:.3f}".format(v), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=11)
    ax.set_ylabel("RMSE (标准化)", fontsize=11)
    ax.set_title("d03 真实场景同域训练评估 (d03 test)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_d03_samedomain.png"), dpi=150)
    plt.close(fig)
    print("saved fig_d03_samedomain.png")


def fig_phys_mid_vs_final(out_dir):
    """pinn 中途(171ep) vs 最终(300ep) checkpoint 对比——训练不稳定证据。"""
    labels = ["171ep\n(中途)", "300ep\n(最终)"]
    rmse = [0.1626, 0.2053]
    bias = [0.0075, 0.0162]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, vals, title in [(axes[0], rmse, "Overall RMSE"),
                            (axes[1], bias, "Bias (系统性偏差)")]:
        bars = ax.bar(labels, vals, width=0.5, color=["#4C9BE8", "#E84C4C"], alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    "{:.4f}".format(v), ha="center", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("物理约束训练后期不稳定:最终 checkpoint 明显劣于中途 (pinn, d04)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(out_dir, "fig_phys_mid_vs_final.png"), dpi=150)
    plt.close(fig)
    print("saved fig_phys_mid_vs_final.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "组会汇报", "2026-08-28"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fig_train_losses(args.out_dir)
    fig_cross_matrix(args.out_dir)
    fig_cross_domain(args.out_dir)
    fig_d03_samedomain(args.out_dir)
    fig_phys_mid_vs_final(args.out_dir)


if __name__ == "__main__":
    main()
