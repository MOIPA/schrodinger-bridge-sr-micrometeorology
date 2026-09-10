# -*- coding: utf-8 -*-
"""
A1 功率谱图:真值 / 模型 / 插值基线的径向平均波数谱(log-log),分 U/V/W。

数据:results/A_group/*_diag.json 的 spectra 字段(服务器评估产出)。
用法:
  python scripts/plot_A1_spectra.py --json results/A_group/A_group_baseline_diag.json \
      --out 组会汇报/2026-09-11/fig_A1_spectra.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Heiti TC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COMP_LABELS = {"U": "U 东西风", "V": "V 南北风", "W": "W 垂直风"}
COLORS = {"truth": "#222222", "model": "#1f77b4", "interp": "#d62728"}
STYLES = {"truth": "-", "model": "-", "interp": "--"}
LINES = {"truth": "1km 真值", "model": "模型预测", "interp": "插值基线"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="A_group_*_diag.json 路径")
    parser.add_argument("--out", default="fig_A1_spectra.png")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    spec = data["spectra"]
    tag = data.get("tag", "model")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, comp in zip(axes, ["U", "V", "W"]):
        s = spec[comp]
        k = np.array(s["k"])
        k = k[k > 0]
        for key in ["truth", "model", "interp"]:
            v = np.array(s[key])[: len(k)]
            ax.loglog(k, 10 ** v, color=COLORS[key], linestyle=STYLES[key],
                      linewidth=1.5, label=LINES[key])
        ax.set_title("{} (tag={})".format(COMP_LABELS[comp], tag),
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("波数 (1/网格)", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("功率谱密度 (log)", fontsize=10)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("径向平均功率谱:模型是否还原小尺度结构", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("saved {}".format(args.out))


if __name__ == "__main__":
    main()
