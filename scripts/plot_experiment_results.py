#!/usr/bin/env python3
"""
Experiment Results Visualization
================================
读取各实验 metrics.json，生成对比图。以 Original 模型为 baseline。

用法:
  python scripts/plot_experiment_results.py --results_dir results/
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 配色方案 ──
COLORS = {
    "Original":         "#999999",
    "W-weighted":       "#F58518",
    "Ablation:All":     "#54A24B",
    "Abl:NoTerrain":    "#E45756",
    "Abl:NoThermal":    "#72B7B2",
    "Abl:NoPBLH":       "#B279A2",
    "Abl:NoPressure":   "#FF9DA6",
}

LABELS_CN = {
    "Original":         "原始模型(Baseline)",
    "W-weighted":       "W加权模型",
    "Ablation:All":     "消融:全变量",
    "Abl:NoTerrain":    "消融:无地形",
    "Abl:NoThermal":    "消融:无热力",
    "Abl:NoPBLH":       "消融:无PBLH",
    "Abl:NoPressure":   "消融:无气压",
}

WIND_COMPONENTS = ["U", "V", "W"]
ALTITUDE_LEVELS = ["ml0", "ml1", "ml2", "ml3", "ml5", "ml10"]

EXP_DIR_TO_LABEL = OrderedDict([
    ("eval_original",              "Original"),
    ("eval_wloss",                 "W-weighted"),
    ("eval_ablation_all",          "Ablation:All"),
    ("eval_ablation_no_terrain",   "Abl:NoTerrain"),
    ("eval_ablation_no_thermal",   "Abl:NoThermal"),
    ("eval_ablation_no_pblh",      "Abl:NoPBLH"),
    ("eval_ablation_no_pressure",  "Abl:NoPressure"),
])


def setup_chinese_font():
    try:
        plt.rcParams["font.sans-serif"] = [
            "PingFang HK", "Heiti TC", "STHeiti", "Hiragino Sans GB",
            "SimHei", "WenQuanYi Micro Hei", "Microsoft YaHei",
            "Arial Unicode MS", "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def load_all_metrics(results_dir: str) -> dict:
    base = Path(results_dir)
    experiments = OrderedDict()
    for dirname, label in EXP_DIR_TO_LABEL.items():
        json_path = base / dirname / "metrics.json"
        if json_path.exists():
            with open(json_path) as f:
                experiments[label] = json.load(f)
    return experiments


def get_var_names(experiments: dict) -> list:
    for data in experiments.values():
        return list(data["model"].keys())
    return []


# ==============================================================
# Fig 1: Overall RMSE/SSIM（无 bicubic）
# ==============================================================
def plot_overall_comparison(experiments, var_names, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = []
    rmse_vals = []
    ssim_vals = []
    colors = []

    for label, data in experiments.items():
        model = data["model"]
        labels.append(LABELS_CN.get(label, label))
        rmse_vals.append(np.mean([model[v]["rmse"] for v in var_names]))
        ssim_vals.append(np.mean([model[v]["ssim"] for v in var_names]))
        colors.append(COLORS.get(label, "#333333"))

    x = np.arange(len(labels))

    # RMSE — 越低越好，标注改进百分比
    ax = axes[0]
    bars = ax.bar(x, rmse_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("RMSE (m/s)", fontsize=12)
    ax.set_title("各实验 RMSE 对比（全通道平均）", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    baseline_rmse = rmse_vals[0]  # Original
    for i, (bar, val) in enumerate(zip(bars, rmse_vals)):
        txt = f"{val:.3f}"
        if i > 0:
            pct = (baseline_rmse - val) / baseline_rmse * 100
            txt += f"\n(↓{pct:.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                txt, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(rmse_vals) * 1.25)

    # SSIM — 越高越好
    ax = axes[1]
    bars = ax.bar(x, ssim_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("SSIM", fontsize=12)
    ax.set_title("各实验 SSIM 对比（全通道平均）", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    baseline_ssim = ssim_vals[0]
    for i, (bar, val) in enumerate(zip(bars, ssim_vals)):
        txt = f"{val:.3f}"
        if i > 0:
            pct = (val - baseline_ssim) / baseline_ssim * 100
            txt += f"\n(↑{pct:.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                txt, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(ssim_vals) * 1.25)

    plt.tight_layout()
    plt.savefig(save_dir / "fig1_overall_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig1_overall_comparison.png")


# ==============================================================
# Fig 2 & 3: Component-wise（无 bicubic）
# ==============================================================
def plot_component_bars(experiments, var_names, save_dir, metric="rmse"):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    exp_labels = list(experiments.keys())
    n_exp = len(exp_labels)
    bar_width = 0.8 / n_exp
    x = np.arange(len(WIND_COMPONENTS))

    for i, label in enumerate(exp_labels):
        model = experiments[label]["model"]
        vals = []
        for comp in ["u", "v", "w"]:
            keys = [v for v in var_names if f"_{comp}_" in v]
            vals.append(np.mean([model[k][metric] for k in keys]))
        offset = -0.4 + bar_width * (i + 0.5)
        bars = ax.bar(x + offset, vals, bar_width,
                      label=LABELS_CN.get(label, label),
                      color=COLORS.get(label, "#333"), edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(WIND_COMPONENTS, fontsize=14, fontweight="bold")
    ylabel = "RMSE (m/s)" if metric == "rmse" else "SSIM"
    title_cn = "各实验 U/V/W 分量 RMSE 对比" if metric == "rmse" else "各实验 U/V/W 分量 SSIM 对比"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title_cn, fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    plt.tight_layout()
    fig_name = f"fig2_component_{metric}.png" if metric == "rmse" else f"fig3_component_{metric}.png"
    plt.savefig(save_dir / fig_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_name}")


# ==============================================================
# Fig 4: Ablation Delta（不变，本来就没 bicubic）
# ==============================================================
def plot_ablation_delta(experiments, var_names, save_dir):
    if "Ablation:All" not in experiments:
        print("  [Skip] fig4 — no Ablation:All baseline found")
        return

    baseline = experiments["Ablation:All"]["model"]
    ablation_exps = OrderedDict()
    for label in ["Abl:NoTerrain", "Abl:NoThermal", "Abl:NoPBLH", "Abl:NoPressure"]:
        if label in experiments:
            ablation_exps[label] = experiments[label]["model"]

    if not ablation_exps:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    components = ["u", "v", "w", "all"]
    comp_labels = ["U", "V", "W", "Overall"]
    n_abl = len(ablation_exps)
    bar_width = 0.8 / n_abl
    x = np.arange(len(components))

    for i, (label, model) in enumerate(ablation_exps.items()):
        deltas = []
        for comp in ["u", "v", "w"]:
            keys = [v for v in var_names if f"_{comp}_" in v]
            bl_val = np.mean([baseline[k]["rmse"] for k in keys])
            m_val = np.mean([model[k]["rmse"] for k in keys])
            deltas.append(m_val - bl_val)
        bl_all = np.mean([baseline[k]["rmse"] for k in var_names])
        m_all = np.mean([model[k]["rmse"] for k in var_names])
        deltas.append(m_all - bl_all)

        offset = -0.4 + bar_width * (i + 0.5)
        color = COLORS.get(label, "#333")
        bars = ax.bar(x + offset, deltas, bar_width,
                      label=LABELS_CN.get(label, label), color=color, edgecolor="white")
        for bar, val in zip(bars, deltas):
            sign = "+" if val > 0 else ""
            y_pos = val + 0.001 if val >= 0 else val - 0.003
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{sign}{val:.4f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("RMSE 变化量 (m/s)", fontsize=12)
    ax.set_title("消融实验：去掉变量后 RMSE 的变化\n（+ 表示变差，- 表示变好）",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_dir / "fig4_ablation_delta.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig4_ablation_delta.png")


# ==============================================================
# Fig 5: Level-wise RMSE Heatmap（去掉 bicubic 列）
# ==============================================================
def plot_level_heatmap(experiments, var_names, save_dir):
    exp_labels = list(experiments.keys())
    all_labels = [LABELS_CN.get(l, l) for l in exp_labels]

    data_matrix = []

    for lvl in ALTITUDE_LEVELS:
        keys = [v for v in var_names if lvl in v]
        row = []
        for label in exp_labels:
            model = experiments[label]["model"]
            row.append(np.mean([model[k]["rmse"] for k in keys if k in model]))
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(all_labels)))
    ax.set_xticklabels(all_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ALTITUDE_LEVELS)))
    ax.set_yticklabels(ALTITUDE_LEVELS, fontsize=11)

    for i in range(len(ALTITUDE_LEVELS)):
        for j in range(len(all_labels)):
            val = data_matrix[i, j]
            text_color = "white" if val > np.median(data_matrix) else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_title("各高度层 × 各实验 RMSE", fontsize=13, fontweight="bold")
    ax.set_ylabel("高度层", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("RMSE (m/s)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir / "fig5_level_rmse.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig5_level_rmse.png")


# ==============================================================
# Fig 6: W-weighted vs Original（不变）
# ==============================================================
def plot_wloss_comparison(experiments, var_names, save_dir):
    if "Original" not in experiments or "W-weighted" not in experiments:
        print("  [Skip] fig6 — need both Original and W-weighted")
        return

    w_keys = [v for v in var_names if "_w_" in v]
    u_keys = [v for v in var_names if "_u_" in v]
    v_keys = [v for v in var_names if "_v_" in v]

    orig = experiments["Original"]["model"]
    wloss = experiments["W-weighted"]["model"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (comp_keys, comp_name) in zip(axes, [(u_keys, "U"), (v_keys, "V"), (w_keys, "W")]):
        levels = []
        orig_rmse = []
        wloss_rmse = []

        for k in comp_keys:
            for lvl in ALTITUDE_LEVELS:
                if lvl in k:
                    levels.append(lvl)
                    break
            orig_rmse.append(orig[k]["rmse"])
            wloss_rmse.append(wloss[k]["rmse"])

        x = np.arange(len(levels))
        w = 0.35
        bars1 = ax.bar(x - w/2, orig_rmse, w, label="原始模型",
                        color=COLORS["Original"], edgecolor="white")
        bars2 = ax.bar(x + w/2, wloss_rmse, w, label="W加权模型",
                        color=COLORS["W-weighted"], edgecolor="white")

        for bar, val in zip(bars1, orig_rmse):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars2, wloss_rmse):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=10)
        ax.set_ylabel("RMSE (m/s)", fontsize=11)
        ax.set_title(f"{comp_name} 分量", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("W 加权 Loss 效果：逐高度层 RMSE 对比",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "fig6_wloss_w_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig6_wloss_w_comparison.png")


# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/")
    args = parser.parse_args()

    setup_chinese_font()

    print("\n" + "=" * 60)
    print("  Experiment Results — Figure Generator")
    print("=" * 60)

    experiments = load_all_metrics(args.results_dir)
    if not experiments:
        print(f"[ERROR] No metrics.json found under {args.results_dir}/eval_*/")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments:")
    for label in experiments:
        print(f"  - {label}")

    var_names = get_var_names(experiments)

    save_dir = Path(args.results_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating figures → {save_dir}/\n")

    plot_overall_comparison(experiments, var_names, save_dir)
    plot_component_bars(experiments, var_names, save_dir, metric="rmse")
    plot_component_bars(experiments, var_names, save_dir, metric="ssim")
    plot_ablation_delta(experiments, var_names, save_dir)
    plot_level_heatmap(experiments, var_names, save_dir)
    plot_wloss_comparison(experiments, var_names, save_dir)

    print(f"\nAll figures saved to {save_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
