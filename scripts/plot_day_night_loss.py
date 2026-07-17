"""
Plot training loss curves for day/night ablation experiments from .out log files.

Usage:
  python scripts/plot_day_night_loss.py --log_dir logs-day-night-log --output_dir results/ablation_day_night
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOG_PATTERN = re.compile(r"train error: avg loss = ([\d.]+)")
# Map ablation key to display name
ABL_LABELS = {
    "all": "All10",
    "no_terrain": "NoTerrain",
    "no_thermal": "NoThermal",
    "no_pblh": "NoPBLH",
    "no_pressure": "NoPSFC",
}
FILTER_LABELS = {"day": "Day", "night": "Night"}
COLORS = {
    "all": "#1f77b4",
    "no_terrain": "#ff7f0e",
    "no_thermal": "#d62728",
    "no_pblh": "#2ca02c",
    "no_pressure": "#9467bd",
}
LINE_STYLES = {"day": "-", "night": "--"}


def parse_losses(log_path):
    """Extract training loss values from a .out log file."""
    losses = []
    with open(log_path) as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                losses.append(float(m.group(1)))
    return losses


def parse_best_loss(log_path):
    """Extract the best (minimum) training loss."""
    losses = parse_losses(log_path)
    return min(losses) if losses else None


def find_log_files(log_dir):
    """Find all .out files and group by ablation type and filter."""
    files = {}
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".out"):
            continue
        # Patterns: abl_day_all_13742366.out or abl_large_day_all_13763903.out
        name = fname.replace(".out", "")
        # Remove job ID suffix
        name = re.sub(r"_\d+$", "", name)
        # Split and extract filter + ablation
        parts = name.split("_")  # ['abl'] or ['abl','large']
        if "large" in parts:
            filt = parts[2]   # abl_large_day_all -> day
            abl = "_".join(parts[3:])  # all, no_terrain, etc.
        else:
            filt = parts[1]   # abl_day_all -> day
            abl = "_".join(parts[2:])
        if filt not in ("day", "night"):
            continue
        if abl not in ABL_LABELS:
            continue
        key = (filt, abl)
        files[key] = os.path.join(log_dir, fname)
    return files


def plot_loss_curves(files, output_dir):
    """Plot training loss curves for all experiments."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Day vs Night grouped by ablation (5 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    abl_list = ["all", "no_terrain", "no_thermal", "no_pblh", "no_pressure"]

    for idx, abl in enumerate(abl_list):
        ax = axes[idx]
        for filt in ["day", "night"]:
            key = (filt, abl)
            if key not in files:
                continue
            losses = parse_losses(files[key])
            if not losses:
                continue
            label = "{} ({})".format(ABL_LABELS[abl], FILTER_LABELS[filt])
            ls = LINE_STYLES[filt]
            ax.plot(losses, label=label, color=COLORS[abl],
                    linestyle=ls, linewidth=1.5, alpha=0.9)
        ax.set_title(ABL_LABELS[abl], fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide extra subplot
    axes[5].set_visible(False)

    fig.suptitle("Day vs Night Training Loss by Ablation Type", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_by_ablation.png"), dpi=150)
    plt.close(fig)

    # 2. All curves on one plot
    fig, ax = plt.subplots(figsize=(14, 7))
    for (filt, abl), fpath in sorted(files.items()):
        losses = parse_losses(fpath)
        if not losses:
            continue
        label = "{} ({})".format(ABL_LABELS[abl], FILTER_LABELS[filt])
        ax.plot(losses, label=label, color=COLORS[abl],
                linestyle=LINE_STYLES[filt], linewidth=1.2, alpha=0.8)
    ax.set_title("All Day/Night Ablation Training Loss Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_all_curves.png"), dpi=150)
    plt.close(fig)

    # 3. Log scale for better detail
    fig, ax = plt.subplots(figsize=(14, 7))
    for (filt, abl), fpath in sorted(files.items()):
        losses = parse_losses(fpath)
        if not losses:
            continue
        label = "{} ({})".format(ABL_LABELS[abl], FILTER_LABELS[filt])
        ax.semilogy(losses, label=label, color=COLORS[abl],
                    linestyle=LINE_STYLES[filt], linewidth=1.2, alpha=0.8)
    ax.set_title("All Day/Night Ablation Training Loss (Log Scale)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (log)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_all_curves_log.png"), dpi=150)
    plt.close(fig)

    # 4. Best loss bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(abl_list))
    width = 0.35

    day_bests = []
    night_bests = []
    for abl in abl_list:
        day_bests.append(parse_best_loss(files.get(("day", abl), "")) or 0)
        night_bests.append(parse_best_loss(files.get(("night", abl), "")) or 0)

    bars1 = ax.bar(x - width/2, day_bests, width, label="Day",
                   color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, night_bests, width, label="Night",
                   color="#3498db", alpha=0.85)

    for bar, val in zip(bars1, day_bests):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                "{:.4f}".format(val), ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, night_bests):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                "{:.4f}".format(val), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([ABL_LABELS[a] for a in abl_list], fontsize=12)
    ax.set_ylabel("Best Training Loss")
    ax.set_title("Best Training Loss: Day vs Night by Ablation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_best_comparison.png"), dpi=150)
    plt.close(fig)

    print("Saved plots:")
    print("  {}/loss_by_ablation.png".format(output_dir))
    print("  {}/loss_all_curves.png".format(output_dir))
    print("  {}/loss_all_curves_log.png".format(output_dir))
    print("  {}/loss_best_comparison.png".format(output_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs-day-night-log")
    parser.add_argument("--output_dir", type=str, default="results/ablation_day_night")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print("ERROR: log dir not found: {}".format(args.log_dir))
        sys.exit(1)

    files = find_log_files(args.log_dir)
    print("Found {} log files".format(len(files)))

    for (filt, abl), fpath in sorted(files.items()):
        losses = parse_losses(fpath)
        best = min(losses) if losses else None
        print("  {}_{}: {} epochs, best loss = {}".format(filt, abl, len(losses), best))

    plot_loss_curves(files, args.output_dir)


if __name__ == "__main__":
    main()
