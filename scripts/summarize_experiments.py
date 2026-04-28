#!/usr/bin/env python3
"""
Summarize & Compare All Experiment Results
==========================================
读取所有实验的 metrics.json，生成跨实验对比表（终端 + CSV）。

用法:
  python scripts/summarize_experiments.py --results_dir results/

输出:
  results/summary_all.csv         — 逐通道 × 逐实验 完整表
  results/summary_component.csv   — 按风分量 (U/V/W) 汇总
  results/summary_level.csv       — 按高度层 (ml0..ml10) 汇总
  results/summary_overall.csv     — 总体汇总（一行一个实验）
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

# fmt: off
# ── 实验目录名 → 显示名映射 ──
EXPERIMENT_LABELS = OrderedDict([
    ("eval_original",              "Original"),
    ("eval_wloss",                 "W-weighted"),
    ("eval_ablation_all",          "Ablation:All"),
    ("eval_ablation_no_terrain",   "Abl:NoTerrain"),
    ("eval_ablation_no_thermal",   "Abl:NoThermal"),
    ("eval_ablation_no_pblh",      "Abl:NoPBLH"),
    ("eval_ablation_no_pressure",  "Abl:NoPressure"),
])
# fmt: on

WIND_COMPONENTS = ["u", "v", "w"]
ALTITUDE_LEVELS = ["ml0", "ml1", "ml2", "ml3", "ml5", "ml10"]
METRICS = ["rmse", "mae", "ssim", "corr"]


def load_all_results(results_dir: str) -> dict:
    """
    扫描 results_dir 下所有 eval_* 子目录，加载 metrics.json。
    返回: {exp_name: {"model": {...}, "bicubic_baseline": {...}}}
    """
    base = Path(results_dir)
    experiments = {}

    # 先按预定义顺序加载
    for dirname in EXPERIMENT_LABELS:
        json_path = base / dirname / "metrics.json"
        if json_path.exists():
            with open(json_path) as f:
                experiments[dirname] = json.load(f)

    # 再扫描其余（未预定义的实验）
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.is_dir() and p.name.startswith("eval_") and p.name not in experiments:
                json_path = p / "metrics.json"
                if json_path.exists():
                    with open(json_path) as f:
                        experiments[p.name] = json.load(f)

    return experiments


def get_label(exp_name: str) -> str:
    return EXPERIMENT_LABELS.get(exp_name, exp_name)


def get_var_names(experiments: dict) -> list:
    """从任意一个实验中取出变量名列表（保持原顺序）。"""
    for data in experiments.values():
        return list(data["model"].keys())
    return []


# ── 1. 逐通道 × 逐实验完整表 ──
def make_full_table(experiments: dict, var_names: list) -> list:
    """返回行列表，每行是一个 dict。"""
    rows = []
    for var in var_names:
        row = {"variable": var}
        # bicubic baseline（各实验理论相同，取第一个有的）
        for exp_data in experiments.values():
            bl = exp_data.get("bicubic_baseline", {}).get(var)
            if bl:
                for m in METRICS:
                    row[f"bicubic_{m}"] = bl[m]
                break

        for exp_name, exp_data in experiments.items():
            label = get_label(exp_name)
            model = exp_data.get("model", {}).get(var, {})
            for m in METRICS:
                row[f"{label}_{m}"] = model.get(m, float("nan"))
        rows.append(row)
    return rows


# ── 2. 按风分量汇总 ──
def make_component_table(experiments: dict, var_names: list) -> list:
    rows = []
    for comp in WIND_COMPONENTS:
        keys = [v for v in var_names if f"_{comp}_" in v]
        if not keys:
            continue
        row = {"component": comp.upper(), "n_channels": len(keys)}

        # bicubic baseline
        for exp_data in experiments.values():
            bl = exp_data.get("bicubic_baseline", {})
            for m in METRICS:
                vals = [bl[k][m] for k in keys if k in bl]
                row[f"bicubic_{m}_mean"] = np.mean(vals) if vals else float("nan")
            break

        for exp_name, exp_data in experiments.items():
            label = get_label(exp_name)
            model = exp_data.get("model", {})
            for m in METRICS:
                vals = [model[k][m] for k in keys if k in model]
                row[f"{label}_{m}_mean"] = np.mean(vals) if vals else float("nan")
                row[f"{label}_{m}_std"] = np.std(vals) if vals else float("nan")
        rows.append(row)
    return rows


# ── 3. 按高度层汇总 ──
def make_level_table(experiments: dict, var_names: list) -> list:
    rows = []
    for lvl in ALTITUDE_LEVELS:
        keys = [v for v in var_names if lvl in v]
        if not keys:
            continue
        row = {"level": lvl, "n_channels": len(keys)}

        for exp_data in experiments.values():
            bl = exp_data.get("bicubic_baseline", {})
            for m in METRICS:
                vals = [bl[k][m] for k in keys if k in bl]
                row[f"bicubic_{m}_mean"] = np.mean(vals) if vals else float("nan")
            break

        for exp_name, exp_data in experiments.items():
            label = get_label(exp_name)
            model = exp_data.get("model", {})
            for m in METRICS:
                vals = [model[k][m] for k in keys if k in model]
                row[f"{label}_{m}_mean"] = np.mean(vals) if vals else float("nan")
                row[f"{label}_{m}_std"] = np.std(vals) if vals else float("nan")
        rows.append(row)
    return rows


# ── 4. 总体汇总 ──
def make_overall_table(experiments: dict, var_names: list) -> list:
    """每个实验一行，所有通道平均。"""
    rows = []

    # bicubic baseline
    for exp_data in experiments.values():
        bl = exp_data.get("bicubic_baseline", {})
        row = {"experiment": "Bicubic Baseline"}
        for m in METRICS:
            vals = [bl[k][m] for k in var_names if k in bl]
            row[f"{m}_mean"] = np.mean(vals) if vals else float("nan")
            row[f"{m}_std"] = np.std(vals) if vals else float("nan")
        # 再按分量拆
        for comp in WIND_COMPONENTS:
            keys_c = [v for v in var_names if f"_{comp}_" in v]
            for m in ["rmse", "ssim"]:
                vals = [bl[k][m] for k in keys_c if k in bl]
                row[f"{comp.upper()}_{m}"] = np.mean(vals) if vals else float("nan")
        rows.append(row)
        break

    for exp_name, exp_data in experiments.items():
        label = get_label(exp_name)
        model = exp_data.get("model", {})
        row = {"experiment": label}
        for m in METRICS:
            vals = [model[k][m] for k in var_names if k in model]
            row[f"{m}_mean"] = np.mean(vals) if vals else float("nan")
            row[f"{m}_std"] = np.std(vals) if vals else float("nan")
        # 按分量
        for comp in WIND_COMPONENTS:
            keys_c = [v for v in var_names if f"_{comp}_" in v]
            for m in ["rmse", "ssim"]:
                vals = [model[k][m] for k in keys_c if k in model]
                row[f"{comp.upper()}_{m}"] = np.mean(vals) if vals else float("nan")
        rows.append(row)
    return rows


# ==============================================================
# 终端打印
# ==============================================================
def print_overall_table(rows):
    """打印总体对比表。"""
    sep = "=" * 120
    print(f"\n{sep}")
    print("  Overall Summary — All Experiments")
    print(sep)

    header = (f"{'Experiment':<18} | "
              f"{'RMSE(all)':>10} {'MAE(all)':>10} {'SSIM(all)':>10} {'Corr(all)':>10} | "
              f"{'U_RMSE':>8} {'V_RMSE':>8} {'W_RMSE':>8} | "
              f"{'U_SSIM':>8} {'V_SSIM':>8} {'W_SSIM':>8}")
    print(header)
    print("-" * 120)

    for row in rows:
        line = (f"{row['experiment']:<18} | "
                f"{row.get('rmse_mean', 0):>10.4f} "
                f"{row.get('mae_mean', 0):>10.4f} "
                f"{row.get('ssim_mean', 0):>10.4f} "
                f"{row.get('corr_mean', 0):>10.4f} | "
                f"{row.get('U_rmse', 0):>8.4f} "
                f"{row.get('V_rmse', 0):>8.4f} "
                f"{row.get('W_rmse', 0):>8.4f} | "
                f"{row.get('U_ssim', 0):>8.4f} "
                f"{row.get('V_ssim', 0):>8.4f} "
                f"{row.get('W_ssim', 0):>8.4f}")
        print(line)
    print(sep)


def print_component_table(rows, experiments):
    """打印分量汇总表。"""
    sep = "=" * 100
    print(f"\n{sep}")
    print("  Component Summary (mean RMSE | mean SSIM)")
    print(sep)

    exp_labels = [get_label(e) for e in experiments]
    header = f"{'Comp':<6} | {'Bicubic':>14} | " + " | ".join(f"{l:>14}" for l in exp_labels)
    print(header)
    print("-" * 100)

    for row in rows:
        parts = [f"{row['component']:<6}"]
        # bicubic
        bic_str = (f"{row.get('bicubic_rmse_mean', 0):.4f}/"
                   f"{row.get('bicubic_ssim_mean', 0):.4f}")
        parts.append(f"{bic_str:>14}")

        for exp_name in experiments:
            label = get_label(exp_name)
            rmse_val = row.get(f"{label}_rmse_mean", float("nan"))
            ssim_val = row.get(f"{label}_ssim_mean", float("nan"))
            s = f"{rmse_val:.4f}/{ssim_val:.4f}"
            parts.append(f"{s:>14}")
        print(" | ".join(parts))
    print(sep)


def print_ablation_delta(experiments, var_names):
    """打印消融实验相对于 baseline (ablation_all) 的变化量。"""
    if "eval_ablation_all" not in experiments:
        return

    baseline = experiments["eval_ablation_all"]["model"]
    ablation_exps = {k: v for k, v in experiments.items()
                     if k.startswith("eval_ablation_") and k != "eval_ablation_all"}

    if not ablation_exps:
        return

    sep = "=" * 100
    print(f"\n{sep}")
    print("  Ablation Delta (vs Ablation:All baseline)")
    print(f"  + means worse (higher RMSE), - means better (lower RMSE)")
    print(sep)

    header = f"{'Component':<10} | " + " | ".join(
        f"{get_label(e):>16}" for e in ablation_exps)
    print(header)
    print("-" * 100)

    for comp in WIND_COMPONENTS:
        keys = [v for v in var_names if f"_{comp}_" in v]
        if not keys:
            continue

        bl_rmse = np.mean([baseline[k]["rmse"] for k in keys if k in baseline])
        bl_ssim = np.mean([baseline[k]["ssim"] for k in keys if k in baseline])

        parts = [f"{comp.upper():<10}"]
        for exp_name, exp_data in ablation_exps.items():
            model = exp_data["model"]
            m_rmse = np.mean([model[k]["rmse"] for k in keys if k in model])
            m_ssim = np.mean([model[k]["ssim"] for k in keys if k in model])
            d_rmse = m_rmse - bl_rmse
            d_ssim = m_ssim - bl_ssim
            sign_r = "+" if d_rmse > 0 else ""
            sign_s = "+" if d_ssim > 0 else ""
            s = f"R:{sign_r}{d_rmse:.4f} S:{sign_s}{d_ssim:.4f}"
            parts.append(f"{s:>16}")
        print(" | ".join(parts))

    # 全局
    bl_rmse_all = np.mean([baseline[k]["rmse"] for k in var_names if k in baseline])
    parts = [f"{'Overall':<10}"]
    for exp_name, exp_data in ablation_exps.items():
        model = exp_data["model"]
        m_rmse = np.mean([model[k]["rmse"] for k in var_names if k in model])
        d = m_rmse - bl_rmse_all
        sign = "+" if d > 0 else ""
        parts.append(f"{'RMSE:' + sign + f'{d:.4f}':>16}")
    print("-" * 100)
    print(" | ".join(parts))
    print(sep)


# ==============================================================
# CSV 输出
# ==============================================================
def save_csv(rows: list, filepath: str):
    """将 list[dict] 保存为 CSV（不依赖 pandas）。"""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(filepath, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"  Saved: {filepath}")


# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Summarize all experiment results")
    parser.add_argument("--results_dir", type=str, default="results/",
                        help="Directory containing eval_* subdirectories")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Experiment Summary Generator")
    print("=" * 60)

    experiments = load_all_results(args.results_dir)

    if not experiments:
        print(f"\n[ERROR] No metrics.json found in {args.results_dir}/eval_*/")
        print("Please run evaluate_wind_3d.py first.")
        sys.exit(1)

    print(f"\nFound {len(experiments)} experiments:")
    for name in experiments:
        print(f"  - {name} ({get_label(name)})")

    var_names = get_var_names(experiments)
    print(f"Variables: {len(var_names)}")

    # ---- 生成表格 ----
    full_rows = make_full_table(experiments, var_names)
    comp_rows = make_component_table(experiments, var_names)
    level_rows = make_level_table(experiments, var_names)
    overall_rows = make_overall_table(experiments, var_names)

    # ---- 终端打印 ----
    print_overall_table(overall_rows)
    print_component_table(comp_rows, experiments)
    print_ablation_delta(experiments, var_names)

    # ---- 保存 CSV ----
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("\nSaving CSV files...")
    save_csv(full_rows, str(out / "summary_all.csv"))
    save_csv(comp_rows, str(out / "summary_component.csv"))
    save_csv(level_rows, str(out / "summary_level.csv"))
    save_csv(overall_rows, str(out / "summary_overall.csv"))

    # ---- 保存完整 JSON ----
    summary_json = {
        "experiments_found": list(experiments.keys()),
        "overall": overall_rows,
        "by_component": comp_rows,
        "by_level": level_rows,
    }
    json_path = out / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")

    print(f"\nDone! All summary files in {out}/")


if __name__ == "__main__":
    main()
