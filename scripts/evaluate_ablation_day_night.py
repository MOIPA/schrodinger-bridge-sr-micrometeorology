"""
Evaluate all day/night ablation experiments and produce comparison summary.

For each of the 5 ablation types x 2 day/night filters = 10 models:
  1. Load checkpoint and run inference on the full test set
  2. Compute per-component and per-level metrics
  3. Output a comparison table

Key analysis: ΔRMSE (vs All10 baseline) separately for day and night models,
to identify which condition variables matter more during daytime vs nighttime.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.ddpm.unet_ddpm_v01 import UNetDDPMVer01
from src.dl_model.si_follmer.si_follmer_framework import SIFollmer
from src.utils.random_seed_helper import seed_worker, get_torch_generator

logger = getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ABLATIONS = ["all", "no_terrain", "no_thermal", "no_pblh", "no_pressure"]
FILTERS = ["day", "night"]
COMPONENTS = ["U", "V", "W"]
LEVELS = ["ml0", "ml1", "ml2", "ml3", "ml5", "ml10"]

EXPERIMENT_NAME = "ExperimentSchrodingerBridge3dWind"


def compute_metrics(pred, target):
    """Compute per-channel RMSE, MAE, SSIM, Correlation, Bias."""
    n_channels = pred.shape[1]
    rmse = np.zeros(n_channels)
    mae = np.zeros(n_channels)
    ssim = np.zeros(n_channels)
    corr = np.zeros(n_channels)
    bias = np.zeros(n_channels)

    for c in range(n_channels):
        p = pred[:, c]
        t = target[:, c]
        diff = p - t
        rmse[c] = np.sqrt(np.mean(diff ** 2))
        mae[c] = np.mean(np.abs(diff))
        bias[c] = np.mean(p) - np.mean(t)

        # SSIM (simple implementation without windowing for speed)
        mu_p, mu_t = np.mean(p), np.mean(t)
        sigma_p, sigma_t = np.std(p), np.std(t)
        # Use global statistics as approximation
        C1, C2 = 1e-4, 9e-4
        numerator = (2 * mu_p * mu_t + C1) * (2 * sigma_p * sigma_t + C2)
        denominator = (mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2)
        ssim[c] = float(numerator / (denominator + 1e-8))

        # Pearson correlation
        p_flat = p.flatten()
        t_flat = t.flatten()
        p_mean = np.mean(p_flat)
        t_mean = np.mean(t_flat)
        numerator = np.sum((p_flat - p_mean) * (t_flat - t_mean))
        denominator = np.sqrt(np.sum((p_flat - p_mean)**2) * np.sum((t_flat - t_mean)**2))
        corr[c] = float(numerator / (denominator + 1e-8))

    return rmse, mae, ssim, corr, bias


def summarize_by_component(metrics_per_ch, target_names):
    """Aggregate per-channel metrics into per-component (U/V/W) averages."""
    result = {}
    for comp in COMPONENTS:
        indices = [i for i, name in enumerate(target_names) if f"_{comp.lower()}_" in name
                   or name.endswith(f"_{comp.lower()}")]
        if not indices:
            # Try matching hr_{comp}_ pattern
            indices = [i for i, name in enumerate(target_names)
                       if name == f"hr_{comp.lower()}_ml0" or f"hr_{comp.lower()}_ml" in name]
        if not indices:
            continue
        result[comp] = {
            "rmse": float(np.mean([metrics_per_ch["rmse"][i] for i in indices])),
            "mae": float(np.mean([metrics_per_ch["mae"][i] for i in indices])),
            "ssim": float(np.mean([metrics_per_ch["ssim"][i] for i in indices])),
            "corr": float(np.mean([metrics_per_ch["corr"][i] for i in indices])),
            "bias": float(np.mean([metrics_per_ch["bias"][i] for i in indices])),
        }
    # Overall
    result["Overall"] = {
        "rmse": float(np.mean(metrics_per_ch["rmse"])),
        "mae": float(np.mean(metrics_per_ch["mae"])),
        "ssim": float(np.mean(metrics_per_ch["ssim"])),
        "corr": float(np.mean(metrics_per_ch["corr"])),
        "bias": float(np.mean(metrics_per_ch["bias"])),
    }
    return result


def summarize_by_level(metrics_per_ch, target_names):
    """Aggregate per-channel metrics into per-level (ml0-ml10) averages."""
    result = {}
    for level in LEVELS:
        indices = [i for i, name in enumerate(target_names) if level in name.lower()
                   or f"_{level}" in name or level in name]
        if not indices:
            continue
        result[level] = {
            "rmse": float(np.mean([metrics_per_ch["rmse"][i] for i in indices])),
            "mae": float(np.mean([metrics_per_ch["mae"][i] for i in indices])),
            "ssim": float(np.mean([metrics_per_ch["ssim"][i] for i in indices])),
            "corr": float(np.mean([metrics_per_ch["corr"][i] for i in indices])),
            "bias": float(np.mean([metrics_per_ch["bias"][i] for i in indices])),
        }
    return result


def evaluate_checkpoint(config_path, checkpoint_path, device):
    """Load model from checkpoint and evaluate on test set (full, unfiltered)."""
    config = load_config(EXPERIMENT_NAME, config_path)

    # Temporarily set filter to "all" so we evaluate on full test set
    original_filter = getattr(config.data, "day_night_filter", "all")
    config.data.day_night_filter = "all"

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT_DIR,
        loader_config=config.loader,
        dataset_config=config.data,
        world_size=None,
        rank=None,
        train_valid_test_kinds=["test"],
    )
    test_loader = dict_loaders["test"]

    # Restore original filter
    config.data.day_night_filter = original_filter

    # Build model
    model = UNetDDPMVer01(config.model)
    si_follmer = SIFollmer(config.si)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Inference
    all_pred = []
    all_target = []
    swdown_vals = []

    # Find swdown index in input variables
    swdown_idx = config.data.input_variable_names.index("swdown")

    for batch_data in tqdm(test_loader, desc="Inference"):
        y_cond = batch_data["x"].to(device)
        y0 = batch_data["y0"].to(device)

        swdown_norm = y_cond[:, swdown_idx].mean(dim=[1, 2]).cpu().numpy()
        swdown_bias = config.data.biases["swdown"]
        swdown_scale = config.data.scales["swdown"]
        swdown_phys = swdown_norm * swdown_scale + swdown_bias
        swdown_vals.append(swdown_phys)

        with torch.no_grad():
            y_est = si_follmer.sample(y_cond, y0, model)
        all_pred.append(y_est.cpu().numpy())
        all_target.append(batch_data["y"].cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)
    swdown_vals = np.concatenate(swdown_vals, axis=0)

    # Split into day/night
    day_mask = swdown_vals > 50.0
    night_mask = swdown_vals < 5.0

    target_names = config.data.target_variable_names

    results = {}
    for group_name, mask in [("all", slice(None)), ("day", day_mask), ("night", night_mask)]:
        if mask is not None and not np.any(mask):
            results[group_name] = None
            continue
        p = all_pred[mask] if mask is not None else all_pred
        t = all_target[mask] if mask is not None else all_target
        if len(p) == 0:
            results[group_name] = None
            continue

        rmse, mae, ssim, corr, bias = compute_metrics(p, t)
        metrics = {"rmse": rmse, "mae": mae, "ssim": ssim, "corr": corr, "bias": bias}
        results[group_name] = {
            "n_samples": len(p),
            "by_component": summarize_by_component(metrics, target_names),
            "by_level": summarize_by_level(metrics, target_names),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(ROOT_DIR, "results", "ablation_day_night"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_base_dir", type=str,
                        default=os.path.join(ROOT_DIR, "data", "DL_result"))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    all_summaries = {}

    for abl in ABLATIONS:
        for filt in FILTERS:
            config_base = f"config_wind_3d_ablation_{abl}"
            config_path = f"configs/{config_base}_{filt}.yml"
            exp_name = f"abl_{filt}_{abl}"

            # Find checkpoint
            checkpoint_dir = os.path.join(args.checkpoint_base_dir, exp_name)
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

            if not os.path.exists(checkpoint_path):
                print(f"[SKIP] Checkpoint not found: {checkpoint_path}")
                continue

            print(f"\n{'='*60}")
            print(f"Evaluating: {exp_name}")
            print(f"Config:    {config_path}")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"{'='*60}")

            results = evaluate_checkpoint(config_path, checkpoint_path, args.device)

            summary = {
                "config": config_path,
                "checkpoint": checkpoint_path,
                "ablation": abl,
                "filter": filt,
                "results": results,
            }
            all_summaries[exp_name] = summary

            # Save individual result
            out_path = os.path.join(args.results_dir, f"{exp_name}_metrics.json")
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Saved: {out_path}")

    # Build comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE: Overall RMSE by Ablation x Day/Night")
    print(f"{'='*80}")

    # Header
    header = f"{'Ablation':<16} {'Day RMSE':>10} {'Night RMSE':>10} {'Day-Night':>10}"
    print(header)
    print("-" * len(header))

    for abl in ABLATIONS:
        day_key = f"abl_day_{abl}"
        night_key = f"abl_night_{abl}"

        day_rmse = all_summaries.get(day_key, {}).get("results", {}).get("day", {})
        night_rmse = all_summaries.get(night_key, {}).get("results", {}).get("night", {})

        if day_rmse and night_rmse:
            day_val = day_rmse["by_component"]["Overall"]["rmse"]
            night_val = night_rmse["by_component"]["Overall"]["rmse"]
            diff = day_val - night_val
            print(f"{abl:<16} {day_val:>10.4f} {night_val:>10.4f} {diff:>+10.4f}")

    # Delta RMSE table (how much each ablation degrades performance vs All10)
    print(f"\n{'='*80}")
    print("DELTA RMSE TABLE: Degradation vs All10 Baseline")
    print(f"Positive = worse than All10 (variable is important)")
    print(f"{'='*80}")

    header = f"{'Ablation':<16} {'Day ΔRMSE':>10} {'Night ΔRMSE':>12} {'ΔDiff':>10} {'Interpretation':>25}"
    print(header)
    print("-" * len(header))

    day_all_rmse = None
    night_all_rmse = None

    if "abl_day_all" in all_summaries:
        day_all_rmse = all_summaries["abl_day_all"]["results"]["day"]["by_component"]["Overall"]["rmse"]
    if "abl_night_all" in all_summaries:
        night_all_rmse = all_summaries["abl_night_all"]["results"]["night"]["by_component"]["Overall"]["rmse"]

    for abl in ABLATIONS:
        if abl == "all":
            continue

        day_key = f"abl_day_{abl}"
        night_key = f"abl_night_{abl}"

        day_rmse = all_summaries.get(day_key, {}).get("results", {}).get("day", {})
        night_rmse = all_summaries.get(night_key, {}).get("results", {}).get("night", {})

        if day_rmse and night_rmse and day_all_rmse and night_all_rmse:
            day_delta = day_rmse["by_component"]["Overall"]["rmse"] - day_all_rmse
            night_delta = night_rmse["by_component"]["Overall"]["rmse"] - night_all_rmse
            delta_diff = day_delta - night_delta

            if delta_diff > 0.005:
                interp = "Daytime more important"
            elif delta_diff < -0.005:
                interp = "Nighttime more important"
            else:
                interp = "Similar importance"

            print(f"{abl:<16} {day_delta:>+10.4f} {night_delta:>+12.4f} {delta_diff:>+10.4f} {interp:>25}")

    # W component breakdown
    print(f"\n{'='*80}")
    print("W COMPONENT RMSE: Day vs Night Comparison")
    print(f"{'='*80}")

    header = f"{'Ablation':<16} {'Day W':>10} {'Night W':>10} {'Diff':>10}"
    print(header)
    print("-" * len(header))

    for abl in ABLATIONS:
        day_key = f"abl_day_{abl}"
        night_key = f"abl_night_{abl}"

        day_w = all_summaries.get(day_key, {}).get("results", {}).get("day", {})
        night_w = all_summaries.get(night_key, {}).get("results", {}).get("night", {})

        if day_w and night_w:
            day_val = day_w["by_component"].get("W", {}).get("rmse", float("nan"))
            night_val = night_w["by_component"].get("W", {}).get("rmse", float("nan"))
            diff = day_val - night_val
            print(f"{abl:<16} {day_val:>10.4f} {night_val:>10.4f} {diff:>+10.4f}")

    # Save combined summary
    combined_path = os.path.join(args.results_dir, "all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nCombined summary saved: {combined_path}")


if __name__ == "__main__":
    main()
