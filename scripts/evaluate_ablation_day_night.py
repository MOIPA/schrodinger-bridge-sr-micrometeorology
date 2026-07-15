"""
Evaluate all day/night ablation experiments and produce comparison summary.

For each of the 5 ablation types x 2 day/night filters = 10 models:
  1. Load checkpoint, run inference on the model's own test set (day model -> day test)
  2. Compute per-component and per-level metrics (RMSE, SSIM, MAE, Corr, Bias)
  3. Output comparison tables showing delta-RMSE for day vs night

Key analysis: for each ablation, compute delta_RMSE vs All10 separately for day
and night, to identify which condition variables matter more at different times.
"""
import argparse
import json
import os
import sys
from logging import getLogger

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.model_maker import make_model
from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer

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

        mu_p = np.mean(p)
        mu_t = np.mean(t)
        sigma_p = np.std(p)
        sigma_t = np.std(t)
        C1, C2 = 1e-4, 9e-4
        numerator = (2 * mu_p * mu_t + C1) * (2 * sigma_p * sigma_t + C2)
        denominator = (mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2)
        ssim[c] = float(numerator / (denominator + 1e-8))

        p_flat = p.flatten()
        t_flat = t.flatten()
        p_mean = np.mean(p_flat)
        t_mean = np.mean(t_flat)
        num = np.sum((p_flat - p_mean) * (t_flat - t_mean))
        den = np.sqrt(np.sum((p_flat - p_mean)**2) * np.sum((t_flat - t_mean)**2))
        corr[c] = float(num / (den + 1e-8))

    return rmse, mae, ssim, corr, bias


def summarize_by_component(metrics_per_ch, target_names):
    """Aggregate per-channel metrics into per-component (U/V/W) averages."""
    result = {}
    for comp in COMPONENTS:
        indices = [i for i, name in enumerate(target_names)
                   if "_{}_".format(comp.lower()) in name]
        if not indices:
            continue
        result[comp] = {
            "rmse": float(np.mean([metrics_per_ch["rmse"][i] for i in indices])),
            "mae": float(np.mean([metrics_per_ch["mae"][i] for i in indices])),
            "ssim": float(np.mean([metrics_per_ch["ssim"][i] for i in indices])),
            "corr": float(np.mean([metrics_per_ch["corr"][i] for i in indices])),
            "bias": float(np.mean([metrics_per_ch["bias"][i] for i in indices])),
        }
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
        indices = [i for i, name in enumerate(target_names) if level in name]
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


def evaluate_checkpoint(config_path, checkpoint_path, device, eval_filter):
    """Load model and evaluate on its own test set (day or night)."""
    config = load_config(EXPERIMENT_NAME, config_path)

    # Use the model's own filter for evaluation
    config.data.day_night_filter = eval_filter

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT_DIR,
        loader_config=config.loader,
        dataset_config=config.data,
        world_size=None,
        rank=None,
        train_valid_test_kinds=["test"],
    )
    test_loader = dict_loaders["test"]

    print("  Test samples: {}".format(len(test_loader.dataset)))

    # Build model
    model = make_model(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    si_follmer = StochasticInterpolantFollmer(config=config.si, neural_net=model)

    # Inference
    all_pred = []
    all_target = []

    for batch_data in tqdm(test_loader, desc="  Inference"):
        y_cond = batch_data["x"].to(device)
        y0 = batch_data["y0"].to(device)

        y_est, _ = si_follmer.sample_y1_bare_diffusion(y0=y0, y_cond=y_cond)
        all_pred.append(y_est.cpu().numpy())
        all_target.append(batch_data["y"].cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    target_names = config.data.target_variable_names

    rmse, mae, ssim, corr, bias = compute_metrics(all_pred, all_target)
    metrics = {"rmse": rmse, "mae": mae, "ssim": ssim, "corr": corr, "bias": bias}

    return {
        "n_samples": len(all_pred),
        "by_component": summarize_by_component(metrics, target_names),
        "by_level": summarize_by_level(metrics, target_names),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(ROOT_DIR, "results", "ablation_day_night"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_base_dir", type=str,
                        default=os.path.join(ROOT_DIR, "data", "DL_result",
                                             "ExperimentSchrodingerBridge3dWind"))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    all_results = {}

    for abl in ABLATIONS:
        for filt in FILTERS:
            config_path = "configs/config_wind_3d_ablation_{}_{}.yml".format(abl, filt)
            exp_name = "abl_{}_{}".format(filt, abl)
            checkpoint_path = os.path.join(
                args.checkpoint_base_dir,
                "config_wind_3d_ablation_{}_{}".format(abl, filt),
                "checkpoint.pth",
            )

            if not os.path.exists(checkpoint_path):
                print("[SKIP] Checkpoint not found: {}".format(checkpoint_path))
                continue

            print("")
            print("=" * 60)
            print("Evaluating: {}  (filter={})".format(exp_name, filt))
            print("  Config:     {}".format(config_path))
            print("  Checkpoint: {}".format(checkpoint_path))
            print("=" * 60)

            results = evaluate_checkpoint(config_path, checkpoint_path,
                                          args.device, filt)

            all_results[exp_name] = {
                "config": config_path,
                "checkpoint": checkpoint_path,
                "ablation": abl,
                "filter": filt,
                "results": results,
            }

            # Save individual result
            out_path = os.path.join(args.results_dir,
                                    "{}_metrics.json".format(exp_name))
            with open(out_path, "w") as f:
                json.dump(all_results[exp_name], f, indent=2)
            print("  Saved: {}".format(out_path))

            # Quick summary
            print("  Overall RMSE: {:.4f}".format(
                results["by_component"]["Overall"]["rmse"]))
            for comp in ["U", "V", "W"]:
                if comp in results["by_component"]:
                    print("  {} RMSE: {:.4f}  SSIM: {:.4f}".format(
                        comp,
                        results["by_component"][comp]["rmse"],
                        results["by_component"][comp]["ssim"]))

    # =============================================
    # Comparison tables
    # =============================================
    print("")
    print("=" * 80)
    print("COMPARISON: Overall RMSE by Ablation x Day/Night")
    print("=" * 80)
    header = "{:<16} {:>10} {:>10}".format("Ablation", "Day RMSE", "Night RMSE")
    print(header)
    print("-" * len(header))

    for abl in ABLATIONS:
        day_key = "abl_day_{}".format(abl)
        night_key = "abl_night_{}".format(abl)
        day_r = all_results.get(day_key, {}).get("results", {})
        night_r = all_results.get(night_key, {}).get("results", {})
        if day_r and night_r:
            day_val = day_r["by_component"]["Overall"]["rmse"]
            night_val = night_r["by_component"]["Overall"]["rmse"]
            print("{:<16} {:>10.4f} {:>10.4f}".format(abl, day_val, night_val))

    # Delta RMSE: how much worse than All10 baseline
    print("")
    print("=" * 80)
    print("DELTA RMSE: Degradation vs All10 (positive = variable is important)")
    print("=" * 80)
    header = "{:<16} {:>12} {:>12} {:>10}".format(
        "Ablation", "Day delta", "Night delta", "Diff")
    print(header)
    print("-" * len(header))

    day_all = all_results.get("abl_day_all", {}).get("results", {})
    night_all = all_results.get("abl_night_all", {}).get("results", {})

    if day_all and night_all:
        day_base = day_all["by_component"]["Overall"]["rmse"]
        night_base = night_all["by_component"]["Overall"]["rmse"]

        for abl in ABLATIONS:
            if abl == "all":
                continue
            day_r = all_results.get("abl_day_{}".format(abl), {}).get("results", {})
            night_r = all_results.get("abl_night_{}".format(abl), {}).get("results", {})

            if day_r and night_r:
                day_d = day_r["by_component"]["Overall"]["rmse"] - day_base
                night_d = night_r["by_component"]["Overall"]["rmse"] - night_base
                diff = day_d - night_d

                if diff > 0.005:
                    interp = "Day > Night"
                elif diff < -0.005:
                    interp = "Night > Day"
                else:
                    interp = "Similar"

                print("{:<16} {:>+12.4f} {:>+12.4f} {:>+10.4f}  [{}]".format(
                    abl, day_d, night_d, diff, interp))

    # Per-component breakdown
    for comp in ["U", "V", "W"]:
        print("")
        print("=" * 80)
        print("{} RMSE: Day vs Night by Ablation".format(comp))
        print("=" * 80)
        header = "{:<16} {:>10} {:>10}".format("Ablation", "Day", "Night")
        print(header)
        print("-" * len(header))
        for abl in ABLATIONS:
            day_r = all_results.get("abl_day_{}".format(abl), {}).get("results", {})
            night_r = all_results.get("abl_night_{}".format(abl), {}).get("results", {})
            if day_r and night_r and comp in day_r["by_component"]:
                dv = day_r["by_component"][comp]["rmse"]
                nv = night_r["by_component"][comp]["rmse"]
                print("{:<16} {:>10.4f} {:>10.4f}".format(abl, dv, nv))

    # Per-level breakdown
    for level in LEVELS:
        print("")
        print("=" * 80)
        print("Level {} RMSE: Day vs Night by Ablation".format(level))
        print("=" * 80)
        header = "{:<16} {:>10} {:>10}".format("Ablation", "Day", "Night")
        print(header)
        print("-" * len(header))
        for abl in ABLATIONS:
            day_r = all_results.get("abl_day_{}".format(abl), {}).get("results", {})
            night_r = all_results.get("abl_night_{}".format(abl), {}).get("results", {})
            if day_r and night_r and level in day_r.get("by_level", {}):
                dv = day_r["by_level"][level]["rmse"]
                nv = night_r["by_level"][level]["rmse"]
                print("{:<16} {:>10.4f} {:>10.4f}".format(abl, dv, nv))

    # Save combined
    combined_path = os.path.join(args.results_dir, "all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nCombined summary saved: {}".format(combined_path))


if __name__ == "__main__":
    main()
