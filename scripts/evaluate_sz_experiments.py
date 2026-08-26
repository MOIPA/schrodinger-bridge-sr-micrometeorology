"""
Evaluate all Shenzhen experiments: baseline / lrcond / day / night / pinn.
=======================================================================
For full-data models (baseline, lrcond, pinn): evaluate on full test set
plus day / night subsets (to compare against the day/night models).
For day/night models: evaluate on their own test subset.

Output per-experiment JSON + combined summary JSON + comparison tables.

用法:
  python scripts/evaluate_sz_experiments.py --device cuda:0 \
      --results_dir results/sz_eval
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.model_maker import make_model
from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer

EXPERIMENT_NAME = "ExperimentSchrodingerBridge3dWind"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_SUBDIR = "深圳"
EXPERIMENTS = ["baseline", "lrcond", "day", "night", "pinn"]
FULL_MODELS = ["baseline", "lrcond", "pinn"]  # 全量模型:额外评估 day/night 子集
FILTERS = ["all", "day", "night"]
COMPONENTS = ["U", "V", "W"]


# ==============================================================
# 指标计算(与 evaluate_ablation_day_night.py 一致,结果可比)
# ==============================================================
def compute_metrics(pred, target):
    """逐通道 RMSE, MAE, SSIM, Corr, Bias。pred/target: [N, C, H, W]"""
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

        mu_p, mu_t = np.mean(p), np.mean(t)
        sigma_p, sigma_t = np.std(p), np.std(t)
        C1, C2 = 1e-4, 9e-4
        numerator = (2 * mu_p * mu_t + C1) * (2 * sigma_p * sigma_t + C2)
        denominator = (mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2)
        ssim[c] = float(numerator / (denominator + 1e-8))

        p_flat, t_flat = p.flatten(), t.flatten()
        p_mean, t_mean = np.mean(p_flat), np.mean(t_flat)
        num = np.sum((p_flat - p_mean) * (t_flat - t_mean))
        den = np.sqrt(np.sum((p_flat - p_mean)**2) * np.sum((t_flat - t_mean)**2))
        corr[c] = float(num / (den + 1e-8))

    return rmse, mae, ssim, corr, bias


def summarize_by_component(metrics_per_ch, target_names):
    """按 U/V/W 分量聚合。"""
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


# ==============================================================
# 单个 checkpoint 评估
# ==============================================================
def evaluate_checkpoint(config_path, checkpoint_path, device, eval_filter):
    """加载模型,在 eval_filter(all/day/night) 过滤的 test 集上推理评估。"""
    config = load_config(EXPERIMENT_NAME, config_path)
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
    n_test = len(test_loader.dataset)
    print("  Test samples: {}".format(n_test))

    model = make_model(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    si_follmer = StochasticInterpolantFollmer(config=config.si, neural_net=model)

    all_pred = []
    all_target = []
    for batch_data in tqdm(test_loader, desc="  Inference"):
        y_cond = batch_data["x"].to(device)
        y0 = batch_data["y0"].to(device)
        with torch.no_grad():
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
    }


# ==============================================================
# 对比表打印
# ==============================================================
def print_table(all_results):
    print("")
    print("=" * 90)
    print("SHENZHEN EVALUATION — Overall Metrics by Experiment x Filter")
    print("=" * 90)
    header = "{:<16} {:>8} {:>10} {:>10} {:>10} {:>10}".format(
        "Exp (filter)", "n", "RMSE", "MAE", "SSIM", "Corr")
    print(header)
    print("-" * len(header))
    for key in sorted(all_results.keys()):
        r = all_results[key]["results"]
        o = r["by_component"]["Overall"]
        print("{:<16} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            key, r["n_samples"], o["rmse"], o["mae"], o["ssim"], o["corr"]))

    # 分量分解(全量评估)
    print("")
    print("=" * 90)
    print("Per-Component RMSE / SSIM (full-test evaluations)")
    print("=" * 90)
    header = "{:<16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Exp", "U RMSE", "V RMSE", "W RMSE", "U SSIM", "V SSIM", "W SSIM")
    print(header)
    print("-" * len(header))
    for exp in ["baseline", "lrcond", "pinn"]:
        r = all_results.get(exp, {})
        if not r:
            continue
        bc = r["results"]["by_component"]
        print("{:<16} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            exp, bc["U"]["rmse"], bc["V"]["rmse"], bc["W"]["rmse"],
            bc["U"]["ssim"], bc["V"]["ssim"], bc["W"]["ssim"]))

    # 关键对比 1: day/night 子集上 baseline vs 专门模型
    print("")
    print("=" * 90)
    print("KEY COMPARISON 1: Day/Night subset — baseline vs dedicated model")
    print("=" * 90)
    for filt, model in [("day", "day"), ("night", "night")]:
        base_key = "baseline_{}".format(filt)
        ded_key = model
        if base_key not in all_results or ded_key not in all_results:
            continue
        base_o = all_results[base_key]["results"]["by_component"]["Overall"]
        ded_o = all_results[ded_key]["results"]["by_component"]["Overall"]
        print("[{} subset]".format(filt.upper()))
        print("  baseline(全量训练): RMSE {:.4f}  SSIM {:.4f}  Corr {:.4f}".format(
            base_o["rmse"], base_o["ssim"], base_o["corr"]))
        print("  {}模型(专用训练): RMSE {:.4f}  SSIM {:.4f}  Corr {:.4f}".format(
            filt, ded_o["rmse"], ded_o["ssim"], ded_o["corr"]))
        print("  delta RMSE: {:+.4f}  ({})".format(
            ded_o["rmse"] - base_o["rmse"],
            "专用模型更好" if ded_o["rmse"] < base_o["rmse"] else "baseline 更好"))

    # 关键对比 2: 全量 test 上 baseline vs lrcond vs pinn
    print("")
    print("=" * 90)
    print("KEY COMPARISON 2: Full test — baseline vs lrcond vs pinn")
    print("=" * 90)
    ref = all_results.get("baseline", {}).get("results", {}).get("by_component", {})
    if ref:
        ref_o = ref["Overall"]
        print("  baseline: RMSE {:.4f}  SSIM {:.4f}  Corr {:.4f}".format(
            ref_o["rmse"], ref_o["ssim"], ref_o["corr"]))
        for exp in ["lrcond", "pinn"]:
            r = all_results.get(exp, {}).get("results", {}).get("by_component", {})
            if not r:
                continue
            o = r["Overall"]
            print("  {:8s}: RMSE {:.4f} ({:+.4f})  SSIM {:.4f} ({:+.4f})  Corr {:.4f}".format(
                exp, o["rmse"], o["rmse"] - ref_o["rmse"],
                o["ssim"], o["ssim"] - ref_o["ssim"], o["corr"]))
    print("")
    print("=" * 90)


# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Shenzhen experiments")
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(ROOT_DIR, "results", "sz_eval"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_base_dir", type=str,
                        default=os.path.join(ROOT_DIR, "data", "DL_result",
                                             "ExperimentSchrodingerBridge3dWind"))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    all_results = {}

    for exp in EXPERIMENTS:
        filters = FILTERS if exp in FULL_MODELS else [exp]
        config_path = os.path.join(ROOT_DIR, "configs", CONFIG_SUBDIR,
                                   "config_wind_3d_sz_{}.yml".format(exp))
        checkpoint_path = os.path.join(args.checkpoint_base_dir,
                                       "config_wind_3d_sz_{}".format(exp),
                                       "checkpoint.pth")

        if not os.path.exists(checkpoint_path):
            print("[SKIP] Checkpoint not found: {}".format(checkpoint_path))
            continue

        for filt in filters:
            key = "{}_{}".format(exp, filt) if filt != "all" else exp
            print("")
            print("=" * 60)
            print("Evaluating: {} (filter={})".format(exp, filt))
            print("  Config:     {}".format(config_path))
            print("  Checkpoint: {}".format(checkpoint_path))
            print("=" * 60)

            results = evaluate_checkpoint(config_path, checkpoint_path,
                                          args.device, filt)

            all_results[key] = {
                "experiment": exp,
                "filter": filt,
                "config": config_path,
                "checkpoint": checkpoint_path,
                "results": results,
            }
            out_path = os.path.join(args.results_dir, "{}_metrics.json".format(key))
            with open(out_path, "w") as f:
                json.dump(all_results[key], f, indent=2)
            print("  Saved: {}".format(out_path))

    print_table(all_results)

    combined_path = os.path.join(args.results_dir, "all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Combined summary saved: {}".format(combined_path))
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
