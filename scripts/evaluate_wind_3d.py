#!/usr/bin/env python3
"""
3D Wind Field Super-Resolution — Evaluation Script
===================================================
在 test 集上评估模型，同时与 bicubic baseline 对比。
输出逐通道 RMSE / MAE / SSIM，按风分量和层级汇总。

用法:
  python scripts/evaluate_wind_3d.py \
    --config_path configs/config_wind_3d.yml \
    --checkpoint_path data/DL_result/ExperimentSchrodingerBridge3dWind/config_wind_3d/checkpoint.pth \
    --device cpu \
    --output_dir results/eval_3d_wind
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

import argparse
import copy
import gc
import json
import os
import time
from logging import INFO, FileHandler, StreamHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.model_maker import make_model
from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer
from src.utils.random_seed_helper import set_seeds

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.resolve())

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)


# ==============================================================
# SSIM（不依赖第三方库，纯 PyTorch 实现）
# ==============================================================
def _ssim_2d(pred: torch.Tensor, target: torch.Tensor,
             window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """计算两张单通道 2D 图的 SSIM（取值 -1~1，越高越好）。"""
    # pred, target: [H, W]
    pred = pred.unsqueeze(0).unsqueeze(0).float()   # [1,1,H,W]
    target = target.unsqueeze(0).unsqueeze(0).float()

    # Gaussian-like uniform window
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device) / (window_size ** 2)

    mu_p = F.conv2d(pred, kernel, padding=window_size // 2)
    mu_t = F.conv2d(target, kernel, padding=window_size // 2)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(pred ** 2, kernel, padding=window_size // 2) - mu_p_sq
    sigma_t_sq = F.conv2d(target ** 2, kernel, padding=window_size // 2) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, kernel, padding=window_size // 2) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))

    return ssim_map.mean().item()


def compute_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 batch 中所有样本的平均 SSIM。pred/target: [N, H, W]"""
    ssim_vals = []
    for i in range(pred.shape[0]):
        ssim_vals.append(_ssim_2d(pred[i], target[i]))
    return float(np.mean(ssim_vals))


# ==============================================================
# 指标计算
# ==============================================================
def compute_metrics_per_channel(pred: torch.Tensor, truth: torch.Tensor,
                                var_names: list) -> dict:
    """
    逐通道计算指标。
    pred, truth: [N, C, H, W]  (物理单位)
    返回 dict: {var_name: {rmse, mae, ssim, corr, bias, ...}}
    """
    results = {}
    for ch, name in enumerate(var_names):
        p = pred[:, ch]   # [N, H, W]
        t = truth[:, ch]

        mse = torch.mean((p - t) ** 2).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(p - t)).item()

        # SSIM
        ssim = compute_ssim_batch(p, t)

        # Pearson correlation (flatten all pixels)
        pf = p.flatten().numpy()
        tf = t.flatten().numpy()
        corr = float(np.corrcoef(pf, tf)[0, 1]) if len(pf) > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0

        # bias & std
        bias = (p.mean() - t.mean()).item()
        pred_std = p.std().item()
        truth_std = t.std().item()

        results[name] = dict(
            rmse=rmse, mae=mae, ssim=ssim, corr=corr,
            bias=bias, pred_std=pred_std, truth_std=truth_std,
        )
    return results


# ==============================================================
# Bicubic Baseline
# ==============================================================
def make_bicubic_baseline(dataset, target_names, device="cpu") -> torch.Tensor:
    """
    从 dataset 中取出所有 LR 数据（y0），反标准化后当作 bicubic baseline。
    y0 本身就是 HR→avgpool→bicubic↑ 得到的，尺寸和 HR 一致，是最直接的 baseline。
    返回 [N, C, H, W] 物理单位。
    """
    lr_names = [n.replace("hr_", "lr_") for n in target_names]
    all_lr = []
    for i in range(len(dataset)):
        sample = dataset[i]
        y0 = sample["y0"]  # [C, H, W] normalized
        # 逐通道反标准化
        y0_phys = torch.zeros_like(y0)
        for ch, (hr_name, lr_name) in enumerate(zip(target_names, lr_names)):
            y0_phys[ch] = dataset._scale_inversely(y0[ch], lr_name)
        all_lr.append(y0_phys)
    return torch.stack(all_lr, dim=0)


# ==============================================================
# 汇总打印
# ==============================================================
def group_metrics(per_var: dict, var_names: list):
    """按风分量 (U/V/W) 和层级 (ml0..ml10) 分组汇总。"""
    groups = {}

    # 按分量
    for comp in ["u", "v", "w"]:
        keys = [v for v in var_names if f"_{comp}_" in v]
        if keys:
            groups[f"Component {comp.upper()}"] = keys

    # 按层级
    for lvl in ["ml0", "ml1", "ml2", "ml3", "ml5", "ml10"]:
        keys = [v for v in var_names if lvl in v]
        if keys:
            groups[f"Level {lvl}"] = keys

    return groups


def print_results_table(model_metrics: dict, baseline_metrics: dict,
                        var_names: list, title: str = ""):
    """打印完整对比表。"""
    sep = "=" * 130
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    header = f"{'Variable':<14} | {'Model RMSE':>11} {'Bicubic':>11} | {'Model MAE':>11} {'Bicubic':>11} | {'Model SSIM':>11} {'Bicubic':>11} | {'Model Corr':>11} {'Bicubic':>11}"
    print(header)
    print("-" * 130)

    for name in var_names:
        m = model_metrics[name]
        b = baseline_metrics[name]
        print(f"{name:<14} | {m['rmse']:>11.4f} {b['rmse']:>11.4f} | "
              f"{m['mae']:>11.4f} {b['mae']:>11.4f} | "
              f"{m['ssim']:>11.4f} {b['ssim']:>11.4f} | "
              f"{m['corr']:>11.4f} {b['corr']:>11.4f}")

    # 分组汇总
    groups = group_metrics(model_metrics, var_names)
    print("-" * 130)
    print("  分组汇总 (mean±std)")
    print("-" * 130)

    for group_name, keys in groups.items():
        m_rmse = [model_metrics[k]["rmse"] for k in keys]
        b_rmse = [baseline_metrics[k]["rmse"] for k in keys]
        m_ssim = [model_metrics[k]["ssim"] for k in keys]
        b_ssim = [baseline_metrics[k]["ssim"] for k in keys]
        m_corr = [model_metrics[k]["corr"] for k in keys]

        print(f"  {group_name:<20}: "
              f"RMSE {np.mean(m_rmse):.4f}±{np.std(m_rmse):.4f} "
              f"(bicubic {np.mean(b_rmse):.4f}±{np.std(b_rmse):.4f}) | "
              f"SSIM {np.mean(m_ssim):.4f}±{np.std(m_ssim):.4f} "
              f"(bicubic {np.mean(b_ssim):.4f}±{np.std(b_ssim):.4f}) | "
              f"Corr {np.mean(m_corr):.4f}")

    print(sep + "\n")


def save_results(model_metrics: dict, baseline_metrics: dict,
                 var_names: list, output_dir: str):
    """保存 JSON + CSV。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON (完整)
    combined = {
        "model": model_metrics,
        "bicubic_baseline": baseline_metrics,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # CSV (便于粘贴到论文 / PPT)
    rows = []
    for name in var_names:
        m = model_metrics[name]
        b = baseline_metrics[name]
        rows.append({
            "variable": name,
            "model_rmse": m["rmse"], "bicubic_rmse": b["rmse"],
            "model_mae": m["mae"], "bicubic_mae": b["mae"],
            "model_ssim": m["ssim"], "bicubic_ssim": b["ssim"],
            "model_corr": m["corr"], "bicubic_corr": b["corr"],
            "model_bias": m["bias"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    logger.info(f"Results saved to {out}/metrics.json and metrics.csv")


# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D wind super-resolution model")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of test samples to use (-1 = all)")
    parser.add_argument("--output_dir", type=str, default="results/eval_3d_wind")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"])
    parser.add_argument("--experiment_name", type=str,
                        default="ExperimentSchrodingerBridge3dWind")
    args = parser.parse_args()

    # ---- Setup ----
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.addHandler(FileHandler(f"{args.output_dir}/evaluation.log"))

    logger.info("=" * 60)
    logger.info("3D Wind Field Super-Resolution — Evaluation")
    logger.info("=" * 60)
    logger.info(f"Config:     {args.config_path}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Device:     {args.device}")
    logger.info(f"Split:      {args.split}")

    # ---- Config ----
    config = load_config(args.experiment_name, args.config_path)
    target_names = config.data.target_variable_names
    logger.info(f"Target variables ({len(target_names)}): {target_names}")

    # ---- Model ----
    config_infer = copy.deepcopy(config)
    config_infer.data.hr_cropped_shape = config_infer.data.hr_data_shape

    model = make_model(config_infer.model).to(args.device)
    ckpt = torch.load(args.checkpoint_path, map_location=args.device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
                     f"best_loss {ckpt.get('best_loss', '?')})")
    else:
        model.load_state_dict(ckpt)
        logger.info("Loaded checkpoint (raw state_dict)")
    model.eval()

    si = StochasticInterpolantFollmer(config=config_infer.si, neural_net=model)

    # ---- Dataset ----
    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT_DIR,
        loader_config=config_infer.loader,
        dataset_config=config_infer.data,
        world_size=None, rank=None,
        train_valid_test_kinds=[args.split],
    )
    dataset = dict_loaders[args.split].dataset
    n_total = len(dataset)
    n_use = n_total if args.num_samples <= 0 else min(args.num_samples, n_total)
    logger.info(f"Dataset: {n_total} samples, using {n_use}")

    # ---- Collect data ----
    set_seeds(42)
    indices = np.random.choice(n_total, n_use, replace=False) if n_use < n_total else np.arange(n_total)

    all_y0, all_y1, all_ycond = [], [], []
    for i in indices:
        sample = dataset[i]
        all_y0.append(sample["y0"])
        all_y1.append(sample["y"])
        all_ycond.append(sample["x"])

    y0 = torch.stack(all_y0)       # [N, 18, H, W] normalized
    y1 = torch.stack(all_y1)       # [N, 18, H, W] normalized
    y_cond = torch.stack(all_ycond) # [N, 10, H, W] normalized
    del all_y0, all_y1, all_ycond

    logger.info(f"y0={y0.shape}, y1={y1.shape}, y_cond={y_cond.shape}")

    # ---- Model Inference ----
    logger.info("Running model inference...")
    t0 = time.time()

    with torch.no_grad():
        pred_scaled, _ = si.sample_y1_bare_diffusion(
            y0=y0.to(args.device),
            y_cond=y_cond.to(args.device),
            n_return_step=None,
            hide_progress_bar=False,
        )

    pred_scaled = pred_scaled.detach().cpu().float()
    y1 = y1.float()
    y0 = y0.float()

    elapsed = time.time() - t0
    logger.info(f"Inference done in {elapsed:.1f}s ({elapsed/n_use:.2f}s per sample)")

    # ---- Denormalize to physical units ----
    logger.info("Denormalizing to physical units...")
    pred_phys = torch.zeros_like(pred_scaled)
    y1_phys = torch.zeros_like(y1)

    lr_names = [n.replace("hr_", "lr_") for n in target_names]
    y0_phys = torch.zeros_like(y0)

    for ch, var_name in enumerate(target_names):
        pred_phys[:, ch] = dataset._scale_inversely(pred_scaled[:, ch], var_name)
        y1_phys[:, ch] = dataset._scale_inversely(y1[:, ch], var_name)
        y0_phys[:, ch] = dataset._scale_inversely(y0[:, ch], lr_names[ch])

    # ---- Compute Metrics ----
    logger.info("Computing model metrics...")
    model_metrics = compute_metrics_per_channel(pred_phys, y1_phys, target_names)

    logger.info("Computing bicubic baseline metrics...")
    baseline_metrics = compute_metrics_per_channel(y0_phys, y1_phys, target_names)

    # ---- Print & Save ----
    print_results_table(model_metrics, baseline_metrics, target_names,
                        title="Model vs Bicubic Baseline — Test Set Evaluation")
    save_results(model_metrics, baseline_metrics, target_names, args.output_dir)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
