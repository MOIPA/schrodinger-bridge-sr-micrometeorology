#!/usr/bin/env python3
"""
昼夜分组的测试集评估
=====================
按 swdown（向下短波辐射）将测试集分为白天/夜间两组，
分别计算模型在两组的 RMSE/SSIM/Corr，输出对比表。

用法:
  python scripts/eval_day_night.py \
    --config_path configs/config_wind_3d_final.yml \
    --checkpoint_path data/DL_result/.../checkpoint.pth \
    --device cuda:0
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

import argparse
import copy
import json
import time
from logging import INFO, StreamHandler, getLogger
from pathlib import Path

import numpy as np
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

# ---- SSIM ----
def _ssim_2d(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    pred = pred.unsqueeze(0).unsqueeze(0).float()
    target = target.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device) / (window_size**2)
    mu_p = F.conv2d(pred, kernel, padding=window_size // 2)
    mu_t = F.conv2d(target, kernel, padding=window_size // 2)
    mu_p_sq, mu_t_sq, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t
    sigma_p_sq = F.conv2d(pred**2, kernel, padding=window_size // 2) - mu_p_sq
    sigma_t_sq = F.conv2d(target**2, kernel, padding=window_size // 2) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, kernel, padding=window_size // 2) - mu_pt
    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean().item()

def compute_ssim_batch(pred, target):
    vals = [_ssim_2d(pred[i], target[i]) for i in range(pred.shape[0])]
    return float(np.mean(vals))

# ---- 按通道计算指标 ----
def compute_metrics(pred, truth, var_names):
    results = {}
    for ch, name in enumerate(var_names):
        p = pred[:, ch]
        t = truth[:, ch]
        mse = torch.mean((p - t) ** 2).item()
        mae = torch.mean(torch.abs(p - t)).item()
        ssim = compute_ssim_batch(p, t)
        pf, tf = p.flatten().numpy(), t.flatten().numpy()
        corr = float(np.corrcoef(pf, tf)[0, 1]) if len(pf) > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0
        bias = (p.mean() - t.mean()).item()
        results[name] = {"rmse": np.sqrt(mse), "mae": mae, "ssim": ssim, "corr": corr, "bias": bias}
    return results

# ---- 汇总 ----
def summarize(results, var_names):
    """按分量 (U/V/W) 和层级汇总"""
    summary = {}
    for label, keys in [
        ("U", [v for v in var_names if "_u_" in v]),
        ("V", [v for v in var_names if "_v_" in v]),
        ("W", [v for v in var_names if "_w_" in v]),
    ]:
        summary[label] = {
            "rmse": float(np.mean([results[k]["rmse"] for k in keys])),
            "ssim": float(np.mean([results[k]["ssim"] for k in keys])),
            "corr": float(np.mean([results[k]["corr"] for k in keys])),
            "mae":  float(np.mean([results[k]["mae"] for k in keys])),
        }
    for label, keys in [
        ("ml0",  [v for v in var_names if "ml0" in v]),
        ("ml1",  [v for v in var_names if "ml1" in v]),
        ("ml2",  [v for v in var_names if "ml2" in v]),
        ("ml3",  [v for v in var_names if "ml3" in v]),
        ("ml5",  [v for v in var_names if "ml5" in v]),
        ("ml10", [v for v in var_names if "ml10" in v]),
    ]:
        summary[label] = {
            "rmse": float(np.mean([results[k]["rmse"] for k in keys])),
            "ssim": float(np.mean([results[k]["ssim"] for k in keys])),
        }
    return summary

# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="results/eval_day_night")
    parser.add_argument("--swdown_threshold", type=float, default=50.0,
                        help="swdown > threshold 为白天，< threshold 为夜间 (W/m²)")
    parser.add_argument("--experiment_name", type=str, default="ExperimentSchrodingerBridge3dWind")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("昼夜分组评估")
    logger.info(f"swdown 阈值: {args.swdown_threshold} W/m²")
    logger.info(f"Config:  {args.config_path}")
    logger.info(f"Device:  {args.device}")

    # ---- 加载配置 & 模型 ----
    config = load_config(args.experiment_name, args.config_path)
    target_names = config.data.target_variable_names
    input_names = config.data.input_variable_names

    # swdown 在输入条件变量中的位置
    swdown_cond_idx = input_names.index("swdown")
    # y_cond 的通道布局: [LR wind (18ch)] + [cond vars (9ch)]
    n_lr = len(target_names)  # 18
    swdown_channel = n_lr + swdown_cond_idx
    logger.info(f"swdown 位于 y_cond 第 {swdown_channel} 通道 (LR={n_lr}ch + cond_idx={swdown_cond_idx})")

    config_infer = copy.deepcopy(config)
    config_infer.data.hr_cropped_shape = config_infer.data.hr_data_shape

    model = make_model(config_infer.model).to(args.device)
    ckpt = torch.load(args.checkpoint_path, map_location=args.device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt)
        logger.info("Loaded checkpoint (raw state_dict)")
    model.eval()
    si = StochasticInterpolantFollmer(config=config_infer.si, neural_net=model)

    # ---- 加载测试集 ----
    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT_DIR, loader_config=config_infer.loader,
        dataset_config=config_infer.data,
        world_size=None, rank=None,
        train_valid_test_kinds=["test"],
    )
    dataset = dict_loaders["test"].dataset
    n_total = len(dataset)
    logger.info(f"测试集样本数: {n_total}")

    # ---- 收集数据 ----
    set_seeds(42)
    all_y0, all_y1, all_ycond = [], [], []
    swdown_vals = []  # 每个样本的 swdown 均值（归一化后的值）
    for i in range(n_total):
        sample = dataset[i]
        all_y0.append(sample["y0"])
        all_y1.append(sample["y"])
        all_ycond.append(sample["x"])
        # 记录 swdown 均值（在归一化的 y_cond 上）
        swdown_vals.append(sample["x"][swdown_channel].mean().item())

    y0 = torch.stack(all_y0)
    y1 = torch.stack(all_y1)
    y_cond = torch.stack(all_ycond)
    swdown_vals = np.array(swdown_vals)
    del all_y0, all_y1, all_ycond

    logger.info(f"swdown 范围: [{swdown_vals.min():.4f}, {swdown_vals.max():.4f}]")

    # ---- 推理 ----
    logger.info("Running inference...")
    t0 = time.time()
    with torch.no_grad():
        pred_scaled, _ = si.sample_y1_bare_diffusion(
            y0=y0.to(args.device), y_cond=y_cond.to(args.device),
            n_return_step=None, hide_progress_bar=False,
        )
    pred_scaled = pred_scaled.detach().cpu().float()
    y1 = y1.float()
    elapsed = time.time() - t0
    logger.info(f"Inference done in {elapsed:.1f}s ({elapsed/n_total:.2f}s per sample)")

    # ---- 反标准化到物理单位 ----
    # swdown 的反标准化参数
    swdown_bias = config_infer.data.biases["swdown"]
    swdown_scale = config_infer.data.scales["swdown"]
    swdown_phys = swdown_vals * swdown_scale + swdown_bias
    logger.info(f"swdown 物理值范围: [{swdown_phys.min():.1f}, {swdown_phys.max():.1f}] W/m²")

    pred_phys = torch.zeros_like(pred_scaled)
    y1_phys = torch.zeros_like(y1)
    for ch, var_name in enumerate(target_names):
        pred_phys[:, ch] = dataset._scale_inversely(pred_scaled[:, ch], var_name)
        y1_phys[:, ch] = dataset._scale_inversely(y1[:, ch], var_name)

    # ---- 昼夜分组 ----
    day_mask = swdown_phys > args.swdown_threshold
    night_mask = swdown_phys < 5.0  # 严格夜间：swdown 近乎为 0
    twilight_mask = ~(day_mask | night_mask)  # 过渡时段（不纳入对比）

    n_day = day_mask.sum()
    n_night = night_mask.sum()
    n_twilight = twilight_mask.sum()
    logger.info(f"白天样本: {n_day}, 夜间样本: {n_night}, 过渡时段: {n_twilight}")

    # ---- 分昼夜计算指标 ----
    results = {}
    for label, mask in [("day", day_mask), ("night", night_mask)]:
        if mask.sum() == 0:
            logger.warning(f"{label} 无样本，跳过")
            continue
        idx = np.where(mask)[0]
        logger.info(f"计算 {label} 组 ({len(idx)} samples)...")
        metrics = compute_metrics(pred_phys[idx], y1_phys[idx], target_names)
        summary = summarize(metrics, target_names)
        results[label] = {"n_samples": int(len(idx)), "per_channel": metrics, "summary": summary}

    # ---- 输出 ----
    print("\n" + "=" * 80)
    print("  昼夜分组评估结果")
    print("=" * 80)
    print(f"  白天 (swdown > {args.swdown_threshold} W/m²): {n_day} 样本")
    print(f"  夜间 (swdown < 5 W/m²):                   {n_night} 样本")
    print(f"  过渡时段:                                  {n_twilight} 样本")

    for label in ["day", "night"]:
        if label not in results:
            continue
        r = results[label]
        s = r["summary"]
        print(f"\n--- {label.upper()} ({r['n_samples']} 样本) ---")
        print(f"  {'分量':<6} {'RMSE':>8} {'MAE':>8} {'SSIM':>8} {'Corr':>8}")
        for comp in ["U", "V", "W"]:
            print(f"  {comp:<6} {s[comp]['rmse']:>8.4f} {s[comp]['mae']:>8.4f} {s[comp]['ssim']:>8.4f} {s[comp]['corr']:>8.4f}")
        print(f"  {'层':<6} {'RMSE':>8} {'SSIM':>8}")
        for lvl in ["ml0", "ml1", "ml2", "ml3", "ml5", "ml10"]:
            print(f"  {lvl:<6} {s[lvl]['rmse']:>8.4f} {s[lvl]['ssim']:>8.4f}")

    # 昼夜差异
    if "day" in results and "night" in results:
        print(f"\n--- 昼夜差异 (Day - Night) ---")
        print(f"  {'分量':<6} {'ΔRMSE':>10} {'ΔSSIM':>10}")
        for comp in ["U", "V", "W"]:
            dr = results["day"]["summary"][comp]["rmse"] - results["night"]["summary"][comp]["rmse"]
            ds = results["day"]["summary"][comp]["ssim"] - results["night"]["summary"][comp]["ssim"]
            print(f"  {comp:<6} {dr:>+10.4f} {ds:>+10.4f}")

    # ---- 保存 ----
    # 只保存 summary（去掉逐通道细节减小文件体积）
    save_data = {
        "swdown_threshold": args.swdown_threshold,
        "day_samples": int(n_day),
        "night_samples": int(n_night),
        "twilight_samples": int(n_twilight),
        "results": {k: {"n_samples": v["n_samples"], "summary": v["summary"]}
                     for k, v in results.items()},
    }
    with open(out / "day_night_summary.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存到 {out}/day_night_summary.json")

    print("\n" + "=" * 80)
    print("  评估完成!")


if __name__ == "__main__":
    main()
