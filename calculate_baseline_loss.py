import sys
import os
import pathlib
import argparse
import random
import glob
import numpy as np
import torch
from tqdm import tqdm

def calculate_baseline_loss_for_sample(npz_path):
    """
    Calculates the loss between the interpolated LR image and the HR ground truth.
    Returns RMSE and MAE.
    """
    try:
        # --- 1. 数据加载 ---
        with np.load(npz_path) as data:
            # 加载插值后的低分图和高分真实图
            hr_tm002m_np = data['hr_tm002m']
            lr_tm002m_np = data['lr_tm002m']
        
        # --- 2. 转换为PyTorch张量以便计算 ---
        hr_truth_tensor = torch.from_numpy(hr_tm002m_np).float()
        interp_lr_tensor = torch.from_numpy(lr_tm002m_np).float()

        # --- 3. 计算损失 ---
        # 确保两个张量的形状一致
        if hr_truth_tensor.shape != interp_lr_tensor.shape:
            raise ValueError(f"Shape mismatch: HR is {hr_truth_tensor.shape}, Interpolated LR is {interp_lr_tensor.shape}")

        # 计算RMSE (均方根误差)
        rmse_loss = torch.sqrt(torch.mean((interp_lr_tensor - hr_truth_tensor) ** 2))
        # 计算MAE (平均绝对误差)
        mae_loss = torch.mean(torch.abs(interp_lr_tensor - hr_truth_tensor))

        return rmse_loss.item(), mae_loss.item()

    except Exception as e:
        print(f"Failed to calculate baseline loss for {os.path.basename(npz_path)}. Error: {e}")
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Evaluation Script (Interpolation vs. Ground Truth)")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing the .npz files for evaluation.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of random samples to evaluate. Default is all files.")
    
    args = parser.parse_args()
    
    # --- 1. 查找和选择文件 ---
    all_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if not all_files:
        print(f"Error: No .npz files found in '{args.npz_dir}'.")
        sys.exit(1)
    
    if args.num_samples is not None:
        random.shuffle(all_files)
        files_to_process = all_files[:min(args.num_samples, len(all_files))]
    else:
        files_to_process = all_files

    print(f"Found {len(all_files)} files. Will evaluate {len(files_to_process)} of them for baseline loss.")

    # --- 2. 循环计算损失 ---
    rmse_losses = []
    mae_losses = []
    for npz_file in tqdm(files_to_process, desc="Calculating Baseline Losses"):
        rmse, mae = calculate_baseline_loss_for_sample(npz_file)
        if rmse is not None and mae is not None:
            # print(f"  File: {os.path.basename(npz_file):<25} | Baseline RMSE: {rmse:.4f} K | Baseline MAE: {mae:.4f} K")
            rmse_losses.append(rmse)
            mae_losses.append(mae)

    # --- 3. 打印最终总结 ---
    if rmse_losses:
        avg_rmse = np.mean(rmse_losses)
        avg_mae = np.mean(mae_losses)
        print("" + "="*60)
        print("---           Baseline Evaluation Results (Interpolation)          ---")
        print(f"  Processed Samples: {len(rmse_losses)}")
        print(f"  Average Baseline RMSE:      {avg_rmse:.4f} K")
        print(f"  Average Baseline MAE:       {avg_mae:.4f} K")
        print("="*60)

    print("Baseline loss calculation complete.")
