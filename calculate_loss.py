import sys
import os
import pathlib
import argparse
import random
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to the Python path
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from src.dl_config.config_loader import load_config
from src.dl_model.model_maker import make_model
from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer
from src.utils.random_seed_helper import set_seeds

def calculate_sample_loss(config, net, si, npz_path):
    """
    Performs inference on a single .npz file and calculates the loss.
    Returns RMSE and MAE.
    """
    device = next(net.parameters()).device
    try:
        # --- 1. 数据加载 ---
        with np.load(npz_path) as data:
            hr_tm002m_torch = torch.from_numpy(data['hr_tm002m']).float()
            lr_tm002m_torch = torch.from_numpy(data['lr_tm002m']).float()
            conditional_vars = config.data.input_variable_names
            x_tensors = [torch.from_numpy(data[var]).float() for var in conditional_vars]

        # --- 2. 归一化与张量准备 ---
        def _get_var_name_for_norm(name: str):
            key = name[:5]
            if '_' in name: key = name
            if name == 'lr_tm002m': key = 'lr_tm'
            return key

        lr_key = _get_var_name_for_norm('lr_tm002m')
        lr_tm002m_norm = (lr_tm002m_torch - config.data.biases.get(lr_key, 0.0)) / config.data.scales.get(lr_key, 1.0)

        x_tensors_normalized = []
        for i, var_name in enumerate(conditional_vars):
            key = _get_var_name_for_norm(var_name)
            bias = config.data.biases.get(key, 0.0)
            scale = config.data.scales.get(key, 1.0)
            x_tensors_normalized.append((x_tensors[i] - bias) / scale)
        
        y0 = lr_tm002m_norm.unsqueeze(0).unsqueeze(0)
        x = torch.stack(x_tensors_normalized, dim=0).unsqueeze(0)

        # --- 3. 缩放与推理 ---
        target_shape = config.data.hr_data_shape
        y0_resized = F.interpolate(y0, size=target_shape, mode='bicubic', align_corners=False)
        x_resized = F.interpolate(x, size=target_shape, mode='bicubic', align_corners=False)
        y0_resized = torch.nan_to_num(y0_resized, nan=0.0)
        x_resized = torch.nan_to_num(x_resized, nan=0.0)
        
        with torch.no_grad():
            pred_hr_normalized, _ = si.sample_y1_bare_diffusion(y0=y0_resized.to(device), y_cond=x_resized.to(device))
        
        # --- 4. 逆归一化 ---
        hr_bias = config.data.biases.get('hr_tm', 0.0)
        hr_scale = config.data.scales.get('hr_tm', 1.0)
        pred_hr_physical = pred_hr_normalized.cpu() * hr_scale + hr_bias
        
        # --- 5. 计算损失 ---
        # 确保真实值也是torch tensor
        hr_truth_tensor = hr_tm002m_torch.unsqueeze(0).unsqueeze(0)
        # 将预测结果插值回原始尺寸以进行比较
        pred_resized_for_loss = F.interpolate(pred_hr_physical, size=hr_truth_tensor.shape[-2:], mode='bicubic', align_corners=False)

        # 计算RMSE
        rmse_loss = torch.sqrt(torch.mean((pred_resized_for_loss - hr_truth_tensor) ** 2))
        # 计算MAE
        mae_loss = torch.mean(torch.abs(pred_resized_for_loss - hr_truth_tensor))

        return rmse_loss.item(), mae_loss.item()

    except Exception as e:
        import traceback
        print(f"Failed to calculate loss for {os.path.basename(npz_path)}. Error: {e}")
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantitative Evaluation Script (calculates loss)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--model_weight_path", type=str, required=True, help="Path to the trained model file (checkpoint or state_dict).")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing the .npz files for evaluation.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of random samples to evaluate. Default is all files.")
    
    args = parser.parse_args()
    
    # --- 1. 加载模型 ---
    print("*** Initializing Model and Framework ***")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config("ExperimentSchrodingerBridgeModel", args.config_path)
    set_seeds(config.train.seed)
    
    loaded_object = torch.load(args.model_weight_path, map_location=device)
    net = make_model(config.model).to(device)
    
    if isinstance(loaded_object, dict) and 'model_state_dict' in loaded_object:
        net.load_state_dict(loaded_object['model_state_dict'])
        print("Loaded model from new checkpoint format.")
    else:
        net.load_state_dict(loaded_object)
        print("Loaded model from old state_dict format.")
    
    net.eval()
    si = StochasticInterpolantFollmer(config=config.si, neural_net=net).to(device)
    print(f"Using device: {device}")

    # --- 2. 查找和选择文件 ---
    all_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if not all_files:
        print(f"Error: No .npz files found in '{args.npz_dir}'.")
        sys.exit(1)
    
    if args.num_samples is not None:
        random.shuffle(all_files)
        files_to_process = all_files[:min(args.num_samples, len(all_files))]
    else:
        files_to_process = all_files

    print(f"Found {len(all_files)} files. Will evaluate {len(files_to_process)} of them.")

    # --- 3. 循环计算损失 ---
    rmse_losses = []
    mae_losses = []
    for npz_file in tqdm(files_to_process, desc="Calculating losses"):
        rmse, mae = calculate_sample_loss(config, net, si, npz_file)
        if rmse is not None and mae is not None:
            print(f"  File: {os.path.basename(npz_file):<25} | RMSE: {rmse:.4f} K | MAE: {mae:.4f} K")
            rmse_losses.append(rmse)
            mae_losses.append(mae)

    # --- 4. 打印最终总结 ---
    if rmse_losses:
        avg_rmse = np.mean(rmse_losses)
        avg_mae = np.mean(mae_losses)
        print("" + "="*50)
        print("---           Overall Evaluation Results           ---")
        print(f"  Processed Samples: {len(rmse_losses)}")
        print(f"  Average RMSE:      {avg_rmse:.4f} K")
        print(f"  Average MAE:       {avg_mae:.4f} K")
        print("="*50)

    print("Loss calculation complete.")
