import sys
import os
import pathlib
import argparse
import random
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import xarray as xr

# Add project root to the Python path
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

# Conditional imports for model-related stuff
try:
    from src.dl_config.config_loader import load_config
    from src.dl_model.model_maker import make_model
    from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer
    from src.utils.random_seed_helper import set_seeds
    MODEL_LIBS_AVAILABLE = True
except ImportError:
    MODEL_LIBS_AVAILABLE = False

def plot_map(ax, data, lon_range, lat_range, title, vmin, vmax):
    """Helper function to plot a single map using imshow."""
    proj = ccrs.PlateCarree()
    ax.set_extent(lon_range + lat_range, crs=proj)

    im = ax.imshow(data, origin='upper', transform=proj, cmap='viridis', 
                   extent=lon_range + lat_range, vmin=vmin, vmax=vmax)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
    
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontsize=14)
    return im

def visualize_data_only(npz_path, output_path):
    """
    Loads and visualizes only the LR and HR data from a .npz file without a model.
    """
    print(f"  -> Visualizing LR/HR data for {os.path.basename(npz_path)}...")
    try:
        with np.load(npz_path) as data:
            hr_plot = data['hr_tm002m'].squeeze()
            lr_plot = data['lr_tm002m'].squeeze()
        
        lon_range = [112.9811, 114.6543]
        lat_range = [21.8413, 23.3858]
        
        fig = plt.figure(figsize=(16, 8))
        proj = ccrs.PlateCarree()
        
        vmin = min(np.nanmin(lr_plot), np.nanmin(hr_plot))
        vmax = max(np.nanmax(lr_plot), np.nanmax(hr_plot))
        
        # Plot LR
        ax1 = fig.add_subplot(1, 2, 1, projection=proj)
        im1 = plot_map(ax1, lr_plot, lon_range, lat_range, f'Low-Resolution Input\n(Shape: {lr_plot.shape})', vmin, vmax)
        ax1.gridlines(draw_labels=True).right_labels = False
        fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1, shrink=0.8)
        
        # Plot HR
        ax2 = fig.add_subplot(1, 2, 2, projection=proj)
        im2 = plot_map(ax2, hr_plot, lon_range, lat_range, f'High-Resolution Ground Truth\n(Shape: {hr_plot.shape})', vmin, vmax)
        ax2.gridlines(draw_labels=True).left_labels = False
        fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1, shrink=0.8)

        plt.tight_layout(pad=3.0)
        plt.savefig(output_path)
        plt.close(fig)

    except Exception as e:
        print(f"Failed to visualize data for {os.path.basename(npz_path)}. Error: {e}")

def visualize_inference(config, net, si, npz_path, output_path):
    """
    Performs full model inference and creates a 2x2 comparison plot.
    Based on the user's trusted working script.
    """
    device = next(net.parameters()).device
    try:
        # --- 1. 数据加载 ---
        with np.load(npz_path) as data:
            hr_tm002m_torch = torch.from_numpy(data['hr_tm002m']).float()
            lr_tm002m_torch = torch.from_numpy(data['lr_tm002m']).float()
            conditional_vars = config.data.input_variable_names
            x_tensors = [torch.from_numpy(data[var]).float() for var in conditional_vars]

        # --- 2. 归一化与张量准备 (来自您的工作代码) ---
        def _get_var_name_for_norm(name: str):
            key = name[:5]
            if '_' in name: key = name
            if name == 'lr_tm002m': key = 'lr_tm'
            return key

        lr_key = _get_var_name_for_norm('lr_tm002m')
        lr_tm002m_norm = (lr_tm002m_torch - config.data.biases[lr_key]) / config.data.scales[lr_key]

        x_tensors_normalized = []
        for i, var_name in enumerate(conditional_vars):
            key = _get_var_name_for_norm(var_name)
            bias = config.data.biases[key]
            scale = config.data.scales[key]
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
            pred_hr, _ = si.sample_y1_bare_diffusion(y0=y0_resized.to(device), y_cond=x_resized.to(device))

        # --- 4. 数据后处理 ---
        hr_bias = config.data.biases.get('hr_tm', 0.0)
        hr_scale = config.data.scales.get('hr_tm', 1.0)
        pred_hr_physical = pred_hr.cpu() * hr_scale + hr_bias
        original_hr_shape = hr_tm002m_torch.shape[-2:]
        pred_plot_resized = F.interpolate(pred_hr_physical, size=original_hr_shape, mode='bicubic', align_corners=False)

        # 准备4个用于绘图的Numpy数组
        hr_plot = hr_tm002m_torch.squeeze().numpy()
        interp_lr_plot = lr_tm002m_torch.squeeze().numpy()
        pred_plot = pred_plot_resized.squeeze().numpy()

        # 【新增】创建原始低分辨率数据 (45x45)
        hr_da = xr.DataArray(hr_plot, dims=("south_north", "west_east"))
        coarse_lr_plot = hr_da.coarsen(south_north=4, west_east=4, boundary="pad").mean().values

        # --- 5. 可视化 (四宫格) ---
        lon_range = [112.9811, 114.6543]
        lat_range = [21.8413, 23.3858]
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()

        plot_info = [
            {'data': coarse_lr_plot, 'title': 'Original Low-Res'},
            {'data': pred_plot, 'title': 'Model Prediction'},
            {'data': interp_lr_plot, 'title': 'Interpolated Low-Res'},
            {'data': hr_plot, 'title': 'High-Res Ground Truth'},
        ]
        plot_order = [0, 1, 2, 3] # 方便对比的绘图顺序

        for i, plot_idx in enumerate(plot_order):
            info = plot_info[plot_idx]
            ax = axes[i]
            vmin = np.nanmin(info['data'])
            vmax = np.nanmax(info['data'])
            title = f"{info['title']}\n(Shape: {info['data'].shape})"
            im = plot_map(ax, info['data'], lon_range, lat_range, title, vmin, vmax)
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)

        # 清理网格线标签
        axes[0].gridlines(draw_labels=True).right_labels = False
        axes[1].gridlines(draw_labels=True).right_labels = False
        axes[2].gridlines(draw_labels=True).right_labels = False
        axes[3].gridlines(draw_labels=True).left_labels = False
        axes[1].gridlines(draw_labels=True).left_labels = False

        plt.suptitle("Super-Resolution Result Comparison (4-Panel)", fontsize=20)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path)
        plt.close(fig)

    except Exception as e:
        import traceback
        print(f"Failed to run full inference for {os.path.basename(npz_path)}. Error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference and Visualization Script with Map Context")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing .npz files.")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save output images.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    parser.add_argument("--data_only", action='store_true', help="If set, only visualize LR/HR data without running the model.")
    parser.add_argument("--config_path", type=str, help="Path to the model config file (required if not --data_only).")
    parser.add_argument("--model_weight_path", type=str, help="Path to the trained checkpoint.pth file (required if not --data_only).")
    
    args = parser.parse_args()
    
    all_files = glob.glob(os.path.join(args.npz_dir, "*.npz"))
    if not all_files:
        print(f"Error: No .npz files found in the directory '{args.npz_dir}'.")
        sys.exit(1)
    
    random.shuffle(all_files)
    selected_files = all_files[:min(args.num_samples, len(all_files))]
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Found {len(all_files)} files. Will visualize {len(selected_files)} of them.")
    print(f"Output images will be saved to '{args.output_dir}' directory.\n")

    if args.data_only:
        print("*** Running in Data-Only Mode ***")
        for npz_file in tqdm(selected_files, desc="Generating data visualizations"):
            base_name = os.path.basename(npz_file).replace('.npz', '.png')
            output_file_path = os.path.join(args.output_dir, f"data_only_{base_name}")
            visualize_data_only(npz_file, output_file_path)
    else:
        if not MODEL_LIBS_AVAILABLE:
            print("Error: Model-related libraries could not be imported.")
            sys.exit(1)
        if not args.config_path or not args.model_weight_path:
            print("Error: --config_path and --model_weight_path are required when not using --data_only.")
            sys.exit(1)

        print("*** Running in Full Inference Mode ***")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = load_config("ExperimentSchrodingerBridgeModel", args.config_path)
        set_seeds(config.train.seed)
        checkpoint = torch.load(args.model_weight_path, map_location=device)
        net = make_model(config.model).to(device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        si = StochasticInterpolantFollmer(config=config.si, neural_net=net).to(device)
        
        for npz_file in tqdm(selected_files, desc="Generating full visualizations"):
            base_name = os.path.basename(npz_file).replace('.npz', '.png')
            output_file_path = os.path.join(args.output_dir, f"inference_{base_name}")
            visualize_inference(config, net, si, npz_file, output_file_path)

    print("\nVisualization complete.")
