from multiprocessing.util import info
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

def plot_wind_map(ax, u, v, lon_range, lat_range, title, vmin, vmax):
    """Helper function to plot a single wind field map with quiver and imshow."""
    proj = ccrs.PlateCarree()
    ax.set_extent(lon_range + lat_range, crs=proj)

    speed = np.sqrt(u**2 + v**2)

    # 使用 imshow, 它通过 extent 参数自动处理坐标和尺寸，无需 lons, lats
    im = ax.imshow(speed, origin='upper', transform=proj, cmap='viridis',
                extent=lon_range + lat_range, vmin=vmin, vmax=vmax)

    # 为 quiver 创建独立的、稀疏的坐标
    lons_sub = np.linspace(lon_range[0], lon_range[1], u.shape[1])
    lats_sub = np.linspace(lat_range[1], lat_range[0], u.shape[0])
    sub = 15
    # 调整箭头美观度
    # ax.quiver(lons_sub[::sub], lats_sub[::sub], u[::sub, ::sub], v[::sub, ::sub], transform=proj, color='white',scale=vmax*40, width=0.003)
    ax.quiver(lons_sub[::sub], lats_sub[::sub], u[::sub, ::sub], v[::sub, ::sub], transform=proj, color='white', width=0.004)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title, fontsize=12)
    return im

def visualize_wind_inference(config, net, si, npz_path, output_path):
    """
    Performs wind model inference and creates a 2x2 comparison plot.
    """
    device = next(net.parameters()).device
    try:
        with np.load(npz_path) as data:
            hr_u10 = torch.from_numpy(data['hr_u10']).float()
            hr_v10 = torch.from_numpy(data['hr_v10']).float()
            lr_u10 = torch.from_numpy(data['lr_u10']).float()
            lr_v10 = torch.from_numpy(data['lr_v10']).float()
            conditional_vars = config.data.input_variable_names
            x_tensors = [torch.from_numpy(data[var]).float() for var in conditional_vars]

        # --- 1. Prepare Tensors for Model (Normalization) ---
        y0_raw = torch.stack([lr_u10, lr_v10], dim=0).unsqueeze(0)
        x_raw = torch.stack(x_tensors, dim=0).unsqueeze(0)

        # Normalize y0 (U and V components of LR wind)
        u_bias_lr = config.data.biases.get('lr_u10', 0.0)
        u_scale_lr = config.data.scales.get('lr_u10', 1.0)
        v_bias_lr = config.data.biases.get('lr_v10', 0.0)
        v_scale_lr = config.data.scales.get('lr_v10', 1.0)
        y0_normalized_u = (y0_raw[:, 0:1, :, :] - u_bias_lr) / u_scale_lr
        y0_normalized_v = (y0_raw[:, 1:2, :, :] - v_bias_lr) / v_scale_lr
        y0_normalized = torch.cat([y0_normalized_u, y0_normalized_v], dim=1)

        # Normalize each channel of x
        x_normalized_channels = []
        for i, var_name in enumerate(conditional_vars):
            key = var_name # Use full name as key by default
            var_bias = config.data.biases.get(key, 0.0)
            var_scale = config.data.scales.get(key, 1.0)
            normalized_channel = (x_raw[:, i:i+1, :, :] - var_bias) / var_scale
            x_normalized_channels.append(normalized_channel)
        x_normalized = torch.cat(x_normalized_channels, dim=1)
        
        # --- 2. Resize and Inference ---
        target_shape = config.data.hr_data_shape
        y0_resized = F.interpolate(y0_normalized, size=target_shape, mode='bicubic', align_corners=False)
        x_resized = F.interpolate(x_normalized, size=target_shape, mode='bicubic', align_corners=False)

        with torch.no_grad():
            pred_hr_normalized, _ = si.sample_y1_bare_diffusion(y0=y0_resized.to(device), y_cond=x_resized.to(device))
        
        # --- 3. Inverse Scale and Process for Plotting ---
        u_bias_hr = config.data.biases.get('hr_u10', 0.0)
        u_scale_hr = config.data.scales.get('hr_u10', 1.0)
        v_bias_hr = config.data.biases.get('hr_v10', 0.0)
        v_scale_hr = config.data.scales.get('hr_v10', 1.0)

        pred_u_physical = pred_hr_normalized[:, 0, :, :].cpu() * u_scale_hr + u_bias_hr
        pred_v_physical = pred_hr_normalized[:, 1, :, :].cpu() * v_scale_hr + v_bias_hr

        original_hr_shape = hr_u10.shape[-2:]
        pred_u_resized = F.interpolate(pred_u_physical.unsqueeze(0), size=original_hr_shape, mode='bicubic', align_corners=False).squeeze()
        pred_v_resized = F.interpolate(pred_v_physical.unsqueeze(0), size=original_hr_shape, mode='bicubic', align_corners=False).squeeze()
        
        # --- 4. Prepare 4 sets of (u, v) numpy arrays for plotting ---
        hr_u_plot, hr_v_plot = hr_u10.squeeze().numpy(), hr_v10.squeeze().numpy()
        interp_lr_u_plot, interp_lr_v_plot = lr_u10.squeeze().numpy(), lr_v10.squeeze().numpy()
        pred_u_plot, pred_v_plot = pred_u_resized.numpy(), pred_v_resized.numpy()

        coarse_lr_u = xr.DataArray(hr_u_plot, dims=("south_north", "west_east")).coarsen(south_north=4, west_east=4, boundary="pad").mean().values
        coarse_lr_v = xr.DataArray(hr_v_plot, dims=("south_north", "west_east")).coarsen(south_north=4, west_east=4, boundary="pad").mean().values
        
        # --- 5. Visualize ---
        lon_range = [112.9811, 114.6543]
        lat_range = [21.8413, 23.3858]
        lons = np.linspace(lon_range[0], lon_range[1], hr_u_plot.shape[1])
        lats = np.linspace(lat_range[1], lat_range[0], hr_u_plot.shape[0])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()

        plot_info = [
            {'u': coarse_lr_u, 'v': coarse_lr_v, 'title': 'Original Low-Res'},
            {'u': hr_u_plot, 'v': hr_v_plot, 'title': 'High-Res Ground Truth'},
            {'u': interp_lr_u_plot, 'v': interp_lr_v_plot, 'title': 'Interpolated Low-Res'},
            {'u': pred_u_plot, 'v': pred_v_plot, 'title': 'Model Prediction'}
        ]
        plot_order = [0, 2, 3, 1]
        
        for i, plot_idx in enumerate(plot_order):
            info = plot_info[plot_idx]
            ax = axes[i]
            # 为每张图独立计算颜色范围 (vmin, vmax)
            current_speed = np.sqrt(info['u']**2 + info['v']**2)
            vmin_local = np.nanmin(current_speed)
            vmax_local = np.nanmax(current_speed)
            title_with_shape = f"{info['title']}\n(Shape: {info['u'].shape})"
            im = plot_wind_map(ax, info['u'], info['v'], lon_range, lat_range, title_with_shape, vmin=vmin_local, vmax=vmax_local)
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, label='Wind Speed (m/s)')
        
        axes[0].gridlines(draw_labels=True).right_labels = False
        axes[1].gridlines(draw_labels=True).right_labels = False
        axes[1].gridlines(draw_labels=True).left_labels = False
        axes[2].gridlines(draw_labels=True).right_labels = False
        axes[3].gridlines(draw_labels=True).left_labels = False
        
        plt.suptitle("Wind Field Super-Resolution Comparison", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        plt.close(fig)

    except Exception as e:
        import traceback
        print(f"Failed to run full inference for {os.path.basename(npz_path)}. Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wind Field Super-Resolution Visualization")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing .npz files.")
    parser.add_argument("--output_dir", type=str, default="visualizations_wind", help="Directory to save output images.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model config file (e.g., config_wind.yml).")
    parser.add_argument("--model_weight_path", type=str, required=True, help="Path to the trained model checkpoint.pth file.")
    
    args = parser.parse_args()
    
    # --- Main execution logic ---
    if not MODEL_LIBS_AVAILABLE:
        print("Error: Core libraries (torch, etc.) could not be imported.")
        sys.exit(1)

    all_files = glob.glob(os.path.join(args.npz_dir, "*.npz"))
    if not all_files:
        print(f"Error: No .npz files found in the directory '{args.npz_dir}'.")
        sys.exit(1)
    
    random.shuffle(all_files)
    selected_files = all_files[:min(args.num_samples, len(all_files))]
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Found {len(all_files)} files. Will visualize {len(selected_files)}.")
    print(f"Output images will be saved to '{args.output_dir}'.\n")

    print("*** Running in Full Inference Mode ***")
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
    
    for npz_file in tqdm(selected_files, desc="Generating visualizations"):
        base_name = os.path.basename(npz_file).replace('.npz', '.png')
        output_file_path = os.path.join(args.output_dir, f"inference_{base_name}")
        visualize_wind_inference(config, net, si, npz_file, output_file_path)

    print("\nVisualization complete.")
