# -*- coding: utf-8 -*-
"""
3D风场超分可视化脚本

布局说明：
1. 水平风场对比图：6行(层级) × 3列(LR / HR真值 / 模型预测)
   - 底色 = 风速 sqrt(u² + v²)，箭头 = quiver(u, v)
2. W分量单独图：6行(层级) × 3列(LR / HR真值 / 模型预测)
3. 垂直剖面图：选取特定经度线，展示u/v/w随高度变化

用法:
  python visualize_wind_3d.py \
    --npz_dir /path/to/npz_files \
    --output_dir visualizations_wind_3d \
    --config_path configs/config_wind_3d.yml \
    --model_weight_path /path/to/checkpoint.pth \
    --num_samples 5
"""

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
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Add project root to the Python path
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

# Conditional imports for model-related stuff
try:
    from src.dl_config.config_loader import load_config
    from src.dl_model.model_maker import make_model
    from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer
    from src.utils.random_seed_helper import set_seeds
    MODEL_LIBS_AVAILABLE = True
except ImportError:
    MODEL_LIBS_AVAILABLE = False

# 层级定义（与 prepare_wind_data_3d.py 一致）
LEVELS = ['ml0', 'ml1', 'ml2', 'ml3', 'ml5', 'ml10']
LEVEL_LABELS = ['Level 0\n(~10m)', 'Level 1\n(~30m)', 'Level 2\n(~50m)',
                'Level 3\n(~70m)', 'Level 5\n(~120m)', 'Level 10\n(~300m)']


def plot_horizontal_wind(ax, u, v, title, vmin=None, vmax=None, cmap='viridis'):
    """绘制水平风场：底色=风速，箭头=quiver(u,v)"""
    speed = np.sqrt(u**2 + v**2)
    if vmin is None:
        vmin = np.nanmin(speed)
    if vmax is None:
        vmax = np.nanmax(speed)

    im = ax.imshow(speed, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    # 稀疏采样画箭头
    ny, nx = u.shape
    sub = max(ny // 10, 1)
    y_idx = np.arange(0, ny, sub)
    x_idx = np.arange(0, nx, sub)
    X, Y = np.meshgrid(x_idx, y_idx)
    ax.quiver(X, Y, u[y_idx][:, x_idx], v[y_idx][:, x_idx],
              color='white', scale_units='inches', scale=None, width=0.004, alpha=0.8)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_w_field(ax, w, title, vmin=None, vmax=None):
    """绘制W分量（垂直速度）"""
    if vmin is None or vmax is None:
        abs_max = max(abs(np.nanmin(w)), abs(np.nanmax(w)))
        vmin, vmax = -abs_max, abs_max

    im = ax.imshow(w, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def normalize_var(data, bias, scale):
    """标准化"""
    return (data - bias) / scale


def denormalize_var(data, bias, scale):
    """反标准化"""
    return data * scale + bias


def visualize_wind_3d_inference(config, net, si, npz_path, output_dir):
    """
    对一个npz样本进行3D风场推理并可视化。
    生成三张图：水平风场对比、W分量对比、垂直剖面。
    """
    device = next(net.parameters()).device
    base_name = os.path.basename(npz_path).replace('.npz', '')

    try:
        with np.load(npz_path) as data:
            npz_data = {key: data[key].copy() for key in data.keys()}
    except Exception as e:
        print(f"无法加载 {npz_path}: {e}")
        return

    # --- 1. 准备模型输入 ---
    target_names = config.data.target_variable_names  # 18个 hr_*
    cond_names = config.data.input_variable_names  # 10个条件变量
    lr_names = [name.replace('hr_', 'lr_') for name in target_names]
    target_shape = config.data.hr_data_shape

    # 标准化 y0 (LR wind)
    y0_channels = []
    for name in lr_names:
        arr = torch.from_numpy(npz_data[name]).float()
        arr = F.interpolate(arr[None, None], size=target_shape, mode='bicubic', align_corners=False).squeeze()
        bias = config.data.biases.get(name, 0.0)
        scale = config.data.scales.get(name, 1.0)
        y0_channels.append((arr - bias) / scale)
    y0 = torch.stack(y0_channels, dim=0).unsqueeze(0)  # [1, 18, H, W]

    # 标准化 x (conditions)
    x_channels = []
    for name in cond_names:
        arr = torch.from_numpy(npz_data[name]).float()
        arr = F.interpolate(arr[None, None], size=target_shape, mode='bicubic', align_corners=False).squeeze()
        bias = config.data.biases.get(name, 0.0)
        scale = config.data.scales.get(name, 1.0)
        x_channels.append((arr - bias) / scale)
    x_cond = torch.stack(x_channels, dim=0).unsqueeze(0)  # [1, 10, H, W]

    # --- 2. 推理 ---
    with torch.no_grad():
        pred_normalized, _ = si.sample_y1_bare_diffusion(
            y0=y0.to(device), y_cond=x_cond.to(device)
        )

    # --- 3. 反标准化预测结果 ---
    pred_physical = {}
    for i, name in enumerate(target_names):
        bias = config.data.biases.get(name, 0.0)
        scale = config.data.scales.get(name, 1.0)
        pred_physical[name] = (pred_normalized[0, i].cpu().numpy() * scale + bias)

    # 准备HR真值和LR数据（resize到target_shape）
    hr_data = {}
    lr_data = {}
    for name in target_names:
        hr_arr = npz_data[name]
        if hr_arr.shape != tuple(target_shape):
            hr_arr = F.interpolate(
                torch.from_numpy(hr_arr).float()[None, None], size=target_shape,
                mode='bicubic', align_corners=False
            ).squeeze().numpy()
        hr_data[name] = hr_arr

    for name in lr_names:
        lr_arr = npz_data[name]
        if lr_arr.shape != tuple(target_shape):
            lr_arr = F.interpolate(
                torch.from_numpy(lr_arr).float()[None, None], size=target_shape,
                mode='bicubic', align_corners=False
            ).squeeze().numpy()
        lr_data[name] = lr_arr

    # ==========================================
    # 图1: 水平风场 (u,v) 对比 — 6行 × 3列
    # ==========================================
    fig, axes = plt.subplots(6, 3, figsize=(18, 32))
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.05)

    for row, level in enumerate(LEVELS):
        hr_u = hr_data[f'hr_u_{level}']
        hr_v = hr_data[f'hr_v_{level}']
        lr_u = lr_data[f'lr_u_{level}']
        lr_v = lr_data[f'lr_v_{level}']
        pred_u = pred_physical[f'hr_u_{level}']
        pred_v = pred_physical[f'hr_v_{level}']

        # 统一色标
        all_speeds = [np.sqrt(hr_u**2 + hr_v**2),
                      np.sqrt(lr_u**2 + lr_v**2),
                      np.sqrt(pred_u**2 + pred_v**2)]
        vmin = min(np.nanmin(s) for s in all_speeds)
        vmax = max(np.nanmax(s) for s in all_speeds)

        plot_horizontal_wind(axes[row, 0], lr_u, lr_v,
                           f'LR - {LEVEL_LABELS[row]}', vmin=vmin, vmax=vmax)
        plot_horizontal_wind(axes[row, 1], hr_u, hr_v,
                           f'HR Truth - {LEVEL_LABELS[row]}', vmin=vmin, vmax=vmax)
        im = plot_horizontal_wind(axes[row, 2], pred_u, pred_v,
                           f'Predicted - {LEVEL_LABELS[row]}', vmin=vmin, vmax=vmax)

        # colorbar放在右侧，不挤占图的空间
        cbar_ax = fig.add_axes([0.90, axes[row, 2].get_position().y0,
                                0.015, axes[row, 2].get_position().height])
        fig.colorbar(im, cax=cbar_ax, label='m/s')

    fig.suptitle(f'Horizontal Wind Field (U,V) Comparison\n{base_name}', fontsize=16, y=0.995)
    plt.savefig(os.path.join(output_dir, f'{base_name}_horizontal_uv.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ==========================================
    # 图2: W分量对比 — 6行 × 3列
    # ==========================================
    fig, axes = plt.subplots(6, 3, figsize=(18, 32))
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.05)

    for row, level in enumerate(LEVELS):
        hr_w = hr_data[f'hr_w_{level}']
        lr_w = lr_data[f'lr_w_{level}']
        pred_w = pred_physical[f'hr_w_{level}']

        # 用HR的95百分位作为色标范围，让细节更明显
        p95 = np.nanpercentile(np.abs(hr_w), 95)
        if p95 < 0.01:
            p95 = 0.01  # 避免色标范围太小
        vmin, vmax = -p95, p95

        plot_w_field(axes[row, 0], lr_w, f'LR W - {LEVEL_LABELS[row]}', vmin=vmin, vmax=vmax)
        plot_w_field(axes[row, 1], hr_w, f'HR W Truth - {LEVEL_LABELS[row]}', vmin=vmin, vmax=vmax)
        im = plot_w_field(axes[row, 2], pred_w, f'Predicted W - {LEVEL_LABELS[row]}', vmin=vmin, vmax=vmax)

        cbar_ax = fig.add_axes([0.90, axes[row, 2].get_position().y0,
                                0.015, axes[row, 2].get_position().height])
        fig.colorbar(im, cax=cbar_ax, label='m/s')

    fig.suptitle(f'Vertical Velocity (W) Comparison\n{base_name}', fontsize=16, y=0.995)
    plt.savefig(os.path.join(output_dir, f'{base_name}_vertical_w.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ==========================================
    # 图3: 垂直剖面图
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 选取几个横截面位置（相对位置）
    ny, nx = target_shape
    cross_positions = [nx // 4, nx // 2, 3 * nx // 4]
    cross_labels = ['x=1/4', 'x=1/2', 'x=3/4']

    for col, (xpos, xlabel) in enumerate(zip(cross_positions, cross_labels)):
        # U分量垂直剖面
        ax_u = axes[0, col]
        for src_label, src_data, ls in [('HR', hr_data, '-'), ('LR', lr_data, '--'), ('Pred', pred_physical, ':')]:
            u_profile = []
            for level in LEVELS:
                key = f'hr_u_{level}' if src_label != 'LR' else f'lr_u_{level}'
                u_profile.append(src_data[key][:, xpos].mean())  # 沿y平均
            ax_u.plot(u_profile, range(len(LEVELS)), ls, label=src_label, linewidth=2)
        ax_u.set_yticks(range(len(LEVELS)))
        ax_u.set_yticklabels([l.replace('\n', ' ') for l in LEVEL_LABELS])
        ax_u.set_xlabel('U (m/s)')
        ax_u.set_title(f'U Profile at {xlabel}')
        ax_u.legend(fontsize=8)
        ax_u.grid(True, alpha=0.3)

        # W分量垂直剖面
        ax_w = axes[1, col]
        for src_label, src_data, ls in [('HR', hr_data, '-'), ('LR', lr_data, '--'), ('Pred', pred_physical, ':')]:
            w_profile = []
            for level in LEVELS:
                key = f'hr_w_{level}' if src_label != 'LR' else f'lr_w_{level}'
                w_profile.append(src_data[key][:, xpos].mean())
            ax_w.plot(w_profile, range(len(LEVELS)), ls, label=src_label, linewidth=2)
        ax_w.set_yticks(range(len(LEVELS)))
        ax_w.set_yticklabels([l.replace('\n', ' ') for l in LEVEL_LABELS])
        ax_w.set_xlabel('W (m/s)')
        ax_w.set_title(f'W Profile at {xlabel}')
        ax_w.legend(fontsize=8)
        ax_w.grid(True, alpha=0.3)

    fig.suptitle(f'Vertical Profiles\n{base_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_vertical_profile.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  已保存: {base_name}_horizontal_uv.png, {base_name}_vertical_w.png, {base_name}_vertical_profile.png")


def visualize_data_only(npz_path, output_dir):
    """
    仅可视化原始数据（不需要模型），用于验证数据准备是否正确。
    """
    base_name = os.path.basename(npz_path).replace('.npz', '')

    try:
        with np.load(npz_path) as data:
            npz_data = {key: data[key].copy() for key in data.keys()}
    except Exception as e:
        print(f"无法加载 {npz_path}: {e}")
        return

    # 打印所有key和shape
    print(f"\n{base_name} 的变量:")
    for key in sorted(npz_data.keys()):
        print(f"  {key}: shape={npz_data[key].shape}, range=[{npz_data[key].min():.4f}, {npz_data[key].max():.4f}]")

    # 水平风场 HR vs LR — 6行 × 2列
    fig, axes = plt.subplots(6, 2, figsize=(12, 32))
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.05)
    for row, level in enumerate(LEVELS):
        hr_u = npz_data.get(f'hr_u_{level}')
        hr_v = npz_data.get(f'hr_v_{level}')
        lr_u = npz_data.get(f'lr_u_{level}')
        lr_v = npz_data.get(f'lr_v_{level}')
        if hr_u is None or lr_u is None:
            continue

        hr_speed = np.sqrt(hr_u**2 + hr_v**2)
        lr_speed = np.sqrt(lr_u**2 + lr_v**2)
        vmin = min(np.nanmin(hr_speed), np.nanmin(lr_speed))
        vmax = max(np.nanmax(hr_speed), np.nanmax(lr_speed))

        plot_horizontal_wind(axes[row, 0], lr_u, lr_v,
                           f'LR - {LEVEL_LABELS[row]}', vmin=vmin, vmax=vmax)
        im = plot_horizontal_wind(axes[row, 1], hr_u, hr_v,
                           f'HR - {LEVEL_LABELS[row]}', vmin=vmin, vmax=vmax)
        cbar_ax = fig.add_axes([0.90, axes[row, 1].get_position().y0,
                                0.015, axes[row, 1].get_position().height])
        fig.colorbar(im, cax=cbar_ax, label='m/s')

    fig.suptitle(f'Data Verification: HR vs LR\n{base_name}', fontsize=14, y=0.995)
    plt.savefig(os.path.join(output_dir, f'{base_name}_data_verify.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {base_name}_data_verify.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Wind Field Super-Resolution Visualization")
    parser.add_argument("--npz_dir", type=str, required=True, help="包含.npz文件的目录")
    parser.add_argument("--output_dir", type=str, default="visualizations_wind_3d", help="输出图片保存目录")
    parser.add_argument("--num_samples", type=int, default=3, help="随机可视化的样本数")
    parser.add_argument("--config_path", type=str, default=None, help="模型配置文件路径（推理模式需要）")
    parser.add_argument("--model_weight_path", type=str, default=None, help="模型权重路径（推理模式需要）")
    parser.add_argument("--data_only", action='store_true', help="仅可视化数据，不做推理")

    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.npz_dir, "*.npz"))
    if not all_files:
        print(f"错误: 在 '{args.npz_dir}' 中未找到 .npz 文件。")
        sys.exit(1)

    random.shuffle(all_files)
    selected_files = all_files[:min(args.num_samples, len(all_files))]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"找到 {len(all_files)} 个文件，将可视化 {len(selected_files)} 个。")
    print(f"输出图片保存到 '{args.output_dir}'")

    if args.data_only:
        print("\n*** 数据验证模式 (不需要模型) ***")
        for npz_file in tqdm(selected_files, desc="可视化数据"):
            visualize_data_only(npz_file, args.output_dir)
    else:
        if not MODEL_LIBS_AVAILABLE:
            print("错误: 核心库无法导入，请检查环境。")
            sys.exit(1)
        if args.config_path is None or args.model_weight_path is None:
            print("错误: 推理模式需要 --config_path 和 --model_weight_path 参数。")
            sys.exit(1)

        print("\n*** 推理模式 ***")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = load_config("ExperimentSchrodingerBridge3dWind", args.config_path)
        set_seeds(config.train.seed)

        loaded_object = torch.load(args.model_weight_path, map_location=device)
        net = make_model(config.model).to(device)

        if isinstance(loaded_object, dict) and 'model_state_dict' in loaded_object:
            net.load_state_dict(loaded_object['model_state_dict'])
            print("从新格式checkpoint加载模型。")
        else:
            net.load_state_dict(loaded_object)
            print("从旧格式state_dict加载模型。")

        net.eval()
        si = StochasticInterpolantFollmer(config=config.si, neural_net=net).to(device)

        for npz_file in tqdm(selected_files, desc="生成可视化"):
            visualize_wind_3d_inference(config, net, si, npz_file, args.output_dir)

    print("\n可视化完成。")
