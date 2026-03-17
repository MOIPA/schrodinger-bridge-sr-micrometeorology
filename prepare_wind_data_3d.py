# -*- coding: utf-8 -*-
"""
3D风场数据准备脚本
从WRF d04输出提取6个eta模式层(ml0,1,2,3,5,10)的U/V/W三风分量，
去交错(Arakawa C网格)后生成HR/LR对，保存为npz文件。

输出：每个时间步一个npz文件，包含：
  - 18个HR风场: hr_u_ml0, hr_v_ml0, hr_w_ml0, ..., hr_u_ml10, hr_v_ml10, hr_w_ml10
  - 18个LR风场: lr_u_ml0, lr_v_ml0, lr_w_ml0, ..., lr_u_ml10, lr_v_ml10, lr_w_ml10
  - 10个条件变量: t2, z, lu, tsk, swdown, glw, hfx, lh, psfc, pblh
"""

import os
import glob
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from netCDF4 import Dataset
import warnings

# ==============================================================================
# ---                       配置部分 (3D风场超分)                          ---
# ==============================================================================

# 1. 路径设置
WRF_DIR = "/public3/home/scg4074/gp_zhaokc/SimulationJuly/RESULT_UCM/SLUCM/"
OUTPUT_DIR = "/public3/home/scg4074/tzq/git/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d"

# 2. 数据处理参数
DOWNSAMPLING_FACTOR = 4

# 3. eta模式层配置: 名称 -> WRF bottom_top索引
LEVELS = {'ml0': 0, 'ml1': 1, 'ml2': 2, 'ml3': 3, 'ml5': 5, 'ml10': 10}

# 4. 条件变量 - 地表 (WRF变量名 -> npz key名)
SURFACE_VARS_WRF = {
    'T2': 't2', 'HGT': 'z', 'LU_INDEX': 'lu', 'TSK': 'tsk', 'SWDOWN': 'swdown',
    'GLW': 'glw', 'HFX': 'hfx', 'LH': 'lh', 'PSFC': 'psfc', 'PBLH': 'pblh'
}

# --- 配置结束 ---
# ==============================================================================


def get_wrf_time(ncfile):
    """从打开的netCDF4数据集中更鲁棒地提取时间戳。"""
    try:
        time_char_array = ncfile.variables['Times'][0]
        times_str = b''.join(time_char_array).decode('utf-8').strip()
        return datetime.strptime(times_str, '%Y-%m-%d_%H:%M:%S')
    except Exception:
        filename = os.path.basename(ncfile.filepath())
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', filename)
        if match:
            time_str = match.group(1)
            return datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S')
        raise ValueError("无法从 'Times' 变量或文件名中提取有效时间。")


def destagger_u(u_stag):
    """
    对U分量去交错（沿west_east方向，即最后一个轴）。
    WRF的U在west_east_stag维度上多1个点。
    u_stag shape: (..., south_north, west_east_stag)
    输出 shape: (..., south_north, west_east)
    """
    return (u_stag[..., :-1] + u_stag[..., 1:]) / 2.0


def destagger_v(v_stag):
    """
    对V分量去交错（沿south_north方向，即倒数第二个轴）。
    WRF的V在south_north_stag维度上多1个点。
    v_stag shape: (..., south_north_stag, west_east)
    输出 shape: (..., south_north, west_east)
    """
    return (v_stag[..., :-1, :] + v_stag[..., 1:, :]) / 2.0


def destagger_w(w_stag_k, w_stag_k1):
    """
    对W分量去交错（沿垂直方向）。
    WRF的W在bottom_top_stag维度上多1个点。
    取相邻两层的平均：(W[k] + W[k+1]) / 2
    """
    return (w_stag_k + w_stag_k1) / 2.0


def create_lr_field(hr_data, base_shape):
    """
    从高分辨率场生成低分辨率场：先降采样再插值回原始尺寸。
    """
    # 使用 average pooling 降采样
    hr_tensor = torch.from_numpy(hr_data).float().unsqueeze(0).unsqueeze(0)
    lr_tensor = F.avg_pool2d(hr_tensor, kernel_size=DOWNSAMPLING_FACTOR)
    # 插值回原始尺寸
    lr_interp = F.interpolate(lr_tensor, size=base_shape, mode='bicubic', align_corners=False)
    return lr_interp.squeeze().numpy()


def process_wrf_file(wrf_file_path):
    """处理单个WRF文件，为3D风场超分任务准备数据。"""
    ncfile = None
    try:
        ncfile = Dataset(wrf_file_path)
        npz_output_data = {}

        # ---- 1. 提取交错网格上的3D风场 ----
        # U: shape (Time, bottom_top, south_north, west_east_stag)
        # V: shape (Time, bottom_top, south_north_stag, west_east)
        # W: shape (Time, bottom_top_stag, south_north, west_east)
        try:
            U_stag = ncfile.variables['U'][0]  # 去掉Time维度
            V_stag = ncfile.variables['V'][0]
            W_stag = ncfile.variables['W'][0]
        except KeyError as e:
            print(f"--> 警告: 在文件 {os.path.basename(wrf_file_path)} 中缺少变量 {e}")
            return None

        # 确定基准shape（质量点网格）
        # U去交错后: (bottom_top, south_north, west_east)
        base_shape_2d = (U_stag.shape[1], U_stag.shape[2])  # (south_north, west_east-1... 不对)

        # 先对一个level做去交错来确定2D基准shape
        u_test = destagger_u(U_stag[0])  # (south_north, west_east)
        base_shape = u_test.shape
        print(f"  基准2D shape: {base_shape}")

        # ---- 2. 逐层提取并去交错U/V/W ----
        for level_name, level_idx in LEVELS.items():
            # U去交错 (沿x方向)
            u_level = destagger_u(U_stag[level_idx])  # (south_north, west_east)
            assert u_level.shape == base_shape, f"U {level_name} shape {u_level.shape} != {base_shape}"

            # V去交错 (沿y方向)
            v_level = destagger_v(V_stag[level_idx])  # (south_north, west_east)
            assert v_level.shape == base_shape, f"V {level_name} shape {v_level.shape} != {base_shape}"

            # W去交错 (沿z方向): W在bottom_top_stag上，需要用level_idx和level_idx+1
            w_level = destagger_w(W_stag[level_idx], W_stag[level_idx + 1])
            assert w_level.shape == base_shape, f"W {level_name} shape {w_level.shape} != {base_shape}"

            # 转为numpy数组（如果还不是的话）
            u_np = np.array(u_level, dtype=np.float32)
            v_np = np.array(v_level, dtype=np.float32)
            w_np = np.array(w_level, dtype=np.float32)

            # 保存HR
            npz_output_data[f'hr_u_{level_name}'] = u_np
            npz_output_data[f'hr_v_{level_name}'] = v_np
            npz_output_data[f'hr_w_{level_name}'] = w_np

            # 生成并保存LR
            npz_output_data[f'lr_u_{level_name}'] = create_lr_field(u_np, base_shape)
            npz_output_data[f'lr_v_{level_name}'] = create_lr_field(v_np, base_shape)
            npz_output_data[f'lr_w_{level_name}'] = create_lr_field(w_np, base_shape)

        # ---- 3. 提取条件变量 ----
        for wrf_name, npz_name in SURFACE_VARS_WRF.items():
            try:
                var_data = ncfile.variables[wrf_name][0]  # 去掉Time维度
                var_np = np.array(var_data, dtype=np.float32)
                # 某些变量可能有额外维度需要squeeze
                if var_np.ndim > 2:
                    var_np = var_np.squeeze()
                # 检查shape兼容性 — 条件变量应与基准shape一致
                if var_np.shape[-2:] != base_shape:
                    print(f"--> 警告: 条件变量 '{npz_name}' shape {var_np.shape} != 基准 {base_shape}，尝试resize")
                    var_tensor = torch.from_numpy(var_np).float().unsqueeze(0).unsqueeze(0)
                    var_np = F.interpolate(var_tensor, size=base_shape, mode='nearest-exact').squeeze().numpy()
                npz_output_data[npz_name] = var_np
            except KeyError:
                print(f"--> 警告: 在文件 {os.path.basename(wrf_file_path)} 中缺少条件变量 '{wrf_name}'")
                return None

        # ---- 4. 尺寸自检 ----
        for key, value in npz_output_data.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                if value.shape[-2:] != base_shape:
                    raise ValueError(f"致命错误: 变量 '{key}' 维度 {value.shape[-2:]} 与基准维度 {base_shape} 不匹配。")

        return npz_output_data

    finally:
        if ncfile:
            ncfile.close()


def main():
    """主函数，处理文件并计算统计数据。"""
    print("启动WRF数据预处理脚本 (3D风场超分)...")
    print(f"输入目录: {WRF_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"降采样因子: {DOWNSAMPLING_FACTOR}")
    print(f"提取层级: {LEVELS}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(WRF_DIR, "wrfout_d04_*")))
    if not all_files:
        print(f"错误：在 '{WRF_DIR}' 中未找到 'wrfout_d04_*' 文件。")
        return

    total_days = len(all_files) // 24
    print(f"找到了 {len(all_files)} 个小时的数据文件, 约等于 {total_days} 天。")

    while True:
        try:
            days_str = input(f"您希望处理多少天的数据？ (输入 1-{total_days}, 或 'all') > ")
            if days_str.lower() == 'all':
                days_to_process = total_days if total_days > 0 else 1
                break
            days_to_process = int(days_str)
            if 1 <= days_to_process <= total_days:
                break
            else:
                print(f"请输入 1 到 {total_days} 之间的有效数字。")
        except ValueError:
            print("输入无效，请输入一个数字。")

    files_to_process = all_files[:days_to_process * 24]
    print(f"将处理 {days_to_process} 天的数据, 共 {len(files_to_process)} 个文件...")

    processed_files = []
    for wrf_file in tqdm(files_to_process, desc="正在处理WRF文件"):
        try:
            with Dataset(wrf_file) as ncfile_time:
                timestamp = get_wrf_time(ncfile_time)

            output_filename = timestamp.strftime('%Y%m%dT%H%M%S') + ".npz"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            if os.path.exists(output_filepath):
                processed_files.append(output_filepath)
                continue

            processed_data = process_wrf_file(wrf_file)

            if processed_data:
                np.savez_compressed(output_filepath, **processed_data)
                processed_files.append(output_filepath)
        except Exception as e:
            print(f"处理文件 {os.path.basename(wrf_file)} 时发生严重错误: {e}")
            import traceback
            traceback.print_exc()

    print(f"处理完成。共生成或找到 {len(processed_files)} 个 .npz 文件。")

    if not processed_files:
        print("没有可用于计算统计数据的文件。")
        return

    # --- 计算并打印所有变量的均值和标准差 ---
    print("\n开始计算所有已处理数据的均值和标准差...")
    all_data = {}

    # 仅加载第一个文件以获取所有变量的键名
    with np.load(processed_files[0]) as data:
        for key in data.keys():
            all_data[key] = []

    # 循环读取所有npz文件，收集数据
    for npz_file in tqdm(processed_files, desc="正在读取NPZ文件统计数据"):
        with np.load(npz_file) as data:
            for key in all_data.keys():
                all_data[key].append(data[key])

    print("\n--- 数据集统计结果 (用于更新 config_wind_3d.yml) ---")
    print("请将以下 'biases' 和 'scales' 内容复制到您的配置文件中：\n")

    print("  biases:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key, values in all_data.items():
            mean_val = np.nanmean(np.stack(values))
            print(f"    {key}: {mean_val:.6f}")

    print("  scales:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key, values in all_data.items():
            std_val = np.nanstd(np.stack(values))
            if std_val == 0:
                std_val = 1.0
                print(f"    {key}: {std_val:.6f}  # WARNING: std=0, 设为1.0")
            else:
                print(f"    {key}: {std_val:.6f}")

    print("\n--- 统计完成 ---")
    print(f"\n验证信息:")
    print(f"  总变量数: {len(all_data)}")
    print(f"  风场变量(HR+LR): {sum(1 for k in all_data if k.startswith('hr_') or k.startswith('lr_'))}")
    print(f"  条件变量: {sum(1 for k in all_data if not k.startswith('hr_') and not k.startswith('lr_'))}")

    # 打印第一个样本的shape用于验证
    with np.load(processed_files[0]) as data:
        print(f"\n  第一个样本的所有key和shape:")
        for key in sorted(data.keys()):
            print(f"    {key}: {data[key].shape}")


if __name__ == "__main__":
    main()
