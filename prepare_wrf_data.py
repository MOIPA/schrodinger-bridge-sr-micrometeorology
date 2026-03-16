# -*- coding: utf-8 -*-
import os
import glob
from datetime import datetime
import numpy as np
from tqdm import tqdm
from netCDF4 import Dataset
# 假设 wrf-python 已通过 conda 安装
from wrf import getvar, interplevel
import torch # 导入torch库
import torch.nn.functional as F # 导入PyTorch的函数式API，用于插值
# ==============================================================================
# ---                           配置部分 (最终版)                      ---
# ==============================================================================

# 1. 路径设置
# TODO: 请根据您的服务器配置修改以下路径
WRF_DIR = "/public3/home/scg4074/gp_zhaokc/SimulationJuly/RESULT_UCM/SLUCM/"  # 存放 wrfout* 文件的目录
OUTPUT_DIR = "/public3/home/scg4074/tzq/schrodinger-bridge-sr-micrometeorology/prepare_npz" # 保存 .npz 文件的目录

# 2. 数据处理参数
DOWNSAMPLING_FACTOR = 4  # 降采样因子
MODEL_LEVELS_TO_EXTRACT = [0, 2, 5] # 从地表向上数的模型层

# 3. 变量定义
TARGET_VAR_HR_WRF = 'T2'
SURFACE_VARS_WRF = {
    'HGT': 'z', 'LU_INDEX': 'lu', 'TSK': 'tsk', 'SWDOWN': 'swdown', 
    'GLW': 'glw', 'HFX': 'hfx', 'LH': 'lh', 'PSFC': 'psfc', 
    'PBLH': 'pblh', 'U10': 'u10', 'V10': 'v10'
}
MODEL_LEVEL_VARS_WRF = ['temp', 'ua', 'va', 'QVAPOR'] 

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


def process_wrf_file(wrf_file_path):
    """处理单个WRF文件，提取、插值并准备数据。"""
    ncfile = None
    try:
        ncfile = Dataset(wrf_file_path)
        npz_output_data = {}

        def safe_getvar(nc, var_name):
            try:
                data = getvar(nc, var_name, meta=True)
                if "Time" in data.dims and data.sizes["Time"] > 0:
                    data = data.isel(Time=0).squeeze()
                return data
            except Exception as e:
                print(f"\n--> 警告: 在文件 {os.path.basename(wrf_file_path)} 中, 无法提取变量 '{var_name}'. 底层错误: {e}")
                return None

        # 1. 提取高分辨率目标
        hr_target = safe_getvar(ncfile, TARGET_VAR_HR_WRF)
        if hr_target is None: return None
        npz_output_data['hr_tm002m'] = hr_target.values
        base_shape = hr_target.shape[-2:]

        # 2. 创建低分辨率目标（已修正）
        lr_target_da = hr_target.coarsen(south_north=DOWNSAMPLING_FACTOR, west_east=DOWNSAMPLING_FACTOR, boundary="pad").mean()
        lr_tensor = torch.from_numpy(lr_target_da.values).float().unsqueeze(0).unsqueeze(0)
        lr_tensor_interp = F.interpolate(lr_tensor, size=base_shape, mode='bicubic', align_corners=False)
        lr_target_interp_np = lr_tensor_interp.squeeze().numpy()
        
            # --- 断言校验，确保尺寸正确 ---
        assert lr_target_interp_np.shape == base_shape, f"致命错误: 插值后的低分辨率图像尺寸{lr_target_interp_np.shape} 与基准尺寸 {base_shape} 不匹配！"
        npz_output_data['lr_tm002m'] = lr_target_interp_np

        # 3. 提取地表变量
        for wrf_name, npz_name in SURFACE_VARS_WRF.items():
            var_data = safe_getvar(ncfile, wrf_name)
            if var_data is None: return None
            npz_output_data[npz_name] = var_data.values

        # 4. 从指定的模型层提取3D变量
        for var_name in MODEL_LEVEL_VARS_WRF:
            var_3d = safe_getvar(ncfile, var_name)
            if var_3d is None: return None
            
            for level_idx in MODEL_LEVELS_TO_EXTRACT:
                var_level = var_3d.isel(bottom_top=level_idx)
                npz_var_prefix = 'q' if var_name == 'QVAPOR' else var_name[0]
                npz_var_name = f"{npz_var_prefix}_ml{level_idx}"
                npz_output_data[npz_var_name] = var_level.values
        
        # --- 尺寸自检 ---
        for key, value in npz_output_data.items():
            if value.ndim >= 2:
                if value.shape[-2:] != base_shape:
                    raise ValueError(f"致命错误: 变量 '{key}' 的空间维度是 {value.shape[-2:]}, 但期望的基准维度是 {base_shape}。")
        
        return npz_output_data

    finally:
        if ncfile:
            ncfile.close()

def main():
    """运行WRF数据预处理流程的主函数。"""
    print("启动WRF数据预处理脚本...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    search_path = os.path.join(WRF_DIR, "wrfout_d04_*")
    all_files = sorted(glob.glob(search_path))

    if not all_files:
        print(f"错误：在 '{WRF_DIR}' 目录中未找到 'wrfout_d04_*' 文件。请检查路径。")
        return

    total_hours = len(all_files)
    total_days = total_hours // 24
    print(f"找到了 {total_hours} 个小时的数据文件, 约等于 {total_days} 天。")

    while True:
        try:
            days_to_process_str = input(f"您希望处理多少天的数据？ (请输入 1 到 {total_days} 之间的数字, 或输入 'all' 处理全部) > ")
            if days_to_process_str.lower() == 'all':
                days_to_process = total_days if total_days > 0 else 1
                break
            days_to_process = int(days_to_process_str)
            if 1 <= days_to_process <= total_days:
                break
            else:
                print(f"请输入一个 1 到 {total_days} 之间的有效数字。")
        except ValueError:
            print("输入无效，请输入一个数字。")

    files_to_process = all_files[:days_to_process * 24]
    print(f"将处理 {days_to_process} 天的数据, 共 {len(files_to_process)} 个文件...")

    for wrf_file in tqdm(files_to_process, desc="正在处理WRF文件"):
        try:
            with Dataset(wrf_file) as ncfile_time:
                timestamp = get_wrf_time(ncfile_time)
            
            output_filename = timestamp.strftime('%Y%m%dT%H%M%S') + ".npz"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            if os.path.exists(output_filepath):
                continue

            processed_data = process_wrf_file(wrf_file)

            if processed_data:
                np.savez_compressed(output_filepath, **processed_data)
        except Exception as e:
            print(f"\n处理文件 {os.path.basename(wrf_file)} 时发生严重错误: {e}")

    print("\n处理完成。")
    print(f"共处理了 {len(files_to_process)} 个文件，并保存在 '{OUTPUT_DIR}' 目录中。")


if __name__ == "__main__":
    main()
