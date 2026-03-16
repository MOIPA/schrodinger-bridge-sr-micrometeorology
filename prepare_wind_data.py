# -*- coding: utf-8 -*-
import os
import glob
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from netCDF4 import Dataset
from wrf import getvar
import warnings

# ==============================================================================
# ---                           配置部分 (风场超分)                    ---
# ==============================================================================

# 1. 路径设置
WRF_DIR = "/public3/home/scg4074/gp_zhaokc/SimulationJuly/RESULT_UCM/SLUCM/"
OUTPUT_DIR = "/public3/home/scg4074/tzq/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind"

# 2. 数据处理参数
DOWNSAMPLING_FACTOR = 4
MODEL_LEVELS_TO_EXTRACT = [0, 2, 5]

# 3. 变量定义
# 目标变量 (高分辨率风场)
TARGET_VARS_HR_WRF = ['U10', 'V10']

# 条件变量 - 地表
SURFACE_VARS_WRF = {
    'T2': 't2', 'HGT': 'z', 'LU_INDEX': 'lu', 'TSK': 'tsk', 'SWDOWN': 'swdown',
    'GLW': 'glw', 'HFX': 'hfx', 'LH': 'lh', 'PSFC': 'psfc', 'PBLH': 'pblh'
}
# 条件变量 - 模型层
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
    """处理单个WRF文件，为风场超分任务准备数据。"""
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
                print(f"--> 警告: 在文件 {os.path.basename(wrf_file_path)} 中, 无法提取变量 '{var_name}'. 底层错误: {e}")
                return None

        # 1. 提取高分辨率目标风场 (U10, V10)
        hr_u10 = safe_getvar(ncfile, 'U10')
        hr_v10 = safe_getvar(ncfile, 'V10')
        if hr_u10 is None or hr_v10 is None: return None
        
        npz_output_data['hr_u10'] = hr_u10.values
        npz_output_data['hr_v10'] = hr_v10.values
        base_shape = hr_u10.shape[-2:]

        # 2. 创建低分辨率输入风场 (lr_u10, lr_v10)
        for hr_da, name in [(hr_u10, 'u10'), (hr_v10, 'v10')]:
            lr_da = hr_da.coarsen(south_north=DOWNSAMPLING_FACTOR, west_east=DOWNSAMPLING_FACTOR, boundary="pad").mean()
            lr_tensor = torch.from_numpy(lr_da.values).float().unsqueeze(0).unsqueeze(0)
            lr_tensor_interp = F.interpolate(lr_tensor, size=base_shape, mode='bicubic', align_corners=False)
            lr_interp_np = lr_tensor_interp.squeeze().numpy()
            assert lr_interp_np.shape == base_shape
            npz_output_data[f'lr_{name}'] = lr_interp_np
        
        # 3. 提取所有条件变量
        for wrf_name, npz_name in SURFACE_VARS_WRF.items():
            var_data = safe_getvar(ncfile, wrf_name)
            if var_data is None: return None
            npz_output_data[npz_name] = var_data.values
        
        for var_name in MODEL_LEVEL_VARS_WRF:
            var_3d = safe_getvar(ncfile, var_name)
            if var_3d is None: return None
            for level_idx in MODEL_LEVELS_TO_EXTRACT:
                var_level = var_3d.isel(bottom_top=level_idx)
                npz_var_prefix = 'q' if var_name == 'QVAPOR' else var_name[0]
                npz_var_name = f"{npz_var_prefix}_ml{level_idx}"
                npz_output_data[npz_var_name] = var_level.values
        
        # 4. 尺寸自检
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
    print("启动WRF数据预处理脚本 (风场超分)...")
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
            if 1 <= days_to_process <= total_days: break
            else: print(f"请输入 1 到 {total_days} 之间的有效数字。")
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

    print(f"处理完成。共生成或找到 {len(processed_files)} 个 .npz 文件。")

    if not processed_files:
        print("没有可用于计算统计数据的文件。")
        return

    # --- 计算并打印所有变量的均值和标准差 ---
    print("开始计算所有已处理数据的均值和标准差...")
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
    
    print("--- 数据集统计结果 (用于更新 config_wind.yml) ---")
    print("请将以下 'biases' 和 'scales' 内容复制到您的配置文件中：")
    
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
            print(f"    {key}: {std_val:.6f}")
    
    print("--- 统计完成 ---")

if __name__ == "__main__":
    main()
