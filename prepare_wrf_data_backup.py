# -*- coding: utf-8 -*-
import os
import glob
from datetime import datetime
import numpy as np
from tqdm import tqdm
from netCDF4 import Dataset
# 假设 wrf-python 已通过 conda 安装
from wrf import getvar, interplevel

# ==============================================================================
# ---                           配置部分 (终极版)                      ---
# ==============================================================================

# 1. 路径设置
# TODO: 请根据您的服务器配置修改以下路径
WRF_DIR = "/public3/home/scg4074/gp_zhaokc/SimulationJuly/RESULT_UCM/SLUCM/"  # 存放 wrfout* 文件的目录
OUTPUT_DIR = "/public3/home/scg4074/tzq/schrodinger-bridge-sr-micrometeorology/prepare_npz" # 保存 .npz 文件的目录

# 2. 数据处理参数
DOWNSAMPLING_FACTOR = 4  # 降采样因子
PRESSURE_LEVELS = [900.0, 850.0]  # 需要插值的等压层 (hPa)

# 3. 变量定义
# 目标变量 (高分辨率)
TARGET_VAR_HR_WRF = 'T2' # 2米气温 (开尔文)

# 地表变量 (Surface)
# 格式: {'WRF变量名': '在npz文件中希望的名称'}
SURFACE_VARS_WRF = {
    'HGT': 'z',          # 地形高度
    'LU_INDEX': 'lu',    # 土地利用类型
    'TSK': 'tsk',        # 地表皮温
    'SWDOWN': 'swdown',  # 向下短波辐射
    'GLW': 'glw',        # 向下长波辐射
    'HFX': 'hfx',        # 感热通量
    'LH': 'lh',          # 潜热通量
    'PSFC': 'psfc',      # 地面气压
    'PBLH': 'pblh',      # 边界层高度
    'U10': 'u10',        # 10米U风
    'V10': 'v10'         # 10米V风
}

# 等压层变量 (Pressure Levels)
# 注意: 'QVAPOR' 必须大写
PRESSURE_VARS_WRF = ['temp', 'ua', 'va', 'QVAPOR'] 

# --- 配置结束 ---
# ==============================================================================


def get_wrf_time(ncfile):
    """
    从打开的netCDF4数据集中更鲁棒地提取时间戳。
    WRF的'Times'变量通常是一个字符数组。
    """
    try:
        # 'Times' 变量通常是 (1, 19) 的形状，包含 b'2', b'0', ...
        time_char_array = ncfile.variables['Times'][0]
        # 使用 b''.join 将字节字符连接成一个字节串，然后解码
        times_str = b''.join(time_char_array).decode('utf-8').strip()
        return datetime.strptime(times_str, '%Y-%m-%d_%H:%M:%S')
    except Exception as e:
        print(f"从 'Times' 变量提取时间戳时出错: {e}")
        # 备用方案：尝试从文件名解析
        filename = os.path.basename(ncfile.filepath())
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', filename)
        if match:
            time_str = match.group(1).replace("_", " ").replace(":", "-")
            print(f"备用方案：从文件名 {filename} 中解析时间。")
            return datetime.strptime(time_str, '%Y-%m-%d %H-%M-%S')
        raise ValueError("无法从 'Times' 变量或文件名中提取有效时间。")


def process_wrf_file(wrf_file_path):
    """
    处理单个WRF文件，提取、插值并准备数据。
    包含详细的错误处理，以诊断问题文件/变量。
    """
    ncfile = None
    try:
        ncfile = Dataset(wrf_file_path)
        npz_output_data = {}

        def safe_getvar(nc, var_name):
            try:
                data = getvar(nc, var_name)
                if "Time" in data.dims and data.sizes["Time"] > 0:
                    data = data.isel(Time=0).squeeze()
                return data
            except Exception as e:
                print(f"\n--> 警告: 在文件 {os.path.basename(wrf_file_path)} 中, 无法提取变量 '{var_name}'. 底层错误: {e}")
                return None

        # --- 逐个提取并检查变量 ---

        hr_target = safe_getvar(ncfile, TARGET_VAR_HR_WRF)
        if hr_target is None: return None 

        lr_target_da = hr_target.coarsen(south_north=DOWNSAMPLING_FACTOR, west_east=DOWNSAMPLING_FACTOR, boundary="pad").mean()
        lr_target_interp = lr_target_da.interp_like(hr_target, method='linear')
        
        npz_output_data['hr_tm002m'] = hr_target.values
        npz_output_data['lr_tm002m'] = lr_target_interp.values

        for wrf_name, npz_name in SURFACE_VARS_WRF.items():
            var_data = safe_getvar(ncfile, wrf_name)
            if var_data is None: return None
            npz_output_data[npz_name] = var_data.values

        pressure = safe_getvar(ncfile, "pressure")
        if pressure is None: return None

        for var_name in PRESSURE_VARS_WRF:
            var_native = safe_getvar(ncfile, var_name)
            if var_native is None: return None
            
            for p_level in PRESSURE_LEVELS:
                var_plevel = interplevel(var_native, pressure, p_level)
                
                npz_var_prefix = var_name[0]
                if var_name == 'QVAPOR':
                    npz_var_prefix = 'q'
                
                npz_var_name = f"{npz_var_prefix}_{int(p_level)}"
                npz_output_data[npz_var_name] = var_plevel.values
        
        return npz_output_data

    finally:
        if ncfile:
            ncfile.close()

def main():
    """
    运行WRF数据预处理流程的主函数。
    """
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
                days_to_process = total_days
                break
            days_to_process = int(days_to_process_str)
            if 1 <= days_to_process <= total_days:
                break
            else:
                print(f"请输入一个 1 到 {total_days} 之间的有效数字。")
        except ValueError:
            print("输入无效。请输入一个数字。")

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