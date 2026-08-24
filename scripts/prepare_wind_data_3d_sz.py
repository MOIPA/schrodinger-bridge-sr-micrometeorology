# -*- coding: utf-8 -*-
"""
深圳 WRF d04 数据预处理脚本（3D 风场超分，输出与香港 prepare_wind_data_3d.py 同格式）

从 case01_Shenzhen 的 meso_202007_myj / meso_202007_ysu 提取 6 个 eta 层
(ml0,1,2,3,5,10) 的 U/V/W，去交错(Arakawa C 网格)后生成 HR/LR 对，保存 npz。

与香港流程的差异：
  1. 去掉 5 点边界缓冲区后微 pad 到 96x112（模型要求 16 的倍数，89x110 -> 96x112）
  2. 该模拟没有 SWDOWN/GLW 输出：swdown 用太阳几何公式合成（仅用于昼夜过滤，
     不作为模型条件变量；模型条件变量为 t2, z, lu, tsk, hfx, lh, psfc, pblh）
  3. 每个 wrfout 文件含 6 个 10-min 时次，全部提取（香港是每小时一个文件）
  4. myj/ysu 时间戳相同，输出文件名加 scheme 前缀（单一目录）

运行（服务器，pytorch-gpu 环境有 netCDF4）：
  /fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/prepare_wind_data_3d_sz.py \
      --scheme both --limit 2   # 冒烟测试
  ... 全量时去掉 --limit
"""
import argparse
import glob
import multiprocessing
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset

# ==============================================================================
# ---                       配置部分 (深圳3D风场超分)                       ---
# ==============================================================================

WRF_BASE = "/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen"
OUTPUT_BASE = "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d_sz"

DOWNSAMPLING_FACTOR = 4
BUFFER = 5  # d01-d04 边界缓冲区网格点数
TARGET_SHAPE = (96, 112)  # 裁剪后 (89, 110) 微 pad 到 16 的倍数

# eta 模式层配置: 名称 -> WRF bottom_top 索引（与香港一致）
LEVELS = {'ml0': 0, 'ml1': 1, 'ml2': 2, 'ml3': 3, 'ml5': 5, 'ml10': 10}

# 条件变量 - 地表 (WRF变量名 -> npz key名)，无 SWDOWN/GLW
SURFACE_VARS_WRF = {
    'T2': 't2', 'HGT': 'z', 'LU_INDEX': 'lu', 'TSK': 'tsk',
    'HFX': 'hfx', 'LH': 'lh', 'PSFC': 'psfc', 'PBLH': 'pblh'
}

# 合成 swdown 用（清晰天空近似）：swdown = SOLAR_CONST * TRANS * max(0, sin(elev))
SOLAR_CONST = 1361.0
CLEAR_SKY_TRANS = 0.7

# --- 配置结束 ---
# ==============================================================================


def solar_elevation(dt_utc, lon_deg, lat_deg):
    """给定 UTC 时刻和经纬度，计算太阳高度角（度）。"""
    doy = dt_utc.timetuple().tm_yday
    decl = 23.44 * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    b_rad = np.radians(360.0 / 365.0 * (doy - 81))
    eot_min = 9.87 * np.sin(2 * b_rad) - 7.53 * np.cos(b_rad) - 1.5 * np.sin(b_rad)
    utc_h = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    lst_h = utc_h + lon_deg / 15.0 + eot_min / 60.0
    hour_angle = 15.0 * (lst_h - 12.0)
    sin_elev = (np.sin(np.radians(lat_deg)) * np.sin(np.radians(decl)) +
                np.cos(np.radians(lat_deg)) * np.cos(np.radians(decl)) *
                np.cos(np.radians(hour_angle)))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))


def synthetic_swdown(elev_deg):
    """清晰天空合成下行短波辐射（W/m2），仅用于昼夜过滤。"""
    return SOLAR_CONST * CLEAR_SKY_TRANS * max(0.0, np.sin(np.radians(elev_deg)))


def get_wrf_times(ncfile):
    """提取文件内全部时间戳。"""
    times = ncfile.variables['Times']
    out = []
    for i in range(times.shape[0]):
        chars = times[i]
        ts = b''.join([c if isinstance(c, bytes) else c.encode('utf-8')
                       for c in chars]).decode('utf-8').strip()
        out.append(datetime.strptime(ts, '%Y-%m-%d_%H:%M:%S'))
    return out


def destagger_u(u_stag):
    """U 沿 west_east 去交错（最后一个轴）。"""
    return (u_stag[..., :-1] + u_stag[..., 1:]) / 2.0


def destagger_v(v_stag):
    """V 沿 south_north 去交错（倒数第二个轴）。"""
    return (v_stag[..., :-1, :] + v_stag[..., 1:, :]) / 2.0


def destagger_w(w_k, w_k1):
    """W 沿垂直方向去交错：相邻两层平均。"""
    return (w_k + w_k1) / 2.0


def crop_pad(field2d):
    """去 5 点缓冲区 -> 微 pad 到 TARGET_SHAPE，边缘复制填充。"""
    field = field2d[BUFFER:-BUFFER, BUFFER:-BUFFER]  # (89, 110)
    target = TARGET_SHAPE
    pad_y = target[0] - field.shape[0]
    pad_x = target[1] - field.shape[1]
    assert pad_y >= 0 and pad_x >= 0, "裁剪后尺寸超过目标形状"
    return np.pad(field, ((0, pad_y), (0, pad_x)), mode='edge')


def create_lr_field(hr_data, base_shape):
    """HR -> LR：平均池化降采样再双三次插值回原始尺寸。"""
    hr_tensor = torch.from_numpy(hr_data).float().unsqueeze(0).unsqueeze(0)
    lr_tensor = F.avg_pool2d(hr_tensor, kernel_size=DOWNSAMPLING_FACTOR)
    lr_interp = F.interpolate(lr_tensor, size=base_shape, mode='bicubic',
                              align_corners=False)
    return lr_interp.squeeze().numpy()


def process_wrf_file(wrf_file_path, out_dir, lon_deg, lat_deg, scheme):
    """处理单个 WRF 文件（含 6 个时次），返回生成的 npz 路径列表。"""
    created = []
    ncfile = None
    try:
        ncfile = Dataset(wrf_file_path)
        timestamps = get_wrf_times(ncfile)
        n_t = len(timestamps)

        U_stag_all = ncfile.variables['U'][:]
        V_stag_all = ncfile.variables['V'][:]
        W_stag_all = ncfile.variables['W'][:]

        surf_cache = {}
        for wrf_name in SURFACE_VARS_WRF:
            if wrf_name not in ncfile.variables:
                print("--> 警告: 缺少条件变量 '{}'，跳过文件 {}".format(
                    wrf_name, os.path.basename(wrf_file_path)))
                return created
            surf_cache[wrf_name] = ncfile.variables[wrf_name][:]

        for t in range(n_t):
            npz_output_data = {}

            U_stag = U_stag_all[t]
            V_stag = V_stag_all[t]
            W_stag = W_stag_all[t]

            # 去交错后基准 shape：(south_north, west_east) = (99, 120)
            base_shape = destagger_u(U_stag[0]).shape
            assert base_shape == (99, 120), \
                "网格尺寸与预期不符: {}".format(base_shape)

            for level_name, level_idx in LEVELS.items():
                u_2d = destagger_u(U_stag[level_idx])
                v_2d = destagger_v(V_stag[level_idx])
                w_2d = destagger_w(W_stag[level_idx], W_stag[level_idx + 1])
                assert u_2d.shape == v_2d.shape == w_2d.shape, "U/V/W shape 不一致"

                u_final = crop_pad(np.array(u_2d, dtype=np.float32))
                v_final = crop_pad(np.array(v_2d, dtype=np.float32))
                w_final = crop_pad(np.array(w_2d, dtype=np.float32))
                assert u_final.shape == TARGET_SHAPE

                npz_output_data['hr_u_' + level_name] = u_final
                npz_output_data['hr_v_' + level_name] = v_final
                npz_output_data['hr_w_' + level_name] = w_final
                npz_output_data['lr_u_' + level_name] = create_lr_field(u_final, TARGET_SHAPE)
                npz_output_data['lr_v_' + level_name] = create_lr_field(v_final, TARGET_SHAPE)
                npz_output_data['lr_w_' + level_name] = create_lr_field(w_final, TARGET_SHAPE)

            for wrf_name, npz_name in SURFACE_VARS_WRF.items():
                var_np = np.array(surf_cache[wrf_name][t], dtype=np.float32)
                if var_np.ndim > 2:
                    var_np = var_np.squeeze()
                if var_np.shape == (99, 120) or var_np.shape[-2:] == (99, 120):
                    npz_output_data[npz_name] = crop_pad(var_np)
                else:
                    print("--> 警告: 条件变量 '{}' shape {} 异常，跳过".format(
                        npz_name, var_np.shape))
                    return created

            # 合成 swdown（太阳几何，仅用于昼夜过滤）
            elev = solar_elevation(timestamps[t], lon_deg, lat_deg)
            swdown_val = synthetic_swdown(elev)
            npz_output_data['swdown'] = np.full(TARGET_SHAPE, swdown_val, dtype=np.float32)

            # 尺寸自检
            for key, value in npz_output_data.items():
                if value.ndim >= 2 and value.shape[-2:] != TARGET_SHAPE:
                    raise ValueError("变量 '{}' 维度 {} 与目标 {} 不匹配".format(
                        key, value.shape, TARGET_SHAPE))

            # myj/ysu 时间戳相同，文件名加 scheme 前缀避免冲突
            output_filename = scheme + "_" + timestamps[t].strftime('%Y%m%dT%H%M%S') + ".npz"
            output_filepath = os.path.join(out_dir, output_filename)
            np.savez_compressed(output_filepath, **npz_output_data)
            created.append(output_filepath)

    finally:
        if ncfile:
            ncfile.close()
    return created


def _worker(args):
    wrf_path, out_dir, lon_deg, lat_deg, scheme = args
    return process_wrf_file(wrf_path, out_dir, lon_deg, lat_deg, scheme)


def get_domain_lon_lat(wrf_dir):
    """从第一个文件读取域平均经纬度（用于太阳几何）。"""
    files = sorted(glob.glob(os.path.join(wrf_dir, "wrfout_d04_*")))
    if not files:
        print("错误：在 '{}' 中未找到 wrfout_d04_* 文件。".format(wrf_dir))
        sys.exit(1)
    with Dataset(files[0]) as nc:
        lon = float(np.mean(nc.variables['XLONG'][:]))
        lat = float(np.mean(nc.variables['XLAT'][:]))
    return lon, lat


def compute_stats(npz_paths, out_dirs):
    """读取全部 npz 计算 biases/scales，并统计昼夜样本数。"""
    print("\n开始计算统计量 ({} 个文件)...".format(len(npz_paths)))
    if not npz_paths:
        print("没有 npz 文件，跳过统计。")
        return

    with np.load(npz_paths[0]) as data:
        keys = sorted(data.keys())

    sums = {k: 0.0 for k in keys}
    sq_sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    n_day = 0
    n_night = 0
    n_other = 0

    for npz_path in npz_paths:
        with np.load(npz_path) as data:
            for k in keys:
                v = data[k].astype(np.float64)
                sums[k] += np.nansum(v)
                sq_sums[k] += np.nansum(v * v)
                counts[k] += np.isfinite(v).sum()
            sw = float(np.mean(data['swdown']))
            if sw > 50.0:
                n_day += 1
            elif sw < 5.0:
                n_night += 1
            else:
                n_other += 1

    print("\n--- 数据集统计结果 (用于更新 config) ---")
    print("  biases:")
    for k in keys:
        print("    {}: {:.6f}".format(k, sums[k] / max(counts[k], 1)))
    print("  scales:")
    for k in keys:
        mean = sums[k] / max(counts[k], 1)
        var = sq_sums[k] / max(counts[k], 1) - mean * mean
        std = np.sqrt(max(var, 0.0))
        if std == 0.0:
            std = 1.0
            print("    {}: {:.6f}  # WARNING: std=0, 设为1.0".format(k, std))
        else:
            print("    {}: {:.6f}".format(k, std))

    print("\n  昼夜样本数: day={} night={} twilight/other={}".format(
        n_day, n_night, n_other))
    print("  输出目录: {}".format(out_dirs))


def main():
    parser = argparse.ArgumentParser(description="深圳 WRF d04 -> npz 预处理")
    parser.add_argument("--scheme", default="both", choices=["myj", "ysu", "both"],
                        help="处理哪个 PBL 方案")
    parser.add_argument("--limit", type=int, default=0,
                        help="每个方案只处理前 N 个文件（冒烟测试）")
    parser.add_argument("--output_base", default=OUTPUT_BASE)
    parser.add_argument("--workers", type=int, default=1, help="并行进程数")
    parser.add_argument("--skip_stats", action="store_true",
                        help="跳过统计量计算")
    args = parser.parse_args()

    schemes = ["myj", "ysu"] if args.scheme == "both" else [args.scheme]

    all_npz = []
    os.makedirs(args.output_base, exist_ok=True)
    for scheme in schemes:
        wrf_dir = os.path.join(WRF_BASE, "meso_202007_" + scheme)
        out_dir = args.output_base
        print("输入目录: {}".format(wrf_dir))
        print("输出目录: {}".format(out_dir))

        lon, lat = get_domain_lon_lat(wrf_dir)
        print("域平均经纬度: lon={:.3f} lat={:.3f}".format(lon, lat))

        all_files = sorted(glob.glob(os.path.join(wrf_dir, "wrfout_d04_*")))
        if args.limit > 0:
            all_files = all_files[:args.limit]
        print("待处理文件数: {}".format(len(all_files)))

        tasks = [(f, out_dir, lon, lat, scheme) for f in all_files]
        created = []
        if args.workers > 1 and len(tasks) > 1:
            pool = multiprocessing.Pool(processes=args.workers)
            for i, result in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
                created.extend(result)
                if (i + 1) % 50 == 0:
                    print("  进度: {}/{}".format(i + 1, len(tasks)))
            pool.close()
            pool.join()
        else:
            for i, t in enumerate(tasks):
                created.extend(_worker(t))
                if (i + 1) % 50 == 0:
                    print("  进度: {}/{}".format(i + 1, len(tasks)))

        print("scheme {}: 生成 {} 个 npz".format(scheme, len(created)))
        all_npz.extend(created)

    if not args.skip_stats:
        compute_stats(all_npz, [args.output_base])

    print("\n完成。")


if __name__ == "__main__":
    main()
