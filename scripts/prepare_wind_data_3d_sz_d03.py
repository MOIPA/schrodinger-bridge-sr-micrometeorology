# -*- coding: utf-8 -*-
"""
深圳 WRF d03(原生 3km)真实低精度数据预处理脚本

用途:真实低精度实验(第二阶段,必做)。把 d03 原生场与 d04 HR 场配对,
生成 (lr_* = 原生 d03 重采样, hr_* = d04 真值) 的 npz。

与 d04 预处理脚本(prepare_wind_data_3d_sz.py)的关键差异:
  1. d03 场是真实 3km 输出,不是 d04 HR 降采样 —— "真实退化"而非"完美退化"
  2. d03 域比 d04 大、网格不对齐 —— 用 XLAT/XLONG 把 d03 场插值到
     d04 的 (99,120) 质量点网格,再复用 crop_pad 裁剪到 (96,112)
  3. 时间对齐:d03 小时级输出,只取整点,与 d04 同一时刻配对
  4. 条件变量同样用 d03 版本(lr_t2/lr_z/...)—— 真实全低精度场景
  5. npz key 名与现管线完全一致(hr_*/lr_*/条件名),数据集类零改动:
     - 现有 d04 模型直接吃 d03 npz 的 lr_* 即可做跨域评估(实验#2)
     - 训练 d03 模型只需换 dl_data_ver(实验#3)

运行(服务器,pytorch-gpu 环境有 netCDF4 + scipy):
  /fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/prepare_wind_data_3d_sz_d03.py \
      --scheme both --limit 2   # 冒烟测试
  ... 全量时去掉 --limit(建议 --workers 8)
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

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: 未找到 scipy,无法插值。pip install scipy")

# ==============================================================================
# ---                       配置部分 (d03 真实低精度)                        ---
# ==============================================================================

WRF_BASE = "/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen"
OUTPUT_BASE = "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d_sz_d03"

BUFFER = 5
TARGET_SHAPE = (96, 112)

# eta 模式层配置(与 d04 一致,嵌套模拟垂直层相同)
LEVELS = {'ml0': 0, 'ml1': 1, 'ml2': 2, 'ml3': 3, 'ml5': 5, 'ml10': 10}

# 条件变量 - 地表 (WRF变量名 -> npz key名)
SURFACE_VARS_WRF = {
    'T2': 't2', 'HGT': 'z', 'LU_INDEX': 'lu', 'TSK': 'tsk',
    'HFX': 'hfx', 'LH': 'lh', 'PSFC': 'psfc', 'PBLH': 'pblh'
}

# 合成 swdown 用(清晰天空近似,仅用于昼夜过滤)
SOLAR_CONST = 1361.0
CLEAR_SKY_TRANS = 0.7

# --- 配置结束 ---
# ==============================================================================


def solar_elevation(dt_utc, lon_deg, lat_deg):
    """给定 UTC 时刻和经纬度,计算太阳高度角(度)。"""
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
    """清晰天空合成下行短波辐射(W/m2),仅用于昼夜过滤。"""
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
    """U 沿 west_east 去交错(最后一个轴)。"""
    return (u_stag[..., :-1] + u_stag[..., 1:]) / 2.0


def destagger_v(v_stag):
    """V 沿 south_north 去交错(倒数第二个轴)。"""
    return (v_stag[..., :-1, :] + v_stag[..., 1:, :]) / 2.0


def destagger_w(w_k, w_k1):
    """W 沿垂直方向去交错:相邻两层平均。"""
    return (w_k + w_k1) / 2.0


def crop_pad(field2d):
    """去 BUFFER 点缓冲区 -> 微 pad 到 TARGET_SHAPE,边缘复制填充。"""
    field = field2d[BUFFER:-BUFFER, BUFFER:-BUFFER]
    target = TARGET_SHAPE
    pad_y = target[0] - field.shape[0]
    pad_x = target[1] - field.shape[1]
    assert pad_y >= 0 and pad_x >= 0, "裁剪后尺寸超过目标形状"
    return np.pad(field, ((0, pad_y), (0, pad_x)), mode='edge')


def interpolate_to_d04_grid(field2d, d03_lon, d03_lat, d04_lon, d04_lat,
                            method='linear'):
    """把 d03 网格上的 2D 场插值到 d04 质量点网格。

    field2d: (d03_sn, d03_we) 去交错后质量点场
    返回: (d04_sn, d04_we) 数组;线性插值溢出点用最近邻填充。
    """
    pts = np.stack([d03_lon.ravel(), d03_lat.ravel()], axis=1)
    xi = np.stack([d04_lon.ravel(), d04_lat.ravel()], axis=1)
    vals = field2d.ravel()
    out = griddata(pts, vals, xi, method=method)
    # 线性插值在凸包外会产生 nan -> 最近邻填补(保证全覆盖)
    if method == 'linear' and np.any(~np.isfinite(out)):
        nn = griddata(pts, vals, xi, method='nearest')
        out[~np.isfinite(out)] = nn[~np.isfinite(out)]
    return out.reshape(d04_lon.shape)


def build_d04_index(wrf_dir):
    """构建 d04 时次索引: {timestamp: (file_path, t_idx)}。

    返回 (index, d04_lon, d04_lat):d04_lon/lat 为参考网格(99,120)。
    """
    d04_files = sorted(glob.glob(os.path.join(wrf_dir, "wrfout_d04_*")))
    index = {}
    ref_lon = ref_lat = None
    for fp in d04_files:
        with Dataset(fp) as nc:
            if ref_lon is None:
                ref_lon = np.array(nc.variables['XLONG'][0], dtype=np.float64)
                ref_lat = np.array(nc.variables['XLAT'][0], dtype=np.float64)
            times = get_wrf_times(nc)
            for t, ts in enumerate(times):
                index[ts] = (fp, t)
    return index, ref_lon, ref_lat


def load_d04_sample(index, timestamp):
    """按时间戳取 d04 的 HR 场(全部变量,裁剪前网格)。"""
    fp, t = index[timestamp]
    with Dataset(fp) as nc:
        out = {}
        U = nc.variables['U'][t]
        V = nc.variables['V'][t]
        W = nc.variables['W'][t]
        for lname, lidx in LEVELS.items():
            u2d = destagger_u(U[lidx])
            v2d = destagger_v(V[lidx])
            w2d = destagger_w(W[lidx], W[lidx + 1])
            out['hr_u_' + lname] = np.array(u2d, dtype=np.float32)
            out['hr_v_' + lname] = np.array(v2d, dtype=np.float32)
            out['hr_w_' + lname] = np.array(w2d, dtype=np.float32)
        for wrf_name, npz_name in SURFACE_VARS_WRF.items():
            var = np.array(nc.variables[wrf_name][t], dtype=np.float32)
            if var.ndim > 2:
                var = var.squeeze()
            out[npz_name] = var
    return out


def process_d03_file(wrf_file_path, out_dir, lon_deg, lat_deg, scheme,
                     d04_index, d04_lon, d04_lat):
    """处理单个 d03 文件:每个时次插值到 d04 网格、与 d04 配对、写 npz。"""
    created = []
    ncfile = None
    try:
        ncfile = Dataset(wrf_file_path)
        timestamps = get_wrf_times(ncfile)
        n_t = len(timestamps)

        # d03 参考网格坐标(质量点)
        d03_lon = np.array(ncfile.variables['XLONG'][0], dtype=np.float64)
        d03_lat = np.array(ncfile.variables['XLAT'][0], dtype=np.float64)
        # 去交错后 U 基准 shape(质量点网格)
        base_shape = destagger_u(ncfile.variables['U'][0]).shape
        # XLONG 若带 Time 轴则取第一个时次
        if d03_lon.ndim > 2:
            d03_lon, d03_lat = d03_lon[0], d03_lat[0]
        assert d03_lon.shape == base_shape, \
            "d03 质量点网格 {} 与去交错场 {} 不符".format(d03_lon.shape, base_shape)
        print("  d03 网格: {} (去交错后 {})".format(d03_lon.shape, base_shape))

        surf_cache = {}
        for wrf_name in SURFACE_VARS_WRF:
            if wrf_name not in ncfile.variables:
                print("--> 警告: 缺少条件变量 '{}',跳过文件 {}".format(
                    wrf_name, os.path.basename(wrf_file_path)))
                return created
            surf_cache[wrf_name] = ncfile.variables[wrf_name][:]

        for t in range(n_t):
            ts = timestamps[t]
            # 只处理整点(d03 小时级输出;防 10-min 残余)
            if (ts.minute != 0) or (ts.second != 0):
                continue
            if ts not in d04_index:
                print("--> 跳过 {}: d04 无对应时次".format(ts))
                continue

            d04_sample = load_d04_sample(d04_index, ts)
            npz_output_data = dict(d04_sample)  # hr_* + HR 条件(裁剪前网格)

            U = ncfile.variables['U'][t]
            V = ncfile.variables['V'][t]
            W = ncfile.variables['W'][t]

            # --- 风场:d03 原生 -> 插值到 d04 网格 ---
            for lname, lidx in LEVELS.items():
                u2d = np.array(destagger_u(U[lidx]), dtype=np.float32)
                v2d = np.array(destagger_v(V[lidx]), dtype=np.float32)
                w2d = np.array(destagger_w(W[lidx], W[lidx + 1]), dtype=np.float32)
                for prefix, f in [('lr_u_', u2d), ('lr_v_', v2d), ('lr_w_', w2d)]:
                    interp = interpolate_to_d04_grid(f, d03_lon, d03_lat,
                                                     d04_lon, d04_lat)
                    npz_output_data[prefix + lname] = crop_pad(interp)

            # --- 条件变量:d03 版本(真实全低精度场景) ---
            for wrf_name, npz_name in SURFACE_VARS_WRF.items():
                var = np.array(surf_cache[wrf_name][t], dtype=np.float32)
                if var.ndim > 2:
                    var = var.squeeze()
                if var.shape != d03_lon.shape:
                    print("--> 警告: 条件变量 '{}' shape {} 与网格 {} 不符,跳过".format(
                        npz_name, var.shape, d03_lon.shape))
                    return created
                # lu 是类别变量用最近邻,其余线性
                method = 'nearest' if npz_name == 'lu' else 'linear'
                interp = interpolate_to_d04_grid(var, d03_lon, d03_lat,
                                                 d04_lon, d04_lat, method=method)
                npz_output_data['lr_' + npz_name] = crop_pad(interp)

            # --- 裁剪所有 d04 场到 (96,112) ---
            for key, value in list(npz_output_data.items()):
                if value.ndim >= 2 and value.shape[-2:] == d04_lon.shape:
                    npz_output_data[key] = crop_pad(value)
                elif value.ndim >= 2 and value.shape[-2:] != TARGET_SHAPE:
                    raise ValueError("变量 '{}' 维度 {} 异常".format(
                        key, value.shape))

            # --- 合成 swdown(仅用于昼夜过滤) ---
            elev = solar_elevation(ts, lon_deg, lat_deg)
            swdown_val = synthetic_swdown(elev)
            npz_output_data['swdown'] = np.full(TARGET_SHAPE, swdown_val,
                                                dtype=np.float32)

            # 尺寸自检
            for key, value in npz_output_data.items():
                if value.ndim >= 2 and value.shape[-2:] != TARGET_SHAPE:
                    raise ValueError("变量 '{}' 维度 {} 与目标 {} 不匹配".format(
                        key, value.shape, TARGET_SHAPE))

            output_filename = scheme + "_" + ts.strftime('%Y%m%dT%H%M%S') + ".npz"
            output_filepath = os.path.join(out_dir, output_filename)
            if os.path.exists(output_filepath):
                continue  # 断点续跑:已生成的跳过
            np.savez_compressed(output_filepath, **npz_output_data)
            created.append(output_filepath)
    finally:
        if ncfile:
            ncfile.close()
    return created


def _worker(args):
    (wrf_path, out_dir, lon_deg, lat_deg, scheme,
     d04_index, d04_lon, d04_lat) = args
    return process_d03_file(wrf_path, out_dir, lon_deg, lat_deg, scheme,
                            d04_index, d04_lon, d04_lat)


def get_domain_lon_lat(wrf_dir):
    """从第一个 d03 文件读取域平均经纬度(用于太阳几何)。"""
    files = sorted(glob.glob(os.path.join(wrf_dir, "wrfout_d03_*")))
    if not files:
        print("错误:在 '{}' 中未找到 wrfout_d03_* 文件。".format(wrf_dir))
        sys.exit(1)
    with Dataset(files[0]) as nc:
        lon = float(np.mean(nc.variables['XLONG'][:]))
        lat = float(np.mean(nc.variables['XLAT'][:]))
    return lon, lat


def compute_stats(npz_paths, out_dirs):
    """读取全部 npz 计算 biases/scales,并统计昼夜样本数。"""
    print("\n开始计算统计量 ({} 个文件)...".format(len(npz_paths)))
    if not npz_paths:
        print("没有 npz 文件,跳过统计。")
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
    parser = argparse.ArgumentParser(description="深圳 WRF d03 -> npz 预处理")
    parser.add_argument("--scheme", default="both", choices=["myj", "ysu", "both"])
    parser.add_argument("--limit", type=int, default=0,
                        help="每个方案只处理前 N 个文件(冒烟测试)")
    parser.add_argument("--output_base", default=OUTPUT_BASE)
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    parser.add_argument("--skip_stats", action="store_true")
    args = parser.parse_args()

    if not HAS_SCIPY:
        print("错误:需要 scipy 做插值。退出。")
        sys.exit(1)

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

        # d04 索引(配对目标)+ 参考网格
        print("构建 d04 时次索引...")
        d04_index, d04_lon, d04_lat = build_d04_index(wrf_dir)
        print("  d04 时次数: {}, 参考网格: {}".format(len(d04_index), d04_lon.shape))

        all_files = sorted(glob.glob(os.path.join(wrf_dir, "wrfout_d03_*")))
        if args.limit > 0:
            all_files = all_files[:args.limit]
        print("待处理 d03 文件数: {}".format(len(all_files)))

        tasks = [(f, out_dir, lon, lat, scheme, d04_index, d04_lon, d04_lat)
                 for f in all_files]
        created = []
        if args.workers > 1 and len(tasks) > 1:
            pool = multiprocessing.Pool(processes=args.workers)
            for i, result in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
                created.extend(result)
                if (i + 1) % 20 == 0:
                    print("  进度: {}/{}".format(i + 1, len(tasks)))
            pool.close()
            pool.join()
        else:
            for i, t in enumerate(tasks):
                created.extend(_worker(t))
                if (i + 1) % 20 == 0:
                    print("  进度: {}/{}".format(i + 1, len(tasks)))

        print("scheme {}: 生成 {} 个 npz".format(scheme, len(created)))
        all_npz.extend(created)

    if not args.skip_stats:
        compute_stats(all_npz, [args.output_base])

    print("\n完成。")


if __name__ == "__main__":
    main()
