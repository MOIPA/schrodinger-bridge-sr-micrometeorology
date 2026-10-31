# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · T0.2 粗端抽取(默认 d02/9 km;--domain d01 可做 27 km 档)

原生 Arakawa C 网格,不插质量点,逐时。每时刻一个 npz:
  c_{scheme}_{YYYYMMDDTHHMMSS}.npz
  keys: c_u (40,120,151), c_v (40,121,150), c_w (41,120,150),
        c_theta (40,120,150)   # 位温 = T(扰动) + 300
        c_ph (41,120,150)      # 逐时扰动位势(可选消融输入)
        c_rmol/ust/pblh/hfx/t2/psfc (120,150)
        c_coszen (120,150)     # 天文 cos(SZA),max(cos,0)
        c_coszen_wrf (120,150) # 仅 ysu 有(用于 T0.2 核实 SZA 是否纯天文量)

运行(pytorch-gpu 环境):
  python scripts/outline/21_extract_coarse.py --scheme both --workers 4
"""
import argparse
import multiprocessing
import os
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline_common import (N_IFACE, N_MASS, OUT_COARSE, SCHEMES,
                            cos_sza_field, domain_files, ensure_dir,
                            read_times, stamp)

SURFACE_VARS = {
    'RMOL': 'c_rmol',
    'UST': 'c_ust',
    'PBLH': 'c_pblh',
    'HFX': 'c_hfx',
    'T2': 'c_t2',
    'PSFC': 'c_psfc',
}
def shapes_for(ny_mass, nx_mass):
    """按文件里的质量点网格 (ny, nx) 给出 C 网格各面形状(不写死域尺寸)。

    WRF 原生交错: U 面 nx+1 列、V 面 ny+1 行、W 面水平与质量点同网格(只垂直到 nz+1)。
    """
    return {
        'u': (N_MASS, ny_mass, nx_mass + 1),
        'v': (N_MASS, ny_mass + 1, nx_mass),
        'w': (N_IFACE, ny_mass, nx_mass),      # W 水平不交错(只垂直交错)
        'theta': (N_MASS, ny_mass, nx_mass),
        '2d': (ny_mass, nx_mass),
    }


def process_file(task):
    path, out_dir, scheme = task
    created = 0
    skipped = 0
    with Dataset(path) as nc:
        times = read_times(nc)
        u = np.asarray(nc.variables['U'][:, :N_MASS], dtype=np.float32)
        v = np.asarray(nc.variables['V'][:, :N_MASS], dtype=np.float32)
        w = np.asarray(nc.variables['W'][:, :N_IFACE], dtype=np.float32)
        theta = np.asarray(nc.variables['T'][:, :N_MASS], dtype=np.float32) + np.float32(300.0)
        ph = np.asarray(nc.variables['PH'][:, :N_IFACE], dtype=np.float32)
        surf = {}
        for wrf_name, key in SURFACE_VARS.items():
            if wrf_name in nc.variables:
                surf[key] = np.asarray(nc.variables[wrf_name][:], dtype=np.float32)
        xlat = np.asarray(nc.variables['XLAT'][0], dtype=np.float64)
        xlong = np.asarray(nc.variables['XLONG'][0], dtype=np.float64)
        coszen_wrf = None
        if 'COSZEN' in nc.variables:
            coszen_wrf = np.asarray(nc.variables['COSZEN'][:], dtype=np.float32)
    shp = shapes_for(theta.shape[-2], theta.shape[-1])   # theta: (t, lev, ny, nx)
    assert u.shape[1:] == shp['u'], "U shape {} != {}".format(u.shape, shp['u'])
    assert v.shape[1:] == shp['v'], "V shape {} != {}".format(v.shape, shp['v'])
    assert w.shape[1:] == shp['w'], "W shape {} != {}".format(w.shape, shp['w'])
    assert theta.shape[1:] == shp['theta'], "T shape {}".format(theta.shape)
    for t, dt in enumerate(times):
        out_path = os.path.join(out_dir, "c_{}_{}.npz".format(scheme, stamp(dt)))
        if os.path.exists(out_path):
            skipped += 1
            continue
        payload = {
            'c_u': u[t], 'c_v': v[t], 'c_w': w[t],
            'c_theta': theta[t], 'c_ph': ph[t],
            'c_coszen': cos_sza_field(xlat, xlong, dt),
        }
        for key, arr in surf.items():
            if arr.shape[1:] == shp['2d']:
                payload[key] = arr[t]
        if coszen_wrf is not None:
            payload['c_coszen_wrf'] = coszen_wrf[t]
        np.savez_compressed(out_path, **payload)
        created += 1
    return created, skipped


def main():
    parser = argparse.ArgumentParser(description="d02 粗端抽取(原生 C 网格)")
    parser.add_argument("--scheme", default="both", choices=["myj", "ysu", "both"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", default="d02", choices=["d01", "d02", "d03"],
                        help="粗端来源域;27 km 档用 d01,开发档默认 d02")
    parser.add_argument("--out_dir", default=OUT_COARSE)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    schemes = SCHEMES if args.scheme == "both" else [args.scheme]
    for scheme in schemes:
        files = domain_files(scheme, args.domain)
        if args.limit > 0:
            files = files[:args.limit]
        print("scheme {} [{}]: {} files -> {}".format(scheme, args.domain, len(files), args.out_dir))
        tasks = [(f, args.out_dir, scheme) for f in files]
        n_created = 0
        n_skipped = 0
        if args.workers > 1 and len(tasks) > 1:
            pool = multiprocessing.Pool(args.workers)
            for i, (c, s) in enumerate(pool.imap_unordered(process_file, tasks, chunksize=1)):
                n_created += c
                n_skipped += s
                if (i + 1) % 10 == 0:
                    print("  {}/{}".format(i + 1, len(tasks)))
            pool.close()
            pool.join()
        else:
            for i, t in enumerate(tasks):
                c, s = process_file(t)
                n_created += c
                n_skipped += s
                if (i + 1) % 10 == 0:
                    print("  {}/{}".format(i + 1, len(tasks)))
        print("scheme {}: created {} skipped {}".format(scheme, n_created, n_skipped))
    print("done")


if __name__ == "__main__":
    main()
