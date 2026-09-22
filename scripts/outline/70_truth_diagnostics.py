# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · T0.5 真值诊断

① 各档位动能谱并标注约 7Δx 有效分辨率
② d04 原生 C 网格可压缩散度 ∇·(ρu) 残差统计(阶段 2 约束权重定标基准)
③ d02/d03 相对 d04 的位置偏移(强风时刻,互相关峰值位移)
④ myj 与 ysu 两套参数化配置输出的差异量级

运行(pytorch-gpu 环境,需 20/21 全量抽取完成):
  python scripts/outline/70_truth_diagnostics.py
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from outline_common import (OUT_COARSE, OUT_FINE, OUT_STATIC, SCHEME_DIRS, WRF_BASE,
                            domain_files)
from src.dl_data.wind_canvas_statics import FINE_SHAPES, CanvasStatics

DX = {'d01': 27000.0, 'd02': 9000.0, 'd03': 3000.0, 'd04': 1000.0}
RD, RV, CP, P0, G = 287.0, 461.6, 1004.5, 100000.0, 9.81


def parse_stamp(name):
    m = re.search(r'_(\d{8}T\d{6})\.npz$', name)
    return m.group(1) if m else None


def radial_spectrum(field):
    """2D FFT -> 径向平均功率谱,(k, P)。"""
    f = field - field.mean()
    nx, ny = f.shape[1], f.shape[0]
    p = np.abs(np.fft.fft2(f)) ** 2 / (nx * ny)
    kx = np.fft.fftfreq(nx) * nx      # 周期数
    ky = np.fft.fftfreq(ny) * ny
    kx2, ky2 = np.meshgrid(kx, ky)
    kr = np.sqrt(kx2 ** 2 + ky2 ** 2).ravel()
    pr = p.ravel()
    nb = int(min(nx, ny) / 2)
    idx = np.clip(np.round(kr).astype(int), 0, nb)
    out = np.bincount(idx, weights=pr, minlength=nb + 1)
    cnt = np.bincount(idx, minlength=nb + 1)
    k = np.arange(nb + 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return k[1:], (out / np.maximum(cnt, 1))[1:]


def band_slope(k, P, k1, k2):
    """log-log 谱斜率(k∈[k1,k2] 最小二乘);用于看 7Δx 前后是否变平(噪声/不解析)。"""
    m = (k >= k1) & (k <= k2)
    if int(m.sum()) < 3:
        return None
    return float(np.polyfit(np.log(k[m]), np.log(P[m]), 1)[0])


def tier_spectra(scheme, stamp, level=0):
    rep = {}
    day = '{}-{}-{}'.format(stamp[0:4], stamp[4:6], stamp[6:8])
    for dom in ('d01', 'd02', 'd03', 'd04'):
        files = [f for f in domain_files(scheme, dom) if pd(f) == day]
        if not files:
            continue
        with Dataset(files[0]) as nc:
            n_t = nc.variables['U'].shape[0]
            t = (int(stamp[11:13]) // 10) if dom == 'd04' else int(stamp[9:11])
            t = min(t, n_t - 1)
            u = np.array(nc.variables['U'][t, level], dtype=np.float64)
            v = np.array(nc.variables['V'][t, level], dtype=np.float64)
            # 统一到质量点尺寸以便同口径比较:去交错
            um = 0.5 * (u[:, :-1] + u[:, 1:])
            vm = 0.5 * (v[:-1, :] + v[1:, :])
            e = um ** 2 + vm ** 2
        k, pk = radial_spectrum(e)
        # 原始 E(k) 是"红谱",峰值恒在最大尺度(k=1)无信息;能量含尺度看 k·E(k) 峰值
        kp = k * pk
        k_peak = int(k[np.argmax(kp)])
        k7 = max(1, int(round(e.shape[1] / 7.0)))
        rep[dom] = {'k': k.tolist(), 'P': pk.tolist(), 'kP': kp.tolist(),
                    'wavelength_7dx_km': 7.0 * DX[dom] / 1000.0,
                    'k_7dx': k7, 'P_at_7dx': float(pk[min(k7, len(pk)) - 1]),
                    'k_peak_energy': k_peak,
                    'wavelength_peak_km': e.shape[1] * DX[dom] / 1000.0 / k_peak,
                    'slope_meso_2_to_7dx': band_slope(k, pk, 2, k7),
                    'slope_beyond_7dx': band_slope(k, pk, k7 + 1, int(k[-1])),
                    'n_cells': list(e.shape)}
    return rep


def pd(name):
    m = re.search(r'wrfout_d\d+_(\d{4}-\d{2}-\d{2})', name)
    return m.group(1) if m else None


def divergence_residual(scheme, stamp, levels=(1, 5, 10, 20)):
    """d04 原生 C 网格 ∇·(ρu) 残差统计(可压缩形式,内部点,物理单位)。

    ρ 由 P/PB/T/QVAPOR 诊断后按原生交错索引平均到 U/V/W 点,再中心差分;
    水平用 d04 dx,垂直 dz 取界面高度差(PHB+PH)。只在质量点内部(去掉一层边界)统计;
    除绝对量(kg m^-3 s^-1)外给出 p95/ρ 的归一值(s^-1),作阶段 2 约束权重定标基准。
    """
    day = '{}-{}-{}'.format(stamp[0:4], stamp[4:6], stamp[6:8])
    hh = stamp[9:11]
    wpath = os.path.join(WRF_BASE, SCHEME_DIRS[scheme],
                         "wrfout_d04_{}_{}:00:00".format(day, hh))
    tidx = int(stamp[11:13]) // 10
    with Dataset(wpath) as nc:
        phb = np.array(nc.variables['PHB'][tidx], dtype=np.float64)
        phi = np.array(nc.variables['PH'][tidx], dtype=np.float64)
        u = np.array(nc.variables['U'][tidx], dtype=np.float64)   # (nz, ny, nx+1)
        v = np.array(nc.variables['V'][tidx], dtype=np.float64)   # (nz, ny+1, nx)
        w = np.array(nc.variables['W'][tidx], dtype=np.float64)   # (nz+1, ny, nx)
        p = np.array(nc.variables['P'][tidx], dtype=np.float64) + \
            np.array(nc.variables['PB'][tidx], dtype=np.float64)
        theta = np.array(nc.variables['T'][tidx], dtype=np.float64) + 300.0
        qv = np.maximum(np.array(nc.variables['QVAPOR'][tidx], dtype=np.float64), 0.0)
    z = (phb + phi) / G                                    # (nz+1, ny, nx) 界面高度
    t_full = theta * np.power(p / P0, RD / CP)
    rho = p / (RD * t_full * (1.0 + 0.61 * qv))            # (nz, ny, nx) 质量点密度
    nz = rho.shape[0]
    dx = DX['d04']
    # 水平:ρ 平均到 U/V 点(内部点 a <-> U 点 a+1),通量差 -> 质量点 i/j = 1..n-2
    rho_u = 0.5 * (rho[:, :, :-1] + rho[:, :, 1:])         # (nz, ny, nx-1) <-> U 点 1..nx-1
    du = (rho_u[:, :, 1:] * u[:, :, 2:-1] - rho_u[:, :, :-1] * u[:, :, 1:-2]) / dx
    rho_v = 0.5 * (rho[:, :-1, :] + rho[:, 1:, :])         # (nz, ny-1, nx) <-> V 点 1..ny-1
    dv = (rho_v[:, 1:, :] * v[:, 2:-1, :] - rho_v[:, :-1, :] * v[:, 1:-2, :]) / dx
    # 垂直:ρ 平均到界面 k=1..nz-1,通量差 / dz -> 质量层 k = 1..nz-2
    rho_w = 0.5 * (rho[:-1] + rho[1:])                     # (nz-1, ny, nx) <-> 界面 1..nz-1
    dz = z[1:nz + 1] - z[0:nz]                             # (nz, ny, nx) 质量层厚度
    dw = (rho_w[1:] * w[2:nz] - rho_w[:-1] * w[1:nz - 1]) / np.maximum(dz[1:-1], 1e-3)
    # 汇总到同一内部体元(质量层 1..nz-2、j=1..ny-2、i=1..nx-2)
    div = du[1:-1, 1:-1, :] + dv[1:-1, :, 1:-1] + dw[:, 1:-1, 1:-1]
    rho_in = rho[1:-1, 1:-1, 1:-1]
    out = {'shape': list(div.shape), 'dx_m': dx}
    for k in levels:
        if not 1 <= k <= nz - 2:
            continue
        d = div[k - 1]
        a = np.abs(d).ravel()
        rm = float(rho_in[k - 1].mean())
        out[str(k)] = {'mean': float(d.mean()), 'std': float(d.std()),
                       'p50_abs': float(np.percentile(a, 50)),
                       'p95_abs': float(np.percentile(a, 95)),
                       'p99_abs': float(np.percentile(a, 99)),
                       'rho_mean': rm,
                       'p95_abs_over_rho': float(np.percentile(a, 95) / max(rm, 1e-9))}
    return out


def mass_speed(cu, cv):
    """U/V 原生交错 -> 质量点风速(水平平均去交错)。"""
    u = 0.5 * (cu[:, :-1] + cu[:, 1:])
    v = 0.5 * (cv[:-1, :] + cv[1:, :])
    return np.sqrt(u ** 2 + v ** 2)


def block_mean_9(field, block=9):
    """1 km 场按 block×block 块平均到 ~9 km(与粗端同尺度比形态)。"""
    ny, nx = field.shape
    ny_b, nx_b = ny // block, nx // block
    return field[:ny_b * block, :nx_b * block].reshape(ny_b, block, nx_b, block).mean((1, 3))


def coarse_speed_aligned(stat, cu, cv):
    """粗端交错风速 -> 用重网格权重对齐到细端 99x120 质量网格。"""
    spd = mass_speed(cu, cv)                      # (ny_s, nx_s) 粗端质量点
    al = stat.regrid(spd.reshape(1, -1), 'mass')  # (1, 99*120) -> 细端质量网格
    return al.reshape(FINE_SHAPES['mass'])


def shift_search(a, b, max_shift=8):
    """b 相对 a 的整数偏移(最大相关)。"""
    best, arg = -1e18, (0, 0)
    for dj in range(-max_shift, max_shift + 1):
        for di in range(-max_shift, max_shift + 1):
            a2 = a[max(0, dj):a.shape[0] + min(0, dj), max(0, di):a.shape[1] + min(0, di)]
            b2 = b[max(0, -dj):b.shape[0] + min(0, -dj), max(0, -di):b.shape[1] + min(0, -di)]
            if a2.size < 0.4 * a.size:
                continue
            c = float(np.corrcoef(a2.ravel(), b2.ravel())[0, 1])
            if c > best:
                best, arg = c, (dj, di)
    return arg, best


def main():
    parser = argparse.ArgumentParser(description="阶段0 T0.5 真值诊断")
    parser.add_argument("--scheme", default="myj")
    parser.add_argument("--static_dir", default=OUT_STATIC)
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()
    rep = {}

    # ① 各档位动能谱(取月中一个整点)
    stamp = '20200715T120000'
    rep['spectra'] = tier_spectra(args.scheme, stamp, level=0)
    print("① 能谱(k·E 峰值 / 谱斜率:7Δx 前 -> 后): " + ", ".join(
        "{} 7dx={:.0f}km lam_pk={:.0f}km slope {:.2f}->{:.2f}".format(
            d, v['wavelength_7dx_km'], v['wavelength_peak_km'],
            v['slope_meso_2_to_7dx'] or float('nan'),
            v['slope_beyond_7dx'] or float('nan'))
        for d, v in rep['spectra'].items()))

    # ② 散度残差统计(三个时刻)
    div = {}
    for s in ('20200715T120000', '20200710T000000', '20200720T060000'):
        div[s] = divergence_residual(args.scheme, s)
    rep['divergence_residual'] = div
    print("② d04 ∇·(ρu) 残差(层 10,P95;绝对 kg m^-3 s^-1 / 除以 ρ 的 s^-1): " + json.dumps(
        {k: [round(v['10']['p95_abs'], 6), round(v['10']['p95_abs_over_rho'], 6)]
         for k, v in div.items()}))

    # ③ 位置偏移:粗端场先用重网格权重对齐到 d04 网格(消除网格错位),
    #    再各自块平均到 9 km 比形态;互相关峰偏离 (0,0) 才是真实的特征位移
    stat = CanvasStatics(args.static_dir)
    c_files = sorted(glob.glob(os.path.join(OUT_COARSE, "c_{}_*.npz".format(args.scheme))))
    speeds = {}
    for p in c_files[:72]:
        s = parse_stamp(os.path.basename(p))
        with np.load(p) as d:
            speeds[s] = (d['c_u'][0].copy(), d['c_v'][0].copy())
    top = sorted(speeds, key=lambda k: -mass_speed(*speeds[k]).mean())[:3]
    rep['shifts'] = {}
    for s in top:
        cu, cv = speeds[s]
        ca = coarse_speed_aligned(stat, cu, cv)          # (99,120) 9 km -> 1 km 网格
        with np.load(os.path.join(OUT_FINE, "f_{}_{}.npz".format(args.scheme, s))) as d:
            fspd_m = mass_speed(d['f_u'][0], d['f_v'][0])
        ca_b = block_mean_9(ca)
        fi_b = block_mean_9(fspd_m)
        (dj, di), c = shift_search(ca_b, fi_b, 3)
        rep['shifts'][s] = {'coarse_mean_speed': float(mass_speed(cu, cv).mean()),
                            'shift_9km_cells': [dj, di],
                            'shift_km': [dj * 9.0, di * 9.0], 'corr': c}
    print("③ 强风时刻 d02 相对 d04 偏移(已网格对齐,9 km 块平均): " + json.dumps(
        {k: v['shift_km'] for k, v in rep['shifts'].items()}))

    # ④ myj vs ysu 差异
    diffs = []
    for p in c_files[:72]:
        s = parse_stamp(os.path.basename(p))
        q = os.path.join(OUT_COARSE, "c_ysu_{}.npz".format(s))
        if not os.path.isfile(q):
            continue
        with np.load(p) as a, np.load(q) as b:
            spd_m = mass_speed(a['c_u'][0], a['c_v'][0])
            spd_y = mass_speed(b['c_u'][0], b['c_v'][0])
            diffs.append([float(np.sqrt(((spd_m - spd_y) ** 2).mean())),
                          float(abs(a['c_pblh'].mean() - b['c_pblh'].mean())),
                          float(np.sqrt(((a['c_ust'] - b['c_ust']) ** 2).mean()))])
    if diffs:
        arr = np.array(diffs)
        rep['scheme_diff'] = {
            'n_hours': int(arr.shape[0]),
            'wind_speed_rmse_mean': float(arr[:, 0].mean()),
            'wind_speed_rmse_p95': float(np.percentile(arr[:, 0], 95)),
            'pblh_mean_abs_diff': float(arr[:, 1].mean()),
            'ust_rmse_mean': float(arr[:, 2].mean()),
        }
        print("④ myj vs ysu: " + json.dumps(rep['scheme_diff']))

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print("写出: " + args.json_out)


if __name__ == "__main__":
    main()
