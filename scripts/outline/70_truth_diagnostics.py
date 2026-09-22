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
from outline_common import OUT_COARSE, OUT_FINE, SCHEME_DIRS, WRF_BASE, domain_files

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
        rep[dom] = {'k': k.tolist(), 'P': pk.tolist(),
                    'wavelength_7dx_km': 7.0 * DX[dom] / 1000.0,
                    'n_cells': list(e.shape)}
    return rep


def pd(name):
    m = re.search(r'wrfout_d\d+_(\d{4}-\d{2}-\d{2})', name)
    return m.group(1) if m else None


def divergence_residual(scheme, stamp, levels=(0, 5, 10, 20)):
    """d04 原生 C 网格 ∇·(ρu) 残差统计(可压缩形式,物理单位)。"""
    day = '{}-{}-{}'.format(stamp[0:4], stamp[4:6], stamp[6:8])
    hh = stamp[9:11]
    wpath = os.path.join(WRF_BASE, SCHEME_DIRS[scheme],
                         "wrfout_d04_{}_{}:00:00".format(day, hh))
    tidx = int(stamp[11:13]) // 10
    with Dataset(wpath) as nc:
        phb = np.array(nc.variables['PHB'][tidx], dtype=np.float64)
        hgt = np.array(nc.variables['HGT'][tidx], dtype=np.float64)
        phi = np.array(nc.variables['PH'][tidx], dtype=np.float64)
        z = (phb + phi) / G
        u = np.array(nc.variables['U'][tidx], dtype=np.float64)
        v = np.array(nc.variables['V'][tidx], dtype=np.float64)
        w = np.array(nc.variables['W'][tidx], dtype=np.float64)
        p = np.array(nc.variables['P'][tidx], dtype=np.float64) + \
            np.array(nc.variables['PB'][tidx], dtype=np.float64)
        theta = np.array(nc.variables['T'][tidx], dtype=np.float64) + 300.0
        qv = np.maximum(np.array(nc.variables['QVAPOR'][tidx], dtype=np.float64), 0.0)
    t_full = theta * np.power(p / P0, RD / CP)
    tv = t_full * (1.0 + 0.61 * qv)
    rho = p / (RD * tv)                     # 质量点干空气密度近似(含虚温修正)
    rho_u = 0.5 * (rho[:, :, :-1] + rho[:, :, 1:])   # -> U 点
    rho_v = 0.5 * (rho[:, :-1, :] + rho[:, 1:, :])   # -> V 点
    rho_w = 0.5 * (rho[:-1, :, :] + rho[1:, :, :])   # -> W 点(界面)
    dz = z[1:] - z[:-1]
    out = {}
    for k in levels:
        if k + 1 >= u.shape[0]:
            continue
        du = (rho_u[k, :, 1:] * u[k, :, 1:] - rho_u[k, :, :-1] * u[k, :, :-1]) / 1000.0
        dv = (rho_v[k, 1:, :] * v[k, 1:, :] - rho_v[k, :-1, :] * v[k, :-1, :]) / 1000.0
        dw = (rho_w[k + 1, :, :] * w[k + 1, :, :] - rho_w[k, :, :] * w[k, :, :]) \
            / np.maximum(dz[k, :, :], 1e-3)
        div = du[:, :] + dv[:, :] + dw[1:, 1:]
        a = np.abs(div).ravel()
        out[str(k)] = {'mean': float(div.mean()), 'std': float(div.std()),
                       'p50_abs': float(np.percentile(a, 50)),
                       'p95_abs': float(np.percentile(a, 95)),
                       'p99_abs': float(np.percentile(a, 99))}
    return out


def mass_speed(cu, cv):
    """U/V 原生交错 -> 质量点风速(水平平均去交错)。"""
    u = 0.5 * (cu[:, :-1] + cu[:, 1:])
    v = 0.5 * (cv[:-1, :] + cv[1:, :])
    return np.sqrt(u ** 2 + v ** 2)


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
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()
    rep = {}

    # ① 各档位动能谱(取月中一个整点)
    stamp = '20200715T120000'
    rep['spectra'] = tier_spectra(args.scheme, stamp, level=0)
    print("① 能谱: " + ", ".join(
        "{} 7dx={:.0f}km k_pk={}".format(d, v['wavelength_7dx_km'],
                                         int(np.argmax(v['P'])))
        for d, v in rep['spectra'].items()))

    # ② 散度残差统计(三个时刻)
    div = {}
    for s in ('20200715T120000', '20200710T000000', '20200720T060000'):
        div[s] = divergence_residual(args.scheme, s)
    rep['divergence_residual'] = div
    print("② d04 ∇·(ρu) 残差(kg m^-2 s^-1 量级,P95): " + json.dumps(
        {k: round(v['10']['p95_abs'], 6) for k, v in div.items()}))

    # ③ 位置偏移:细端 laplacian 平滑后的风速与粗端重网格场互相关
    c_files = sorted(glob.glob(os.path.join(OUT_COARSE, "c_{}_*.npz".format(args.scheme))))
    speeds = {}
    for p in c_files[:72]:
        with np.load(p) as d:
            speeds[parse_stamp(os.path.basename(p))] = mass_speed(d['c_u'][0], d['c_v'][0])
    top = sorted(speeds, key=lambda k: -speeds[k].mean())[:3]
    rep['shifts'] = {}
    for s in top:
        with np.load(os.path.join(OUT_FINE, "f_{}_{}.npz".format(args.scheme, s))) as d:
            fspd_m = mass_speed(d['f_u'][0], d['f_v'][0])
        cs = speeds[s]
        cs_m = cs
        # 粗端 9km 场在细端网格上按 9x9 抽样代表,直接比形态:
        fine_ds = fspd_m[::9, ::9]
        n = min(fine_ds.shape[0], cs_m.shape[0]), min(fine_ds.shape[1], cs_m.shape[1])
        (dj, di), c = shift_search(cs_m[:n[0], :n[1]], fine_ds[:n[0], :n[1]], 3)
        rep['shifts'][s] = {'coarse_mean_speed': float(cs.mean()),
                            'shift_coarse_cells': [dj, di],
                            'shift_km': [dj * 9.0, di * 9.0], 'corr': c}
    print("③ 强风时刻 d02 相对 d04 偏移: " + json.dumps(
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
