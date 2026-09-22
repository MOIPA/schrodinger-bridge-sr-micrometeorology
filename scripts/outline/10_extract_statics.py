# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · T0.2 静态场与几何表

输出 OUT_STATIC/statics.npz + meta.json:
  几何   : z_agl 逐层(粗/细端,质量层与界面层)、HGT、eta(ZNU/ZNW)
  下垫面 : landuse 21 类分数、urban/water 分数、VEGFRA(7 月)、log(z0)
           (z0 按 WRF Noah 陆面方案 Z0BRD 公式离线重建,VEGPARM.TBL 查表)
  映射   : d02 -> d04 重网格权重表(mass/U/V 三类交错,原生位置)
  评估   : AGL 11 层插值表(质量层与界面层各一套)

不含任何逐时刻量,全部符合"规则二 d04 只提供静态场"。

运行(pytorch-gpu 环境):
  python scripts/outline/10_extract_statics.py
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline_common import (GEOG_BASE, N_IFACE, N_MASS, OUT_STATIC,
                           domain_files, ensure_dir)
from wps_geog import read_geog_region

G = 9.81
R_EARTH = 6370000.0
TARGET_AGL = [10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]

LANDUSE_DIR = "modis_landuse_20class_15s_with_lakes"
GREENFRAC_DIR = "greenfrac_fpar_modis"
JULY = 6  # 0 起算的月份索引

# WRF v4.2.1 VEGPARM.TBL,MODIFIED_IGBP_MODIS_NOAH 段 Z0MIN/Z0MAX(米)
# 类 21 = 湖泊(with_lakes 数据集的第 21 类),按水面表值处理
Z0_MIN = {1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.50, 6: 0.01, 7: 0.01, 8: 0.01,
          9: 0.15, 10: 0.10, 11: 0.30, 12: 0.05, 13: 0.50, 14: 0.05, 15: 0.55,
          16: 0.01, 17: 0.0001, 18: 0.30, 19: 0.15, 20: 0.05, 21: 0.0001}
Z0_MAX = {1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.50, 6: 0.05, 7: 0.06, 8: 0.05,
          9: 0.15, 10: 0.12, 11: 0.30, 12: 0.15, 13: 0.50, 14: 0.14, 15: 0.70,
          16: 0.01, 17: 0.0001, 18: 0.30, 19: 0.15, 20: 0.10, 21: 0.0001}

# namelist.input_SZ.6d 的 62 个 eta 值(用于核对 ZNW)
ETA_NAMELIST = [
    1.00000, 0.99377, 0.98758, 0.98141, 0.97528, 0.96917, 0.96309, 0.95705,
    0.95103, 0.94504, 0.93908, 0.93315, 0.92724, 0.92137, 0.91552, 0.90971,
    0.90392, 0.89816, 0.89243, 0.88672, 0.88105, 0.87540, 0.86978, 0.86418,
    0.85862, 0.85308, 0.84757, 0.84209, 0.83663, 0.83120, 0.82580, 0.82042,
    0.81507, 0.80975, 0.80445, 0.79918, 0.79394, 0.78872, 0.78353, 0.77837,
    0.77323, 0.76811, 0.76230, 0.75451, 0.74355, 0.72829, 0.70778, 0.68126,
    0.64827, 0.60864, 0.56262, 0.51355, 0.46245, 0.41149, 0.36046, 0.30938,
    0.25827, 0.20710, 0.15595, 0.10471, 0.05342, 0.00000,
]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


# ---------------------------------------------------------------- 投影与映射
def lcc_xy(lat, lon, truelat1=30.0, truelat2=30.0, stand_lon=113.0, r=R_EARTH):
    """WRF Lambert 共形圆锥投影(球体,半径 6370 km),返回投影平面坐标(米)。"""
    phi1, phi2 = np.radians(truelat1), np.radians(truelat2)
    lam0 = np.radians(stand_lon)
    if abs(truelat1 - truelat2) < 1e-10:
        n = np.sin(phi1)
    else:
        n = (np.log(np.cos(phi1) / np.cos(phi2)) /
             np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2)))
    f = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n
    rho = r * f / np.tan(np.pi / 4 + np.radians(lat) / 2) ** n
    theta = n * (np.radians(lon) - lam0)
    return rho * np.sin(theta), -rho * np.cos(theta)


def grid_lattice(lat2d, lon2d):
    """网格在投影平面应为规则格点;返回原点/间距与残差(自检)。"""
    x, y = lcc_xy(lat2d, lon2d)
    dx = float(np.median(np.diff(x[0, :])))
    dy = float(np.median(np.diff(y[:, 0])))
    x00, y00 = float(x[0, 0]), float(y[0, 0])
    ii = np.arange(x.shape[1])[None, :].astype(np.float64)
    jj = np.arange(y.shape[0])[:, None].astype(np.float64)
    resx = float(np.abs(x - (x00 + ii * dx)).max())
    resy = float(np.abs(y - (y00 + jj * dy)).max())
    return {"x00": x00, "y00": y00, "dx": dx, "dy": dy, "resx": resx, "resy": resy}


def regrid_weights(src_lat, src_lon, dst_lat, dst_lon):
    """d02 -> d04 双线性重网格权重(按交错类别各自调用)。

    返回 idx (n_dst,4) int32、w (n_dst,4) float32、valid (n_dst,) bool。
    """
    sl = grid_lattice(src_lat, src_lon)
    xd, yd = lcc_xy(dst_lat, dst_lon)
    fx = (xd - sl["x00"]) / sl["dx"]
    fy = (yd - sl["y00"]) / sl["dy"]
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    tx = fx - i0
    ty = fy - j0
    nys, nxs = src_lat.shape
    valid = (i0 >= 0) & (i0 <= nxs - 2) & (j0 >= 0) & (j0 <= nys - 2)
    i0c = np.clip(i0, 0, nxs - 2)
    j0c = np.clip(j0, 0, nys - 2)
    base = (j0c * nxs + i0c).reshape(-1)
    idx = np.stack([base, base + 1, base + nxs, base + nxs + 1], axis=-1).astype(np.int32)
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty],
                 axis=-1).reshape(-1, 4).astype(np.float32)
    return idx, w, valid.reshape(-1), sl


def apply_regrid(field, idx, w):
    """field (..., n_src_flat) -> (..., n_dst):重量表应用(供 Dataset 复用)。"""
    flat = field.reshape(-1, field.shape[-1])
    return np.einsum('ln,nd->ld', flat[:, idx], w).reshape(field.shape[:-1] + (idx.shape[0],))


# ---------------------------------------------------------------- AGL 插值表
def agl_table(z_agl, targets, use_10m_anchor):
    """逐像素 AGL 插值表(ln(z) 线性)。

    idx=0..nlev-2: value = w*f[idx] + (1-w)*f[idx+1]
    idx=-1       : 低于首层(仅质量层): value = w*f[0] + (1-w)*f_10m
    idx=-2       : 目标 10 m,直接取 10 m 通道
    idx=-3       : 无解(不应出现)
    """
    nlev, ny, nx = z_agl.shape
    lnz = np.log(np.maximum(z_agl.astype(np.float64), 1e-3))
    nt = len(targets)
    idx = np.full((nt, ny, nx), -3, dtype=np.int16)
    w = np.zeros((nt, ny, nx), dtype=np.float32)
    stats = {}
    for ti, h in enumerate(targets):
        if use_10m_anchor and h == 10:
            idx[ti] = -2
            stats[str(h)] = {"below_first": 0, "above_top": 0,
                             "use_10m_channel": int(ny * nx)}
            continue
        k = np.sum(z_agl <= h, axis=0).astype(np.int64) - 1
        below = k < 0
        above = (~below) & (k >= nlev - 1)
        kc = np.clip(k, 0, nlev - 2)
        z1 = np.take_along_axis(lnz, (kc + 1)[None], axis=0)[0]
        z0 = np.take_along_axis(lnz, kc[None], axis=0)[0]
        ww = np.clip((z1 - np.log(h)) / np.maximum(z1 - z0, 1e-12), 0.0, 1.0)
        idx[ti] = kc.astype(np.int16)
        w[ti] = ww.astype(np.float32)
        if below.any():
            if use_10m_anchor:
                wb = (lnz[0] - np.log(float(h))) / (lnz[0] - np.log(10.0))
                idx[ti][below] = -1
                w[ti][below] = np.clip(wb, 0.0, 1.0)[below].astype(np.float32)
        stats[str(h)] = {"below_first": int(below.sum()), "above_top": int(above.sum())}
    return idx, w, stats


# ---------------------------------------------------------------- 地理数据聚合
def pixel_cell_index(lats, lons, tl, ny, nx, chunk=400):
    """geog 像素 -> 目标网格单元(C 序展平索引,-1 表示域外)。分块计算。"""
    out = np.full((len(lats), len(lons)), -1, dtype=np.int64)
    for j0 in range(0, len(lats), chunk):
        j1 = min(j0 + chunk, len(lats))
        lon2d, lat2d = np.meshgrid(lons, lats[j0:j1])
        x, y = lcc_xy(lat2d, lon2d)
        ii = np.floor((x - tl["x00"]) / tl["dx"]).astype(np.int64)
        jj = np.floor((y - tl["y00"]) / tl["dy"]).astype(np.int64)
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        out[j0:j1] = np.where(ok, jj * nx + ii, -1)
    return out.reshape(-1)


def agg_class(cell, vals, ncat, ny, nx):
    """类别数据分箱 -> (ncat, ny, nx) 计数。"""
    ncell = ny * nx
    cls = np.rint(vals).astype(np.int64)
    ok = (cell >= 0) & np.isfinite(vals) & (cls >= 1) & (cls <= ncat)
    comb = (cls[ok] - 1) * ncell + cell[ok]
    return np.bincount(comb, minlength=ncat * ncell).reshape(ncat, ny, nx)


def agg_cont(cell, vals, ny, nx):
    """连续数据分箱 -> (ny, nx) 均值。"""
    ncell = ny * nx
    ok = (cell >= 0) & np.isfinite(vals)
    comb = cell[ok]
    ssum = np.bincount(comb, weights=vals[ok], minlength=ncell).reshape(ny, nx)
    scnt = np.bincount(comb, minlength=ncell).reshape(ny, nx).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)


def noah_z0(dominant, vegfra, shdmin, shdmax):
    """Noah Z0BRD 公式(module_sf_noahlsm.F),SHDFAC=7 月 VEGFRA。"""
    z0min_tab = np.array([Z0_MIN[c] for c in range(1, 22)])
    z0max_tab = np.array([Z0_MAX[c] for c in range(1, 22)])
    ok = dominant >= 1
    dc = np.clip(dominant, 1, 21)
    z0min = z0min_tab[dc - 1]
    z0max = z0max_tab[dc - 1]
    interp = np.clip((vegfra - shdmin) / np.maximum(shdmax - shdmin, 1e-6), 0.0, 1.0)
    z0 = np.where(vegfra >= shdmax, z0max,
                  np.where(vegfra <= shdmin, z0min,
                           np.where(shdmax > shdmin, z0min + interp * (z0max - z0min),
                                    0.5 * (z0min + z0max))))
    return np.where(ok, z0, np.nan).astype(np.float32)


# ---------------------------------------------------------------- 读取几何
def load_geom(path):
    with Dataset(path) as nc:
        d = {
            'hgt': np.array(nc.variables['HGT'][0], dtype=np.float64),
            'phb': np.array(nc.variables['PHB'][0], dtype=np.float64),
            'xlat': np.array(nc.variables['XLAT'][0], dtype=np.float64),
            'xlong': np.array(nc.variables['XLONG'][0], dtype=np.float64),
            'xlat_u': np.array(nc.variables['XLAT_U'][0], dtype=np.float64),
            'xlong_u': np.array(nc.variables['XLONG_U'][0], dtype=np.float64),
            'xlat_v': np.array(nc.variables['XLAT_V'][0], dtype=np.float64),
            'xlong_v': np.array(nc.variables['XLONG_V'][0], dtype=np.float64),
        }
        for name in ('ZNU', 'ZNW'):
            d[name.lower()] = (np.array(nc.variables[name][0], dtype=np.float64)
                               if name in nc.variables else None)
        d['attrs'] = {}
        for k in ['MAP_PROJ', 'TRUELAT1', 'TRUELAT2', 'STAND_LON', 'DX', 'DY',
                  'CEN_LAT', 'CEN_LON', 'MMINLU']:
            if k in nc.ncattrs():
                v = nc.getncattr(k)
                d['attrs'][k] = v if isinstance(v, str) else float(v)
    z_if = d['phb'] / G - d['hgt'][None, :, :]
    d['z_if'] = z_if
    d['z_mass'] = 0.5 * (z_if[:-1] + z_if[1:])
    return d


def main():
    parser = argparse.ArgumentParser(description="阶段0 T0.2 静态场与几何表")
    parser.add_argument("--fine_file", default=None, help="d04 文件(默认 myj 第一个)")
    parser.add_argument("--coarse_file", default=None, help="d02 文件(默认 myj 第一个)")
    parser.add_argument("--out_dir", default=OUT_STATIC)
    parser.add_argument("--geog_dir", default=GEOG_BASE)
    args = parser.parse_args()

    fine_file = args.fine_file or domain_files("myj", "d04")[0]
    coarse_file = args.coarse_file or domain_files("myj", "d02")[0]
    ensure_dir(args.out_dir)
    rep = {}
    out = {}

    print("fine   : " + fine_file)
    print("coarse : " + coarse_file)
    fine = load_geom(fine_file)
    coarse = load_geom(coarse_file)
    ny_f, nx_f = fine['hgt'].shape
    ny_c, nx_c = coarse['hgt'].shape

    # 两方案地形一致性
    try:
        with Dataset(domain_files("ysu", "d04")[0]) as nc:
            hgt_ysu = np.array(nc.variables['HGT'][0], dtype=np.float64)
        rep['hgt_scheme_maxdiff_fine_m'] = float(np.abs(hgt_ysu - fine['hgt']).max())
    except Exception as e:  # noqa: BLE001
        rep['hgt_scheme_check'] = str(e)

    if fine['znw'] is not None:
        rep['eta_znw_maxdiff_vs_namelist'] = float(
            np.abs(fine['znw'] - np.array(ETA_NAMELIST)).max())
    rep['z_iface0_abs_max_m'] = float(np.abs(fine['z_if'][0]).max())
    rep['attrs_fine'] = fine['attrs']
    assert fine['attrs']['MAP_PROJ'] == coarse['attrs']['MAP_PROJ'] == 3
    proj = {'truelat1': fine['attrs']['TRUELAT1'], 'truelat2': fine['attrs']['TRUELAT2'],
            'stand_lon': fine['attrs']['STAND_LON']}

    # ------------------------------------------------------------ 重网格权重
    tl_fine = grid_lattice(fine['xlat'], fine['xlong'])
    tl_c = grid_lattice(coarse['xlat'], coarse['xlong'])
    rep['lattice_fine_mass'] = {k: tl_fine[k] for k in ('resx', 'resy', 'dx', 'dy')}
    rep['lattice_coarse_mass'] = {k: tl_c[k] for k in ('resx', 'resy', 'dx', 'dy')}
    classes = {
        'mass': (fine['xlat'], fine['xlong'], coarse['xlat'], coarse['xlong']),
        'u': (fine['xlat_u'], fine['xlong_u'], coarse['xlat_u'], coarse['xlong_u']),
        'v': (fine['xlat_v'], fine['xlong_v'], coarse['xlat_v'], coarse['xlong_v']),
    }
    for cls, (flat, flon, clat, clon) in classes.items():
        idx, w, valid, sl = regrid_weights(clat, clon, flat, flon)
        out['regrid_idx_' + cls] = idx
        out['regrid_w_' + cls] = w
        out['regrid_valid_' + cls] = valid
        out['regrid_shape_' + cls] = np.array(clat.shape, dtype=np.int32)
        lat_r = (w * clat.reshape(-1)[idx]).sum(axis=1).reshape(flat.shape)
        lon_r = (w * clon.reshape(-1)[idx]).sum(axis=1).reshape(flon.shape)
        rep['regrid_' + cls] = {
            'invalid': int((~valid).sum()),
            'lat_err_max_m': float(np.abs(lat_r - flat).max() * 111320.0),
            'lon_err_max_m': float(np.abs(lon_r - flon).max() * 111320.0
                                   * np.cos(np.radians(float(np.nanmean(flat))))),
            'src_lattice_res_m': max(sl['resx'], sl['resy']),
        }
    print("重网格校验(mass): " + json.dumps(jsonable(rep['regrid_mass'])))

    # ------------------------------------------------------------ AGL 表
    agl_idx_m, agl_w_m, st_m = agl_table(fine['z_mass'], TARGET_AGL, True)
    agl_idx_i, agl_w_i, st_i = agl_table(fine['z_if'], TARGET_AGL, False)
    out['agl_targets'] = np.array(TARGET_AGL, dtype=np.float32)
    out['agl_idx_mass'] = agl_idx_m
    out['agl_w_mass'] = agl_w_m
    out['agl_idx_iface'] = agl_idx_i
    out['agl_w_iface'] = agl_w_i
    rep['agl_mass'] = st_m
    rep['agl_iface'] = st_i

    # ------------------------------------------------------------ 几何
    out['hgt_fine'] = fine['hgt'].astype(np.float32)
    out['zagl_mass_fine'] = fine['z_mass'][:N_MASS].astype(np.float32)
    out['zagl_iface_fine'] = fine['z_if'][:N_IFACE].astype(np.float32)
    out['xlat_fine'] = fine['xlat'].astype(np.float32)
    out['xlong_fine'] = fine['xlong'].astype(np.float32)
    out['hgt_coarse'] = coarse['hgt'].astype(np.float32)
    out['zagl_mass_coarse'] = coarse['z_mass'][:N_MASS].astype(np.float32)
    out['zagl_iface_coarse'] = coarse['z_if'][:N_IFACE].astype(np.float32)
    out['xlat_coarse'] = coarse['xlat'].astype(np.float32)
    out['xlong_coarse'] = coarse['xlong'].astype(np.float32)
    if fine['znu'] is not None:
        out['znu_fine'] = fine['znu'].astype(np.float32)
        out['znw_fine'] = fine['znw'].astype(np.float32)
    if coarse['znu'] is not None:
        out['znu_coarse'] = coarse['znu'].astype(np.float32)
        out['znw_coarse'] = coarse['znw'].astype(np.float32)

    # ------------------------------------------------------------ 下垫面
    lat_min = min(fine['xlat'].min(), coarse['xlat'].min())
    lat_max = max(fine['xlat'].max(), coarse['xlat'].max())
    lon_min = min(fine['xlong'].min(), coarse['xlong'].min())
    lon_max = max(fine['xlong'].max(), coarse['xlong'].max())

    lu, lats, lons, lu_idx = read_geog_region(
        os.path.join(args.geog_dir, LANDUSE_DIR), lat_min, lat_max, lon_min, lon_max)
    lu = lu[0]
    ncat = int(lu_idx.get('category_max', 21))
    print("landuse 区域 {} 类别 1..{}".format(lu.shape, ncat))
    cell_f = pixel_cell_index(lats, lons, tl_fine, ny_f, nx_f)
    cell_c = pixel_cell_index(lats, lons, tl_c, ny_c, nx_c)
    frac_f = agg_class(cell_f, lu.reshape(-1), ncat, ny_f, nx_f)
    frac_c = agg_class(cell_c, lu.reshape(-1), ncat, ny_c, nx_c)
    rep['landuse'] = {
        'dataset': LANDUSE_DIR, 'ncat': ncat,
        'frac_sum_min': float(frac_f.sum(axis=0).min()),
        'frac_sum_max': float(frac_f.sum(axis=0).max()),
        'empty_cells_fine': int((frac_f.sum(axis=0) == 0).sum()),
        'empty_cells_coarse': int((frac_c.sum(axis=0) == 0).sum()),
    }

    green, glats, glons, gr_idx = read_geog_region(
        os.path.join(args.geog_dir, GREENFRAC_DIR), lat_min, lat_max, lon_min, lon_max)
    print("greenfrac 区域 {} (12 月)".format(green.shape))
    cell_fg = pixel_cell_index(glats, glons, tl_fine, ny_f, nx_f)
    cell_cg = pixel_cell_index(glats, glons, tl_c, ny_c, nx_c)
    veg_month_f = [agg_cont(cell_fg, green[m].reshape(-1), ny_f, nx_f) for m in range(12)]
    veg_month_c = [agg_cont(cell_cg, green[m].reshape(-1), ny_c, nx_c) for m in range(12)]
    rep['greenfrac_month_mean_region'] = [round(float(np.nanmean(green[m])), 4)
                                          for m in range(12)]
    veg_f = veg_month_f[JULY]
    veg_c = veg_month_c[JULY]
    shdmin_f = np.nanmin(np.stack(veg_month_f), axis=0)
    shdmax_f = np.nanmax(np.stack(veg_month_f), axis=0)
    shdmin_c = np.nanmin(np.stack(veg_month_c), axis=0)
    shdmax_c = np.nanmax(np.stack(veg_month_c), axis=0)

    dom_f = np.where(frac_f.sum(axis=0) > 0, np.argmax(frac_f, axis=0) + 1, 0)
    dom_c = np.where(frac_c.sum(axis=0) > 0, np.argmax(frac_c, axis=0) + 1, 0)
    z0_f = noah_z0(dom_f, veg_f, shdmin_f, shdmax_f)
    z0_c = noah_z0(dom_c, veg_c, shdmin_c, shdmax_c)
    for tag, z0, dom in (('fine', z0_f, dom_f), ('coarse', z0_c, dom_c)):
        rep['z0_' + tag] = {
            'min': float(np.nanmin(z0)), 'mean': float(np.nanmean(z0)),
            'max': float(np.nanmax(z0)),
            'urban_cells': int((dom == 13).sum()),
            'water_cells': int(((dom == 17) | (dom == 21)).sum()),
        }
    rep['vegfra_july'] = {'fine_mean': float(np.nanmean(veg_f)),
                          'coarse_mean': float(np.nanmean(veg_c))}

    water_f = frac_f[16] + (frac_f[20] if ncat >= 21 else 0.0)
    water_c = frac_c[16] + (frac_c[20] if ncat >= 21 else 0.0)
    out['frac_fine'] = frac_f.astype(np.float32)
    out['frac_coarse'] = frac_c.astype(np.float32)
    out['urban_fine'] = frac_f[12].astype(np.float32)
    out['water_fine'] = water_f.astype(np.float32)
    out['vegfra_fine'] = veg_f.astype(np.float32)
    out['shdmin_fine'] = shdmin_f.astype(np.float32)
    out['shdmax_fine'] = shdmax_f.astype(np.float32)
    out['logz0_fine'] = np.log(np.maximum(z0_f, 1e-6)).astype(np.float32)
    out['urban_coarse'] = frac_c[12].astype(np.float32)
    out['water_coarse'] = water_c.astype(np.float32)
    out['vegfra_coarse'] = veg_c.astype(np.float32)
    out['shdmin_coarse'] = shdmin_c.astype(np.float32)
    out['shdmax_coarse'] = shdmax_c.astype(np.float32)
    out['logz0_coarse'] = np.log(np.maximum(z0_c, 1e-6)).astype(np.float32)

    # ------------------------------------------------------------ 保存
    npz_path = os.path.join(args.out_dir, "statics.npz")
    np.savez_compressed(npz_path, **out)
    meta = {
        "generated": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "fine_file": fine_file, "coarse_file": coarse_file,
        "eta": {
            "e_vert": 62, "p_top_requested": 5000.0,
            "eta_levels": ETA_NAMELIST,
            "base_temp": 290.0, "base_pres": 100000.0, "base_lapse": 50.0,
            "source": "namelist.input_SZ.6d(eta_levels/p_top/base_temp 显式;"
                      "base_pres/base_lapse 为 WRF 默认值,namelist 未显式给出,待导师确认)",
        },
        "projection": {"map_proj": 3, "truelat1": proj['truelat1'],
                       "truelat2": proj['truelat2'], "stand_lon": proj['stand_lon'],
                       "dx_fine": 1000.0, "dy_fine": 1000.0, "dx_coarse": 9000.0},
        "landuse": {"dataset": LANDUSE_DIR, "mminlu": "MODIFIED_IGBP_MODIS_NOAH",
                    "num_land_cat": 21, "iswater": 17, "isurban": 13, "isice": 15,
                    "islake": 21, "z0_table_source": "WRF v4.2.1 VEGPARM.TBL"},
        "greenfrac": {"dataset": GREENFRAC_DIR, "month": 7, "scale_factor": 0.01},
        "z0_formula": "Noah Z0BRD (module_sf_noahlsm.F): SHDFAC>=SHDMAX -> Z0MAX; "
                      "SHDFAC<=SHDMIN -> Z0MIN; 否则线性插值。SHDFAC = 7 月 VEGFRA, "
                      "SHDMIN/SHDMAX = 12 月最小/最大值。水面为表值(真实 ZNT 走 Charnock,"
                      "大纲规定只取月均)。",
        "y0_design": {
            "regrid": "d02 -> d04 原生交错位置的双线性权重(regrid_idx_*/regrid_w_*);"
                      "U/V/质量点各自独立",
            "agl_tables": "idx>=0: w*f[idx]+(1-w)*f[idx+1]; idx=-1: 低于首层,与 10 m 通道"
                          "按对数律插值; idx=-2: 目标 10 m 直接取 10 m 通道",
        },
        "n_levels": {"mass_stored": N_MASS, "iface_stored": N_IFACE, "n_levels_total": 61},
        "validations": rep,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(jsonable(meta), f, indent=2, ensure_ascii=False)
    print("\n写出: {} ({:.1f} MB), {}".format(
        npz_path, os.path.getsize(npz_path) / 1e6, meta_path))
    print("\n=== 校验汇总 ===")
    print(json.dumps(jsonable(rep), indent=1, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()
