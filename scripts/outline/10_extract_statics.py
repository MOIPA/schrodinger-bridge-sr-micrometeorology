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
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline_common import (GEOG_BASE, N_IFACE, N_MASS, OUT_STATIC,
                           domain_files, ensure_dir)
from wps_geog import read_geog_region

G = 9.81
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


# ---------------------------------------------------------------- 网格映射(投影无关)
def _kdtree(lat2d, lon2d):
    """经纬度 KD 树(经度按 cos(lat) 缩放,域内尺度足够小)。"""
    slat = np.asarray(lat2d, dtype=np.float64).reshape(-1)
    slon = np.asarray(lon2d, dtype=np.float64).reshape(-1)
    scale = float(np.cos(np.radians(np.mean(slat))))
    return cKDTree(np.stack([slon * scale, slat], axis=-1)), scale


def _invert_bilinear(p00, p10, p01, p11, p, n_iter=6):
    """逆双线性:p = (1-s)(1-t)p00 + s(1-t)p10 + (1-s)t p01 + s t p11。"""
    s = np.full(p.shape[:-1], 0.5)
    t = np.full(p.shape[:-1], 0.5)
    for _ in range(n_iter):
        a = (1.0 - s)[..., None]
        b = s[..., None]
        c = (1.0 - t)[..., None]
        d = t[..., None]
        f = a * c * p00 + b * c * p10 + a * d * p01 + b * d * p11 - p
        dfs = c * (p10 - p00) + d * (p11 - p01)
        dft = a * (p01 - p00) + b * (p11 - p10)
        det = dfs[..., 0] * dft[..., 1] - dfs[..., 1] * dft[..., 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        ds = (-f[..., 0] * dft[..., 1] + f[..., 1] * dft[..., 0]) / det
        dt = (-dfs[..., 0] * f[..., 1] + dfs[..., 1] * f[..., 0]) / det
        s = s + ds
        t = t + dt
    return s, t


def regrid_weights(src_lat, src_lon, dst_lat, dst_lon):
    """dst 点在 src 网格(原生交错位置)中的双线性权重表。

    投影无关:最近节点起手 + 逆双线性(在经纬度平面上,域内畸变可忽略)。
    返回 idx (n_dst,4) int32、w (n_dst,4) float32、valid (n_dst,) bool。
    """
    nys, nxs = src_lat.shape
    tree, scale = _kdtree(src_lat, src_lon)
    slat = src_lat.reshape(-1).astype(np.float64)
    slon = src_lon.reshape(-1).astype(np.float64)
    dlat = np.asarray(dst_lat, dtype=np.float64).reshape(-1)
    dlon = np.asarray(dst_lon, dtype=np.float64).reshape(-1)
    _, near = tree.query(np.stack([dlon * scale, dlat], axis=-1))
    n_i = (near % nxs).astype(np.int64)
    n_j = (near // nxs).astype(np.int64)
    n_dst = dlat.size
    idx = np.zeros((n_dst, 4), dtype=np.int32)
    w = np.zeros((n_dst, 4), dtype=np.float32)
    valid = np.zeros(n_dst, dtype=bool)
    for dj, di in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        j0 = n_j + dj
        i0 = n_i + di
        can = (j0 >= 0) & (j0 <= nys - 2) & (i0 >= 0) & (i0 <= nxs - 2) & (~valid)
        if not can.any():
            continue
        base = j0[can] * nxs + i0[can]
        p00 = np.stack([slon[base], slat[base]], axis=-1)
        p10 = np.stack([slon[base + 1], slat[base + 1]], axis=-1)
        p01 = np.stack([slon[base + nxs], slat[base + nxs]], axis=-1)
        p11 = np.stack([slon[base + nxs + 1], slat[base + nxs + 1]], axis=-1)
        p = np.stack([dlon[can], dlat[can]], axis=-1)
        s, t = _invert_bilinear(p00, p10, p01, p11, p)
        ok = (s >= -1e-4) & (s <= 1 + 1e-4) & (t >= -1e-4) & (t <= 1 + 1e-4)
        if not ok.any():
            continue
        sel = np.where(can)[0][ok]
        b = base[ok]
        idx[sel, 0] = b
        idx[sel, 1] = b + 1
        idx[sel, 2] = b + nxs
        idx[sel, 3] = b + nxs + 1
        ss = np.clip(s[ok], 0.0, 1.0)
        tt = np.clip(t[ok], 0.0, 1.0)
        w[sel, 0] = ((1 - ss) * (1 - tt)).astype(np.float32)
        w[sel, 1] = (ss * (1 - tt)).astype(np.float32)
        w[sel, 2] = ((1 - ss) * tt).astype(np.float32)
        w[sel, 3] = (ss * tt).astype(np.float32)
        valid[sel] = True
    if (~valid).any():  # 回退最近节点
        fb = np.where(~valid)[0]
        idx[fb, 0] = n_j[fb] * nxs + n_i[fb]
        w[fb, 0] = 1.0
    return idx, w, valid


def apply_regrid(field, idx, w):
    """field (..., n_src_flat) -> (..., n_dst):权重表应用(供 Dataset 复用)。"""
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
def fit_index_map(lat2d, lon2d, deg=3):
    """最小二乘拟合 (lon,lat) -> (i,j) 多项式映射,返回 (func, 最大残差[格])。

    用于海量 geog 像素的分箱(误差 << 1 格即可);风场重网格另有精确权重表。
    """
    x = lon2d.reshape(-1).astype(np.float64)
    y = lat2d.reshape(-1).astype(np.float64)
    ii, jj = np.meshgrid(np.arange(lon2d.shape[1]), np.arange(lat2d.shape[0]))
    ii = ii.reshape(-1).astype(np.float64)
    jj = jj.reshape(-1).astype(np.float64)
    xm, xs = x.mean(), x.std() + 1e-12
    ym, ys = y.mean(), y.std() + 1e-12
    xn, yn = (x - xm) / xs, (y - ym) / ys

    def terms(a, b):
        t = [np.ones_like(a), a, b, a * a, a * b, b * b]
        if deg >= 3:
            t += [a ** 3, a * a * b, a * b * b, b ** 3]
        return np.stack(t, axis=-1)

    A = terms(xn, yn)
    ci = np.linalg.lstsq(A, ii, rcond=None)[0]
    cj = np.linalg.lstsq(A, jj, rcond=None)[0]
    res = max(float(np.abs(A @ ci - ii).max()), float(np.abs(A @ cj - jj).max()))

    def func(glon, glat):
        gx = (np.asarray(glon, dtype=np.float64).reshape(-1) - xm) / xs
        gy = (np.asarray(glat, dtype=np.float64).reshape(-1) - ym) / ys
        B = terms(gx, gy)
        return B @ ci, B @ cj

    return func, res


def pixel_cell_index(lats, lons, imap, ny, nx):
    """geog 像素(lats x lons 规则网) -> 目标单元(C 序展平索引,-1 域外)。"""
    lon2d, lat2d = np.meshgrid(lons, lats)
    fi, fj = imap(lon2d, lat2d)
    ii = np.floor(fi).astype(np.int64)
    jj = np.floor(fj).astype(np.int64)
    ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
    return np.where(ok, jj * nx + ii, -1)


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
    imap_f, imap_res_f = fit_index_map(fine['xlat'], fine['xlong'])
    imap_c, imap_res_c = fit_index_map(coarse['xlat'], coarse['xlong'])
    rep['index_map_res_cell'] = {'fine': imap_res_f, 'coarse': imap_res_c}
    classes = {
        'mass': (fine['xlat'], fine['xlong'], coarse['xlat'], coarse['xlong']),
        'u': (fine['xlat_u'], fine['xlong_u'], coarse['xlat_u'], coarse['xlong_u']),
        'v': (fine['xlat_v'], fine['xlong_v'], coarse['xlat_v'], coarse['xlong_v']),
    }
    for cls, (flat, flon, clat, clon) in classes.items():
        idx, w, valid = regrid_weights(clat, clon, flat, flon)
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
    cell_f = pixel_cell_index(lats, lons, imap_f, ny_f, nx_f)
    cell_c = pixel_cell_index(lats, lons, imap_c, ny_c, nx_c)
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
    cell_fg = pixel_cell_index(glats, glons, imap_f, ny_f, nx_f)
    cell_cg = pixel_cell_index(glats, glons, imap_c, ny_c, nx_c)
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
        "projection": {"map_proj": 3, "map_proj_char": "Mercator",
                       "truelat1": proj['truelat1'], "truelat2": proj['truelat2'],
                       "stand_lon": proj['stand_lon'],
                       "dx_fine": 1000.0, "dy_fine": 1000.0, "dx_coarse": 9000.0,
                       "note": "wrfout MAP_PROJ=3 即 Mercator;重网格不依赖投影公式,"
                               "用最近节点 + 经纬度平面逆双线性(域内畸变可忽略)"},
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
