# -*- coding: utf-8 -*-
"""
WPS geogrid 二进制数据读取(index 头 + 瓦片文件)。

格式(见 WPS_GeoStatic/<dataset>/index):
  type=categorical|continuous, projection=regular_ll, dx/dy(度),
  known_x/known_y/known_lat/known_lon(1 起算的网格原点),
  wordsize(字节), tile_x/tile_y/tile_z(瓦片尺寸), scale_factor, missing_value
瓦片文件名 "{i0:05d}-{i1:05d}.{j0:05d}-{j1:05d}",数据按 z,y,x 行主序,
y 从南到北、x 从西到东;格点(i,j) 经纬度 = known_lon + (i-known_x)*dx 等。
"""
import os

import numpy as np

DTYPES = {1: np.uint8, 2: np.int16, 4: np.float32}


def parse_index(geog_dir):
    """解析 index 文件为 dict(数值项转 float)。"""
    out = {}
    with open(os.path.join(geog_dir, 'index')) as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            v = v.strip().strip('"')
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def index_range(idx, lat_min, lat_max, lon_min, lon_max):
    """返回覆盖 bbox 的全局 1 起算索引范围 (i_min, i_max, j_min, j_max)。"""
    dx, dy = idx['dx'], idx['dy']
    kx, ky = idx['known_x'], idx['known_y']
    lat0, lon0 = idx['known_lat'], idx['known_lon']
    i_min = int(np.floor((lon_min - lon0) / dx + kx))
    i_max = int(np.ceil((lon_max - lon0) / dx + kx))
    j_min = int(np.floor((lat_min - lat0) / dy + ky))
    j_max = int(np.ceil((lat_max - lat0) / dy + ky))
    return i_min, i_max, j_min, j_max


def read_geog_region(geog_dir, lat_min, lat_max, lon_min, lon_max, z_slice=None):
    """读取覆盖 bbox 的 geog 数据。

    返回 (data, lats, lons, idx):
      data: (nz, nj, ni) float32,缺测为 np.nan
      lats/lons: 1 维像素坐标(度),与 data 的 j/i 维对应
    """
    idx = parse_index(geog_dir)
    dx, dy = idx['dx'], idx['dy']
    lat0, lon0 = idx['known_lat'], idx['known_lon']
    kx, ky = idx['known_x'], idx['known_y']
    tx, ty, tz = int(idx['tile_x']), int(idx['tile_y']), int(idx['tile_z'])
    ws = int(idx['wordsize'])
    scale = float(idx.get('scale_factor', 1.0))
    missing = idx.get('missing_value', None)
    i_min, i_max, j_min, j_max = index_range(idx, lat_min, lat_max, lon_min, lon_max)
    ni, nj = i_max - i_min + 1, j_max - j_min + 1
    if z_slice is None:
        z_slice = list(range(tz))
    zs = np.asarray(z_slice, dtype=int)
    data = np.full((len(zs), nj, ni), np.nan, dtype=np.float32)

    i_tiles = range((i_min - 1) // tx, (i_max - 1) // tx + 1)
    j_tiles = range((j_min - 1) // ty, (j_max - 1) // ty + 1)
    for ti in i_tiles:
        for tj in j_tiles:
            i0 = ti * tx + 1
            j0 = tj * ty + 1
            name = "{:05d}-{:05d}.{:05d}-{:05d}".format(i0, i0 + tx - 1, j0, j0 + ty - 1)
            path = os.path.join(geog_dir, name)
            if not os.path.isfile(path):
                raise IOError("geog tile missing: " + path)
            arr = np.fromfile(path, dtype=DTYPES[ws]).reshape(tz, ty, tx).astype(np.float32)
            arr *= scale
            # 全局索引 -> 瓦片内索引
            gi = np.arange(max(i_min, i0), min(i_max, i0 + tx - 1) + 1)
            gj = np.arange(max(j_min, j0), min(j_max, j0 + ty - 1) + 1)
            if gi.size == 0 or gj.size == 0:
                continue
            block = arr[np.ix_(zs, gj - j0, gi - i0)]
            data[np.ix_(np.arange(len(zs)), gj - j_min, gi - i_min)] = block
    if missing is not None:
        data[data == missing * scale] = np.nan
    lons = lon0 + (np.arange(i_min, i_max + 1) - kx) * dx
    lats = lat0 + (np.arange(j_min, j_max + 1) - ky) * dy
    return data, lats, lons, idx
