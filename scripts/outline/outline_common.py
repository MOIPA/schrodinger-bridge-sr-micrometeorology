# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) 公共定义:路径、层集合、时间解析、太阳几何。

层集合约定(探查 51 结论):d04 61 个模式层,层间距约 50 m;
  质量层 z_agl < 1 km 共 20 层、< 2 km 共 40 层。
存储取 0..39(质量层/U/V)与 0..40(W 界面),训练配置再从中选层,
避免因层数口径变化重新抽取(导师大纲"2 km 以下所有层"字面 = 40 层)。
"""
import os
from datetime import datetime

import numpy as np

WRF_BASE = "/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen"
GEOG_BASE = "/fsb/home/yutingwang/share/WPS_GeoStatic"
OUT_ROOT = "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology"
OUT_FINE = os.path.join(OUT_ROOT, "prepare_npz_outline_fine")
OUT_COARSE = os.path.join(OUT_ROOT, "prepare_npz_outline_coarse")
OUT_STATIC = os.path.join(OUT_ROOT, "prepare_npz_outline_static")

SCHEMES = ["myj", "ysu"]
SCHEME_DIRS = {"myj": "meso_202007_myj", "ysu": "meso_202007_ysu"}

N_MASS = 40      # 存储的质量层数(index 0..39, z_agl < 2 km)
N_IFACE = 41     # 存储的界面层数(W, PH 等)

G = 9.81


def read_times(ncfile):
    """读取 wrfout 内全部时间戳(UTC)。"""
    times = ncfile.variables['Times']
    out = []
    for i in range(times.shape[0]):
        chars = times[i]
        ts = b''.join([c if isinstance(c, bytes) else str(c).encode('utf-8')
                       for c in chars]).decode('utf-8').strip()
        out.append(datetime.strptime(ts, '%Y-%m-%d_%H:%M:%S'))
    return out


def stamp(dt):
    return dt.strftime('%Y%m%dT%H%M%S')


def cos_sza_field(xlat, xlong, dt_utc):
    """天文 cos(SZA) 场,取 max(cos,0);与旧管线太阳几何公式一致。

    xlat/xlong: 二维(或标量)经纬度(度);dt_utc: datetime(UTC)
    """
    doy = dt_utc.timetuple().tm_yday
    b_rad = np.radians(360.0 / 365.0 * (doy - 81))
    decl_deg = 23.44 * np.sin(b_rad)
    eot_min = 9.87 * np.sin(2 * b_rad) - 7.53 * np.cos(b_rad) - 1.5 * np.sin(b_rad)
    utc_h = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    lst_h = utc_h + np.asarray(xlong) / 15.0 + eot_min / 60.0
    hour_angle = np.radians(15.0 * (lst_h - 12.0))
    lat_r = np.radians(np.asarray(xlat))
    decl_r = np.radians(decl_deg)
    sin_elev = (np.sin(lat_r) * np.sin(decl_r) +
                np.cos(lat_r) * np.cos(decl_r) * np.cos(hour_angle))
    return np.maximum(sin_elev, 0.0).astype(np.float32)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def domain_files(scheme, dom):
    import glob
    return sorted(glob.glob(os.path.join(WRF_BASE, SCHEME_DIRS[scheme], "wrfout_" + dom + "_*")))
