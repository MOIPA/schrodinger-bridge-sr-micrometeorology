# -*- coding: utf-8 -*-
"""
阶段0(导师大纲)DatasetWindCanvas:原生 C 网格画布 + 分组输入条件。

- 目标(y):d04 原生交错 U/V(W 可选,界面层)+ U10/V10,标准化用细端 σ(μ=0)
- y0:粗端(d02)风场重网格到细端原生位置,与 y 同一标准化空间(细端 σ)
- 条件(x):按 input_groups 组装;粗端逐层量用粗端 σ、细端静态用全域统计
- rho:干空气密度(粗端 psfc/θ 静力积分,重网格到细端),仅供阶段 2 损失项,
  不进网络输入(in_channel 不变,V* 基线可比)
- 时间编码/ cos(SZA) 不处理;所有粗端逐时量只来自粗端(规则二)
"""
import copy
import dataclasses
import json
import os
import re
import typing
from datetime import datetime, timedelta

import numpy as np
import torch
import warnings
from torch.utils.data import Dataset

from src.dl_config.base_config import BaseDatasetConfig
from src.dl_data.wind_canvas_statics import (
    FINE_SHAPES,
    CanvasStatics,
    build_agl_tables_native,
)
from src.utils.random_crop import RandomCrop2D

INPUT_GROUPS = [
    'coarse_wind_uv', 'coarse_w', 'coarse_zagl', 'coarse_logz0',
    'coarse_most', 'coarse_flux', 'coarse_theta', 'coarse_ph',
    'time_enc', 'coszen',
    'fine_geom', 'fine_logz0', 'fine_urban', 'fine_waterveg',
    'geom_diff_zagl', 'geom_diff_hgt', 'coords',
]

# __getitem__ 追加/拆分的 AGL 表后缀(顺序即通道拼接顺序;返回键加 'agl_' 前缀)
AGL_TABLE_SUFFIXES = ('idx_m', 'w_m', 'idx_i', 'w_i')


def build_target_channel_names(L, include_w):
    names = ['y_u_{:02d}'.format(k) for k in L] + ['y_v_{:02d}'.format(k) for k in L]
    if include_w:
        names += ['y_w_{:02d}'.format(k) for k in range(L[0], L[-1] + 2)]
    names += ['y_u10', 'y_v10']
    return names


def build_input_channel_names(groups, L, include_w):
    """与 `_inputs` 的通道拼接顺序严格一致(顺序即索引,两处必须共用本函数)。"""
    WL = list(range(L[0], L[-1] + 2))
    names = []
    if 'coarse_wind_uv' in groups:
        names += ['c_u_{:02d}'.format(k) for k in L] + ['c_v_{:02d}'.format(k) for k in L]
    if 'coarse_w' in groups:
        names += ['c_w_{:02d}'.format(k) for k in WL]
    if 'coarse_zagl' in groups:
        names += ['c_zagl_{:02d}'.format(k) for k in L]
    if 'coarse_logz0' in groups:
        names += ['c_logz0']
    if 'coarse_most' in groups:
        names += ['c_rmol', 'c_ust', 'c_pblh']
    if 'coarse_flux' in groups:
        names += ['c_hfx', 'c_t2', 'c_psfc']
    if 'coarse_theta' in groups:
        names += ['c_theta_{:02d}'.format(k) for k in L]
    if 'coarse_ph' in groups:
        names += ['c_ph_{:02d}'.format(k) for k in WL]
    if 'time_enc' in groups:
        names += ['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']
    if 'coszen' in groups:
        names += ['coszen']
    if 'fine_geom' in groups:
        names += ['f_hgt'] + ['f_zagl_{:02d}'.format(k) for k in L]
    if 'fine_logz0' in groups:
        names += ['f_logz0']
    if 'fine_urban' in groups:
        names += ['f_urban']
    if 'fine_waterveg' in groups:
        names += ['f_water', 'f_vegfra']
    if 'geom_diff_zagl' in groups:
        names += ['d_zagl_{:02d}'.format(k) for k in L]
    if 'geom_diff_hgt' in groups:
        names += ['d_hgt']
    if 'coords' in groups:
        names += ['xlat', 'xlong']
    return names


_STAMP = re.compile(r'_(\d{8}T\d{6})\.npz$')


def _parse_stamp(path):
    m = _STAMP.search(os.path.basename(path))
    return m.group(1) if m else None


def compute_rho_dry(psfc, theta, z_mass, g=9.81, rd=287.05, cp=1004.5, p00=1e5):
    """干空气静力积分密度:psfc (120,150) Pa(地面气压),theta (K,120,150) 质量层,
    z_mass (m,120,150) 质量层高度,须相对 psfc 所在面(即 AGL;层间只用差分,
    与地形高度无关)。

    从最低质量层向上:先 p0 = psfc·exp(−g·z0/(Rd·T0));再逐层
    p_{k+1} = p_k·exp(−g·Δz/(Rd·T̄)),T = θ·(p/p00)^(Rd/cp),
    T̄ 用相邻两层(对 p_{k+1} 迭代一次即可)。返回 ρ (K,120,150) kg/m³。

    粗端 npz 无湿度,采用干空气近似(误差 ~1%);不做 netCDF4/QVAPOR 依赖。
    """
    psfc = np.maximum(np.asarray(psfc, dtype=np.float64), 1.0)      # Pa,防 0
    theta = np.maximum(np.asarray(theta, dtype=np.float64), 1.0)    # K,防 0
    z = np.asarray(z_mass, dtype=np.float64)
    kappa = rd / cp
    nlev = theta.shape[0]
    rho = np.empty_like(theta)

    # p/t 下限只为病态输入兜底(如 80 号合成自检的随机几何会让 p 逐层下溢到 0),
    # 真实大气 p ~1e4–1e5 Pa、T ~250–320 K,离下限很远,不影响数值
    def _exp_int(dz, t):
        # exp(−g·Δz/(Rd·T̄));指数裁剪只防病态输入,真实 |Δz| ≲ 2 km 不触发
        return np.exp(np.clip(-g * dz / (rd * np.maximum(t, 1e-3)), -50.0, 50.0))

    # 最低质量层:p0 对 T0 迭代一次(初猜 T0 用 psfc 近似 p0,即忽略 z0 处的气压亏损)
    t_cur = np.maximum(theta[0] * (psfc / p00) ** kappa, 1e-3)
    p_cur = np.maximum(psfc * _exp_int(z[0], t_cur), 1.0)
    t_cur = np.maximum(theta[0] * (p_cur / p00) ** kappa, 1e-3)
    p_cur = np.maximum(psfc * _exp_int(z[0], t_cur), 1.0)
    t_cur = np.maximum(theta[0] * (p_cur / p00) ** kappa, 1e-3)  # T 由当前 p 计算,保证 ρ=p/(Rd·T) 自洽
    rho[0] = p_cur / (rd * t_cur)

    # 逐层向上:p_{k+1} 用 T̄(相邻两层平均)迭代一次
    for k in range(1, nlev):
        dz = z[k] - z[k - 1]
        p_hat = np.maximum(p_cur * _exp_int(dz, t_cur), 1.0)
        t_bar = 0.5 * (t_cur + np.maximum(theta[k] * (p_hat / p00) ** kappa, 1e-3))
        p_cur = np.maximum(p_cur * _exp_int(dz, t_bar), 1.0)
        t_cur = np.maximum(theta[k] * (p_cur / p00) ** kappa, 1e-3)
        rho[k] = p_cur / (rd * t_cur)

    return rho


@dataclasses.dataclass()
class DatasetWindCanvasConfig(BaseDatasetConfig):
    scheme: str
    coarse_dir: str
    statics_dir: str
    target_levels: list[int]
    input_groups: list[str]
    include_w: bool = True
    normalize_json: typing.Optional[str] = None
    split_manifest: typing.Optional[str] = None
    hr_data_shape: list[int] = dataclasses.field(default_factory=lambda: [99, 120])
    hr_cropped_shape: list[int] = dataclasses.field(default_factory=lambda: [96, 112])
    is_clipped: bool = False
    min_clipped_value: typing.Optional[float] = None
    max_clipped_value: typing.Optional[float] = None
    missing_value: float = 0.0
    dtype: str = "float32"
    day_night_filter: str = "all"
    return_agl_tables: bool = False
    dataset_name: typing.ClassVar[str] = "DatasetWindCanvas"

    def __post_init__(self):
        assert self.scheme in ("myj", "ysu")
        assert self.dtype in ("float16", "float32")
        for g in self.input_groups:
            assert g in INPUT_GROUPS, "unknown input group " + g
        if self.day_night_filter != "all":
            # 细端 npz 不含 swdown,day/night 过滤无法在 canvas 上工作(评估侧按 cos SZA 分层)
            warnings.warn("canvas 数据集不支持 day_night_filter={},已降级为 all".format(
                self.day_night_filter))
            self.day_night_filter = "all"


class DatasetWindCanvas(Dataset):
    def __init__(self, file_paths, config, **kwargs):
        self.c = copy.deepcopy(config)
        # 只用本方案(归一化统计与目标必须来自同一套参数化)的整点帧:
        # stamp='YYYYMMDDTHHMMSS',[9:11]=小时、[11:13]=分钟
        tag = "_" + self.c.scheme + "_"
        self.ps = [p for p in file_paths if p.endswith(".npz") and tag in os.path.basename(p)
                   and _parse_stamp(p) is not None and _parse_stamp(p)[11:13] == "00"]
        self.stat = CanvasStatics(self.c.statics_dir)
        norm_path = self.c.normalize_json or os.path.join(self.c.statics_dir,
                                                          "normalize_config.json")
        with open(norm_path) as f:
            norms = json.load(f)
        self.fn = norms['fine'][self.c.scheme]
        self.cn = norms['coarse'][self.c.scheme]
        self.sn = norms['static']
        self.L = list(self.c.target_levels)
        self.WL = list(range(self.L[0], self.L[-1] + 2))  # W 界面层
        self.dtype = torch.float32 if self.c.dtype == "float32" else torch.float16
        self.crop = RandomCrop2D(img_sz=self.c.hr_data_shape, crop_sz=self.c.hr_cropped_shape)
        # 阶段 3 T3.5:AGL 插值表(原生网格构建 -> place 成画布 (11,100,121)),
        # 常驻 CPU 的 float32 静态张量;return_agl_tables=False 时完全不建
        self.agl_tables = None
        if self.c.return_agl_tables:
            raw = build_agl_tables_native(self.stat.d['zagl_mass_fine'],
                                          self.stat.d['zagl_iface_fine'], self.L)
            self.agl_tables = {k: torch.from_numpy(CanvasStatics.place(raw[k]))
                               for k in AGL_TABLE_SUFFIXES}

    # ------------------------------------------------------------ 通道清单
    def target_channel_names(self):
        return build_target_channel_names(self.L, self.c.include_w)

    def input_channel_names(self):
        return build_input_channel_names(self.c.input_groups, self.L, self.c.include_w)

    def __len__(self):
        return len(self.ps)

    # ------------------------------------------------------------ 组装
    def _std_fine_wind(self, arr, comp):
        sig = np.asarray(self.fn[comp]['sigma'], dtype=np.float32)[self.L]
        return np.asarray(arr, dtype=np.float32)[self.L] / sig[:, None, None]

    def _std_coarse_wind(self, arr, comp):
        sig = np.asarray(self.cn[comp]['sigma'], dtype=np.float32)[self.L]
        return np.asarray(arr, dtype=np.float32)[self.L] / sig[:, None, None]

    def _targets(self, fine):
        sig_uv10 = float(self.fn['u10']['sigma'])
        chans = [CanvasStatics.place(self._std_fine_wind(fine['f_u'], 'u'))]
        chans.append(CanvasStatics.place(self._std_fine_wind(fine['f_v'], 'v')))
        if self.c.include_w:
            sig_w = np.asarray(self.fn['w']['sigma'], dtype=np.float32)[self.WL]
            chans.append(CanvasStatics.place(
                np.asarray(fine['f_w'], dtype=np.float32)[self.WL] / sig_w[:, None, None]))
        chans.append(CanvasStatics.place(np.asarray(fine['f_u10'], dtype=np.float32) / sig_uv10))
        chans.append(CanvasStatics.place(np.asarray(fine['f_v10'], dtype=np.float32) / sig_uv10))
        return torch.from_numpy(np.concatenate(chans, axis=0)).to(self.dtype)

    def _y0(self, co):
        """粗端风场重网格到细端原生位置(与 y 同标准化空间:细端 σ)。"""
        sig = np.asarray(self.fn['u']['sigma'], dtype=np.float32)[self.L]
        u = np.asarray(co['c_u'], dtype=np.float32)[self.L] / sig[:, None, None]
        v = np.asarray(co['c_v'], dtype=np.float32)[self.L] / sig[:, None, None]
        chans = [CanvasStatics.place(self.stat.regrid_field(u, 'u')),
                 CanvasStatics.place(self.stat.regrid_field(v, 'v'))]
        if self.c.include_w:
            sig_w = np.asarray(self.fn['w']['sigma'], dtype=np.float32)[self.WL]
            w = np.asarray(co['c_w'], dtype=np.float32)[self.WL] / sig_w[:, None, None]
            chans.append(CanvasStatics.place(self.stat.regrid_field(w, 'mass')))
        # 10 m 通道代理:粗端最低层风(粗端未输出 U10/V10;去交错到质量点后重网格)
        u10 = 0.5 * (u[0, :, :-1] + u[0, :, 1:])
        v10 = 0.5 * (v[0, :-1, :] + v[0, 1:, :])
        chans.append(CanvasStatics.place(self.stat.regrid_field(u10[None], 'mass')))
        chans.append(CanvasStatics.place(self.stat.regrid_field(v10[None], 'mass')))
        return torch.from_numpy(np.concatenate(chans, axis=0)).to(self.dtype)

    def _rho(self, co):
        """阶段 2:粗端干空气密度静力积分 -> 重网格 -> 画布 (L,100,121) kg/m³。

        仅作散度损失 ∇·(ρu) 的逐样本权重,不进网络输入。
        用 AGL 而非 AGL+地形:psfc 是地面气压(位于地形高度 hgt),从 psfc 到
        最低质量层的积分距离是 AGL;误用 MSL 会多积一个地形高度,高地形列
        ~10% 密度偏差(层间差分与地形无关,不受影响)。
        """
        z_agl = np.asarray(self.stat.d['zagl_mass_coarse'], dtype=np.float64)[self.L]
        rho = compute_rho_dry(co['c_psfc'],
                              np.asarray(co['c_theta'], dtype=np.float64)[self.L], z_agl)
        return torch.from_numpy(CanvasStatics.place(
            self.stat.regrid_field(rho, 'mass'))).to(self.dtype)

    def _inputs(self, co, stamp):
        g = self.c.input_groups
        chans = []

        def add(name, arr):
            chans.append(CanvasStatics.place(arr))

        if 'coarse_wind_uv' in g:
            add('c_u', self.stat.regrid_field(self._std_coarse_wind(co['c_u'], 'u'), 'u'))
            add('c_v', self.stat.regrid_field(self._std_coarse_wind(co['c_v'], 'v'), 'v'))
        if 'coarse_w' in g:
            sig = np.asarray(self.cn['w']['sigma'], dtype=np.float32)[self.WL]
            w = np.asarray(co['c_w'], dtype=np.float32)[self.WL] / sig[:, None, None]
            add('c_w', self.stat.regrid_field(w, 'mass'))
        if 'coarse_zagl' in g:
            s = self.sn['zagl_mass_coarse_log']
            z = np.log(np.maximum(np.asarray(self.stat.d['zagl_mass_coarse'],
                                             dtype=np.float32)[self.L], 1e-3))
            add('c_zagl', self.stat.regrid_field((z - s['mu']) / s['sigma'], 'mass'))
        if 'coarse_logz0' in g:
            s = self.sn['logz0_coarse_raw']
            v = (np.asarray(self.stat.d['logz0_coarse'], dtype=np.float32) - s['mu']) / s['sigma']
            add('c_logz0', self.stat.regrid_field(v[None], 'mass'))
        if 'coarse_most' in g:
            z1 = np.asarray(self.stat.d['zagl_mass_coarse'][0], dtype=np.float32)
            rmol = np.sign(z1 * co['c_rmol']) * np.log1p(np.abs(z1 * co['c_rmol']))
            s = self.cn['rmol']
            add('c_rmol', self.stat.regrid_field(((rmol - s['mu']) / s['sigma'])[None], 'mass'))
            for key in ('c_ust', 'c_pblh'):
                s = self.cn[key[2:]]
                v = (np.log(np.maximum(co[key], 1e-8)) - s['mu']) / s['sigma']
                add(key, self.stat.regrid_field(v[None], 'mass'))
        if 'coarse_flux' in g:
            for key in ('c_hfx', 'c_t2', 'c_psfc'):
                s = self.cn[key[2:]]
                v = (co[key] - s['mu']) / s['sigma']
                add(key, self.stat.regrid_field(v[None], 'mass'))
        if 'coarse_theta' in g:
            mu = np.asarray(self.cn['theta']['mu'], dtype=np.float32)[self.L]
            sg = np.asarray(self.cn['theta']['sigma'], dtype=np.float32)[self.L]
            add('c_theta', self.stat.regrid_field(
                (np.asarray(co['c_theta'], dtype=np.float32)[self.L] - mu[:, None, None])
                / sg[:, None, None], 'mass'))
        if 'coarse_ph' in g:
            mu = np.asarray(self.cn['ph']['mu'], dtype=np.float32)[self.WL]
            sg = np.asarray(self.cn['ph']['sigma'], dtype=np.float32)[self.WL]
            add('c_ph', self.stat.regrid_field(
                (np.asarray(co['c_ph'], dtype=np.float32)[self.WL] - mu[:, None, None])
                / sg[:, None, None], 'mass'))
        if 'time_enc' in g:
            dt = datetime.strptime(stamp, '%Y%m%dT%H%M%S') + timedelta(hours=8)  # 本地时
            hour = dt.hour + dt.minute / 60.0
            doy = dt.timetuple().tm_yday
            for val in (np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)):
                add('time', np.full((1, self.c.hr_data_shape[0], self.c.hr_data_shape[1]),
                                    val, dtype=np.float32))
        if 'coszen' in g:
            r = self.stat.regrid(np.asarray(co['c_coszen'], dtype=np.float32)[None], 'mass')
            add('coszen', r[0].reshape(FINE_SHAPES['mass']))
        if 'fine_geom' in g:
            s = self.sn['hgt_fine_raw']
            add('f_hgt', ((np.asarray(self.stat.d['hgt_fine'], dtype=np.float32)
                           - s['mu']) / s['sigma'])[None])
            s = self.sn['zagl_mass_fine_log']
            z = np.log(np.maximum(np.asarray(self.stat.d['zagl_mass_fine'],
                                             dtype=np.float32)[self.L], 1e-3))
            add('f_zagl', (z - s['mu']) / s['sigma'])
        if 'fine_logz0' in g:
            s = self.sn['logz0_fine_raw']
            add('f_logz0', ((np.asarray(self.stat.d['logz0_fine'], dtype=np.float32)
                             - s['mu']) / s['sigma'])[None])
        if 'fine_urban' in g:
            s = self.sn['urban_fine_raw']
            add('f_urban', ((np.asarray(self.stat.d['urban_fine'], dtype=np.float32)
                             - s['mu']) / s['sigma'])[None])
        if 'fine_waterveg' in g:
            for key, name in (('water_fine', 'f_water'), ('vegfra_fine', 'f_vegfra')):
                s = self.sn[key + '_raw']
                add(name, (((np.asarray(self.stat.d[key], dtype=np.float32) - s['mu'])
                            / s['sigma'])[None]))
        if 'geom_diff_zagl' in g:
            # ln z_fine − 重网格后的 ln z_coarse;差值近零中心,只除以细端 σ(不减 μ)
            s = self.sn['zagl_mass_fine_log']
            zf = np.log(np.maximum(np.asarray(self.stat.d['zagl_mass_fine'],
                                              dtype=np.float32)[self.L], 1e-3))
            zc = np.log(np.maximum(np.asarray(self.stat.d['zagl_mass_coarse'],
                                              dtype=np.float32)[self.L], 1e-3))
            add('d_zagl', (zf - self.stat.regrid_field(zc, 'mass')) / s['sigma'])
        if 'geom_diff_hgt' in g:
            s = self.sn['hgt_fine_raw']
            hf = np.asarray(self.stat.d['hgt_fine'], dtype=np.float32)
            hc = self.stat.regrid_field(
                np.asarray(self.stat.d['hgt_coarse'], dtype=np.float32)[None], 'mass')[0]
            add('d_hgt', ((hf - hc) / s['sigma'])[None])
        if 'coords' in g:
            # T1.7 坐标泄漏对照:全域 min-max 归一到 [0,1]
            for key, name in (('xlat_fine', 'xlat'), ('xlong_fine', 'xlong')):
                v = np.asarray(self.stat.d[key], dtype=np.float32)
                v = (v - float(v.min())) / max(float(v.max() - v.min()), 1e-6)
                add(name, v[None])
        return torch.from_numpy(np.concatenate(chans, axis=0)).to(self.dtype)

    def __getitem__(self, idx):
        path = self.ps[idx]
        stamp = _parse_stamp(path)
        with np.load(path) as f:
            fine = {k: f[k] for k in f.keys()}
        co_path = os.path.join(self.c.coarse_dir,
                              "c_{}_{}.npz".format(self.c.scheme, stamp))
        with np.load(co_path) as f:
            co = {k: f[k] for k in f.keys()}
        y = self._targets(fine)
        y0 = self._y0(co)
        x = self._inputs(co, stamp)
        rho = self._rho(co)
        n, nx = y.shape[0], x.shape[0]
        # rho 与 y/y0/x 拼成同一张量后再切,保证各段共享同一次随机裁剪窗口
        # (rho 通道数为 L,与 y 的通道数不同,切片按各自实际通道数);
        # return_agl_tables=True 时在末尾追加 4 张 AGL 表(同一 crop,窗口一致)
        parts = [y, y0, x, rho]
        if self.agl_tables is not None:
            parts += [self.agl_tables[k] for k in AGL_TABLE_SUFFIXES]
        cropped = self.crop(torch.cat(parts, dim=0))
        base = 2 * n + nx
        out = {'x': cropped[2 * n:base], 'y': cropped[:n], 'y0': cropped[n:2 * n],
               'rho': cropped[base:base + rho.shape[0]]}
        if self.agl_tables is not None:
            at = base + rho.shape[0]
            for k in AGL_TABLE_SUFFIXES:
                out['agl_' + k] = cropped[at:at + self.agl_tables[k].shape[0]]
                at += self.agl_tables[k].shape[0]
        return {k: torch.nan_to_num(v, self.c.missing_value) for k, v in out.items()}
