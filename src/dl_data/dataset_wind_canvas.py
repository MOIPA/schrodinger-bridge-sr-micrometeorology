# -*- coding: utf-8 -*-
"""
阶段0(导师大纲)DatasetWindCanvas:原生 C 网格画布 + 分组输入条件。

- 目标(y):d04 原生交错 U/V(W 可选,界面层)+ U10/V10,标准化用细端 σ(μ=0)
- y0:粗端(d02)风场重网格到细端原生位置,与 y 同一标准化空间(细端 σ)
- 条件(x):按 input_groups 组装;粗端逐层量用粗端 σ、细端静态用全域统计
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
from torch.utils.data import Dataset

from src.dl_config.base_config import BaseDatasetConfig
from src.dl_data.wind_canvas_statics import FINE_SHAPES, CanvasStatics
from src.utils.random_crop import RandomCrop2D

INPUT_GROUPS = [
    'coarse_wind_uv', 'coarse_w', 'coarse_zagl', 'coarse_logz0',
    'coarse_most', 'coarse_flux', 'coarse_theta', 'coarse_ph',
    'time_enc', 'coszen', 'fine_static',
]

_STAMP = re.compile(r'_(\d{8}T\d{6})\.npz$')


def _parse_stamp(path):
    m = _STAMP.search(os.path.basename(path))
    return m.group(1) if m else None


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
    dataset_name: typing.ClassVar[str] = "DatasetWindCanvas"

    def __post_init__(self):
        assert self.scheme in ("myj", "ysu")
        assert self.dtype in ("float16", "float32")
        for g in self.input_groups:
            assert g in INPUT_GROUPS, "unknown input group " + g


class DatasetWindCanvas(Dataset):
    def __init__(self, file_paths, config, **kwargs):
        self.c = copy.deepcopy(config)
        # 只用整点帧:stamp='YYYYMMDDTHHMMSS',[9:11]=小时、[11:13]=分钟
        self.ps = [p for p in file_paths if p.endswith(".npz")
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

    # ------------------------------------------------------------ 通道清单
    def target_channel_names(self):
        names = ['y_u_{:02d}'.format(k) for k in self.L] + \
                ['y_v_{:02d}'.format(k) for k in self.L]
        if self.c.include_w:
            names += ['y_w_{:02d}'.format(k) for k in self.WL]
        names += ['y_u10', 'y_v10']
        return names

    def input_channel_names(self):
        g = self.c.input_groups
        names = []
        if 'coarse_wind_uv' in g:
            names += ['c_u_{:02d}'.format(k) for k in self.L]
            names += ['c_v_{:02d}'.format(k) for k in self.L]
        if 'coarse_w' in g:
            names += ['c_w_{:02d}'.format(k) for k in self.WL]
        if 'coarse_zagl' in g:
            names += ['c_zagl_{:02d}'.format(k) for k in self.L]
        if 'coarse_logz0' in g:
            names += ['c_logz0']
        if 'coarse_most' in g:
            names += ['c_rmol', 'c_ust', 'c_pblh']
        if 'coarse_flux' in g:
            names += ['c_hfx', 'c_t2', 'c_psfc']
        if 'coarse_theta' in g:
            names += ['c_theta_{:02d}'.format(k) for k in self.L]
        if 'coarse_ph' in g:
            names += ['c_ph_{:02d}'.format(k) for k in self.WL]
        if 'time_enc' in g:
            names += ['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']
        if 'coszen' in g:
            names += ['coszen']
        if 'fine_static' in g:
            names += ['f_hgt']
            names += ['f_zagl_{:02d}'.format(k) for k in self.L]
            names += ['f_logz0', 'f_urban', 'f_water', 'f_vegfra']
        return names

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
        if 'coszen' in g:
            r = self.stat.regrid(np.asarray(co['c_coszen'], dtype=np.float32)[None], 'mass')
            add('coszen', r[0].reshape(FINE_SHAPES['mass']))
        if 'fine_static' in g:
            s = self.sn['hgt_fine_raw']
            add('f_hgt', ((np.asarray(self.stat.d['hgt_fine'], dtype=np.float32)
                           - s['mu']) / s['sigma'])[None])
            s = self.sn['zagl_mass_fine_log']
            z = np.log(np.maximum(np.asarray(self.stat.d['zagl_mass_fine'],
                                             dtype=np.float32)[self.L], 1e-3))
            add('f_zagl', (z - s['mu']) / s['sigma'])
            for key, name in (('logz0_fine', 'f_logz0'), ('urban_fine', 'f_urban'),
                              ('water_fine', 'f_water'), ('vegfra_fine', 'f_vegfra')):
                s = self.sn[key + '_raw']
                add(name, (((np.asarray(self.stat.d[key], dtype=np.float32) - s['mu'])
                            / s['sigma'])[None]))
        if 'time_enc' in g:
            dt = datetime.strptime(stamp, '%Y%m%dT%H%M%S') + timedelta(hours=8)  # 本地时
            hour = dt.hour + dt.minute / 60.0
            doy = dt.timetuple().tm_yday
            for val in (np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)):
                add('time', np.full((1, self.c.hr_data_shape[0], self.c.hr_data_shape[1]),
                                    val, dtype=np.float32))
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
        n = y.shape[0]
        cropped = self.crop(torch.cat([y, y0, x], dim=0))
        out = {'x': cropped[2 * n:], 'y': cropped[:n], 'y0': cropped[n:2 * n]}
        return {k: torch.nan_to_num(v, self.c.missing_value) for k, v in out.items()}
