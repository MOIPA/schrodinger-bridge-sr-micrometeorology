# -*- coding: utf-8 -*-
"""阶段0(导师大纲)canvas 数据集工具:静态场、重网格权重、画布组装、AGL 插值表。

画布约定(细端 d04 质量网格 99x120,原生交错):
  质量点/U 面/V 面同处 [j, i];canvas (100, 121);
  U 占 cols 0..120(满列)、V 占 rows 0..99(满行),虚行/列按边缘复制填充。
"""
import os

import numpy as np

MASS_SHAPE = (99, 120)
CANVAS_SHAPE = (100, 121)
FINE_SHAPES = {'mass': MASS_SHAPE, 'u': (99, 121), 'v': (100, 120),
               'w': (99, 120)}

# AGL 评估网格(单一来源):训练侧插值表与评估侧 scripts/outline/agl_eval_common 同参
TARGET_AGL = np.array([10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000],
                      dtype=np.float64)


class CanvasStatics(object):
    """statics.npz 的读取与重网格/画布操作。"""

    def __init__(self, statics_dir):
        path = os.path.join(statics_dir, "statics.npz")
        with np.load(path) as s:
            self.d = {k: s[k] for k in s.files}

    def regrid(self, arr, cls):
        """(..., n_src_flat) 或 (..., ny_s, nx_s) -> (..., n_dst):粗端 -> 细端原生交错位置。

        权重表里的源索引是**扁平**索引,若传入 (..., ny_s, nx_s) 必须先摊平,
        否则会拿扁平索引去索引最后一维(越界或静默取错位置)。
        """
        idx = self.d['regrid_idx_' + cls]        # (n_dst, 4) 源节点扁平索引
        w = self.d['regrid_w_' + cls]            # (n_dst, 4)
        a = np.asarray(arr, dtype=np.float32)
        n_src = int(np.prod(self.d['regrid_shape_' + cls]))
        if a.shape[-1] != n_src:
            a = a.reshape(a.shape[:-2] + (n_src,))
        return (a[..., idx] * w).sum(axis=-1)    # 前置维任意(层/批次)均可

    def regrid_field(self, arr, cls):
        """(nlev, ny_s, nx_s) -> (nlev, ny_d, nx_d)。"""
        out = self.regrid(np.asarray(arr).reshape(arr.shape[0], -1), cls)
        return out.reshape((arr.shape[0],) + FINE_SHAPES[cls])

    @staticmethod
    def place(arr):
        """(..., ny_k, nx_k) -> (..., 100, 121);虚行/列边缘复制。"""
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2:
            a = a[None]
        nyc, nxc = CANVAS_SHAPE
        out = np.empty((a.shape[0], nyc, nxc), dtype=np.float32)
        h, w = a.shape[1], a.shape[2]
        out[:, :h, :w] = a
        if h < nyc:
            out[:, h:, :] = out[:, h - 1:h, :]
        if w < nxc:
            out[:, :, w:] = out[:, :, w - 1:w]
        return out


def build_agl_table(z_agl, targets, use_10m_anchor=True):
    """逐像素 AGL 插值表(ln z 线性;可作用于画布或任意同形网格)。

    idx >= 0 : value = w*f[idx] + (1-w)*f[idx+1]
    idx == -1: 低于首层: w*f[0] + (1-w)*f_10m(仅质量层)
    idx == -2: 目标 10 m,直接取 10 m 通道
    idx == -3: 无解(不应出现)
    """
    z_agl = np.asarray(z_agl, dtype=np.float64)
    nlev, ny, nx = z_agl.shape
    lnz = np.log(np.maximum(z_agl, 1e-3))
    nt = len(targets)
    idx = np.full((nt, ny, nx), -3, dtype=np.int16)
    w = np.zeros((nt, ny, nx), dtype=np.float32)
    for ti, h in enumerate(targets):
        if use_10m_anchor and h == 10:
            idx[ti] = -2
            continue
        k = np.sum(z_agl <= h, axis=0).astype(np.int64) - 1
        below = k < 0
        kc = np.clip(k, 0, nlev - 2)
        z1 = np.take_along_axis(lnz, (kc + 1)[None], axis=0)[0]
        z0 = np.take_along_axis(lnz, kc[None], axis=0)[0]
        ww = np.clip((z1 - np.log(h)) / np.maximum(z1 - z0, 1e-12), 0.0, 1.0)
        idx[ti] = kc.astype(np.int16)
        w[ti] = ww.astype(np.float32)
        if below.any() and use_10m_anchor:
            wb = (lnz[0] - np.log(float(h))) / (lnz[0] - np.log(10.0))
            idx[ti][below] = -1
            w[ti][below] = np.clip(wb, 0.0, 1.0)[below].astype(np.float32)
    return idx, w


def build_agl_tables_native(zagl_mass_fine, zagl_iface_fine, levels, targets=None):
    """原生网格 AGL 插值表(与 scripts/outline/agl_eval_common.load_tables 同源同参)。

    levels: 训练层索引序列(0..22)。
    返回 {'idx_m','w_m','idx_i','w_i'};idx_m/w_m 由 L 个质量层、idx_i/w_i 由 L+1 个
    界面层数组构建,输出均为 (nt,ny,nx)——层轴已编码进 idx(与 load_tables 逐元素相同)。
    """
    levels = list(levels)
    zagl_m = np.asarray(zagl_mass_fine, np.float64)[levels]
    zagl_i = np.asarray(zagl_iface_fine, np.float64)[levels[0]:levels[-1] + 2]
    idx_m, w_m = build_agl_table(zagl_m, targets if targets is not None else TARGET_AGL,
                                 True)
    idx_i, w_i = build_agl_table(zagl_i, targets if targets is not None else TARGET_AGL,
                                 False)
    return {'idx_m': idx_m, 'w_m': w_m, 'idx_i': idx_i, 'w_i': w_i}
