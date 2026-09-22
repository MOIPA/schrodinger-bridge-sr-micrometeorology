# -*- coding: utf-8 -*-
"""阶段0(导师大纲)canvas 数据集工具:静态场、重网格权重、画布组装、AGL 插值。

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


class CanvasStatics(object):
    """statics.npz 的读取与重网格/画布操作。"""

    def __init__(self, statics_dir):
        path = os.path.join(statics_dir, "statics.npz")
        with np.load(path) as s:
            self.d = {k: s[k] for k in s.files}

    def regrid(self, arr, cls):
        """(..., n_src_flat) -> (..., n_dst):粗端 -> 细端原生交错位置。"""
        idx = self.d['regrid_idx_' + cls]
        w = self.d['regrid_w_' + cls]
        a = np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1])
        return np.einsum('ln,nd->ld', a[:, idx], w)

    def regrid_field(self, arr, cls):
        """(nlev, ny_s, nx_s) -> (nlev, ny_d, nx_d)。"""
        out = self.regrid(arr.reshape(arr.shape[0], -1), cls)
        return out.reshape((arr.shape[0],) + FINE_SHAPES[cls])

    @staticmethod
    def place(arr, cls=None):
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
