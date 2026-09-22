# -*- coding: utf-8 -*-
"""阶段 0 合成数据最小自检(秒级):把 DatasetWindCanvas 全路径走一遍。

不读 wrfout / 不读 44 GB 抽取结果:用真实 statics.npz + normalize_config.json,
fine/coarse npz 用随机小数据合成,把每个 input_group、每个目标通道、裁剪、
以及 split_manifest 划分全部执行到,先把 shape/索引类错误照出来。
(教训:上一轮真数据冒烟每次要等几分钟才暴露一个 shape 错,合成自检几秒钟能照出全部。)

运行(wind3d 或 pytorch-gpu 环境,仓库根目录):
  python scripts/outline/80_fixture_check.py
"""
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from outline_common import OUT_STATIC  # noqa: E402
from src.dl_data.block_split import split_paths_by_manifest  # noqa: E402
from src.dl_data.dataset_wind_canvas import (INPUT_GROUPS, DatasetWindCanvas,  # noqa: E402
                                             DatasetWindCanvasConfig)

L = list(range(23))
WL = list(range(0, 25))
FINE_SHAPES = {'f_u': (40, 99, 121), 'f_v': (40, 100, 120), 'f_w': (41, 99, 120),
               'f_u10': (99, 120), 'f_v10': (99, 120)}
COARSE_SHAPES = {'c_u': (40, 120, 151), 'c_v': (40, 121, 150), 'c_w': (41, 120, 150),
                 'c_theta': (40, 120, 150), 'c_ph': (41, 120, 150),
                 'c_rmol': (120, 150), 'c_ust': (120, 150), 'c_pblh': (120, 150),
                 'c_hfx': (120, 150), 'c_t2': (120, 150), 'c_psfc': (120, 150),
                 'c_coszen': (120, 150)}


def pick_train_stamp(static_dir):
    """从真 split.json 里取一个属于 train 的整点,保证 manifest 匹配可测。"""
    with open(os.path.join(static_dir, 'split.json')) as f:
        hours = json.load(f)['hours']
    iso = sorted(hours['train'])[0]
    return iso[:4] + iso[5:7] + iso[8:10] + 'T' + iso[11:13] + '0000', iso


def main():
    static_dir = OUT_STATIC
    with open(os.path.join(static_dir, 'normalize_config.json')) as f:
        norm = json.load(f)
    sig = np.asarray(norm['fine']['myj']['sigma_uv_profile'])
    assert np.isfinite(sig).all() and (sig > 0).all(), \
        'normalize_config.json 含 NaN/0(说明 40 号没真读到数据)'

    stamp, iso = pick_train_stamp(static_dir)
    rng = np.random.RandomState(0)
    fine = {k: rng.randn(*s).astype(np.float32) for k, s in FINE_SHAPES.items()}
    coarse = {k: rng.randn(*s).astype(np.float32) for k, s in COARSE_SHAPES.items()}
    coarse['c_ust'] = np.abs(coarse['c_ust']) * 0.1 + 0.2
    coarse['c_pblh'] = np.abs(coarse['c_pblh']) * 100 + 500
    coarse['c_coszen'] = np.abs(coarse['c_coszen']) * 0.5

    tmp = tempfile.mkdtemp(prefix='canvas_fixture_')
    try:
        fdir, cdir = os.path.join(tmp, 'fine'), os.path.join(tmp, 'coarse')
        os.makedirs(fdir)
        os.makedirs(cdir)
        fpath = os.path.join(fdir, 'f_myj_{}.npz'.format(stamp))
        np.savez_compressed(fpath, **fine)
        cpath = os.path.join(cdir, 'c_myj_{}.npz'.format(stamp))
        np.savez_compressed(cpath, **coarse)

        # 1) split_manifest 划分(真 split.json × 合成的正确命名文件)
        manifest = os.path.join(static_dir, 'split.json')
        dict_paths, unmatched = split_paths_by_manifest([fpath], manifest)
        print('manifest 划分: train={} valid={} test={} unmatched={}'.format(
            len(dict_paths['train']), len(dict_paths['valid']),
            len(dict_paths['test']), len(unmatched)))
        assert not unmatched and len(dict_paths['train']) == 1, \
            '时间戳 {} ({}) 未被 split.json 正确归类'.format(stamp, iso)

        # 2) Dataset 全输入组(比真冒烟多覆盖 w/most/flux/theta/ph/coszen/fine_static)
        cfg = DatasetWindCanvasConfig(
            scheme='myj', coarse_dir=cdir, statics_dir=static_dir,
            target_levels=L, input_groups=list(INPUT_GROUPS), include_w=True,
            hr_data_shape=[99, 120], hr_cropped_shape=[96, 112], dtype='float32')
        ds = DatasetWindCanvas([fpath], cfg)
        assert len(ds) == 1, '整点过滤把合成帧滤掉了'
        out_names, in_names = ds.target_channel_names(), ds.input_channel_names()
        item = ds[0]
        print('通道: 输入 {} / 输出 {}'.format(len(in_names), len(out_names)))
        print('张量: x={} y={} y0={}'.format(tuple(item['x'].shape),
                                            tuple(item['y'].shape), tuple(item['y0'].shape)))
        # 逐组报数,便于定位是哪一组通道数对不上
        for g in INPUT_GROUPS:
            cfg1 = DatasetWindCanvasConfig(
                scheme='myj', coarse_dir=cdir, statics_dir=static_dir,
                target_levels=L, input_groups=[g], include_w=True, dtype='float32')
            print('  {:<16} -> {} 通道'.format(g, len(DatasetWindCanvas([fpath], cfg1)
                                                   .input_channel_names())))
        assert item['x'].shape == (len(in_names), 96, 112)
        assert item['y'].shape == (len(out_names), 96, 112)
        assert item['y0'].shape == (len(out_names), 96, 112)
        for k in ('x', 'y', 'y0'):
            assert torch.isfinite(item[k]).all(), k + ' 含非有限值'
        # y0 是粗端重网格场,y 是真值,两者不应完全相同
        d = float((item['y'] - item['y0']).abs().mean())
        assert d > 0, 'y0 与 y 完全相同(重网格没生效?)'
        print('y-y0 平均绝对差 {:.4f}(随机合成数据,仅验证非退化)'.format(d))
        print('FIXTURE OK')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
