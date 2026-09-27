# -*- coding: utf-8 -*-
"""阶段 1 配置生成:输入变量消融 15 个 run 的 yml(通道数由 dataset 的通道函数算出,不手写)。

固定口径(2026-09-27 定稿):myj / 23 层(0..22)/ L1 损失 / 模式层监督 / 150 epochs + valid 早停。
变体表见 VARIANTS;生成到 configs/深圳/phase1/config_wind_canvas_p1_<tag>.yml。

运行(pytorch-gpu 或 wind3d 环境,仓库根目录):
  python scripts/outline/85_gen_phase1_configs.py
"""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.dl_data.dataset_wind_canvas import (  # noqa: E402
    build_input_channel_names,
    build_target_channel_names,
)

SERVER_ROOT = "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology"
OUT_DIR = os.path.join(ROOT, "configs", "深圳", "phase1")
L = list(range(23))

BASE = ['coarse_wind_uv', 'coarse_zagl', 'coarse_logz0', 'time_enc', 'fine_geom']

# (tag, 输入组, si 覆盖项) —— 与 docs 阶段 1 计划表一致
VARIANTS = [
    ('base', BASE, {}),
    # T1.2 几何通道三选一("不提供" 即 base)
    ('t12_zagldiff', BASE + ['geom_diff_zagl'], {}),
    ('t12_hgtdiff', BASE + ['geom_diff_hgt'], {}),
    # T1.3 增量消融(每次只加一组;组 B 在组 A 基础上)
    ('t13_w', BASE + ['coarse_w'], {}),
    ('t13_most', BASE + ['coarse_most'], {}),
    ('t13_mostflux', BASE + ['coarse_most', 'coarse_flux'], {}),
    ('t13_theta', BASE + ['coarse_theta'], {}),
    ('t13_ph', BASE + ['coarse_ph'], {}),
    # T1.4 太阳强迫编码
    ('t14_cos', BASE + ['coszen'], {}),
    ('t14_noenc_cos', [g for g in BASE if g != 'time_enc'] + ['coszen'], {}),
    # T1.5 下垫面三级(递进)
    ('t15_z0', BASE + ['fine_logz0'], {}),
    ('t15_z0_urban', BASE + ['fine_logz0', 'fine_urban'], {}),
    ('t15_z0_urban_wv', BASE + ['fine_logz0', 'fine_urban', 'fine_waterveg'], {}),
    # T1.6 输出形式
    ('t16_residual', BASE, {'residual_output': True}),
    # T1.7 坐标泄漏对照
    ('t17_coords', BASE + ['coords'], {}),
]


def make_config(groups, si_extra):
    n_out = len(build_target_channel_names(L, True))
    n_cond = len(build_input_channel_names(groups, L, True))
    cfg = {
        'data': {
            'scheme': 'myj',
            'coarse_dir': SERVER_ROOT + '/prepare_npz_outline_coarse',
            'statics_dir': SERVER_ROOT + '/prepare_npz_outline_static',
            'normalize_json': None,
            'split_manifest': SERVER_ROOT + '/prepare_npz_outline_static/split.json',
            'target_levels': list(L),
            'input_groups': list(groups),
            'include_w': True,
            'hr_data_shape': [99, 120],
            'hr_cropped_shape': [96, 112],
            'is_clipped': False,
            'min_clipped_value': None,
            'max_clipped_value': None,
            'missing_value': 0.0,
            'dtype': 'float32',
            'day_night_filter': 'all',
        },
        'loader': {
            'batch_size': 2,
            'dl_data_ver': 'wrf_3d_v2_canvas',
            'num_workers': 4,
            'seed': 42,
            'train_valid_test_ratios': [0.7, 0.1, 0.2],
        },
        'model': {
            'attn_res': [16],
            'channel_mults': [1, 2, 4, 8, 8],
            'channels_each_head': 16,
            'dropout': 0.2,
            'in_channel': n_out + n_cond,
            'inner_channel': 64,
            'max_period': 10.0,
            'out_channel': n_out,
            'res_blocks': 1,
            'resblock_updown': True,
        },
        'si': {
            'eps': 0.2,
            'formula': 'quadratic',
            'loss_type': 'L1',
            'n_timestep': 10,
            'channel_weights': None,
            'divergence_weight': 0.0,
            'vorticity_weight': 0.0,
            'residual_output': False,
            'state_layout': 'canvas',
        },
        'train': {
            'early_stopping_patience': 40,
            'ema_decay': None,
            'epochs': 150,
            'learning_rate': 0.0005,
            'loss': {'loss_name': 'L1'},
            'optim_name': 'AdamW',
            'save_interval': 50,
            'seed': 78268,
            'use_amp': False,
        },
    }
    cfg['si'].update(si_extra)
    return cfg, n_out, n_cond


def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    print('{:<18} {:>5} {:>5} {:>6}  {}'.format('tag', 'out', 'cond', 'in', 'groups'))
    for tag, groups, si_extra in VARIANTS:
        cfg, n_out, n_cond = make_config(groups, si_extra)
        path = os.path.join(OUT_DIR, 'config_wind_canvas_p1_{}.yml'.format(tag))
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print('{:<18} {:>5} {:>5} {:>6}  {}'.format(
            tag, n_out, n_cond, n_out + n_cond, ','.join(groups)))
    print('生成 {} 个配置 -> {}'.format(len(VARIANTS), OUT_DIR))


if __name__ == '__main__':
    main()
