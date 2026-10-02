# -*- coding: utf-8 -*-
"""阶段 2 配置生成:损失函数与物理约束消融 7 个 run 的 yml(通道数由通道函数算出)。

固定 V* 输入口径(phase1r t14_noenc_cos:coarse_wind_uv + coarse_zagl +
coarse_logz0 + fine_geom + coszen,残差输出,canvas 布局,L1,Q10 步),
只改 si 段:
  p2_l2                loss_type=L2
  p2_div_lo/mid/hi     divergence_weight 三档(占位值,96 探针后回填)
  p2_spec              spectral_weight(占位)
  p2_ext               extreme_weight(占位)+ extreme_levels [0..9]
  p2_vort              vorticity_weight(占位)
生成到 configs/深圳/phase2/config_wind_canvas_p2_<tag>.yml。

phys_* 参数不手写,生成时从数据目录读出:
  phys_scale(72)  <- normalize_config.json fine['myj'] 的
                     [u.σ(23), v.σ(23), w.σ(24), u10.σ, v10.σ](通道序同 y);
  phys_dz(23)     <- statics.npz zagl_iface_fine 的水平平均界面高度差分
                     (不水平均匀,用场平均剖面;在 yml 头注释注明);
  phys_div_tau(23)<- results/outline/truth_diagnostics.json 的 divergence_residual
                     p95_abs(三时刻均值),按 ln z 线性插值到层 0..22(范围外钳制);
  phys_dx=1000.0, phys_min_t=0.5。

运行(pytorch-gpu / wind3d 环境,仓库根目录;生成需能读数据目录):
  python scripts/outline/86_gen_phase2_configs.py                       # 写 yml
  python scripts/outline/86_gen_phase2_configs.py --dry_run             # 只打印 si 段
  python scripts/outline/86_gen_phase2_configs.py --data_dir <root>     # 换数据根
本地无服务器数据时可直接用仓库内静态文件(results/outline 本身即 static 目录):
  python scripts/outline/86_gen_phase2_configs.py --data_dir results/outline --dry_run
权重覆盖(探针定标后重生成):
  python scripts/outline/86_gen_phase2_configs.py --div-mid 3e-3 --spectral 0.05
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.dl_data.dataset_wind_canvas import (  # noqa: E402
    build_input_channel_names,
    build_target_channel_names,
)

SERVER_ROOT = "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology"
OUT_DIR = os.path.join(ROOT, "configs", "深圳", "phase2")
TRUTH_JSON = os.path.join(ROOT, "results", "outline", "truth_diagnostics.json")
L = list(range(23))
N_LEVELS = 23

# V* 输入组(phase1r t14_noenc_cos,不加不减)
INPUT_GROUPS = ['coarse_wind_uv', 'coarse_zagl', 'coarse_logz0', 'fine_geom', 'coszen']

# ---- 权重:96 探针(V* 基线, 2026-10-02)定标,目标 = 各物理项量级约为 data 项的 10% ----
# 探针原始值(data=0.1063):div 1.3445, vort 4.25e-4, spectral 0.2749, extreme 0.2463;
# hinge 激活率 fraction(|D|>tau)=0.427,P95(|D|/tau)=3.95(约束有实际压降空间)。
WEIGHTS = {
    'div_lo': 0.00263608,   # = div_mid/3
    'div_mid': 0.00790824,  # 探针建议(≈data 的 10%)
    'div_hi': 0.0237247,    # = div_mid*3
    'spectral': 0.0386746,  # 探针建议(≈data 的 10%)
    'extreme': 0.0431783,   # 探针建议(≈data 的 10%)
    'vorticity': 25.0074,   # 探针建议(≈data 的 10%)
}

SI_BASE = {
    'eps': 0.2,
    'formula': 'quadratic',
    'loss_type': 'L1',
    'n_timestep': 10,
    'channel_weights': None,
    'divergence_weight': 0.0,
    'vorticity_weight': 0.0,
    'residual_output': True,
    'state_layout': 'canvas',
}


def variants(w):
    """(tag, si 覆盖) —— 与 docs 阶段 2 计划表一致;权重全部来自 WEIGHTS。"""
    return [
        ('l2', {'loss_type': 'L2'}),
        ('div_lo', {'divergence_weight': w['div_lo']}),
        ('div_mid', {'divergence_weight': w['div_mid']}),
        ('div_hi', {'divergence_weight': w['div_hi']}),
        ('spec', {'spectral_weight': w['spectral']}),
        ('ext', {'extreme_weight': w['extreme'], 'extreme_levels': list(range(10))}),
        ('vort', {'vorticity_weight': w['vorticity']}),
    ]


# ---------------------------------------------------------------------------
# 物理参数:从数据目录读出(不手写)
# ---------------------------------------------------------------------------
def static_dir_of(data_dir):
    """定位 static 目录:<data_dir>/prepare_npz_outline_static,或 data_dir 本身即 static。"""
    cand = os.path.join(data_dir, 'prepare_npz_outline_static')
    if os.path.isfile(os.path.join(cand, 'normalize_config.json')) and \
            os.path.isfile(os.path.join(cand, 'statics.npz')):
        return cand
    if os.path.isfile(os.path.join(data_dir, 'normalize_config.json')) and \
            os.path.isfile(os.path.join(data_dir, 'statics.npz')):
        return data_dir
    sys.stderr.write(
        "错误: 在 {cand} 与 {data_dir} 下都找不到 normalize_config.json + statics.npz。\n"
        "      服务器上请确认 --data_dir 指向数据根(含 prepare_npz_outline_*);\n"
        "      本地无服务器数据时可用仓库内静态文件:\n"
        "        python scripts/outline/86_gen_phase2_configs.py"
        " --data_dir results/outline --dry_run\n".format(cand=cand, data_dir=data_dir))
    sys.exit(1)


def read_phys_scale(static_dir):
    """长度 72:细端各分量物理 σ,通道序 u(23)/v(23)/w(24)/u10/v10。"""
    with open(os.path.join(static_dir, 'normalize_config.json')) as f:
        norms = json.load(f)
    fn = norms['fine']['myj']
    sig_u = [float(v) for v in fn['u']['sigma'][:N_LEVELS]]
    sig_v = [float(v) for v in fn['v']['sigma'][:N_LEVELS]]
    sig_w = [float(v) for v in fn['w']['sigma'][:N_LEVELS + 1]]
    sig_u10 = float(fn['u10']['sigma'])
    sig_v10 = float(fn['v10']['sigma'])
    out = sig_u + sig_v + sig_w + [sig_u10, sig_v10]
    assert len(out) == 72, 'phys_scale 长度 {} != 72'.format(len(out))
    return out


def read_phys_dz(static_dir):
    """长度 23:zagl_iface_fine 水平平均界面高度差分(非水平均匀,取场平均剖面)。"""
    with np.load(os.path.join(static_dir, 'statics.npz')) as s:
        zi = np.asarray(s['zagl_iface_fine'], dtype=np.float64)
    prof = zi.reshape(zi.shape[0], -1).mean(axis=1)          # (41,) 水平平均
    dz = prof[1:N_LEVELS + 1] - prof[:N_LEVELS]
    return [float(v) for v in dz]


def read_phys_div_tau(static_dir, truth_json):
    """长度 23:真值散度 p95_abs(三时刻均值)按 ln z 插值到层 0..22(范围外钳制)。"""
    with np.load(os.path.join(static_dir, 'statics.npz')) as s:
        zm = np.asarray(s['zagl_mass_fine'], dtype=np.float64)
    zm = zm.reshape(zm.shape[0], -1).mean(axis=1)             # 水平平均层高
    if not os.path.isfile(truth_json):
        sys.stderr.write("错误: 找不到真值诊断 {}".format(truth_json))
        sys.exit(1)
    with open(truth_json) as f:
        div = json.load(f)['divergence_residual']
    stamps = sorted(div.keys())
    keys = sorted([k for k in div[stamps[0]] if str(k).isdigit()], key=int)
    z_obs, tau_obs = [], []
    for k in keys:
        vals = [float(div[st][k]['p95_abs']) for st in stamps if k in div[st]]
        z_obs.append(zm[int(k)])
        tau_obs.append(sum(vals) / len(vals))                 # 先对三时刻取均值
    z_obs, tau_obs = np.asarray(z_obs), np.asarray(tau_obs)
    z_t = zm[:N_LEVELS]
    ln = np.log(np.maximum(z_t, 1e-3))
    ln_obs = np.log(np.maximum(z_obs, 1e-3))
    tau = np.interp(ln, ln_obs, tau_obs)                      # np.interp 端点自动钳制
    print('phys_div_tau 观测层 z(m)={} tau(p95_abs)={} -> 层 0..22:'.format(
        [round(float(z), 1) for z in z_obs], [float('%.3e' % t) for t in tau_obs]))
    print('  tau = [{}]'.format(', '.join('%.4e' % v for v in tau)))
    return [float(v) for v in tau]


def read_phys(static_dir, truth_json):
    return {
        'phys_scale': read_phys_scale(static_dir),
        'phys_dz': read_phys_dz(static_dir),
        'phys_div_tau': read_phys_div_tau(static_dir, truth_json),
        'phys_dx': 1000.0,
        'phys_min_t': 0.5,
    }


# ---------------------------------------------------------------------------
# 配置组装(数据/loader/model/train 段与 V* 模板逐字段一致)
# ---------------------------------------------------------------------------
def make_config(phys, si_extra):
    n_out = len(build_target_channel_names(L, True))
    n_cond = len(build_input_channel_names(INPUT_GROUPS, L, True))
    si = dict(SI_BASE)
    si.update(phys)
    si.update(si_extra)
    cfg = {
        'data': {
            'scheme': 'myj',
            'coarse_dir': SERVER_ROOT + '/prepare_npz_outline_coarse',
            'statics_dir': SERVER_ROOT + '/prepare_npz_outline_static',
            'normalize_json': None,
            'split_manifest': SERVER_ROOT + '/prepare_npz_outline_static/split.json',
            'target_levels': list(L),
            'input_groups': list(INPUT_GROUPS),
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
        'si': si,
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
    return cfg, n_out, n_cond


def yml_header(tag, si_extra):
    """生成 yml 的头部注释(来源与占位状态写清楚,便于复核)。"""
    lines = [
        "# 阶段 2 配置 {}:V* 输入组({})".format(tag, ','.join(INPUT_GROUPS)),
        "# 只改 si 段: {}".format(si_extra),
        "# 权重为占位值,96_weight_probe 探针后回填 86 的 WEIGHTS 并重跑生成。",
        "# phys_scale <- normalize_config.json fine['myj'] 各分量 σ(u/v/w/u10/v10);",
        "# phys_dz   <- statics.npz zagl_iface_fine 的水平平均界面高度差分",
        "#              (非水平均匀,用场平均剖面);",
        "# phys_div_tau <- truth_diagnostics.json divergence_residual p95_abs",
        "#              (三时刻均值,ln z 线性插值到层 0..22,范围外钳制)。",
    ]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description="阶段 2 配置生成(canvas 损失/物理消融)")
    ap.add_argument("--data_dir", default=SERVER_ROOT,
                    help="数据根目录(含 prepare_npz_outline_static),默认服务器路径")
    ap.add_argument("--truth_json", default=TRUTH_JSON,
                    help="真值诊断 json(phys_div_tau 来源),默认 results/outline")
    ap.add_argument("--dry_run", action="store_true", help="只打印将写入的 si 段,不写文件")
    ap.add_argument("--div-lo", type=float, default=None, dest="div_lo")
    ap.add_argument("--div-mid", type=float, default=None, dest="div_mid")
    ap.add_argument("--div-hi", type=float, default=None, dest="div_hi")
    ap.add_argument("--spectral", type=float, default=None)
    ap.add_argument("--extreme", type=float, default=None)
    ap.add_argument("--vorticity", type=float, default=None)
    args = ap.parse_args()

    weights = dict(WEIGHTS)
    for k in weights:
        v = getattr(args, k)
        if v is not None:
            weights[k] = v
    print('权重: {}'.format(
        ', '.join('{}={:g}'.format(k, weights[k]) for k in sorted(weights))))

    data_dir = args.data_dir if os.path.isabs(args.data_dir) \
        else os.path.join(ROOT, args.data_dir)
    static_dir = static_dir_of(data_dir)
    print('static 目录: {}'.format(static_dir))
    phys = read_phys(static_dir, args.truth_json)

    out_dir = OUT_DIR
    if not args.dry_run and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    print('{:<10} {:>5} {:>5} {:>6}  {}'.format('tag', 'out', 'cond', 'in', 'si 覆盖'))
    for tag, si_extra in variants(weights):
        cfg, n_out, n_cond = make_config(phys, si_extra)
        path = os.path.join(out_dir, 'config_wind_canvas_p2_' + tag + '.yml')
        if args.dry_run:
            print('---- {} (dry_run, 未写文件) ----'.format(path))
            print(yml_header(tag, si_extra).rstrip())
            print(yaml.safe_dump(cfg['si'], sort_keys=False, allow_unicode=True).rstrip())
        else:
            with open(path, 'w') as f:
                f.write(yml_header(tag, si_extra))
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print('{:<10} {:>5} {:>5} {:>6}  {}'.format(
            tag, n_out, n_cond, n_out + n_cond,
            ', '.join('{}={}'.format(k, v) for k, v in sorted(si_extra.items()))))
    if args.dry_run:
        print('dry_run: {} 个配置未写盘 -> {}'.format(len(variants(weights)), out_dir))
    else:
        print('生成 {} 个配置 -> {}'.format(len(variants(weights)), out_dir))


if __name__ == '__main__':
    main()
