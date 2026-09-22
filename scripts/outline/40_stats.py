# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · T0.2 标准化统计(表 7)

在训练块上一次性计算,粗细端各自统计:
  风分量 U/V : μ 强制 0,只缩放;σ_k = sqrt(mean(U²+V²)/2) 逐层,U/V 合并
  W          : μ 强制 0,σ 逐层独立
  RMOL       : 先化 z1/L(z1 = 粗端第一模式层 z_agl),再 sign(x)·log(1+|x|),后常规标准化
  UST / PBLH : 先取对数再标准化
  HFX/T2/PSFC/θ 廓线/PH : 常规标准化(廓线逐层)
  z_agl      : 取对数后"全层统一"标准化(唯一不逐层)
  log(z0)/HGT/urban/water/vegfra : 二维静态场,全域统一(与时间无关)
  时间编码 hour/doy sin/cos 与 cos(SZA) : 不处理

输出 normalize_config.json(粗细两端 × 两方案)+ σ 廓线图。

运行(pytorch-gpu 环境,需 20/21/30 已完成):
  python scripts/outline/40_stats.py
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline_common import OUT_COARSE, OUT_FINE, OUT_STATIC, ensure_dir

L2 = lambda x: np.sign(x) * np.log1p(np.abs(x))  # noqa: E731


class Acc(object):
    """流式累加器:按层累加 sum / sumsq / count。"""

    def __init__(self, nlev=1):
        self.nlev = nlev
        self.s = np.zeros(nlev, dtype=np.float64)
        self.q = np.zeros(nlev, dtype=np.float64)
        self.n = 0.0

    def add(self, arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 2:
            self.s[0] += a.sum()
            self.q[0] += (a * a).sum()
        else:
            flat = a.reshape(a.shape[0], -1)
            self.s += flat.sum(axis=1)
            self.q += (flat * flat).sum(axis=1)
        self.n += float(np.prod(a.shape[1:]) if a.ndim > 2 else a.size)

    def result(self):
        n = max(self.n, 1.0)
        mean = self.s / n
        var = np.maximum(self.q / n - mean * mean, 0.0)
        return mean, np.sqrt(var)


def parse_stamp(name):
    m = re.search(r'_(\d{8}T\d{6})\.npz$', name)
    return m.group(1) if m else None


def hour_key(stamp):
    """'YYYYMMDDTHHMMSS' -> 'YYYYMMDDTHH'(npz 文件名的口径)。"""
    return stamp[:11]


def iso_to_hour_key(iso):
    """split.json 里的 'YYYY-MM-DDTHH:MM:SS' -> 'YYYYMMDDTHH'(与 hour_key 同口径)。"""
    return iso[:4] + iso[5:7] + iso[8:10] + 'T' + iso[11:13]


def main():
    parser = argparse.ArgumentParser(description="阶段0 T0.2 标准化统计")
    parser.add_argument("--fine_dir", default=OUT_FINE)
    parser.add_argument("--coarse_dir", default=OUT_COARSE)
    parser.add_argument("--static_dir", default=OUT_STATIC)
    parser.add_argument("--schemes", default="myj,ysu")
    args = parser.parse_args()

    split_path = os.path.join(args.static_dir, "split.json")
    if not os.path.isfile(split_path):
        raise SystemExit("先运行 30_blocks_and_split.py 生成 " + split_path)
    split = json.load(open(split_path))
    train_hours = set(iso_to_hour_key(h) for h in split['hours']['train'])
    print("训练块小时数: {}".format(len(train_hours)))

    statics = np.load(os.path.join(args.static_dir, "statics.npz"))
    schemes = [s.strip() for s in args.schemes.split(',') if s.strip()]
    out = {
        'generated': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'train_blocks': [b for b in split['blocks'] if b['split'] == 'train'],
        'rules': {
            'wind_uv': 'mu=0 (不平移); sigma_k = sqrt(mean(U^2+V^2)/2), 逐层, U/V 共用',
            'w': 'mu=0; sigma 逐层独立',
            'rmol': 'x = z1*RMOL (z1 = 粗端第一模式层 z_agl); 再 sign(x)*log(1+|x|); 后常规标准化',
            'ust/pblh': '先取自然对数,再常规标准化',
            'theta/ph/hfx/t2/psfc': '常规标准化(θ、PH 逐层)',
            'zagl': '取对数后全层统一标准化(不逐层)',
            'logz0/hgt/fractions': '二维静态场,全域统一',
            'time_encodings/coszen': '不处理',
            'note': '统计量只含训练块;评估与物理约束一律先反标准化回物理单位',
        },
        'fine': {}, 'coarse': {}, 'static': {},
    }

    # ---------------- 静态场(与时间无关,全域统计) ----------------
    for tag, names in (('fine', ['zagl_mass_fine', 'hgt_fine', 'logz0_fine', 'urban_fine',
                                'water_fine', 'vegfra_fine']),
                       ('coarse', ['zagl_mass_coarse', 'hgt_coarse', 'logz0_coarse',
                                   'urban_coarse', 'water_coarse', 'vegfra_coarse'])):
        for nm in names:
            a = np.asarray(statics[nm], dtype=np.float64)
            if nm.startswith('zagl'):
                a = np.log(a)
                mode = 'log_global_standard'
            else:
                mode = 'standard'
            out['static']['{}_{}'.format(nm, 'log' if nm.startswith('zagl') else 'raw')] = {
                'mode': mode, 'mu': float(a.mean()), 'sigma': float(a.std()),
                'note': '全层统一' if nm.startswith('zagl') else '全域统一',
            }
    print("静态场统计完成")

    # ---------------- 逐方案统计 ----------------
    z1_coarse = np.asarray(statics['zagl_mass_coarse'][0], dtype=np.float64)
    for scheme in schemes:
        print("\n=== scheme {} ===".format(scheme))
        fine_acc = {'u': Acc(40), 'v': Acc(40), 'w': Acc(41)}  # W 为 41 个界面层
        fine10 = {k: Acc(1) for k in ('u10', 'v10')}
        co_acc = {'u': Acc(40), 'v': Acc(40), 'w': Acc(41)}
        co_theta, co_ph = Acc(40), Acc(41)
        co_2d = {k: Acc(1) for k in ('rmol_x', 'ust_l', 'pblh_l', 'hfx', 't2', 'psfc', 'coszen')}
        n_fine = n_coarse = 0

        ffiles = sorted(glob.glob(os.path.join(args.fine_dir, "f_{}_*.npz".format(scheme))))
        for f in ffiles:
            stamp = parse_stamp(os.path.basename(f))
            if stamp is None or stamp[11:13] != '00':  # 只用整点帧(分钟==00,与空间降尺度训练一致)
                continue
            if hour_key(stamp) not in train_hours:
                continue
            with np.load(f) as d:
                fine_acc['u'].add(d['f_u'])
                fine_acc['v'].add(d['f_v'])
                fine_acc['w'].add(d['f_w'])
                fine10['u10'].add(d['f_u10'])
                fine10['v10'].add(d['f_v10'])
            n_fine += 1
            if n_fine % 200 == 0:
                print("  fine {}/~".format(n_fine))

        cfiles = sorted(glob.glob(os.path.join(args.coarse_dir, "c_{}_*.npz".format(scheme))))
        for f in cfiles:
            stamp = parse_stamp(os.path.basename(f))
            if stamp is None or hour_key(stamp) not in train_hours:
                continue
            with np.load(f) as d:
                co_acc['u'].add(d['c_u'])
                co_acc['v'].add(d['c_v'])
                co_acc['w'].add(d['c_w'])
                co_theta.add(d['c_theta'])
                co_ph.add(d['c_ph'])
                co_2d['rmol_x'].add(L2(z1_coarse * d['c_rmol']))
                co_2d['ust_l'].add(np.log(np.maximum(d['c_ust'], 1e-8)))
                co_2d['pblh_l'].add(np.log(np.maximum(d['c_pblh'], 1e-6)))
                co_2d['hfx'].add(d['c_hfx'])
                co_2d['t2'].add(d['c_t2'])
                co_2d['psfc'].add(d['c_psfc'])
                co_2d['coszen'].add(d['c_coszen'])
            n_coarse += 1
            if n_coarse % 200 == 0:
                print("  coarse {}/~".format(n_coarse))
        print("  读取 fine {} 帧, coarse {} 小时".format(n_fine, n_coarse))
        # 静默漏读会让统计量跑偏且看不出来,这里硬比对训练块小时数
        assert n_fine == len(train_hours), \
            "fine 帧数 {} != 训练小时数 {}".format(n_fine, len(train_hours))
        assert n_coarse == len(train_hours), \
            "coarse 小时数 {} != 训练小时数 {}".format(n_coarse, len(train_hours))

        # U/V 合并 σ(风矢量尺度): sqrt(mean(U²+V²)/2)
        u_mean, u_sig = fine_acc['u'].result()
        _, v_sig = fine_acc['v'].result()
        s_uv2 = (fine_acc['u'].q / fine_acc['u'].n + fine_acc['v'].q / fine_acc['v'].n) / 2.0
        sig_uv_f = np.sqrt(s_uv2)
        _, sig_w_f = fine_acc['w'].result()
        _, sig_u10 = fine10['u10'].result()
        _, sig_v10 = fine10['v10'].result()
        sig_uv10 = np.sqrt((fine10['u10'].q / fine10['u10'].n
                            + fine10['v10'].q / fine10['v10'].n) / 2.0)

        cu_sig = co_acc['u']
        cv_sig = co_acc['v']
        sig_uv_c = np.sqrt((cu_sig.q / cu_sig.n + cv_sig.q / cv_sig.n) / 2.0)
        _, sig_w_c = co_acc['w'].result()
        th_mu, th_sig = co_theta.result()
        ph_mu, ph_sig = co_ph.result()
        two_d = {k: acc.result() for k, acc in co_2d.items()}

        out['fine'][scheme] = {
            'u': {'mode': 'zero_mean_rms_shared_with_v', 'mu': 0.0,
                  'sigma': sig_uv_f.tolist()},
            'v': {'mode': 'zero_mean_rms_shared_with_u', 'mu': 0.0,
                  'sigma': sig_uv_f.tolist()},
            'w': {'mode': 'zero_mean_rms', 'mu': 0.0, 'sigma': sig_w_f.tolist()},
            'u10': {'mode': 'zero_mean_rms_shared_with_v10', 'mu': 0.0,
                    'sigma': float(sig_uv10)},
            'v10': {'mode': 'zero_mean_rms_shared_with_u10', 'mu': 0.0,
                    'sigma': float(sig_uv10)},
            'n_samples': n_fine,
            'sigma_uv_profile': sig_uv_f.tolist(),
            'sigma_w_profile': sig_w_f.tolist(),
        }
        out['coarse'][scheme] = {
            'u': {'mode': 'zero_mean_rms_shared_with_v', 'mu': 0.0,
                  'sigma': sig_uv_c.tolist()},
            'v': {'mode': 'zero_mean_rms_shared_with_u', 'mu': 0.0,
                  'sigma': sig_uv_c.tolist()},
            'w': {'mode': 'zero_mean_rms', 'mu': 0.0, 'sigma': sig_w_c.tolist()},
            'theta': {'mode': 'standard_per_level', 'mu': th_mu.tolist(),
                      'sigma': th_sig.tolist()},
            'ph': {'mode': 'standard_per_level', 'mu': ph_mu.tolist(),
                   'sigma': ph_sig.tolist()},
            'rmol': {'mode': 'signlog_standard', 'pre': 'z1*rmol',
                     'mu': float(two_d['rmol_x'][0][0]),
                     'sigma': float(two_d['rmol_x'][1][0])},
            'ust': {'mode': 'log_standard', 'mu': float(two_d['ust_l'][0][0]),
                    'sigma': float(two_d['ust_l'][1][0])},
            'pblh': {'mode': 'log_standard', 'mu': float(two_d['pblh_l'][0][0]),
                     'sigma': float(two_d['pblh_l'][1][0])},
            'hfx': {'mode': 'standard', 'mu': float(two_d['hfx'][0][0]),
                    'sigma': float(two_d['hfx'][1][0])},
            't2': {'mode': 'standard', 'mu': float(two_d['t2'][0][0]),
                   'sigma': float(two_d['t2'][1][0])},
            'psfc': {'mode': 'standard', 'mu': float(two_d['psfc'][0][0]),
                     'sigma': float(two_d['psfc'][1][0])},
            'coszen': {'mode': 'none',
                       'mu': float(two_d['coszen'][0][0]),
                       'sigma': float(two_d['coszen'][1][0])},
            'n_samples': n_coarse,
            'sigma_uv_profile': sig_uv_c.tolist(),
            'sigma_w_profile': sig_w_c.tolist(),
        }
        print("  fine σ_uv 前5层: " + ", ".join("{:.3f}".format(x) for x in sig_uv_f[:5]))
        print("  fine σ_w  前5层: " + ", ".join("{:.3f}".format(x) for x in sig_w_f[:5]))
        print("  coarse σ_uv 前5层: " + ", ".join("{:.3f}".format(x) for x in sig_uv_c[:5]))
        print("  coarse UST/PBLH/HFX 均值: {:.4f} / {:.1f} / {:.1f}".format(
            np.exp(two_d['ust_l'][0][0]), np.exp(two_d['pblh_l'][0][0]), two_d['hfx'][0][0]))

    out_path = os.path.join(args.static_dir, "normalize_config.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n写出: " + out_path)

    # σ 廓线图(异常排查用)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        zf = np.asarray(statics['zagl_mass_fine']).mean(axis=(1, 2))
        zc = np.asarray(statics['zagl_mass_coarse']).mean(axis=(1, 2))
        # W 在界面层(41 层),与质量层(40 层)不同高度 -> 单独取界面平均高度
        zwf = np.asarray(statics['zagl_iface_fine']).mean(axis=(1, 2))
        zwc = np.asarray(statics['zagl_iface_coarse']).mean(axis=(1, 2))
        ax = axes[0]
        for scheme in schemes:
            ax.plot(out['fine'][scheme]['sigma_uv_profile'], zf, marker='o', ms=3,
                    label='fine {}'.format(scheme))
            ax.plot(out['coarse'][scheme]['sigma_uv_profile'], zc, marker='s', ms=3,
                    label='coarse {}'.format(scheme))
        ax.set_xlabel('sigma_uv (m/s)')
        ax.set_ylabel('z_agl (m)')
        ax.set_ylim(0, 2000)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title('sigma_uv profile')
        ax = axes[1]
        for scheme in schemes:
            ax.plot(out['fine'][scheme]['sigma_w_profile'], zwf, marker='o', ms=3,
                    label='fine {}'.format(scheme))
            ax.plot(out['coarse'][scheme]['sigma_w_profile'], zwc, marker='s', ms=3,
                    label='coarse {}'.format(scheme))
        ax.set_xlabel('sigma_w (m/s)')
        ax.set_ylim(0, 2000)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title('sigma_w profile')
        png = os.path.join(os.path.dirname(args.static_dir), "results", "outline",
                           "sigma_profiles.png")
        png = os.path.abspath(png)
        ensure_dir(os.path.dirname(png))
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        print("σ 廓线图: " + png)
    except Exception as e:  # noqa: BLE001
        print("绘图跳过: " + str(e))


if __name__ == "__main__":
    main()
