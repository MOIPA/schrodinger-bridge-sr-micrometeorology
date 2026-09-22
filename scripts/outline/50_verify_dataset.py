# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · 数据完整性校验

1) 文件数与时间覆盖(细/粗端,两方案);整点对齐
2) 抽样检查形状/dtype/有限性
3) 与原始 wrfout 的逐点数值抽查(最强校验)
4) cos(SZA) 天文性检查(与 myj 的 COSZEN 对比:量级与空间平滑性)
5) 防泄漏断言:细端 npz 只含目标量(无任何用于输入的 d04 逐时刻量)

运行(pytorch-gpu 环境,需 20/21 全量抽取完成):
  python scripts/outline/50_verify_dataset.py
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline_common import (OUT_COARSE, OUT_FINE, N_IFACE, N_MASS, SCHEMES,
                            SCHEME_DIRS, WRF_BASE, domain_files)

FINE_KEYS = {'f_u', 'f_v', 'f_w', 'f_u10', 'f_v10'}
COARSE_KEYS = {'c_u', 'c_v', 'c_w', 'c_theta', 'c_ph', 'c_rmol', 'c_ust',
               'c_pblh', 'c_hfx', 'c_t2', 'c_psfc', 'c_coszen'}
COARSE_OPTIONAL = {'c_coszen_wrf'}


def parse_stamp(name):
    m = re.search(r'_(\d{8}T\d{6})\.npz$', name)
    return m.group(1) if m else None


def wrfout_path_for(scheme, dom, stamp):
    d = SCHEME_DIRS[scheme]
    day = '{}-{}-{}'.format(stamp[0:4], stamp[4:6], stamp[6:8])
    hh, mm = stamp[9:11], stamp[11:13]
    path = os.path.join(WRF_BASE, d, "wrfout_{}_{}_{}:00:00".format(dom, day, hh))
    return path, int(mm) // 10


def main():
    parser = argparse.ArgumentParser(description="阶段0 数据校验")
    parser.add_argument("--fine_dir", default=OUT_FINE)
    parser.add_argument("--coarse_dir", default=OUT_COARSE)
    parser.add_argument("--n_spot", type=int, default=6, help="wrfout 抽查点数")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()
    rep = {}

    for scheme in SCHEMES:
        f_files = sorted(glob.glob(os.path.join(args.fine_dir, "f_{}_*.npz".format(scheme))))
        c_files = sorted(glob.glob(os.path.join(args.coarse_dir, "c_{}_*.npz".format(scheme))))
        f_stamps = [parse_stamp(os.path.basename(p)) for p in f_files]
        c_stamps = [parse_stamp(os.path.basename(p)) for p in c_files]
        f_hours = sorted(set(s[:11] + '0000' for s in f_stamps))
        c_hours = sorted(set(c_stamps))
        entry = {
            'fine_files': len(f_files), 'coarse_files': len(c_files),
            'fine_hours': len(f_hours), 'coarse_hours': len(c_hours),
            'hour_sets_equal': f_hours == c_hours,
            'fine_first_last': [f_stamps[0], f_stamps[-1]],
            'coarse_first_last': [c_stamps[0], c_stamps[-1]],
        }
        # 分钟步长
        mins = sorted(set(s[11:13] for s in f_stamps))
        entry['fine_minutes'] = mins
        rep['counts_' + scheme] = entry

        # --- 抽样形状与有限性 ---
        shapes = Counter()
        bad = 0
        for p in f_files[::max(1, len(f_files) // 20)][:20]:
            with np.load(p) as d:
                keys = set(d.keys())
                assert keys == FINE_KEYS, "细端 keys 异常: {}".format(keys)
                shapes[str({k: d[k].shape for k in ('f_u', 'f_v', 'f_w')})] += 1
                if not all(np.isfinite(d[k]).all() for k in d.keys()):
                    bad += 1
        for p in c_files[::max(1, len(c_files) // 10)][:10]:
            with np.load(p) as d:
                keys = set(d.keys())
                assert keys - COARSE_OPTIONAL == COARSE_KEYS, "粗端 keys 异常: {}".format(keys)
                if not all(np.isfinite(d[k]).all() for k in d.keys()):
                    bad += 1
        rep['shapes_' + scheme] = dict(shapes)
        rep['nonfinite_files_' + scheme] = bad

        # --- 与原始 wrfout 抽查 ---
        spot = []
        rng = np.random.RandomState(0)
        for p in [f_files[i] for i in rng.choice(len(f_files),
                                                 size=min(args.n_spot, len(f_files)),
                                                 replace=False)]:
            stamp = parse_stamp(os.path.basename(p))
            wpath, tidx = wrfout_path_for(scheme, "d04", stamp)
            if not os.path.isfile(wpath):
                continue
            lv = int(rng.randint(0, N_MASS))
            with Dataset(wpath) as nc:
                u_ref = float(nc.variables['U'][tidx, lv, 10, 20])
                w_ref = float(nc.variables['W'][tidx, lv, 10, 20])
                u10_ref = float(nc.variables['U10'][tidx, 10, 20])
            with np.load(p) as d:
                du = abs(float(d['f_u'][lv, 10, 20]) - u_ref)
                dw = abs(float(d['f_w'][lv, 10, 20]) - w_ref)
                du10 = abs(float(d['f_u10'][10, 20]) - u10_ref)
            spot.append({'stamp': stamp, 'level': lv, 'du': du, 'dw': dw, 'du10': du10})
        rep['wrfout_spot_' + scheme] = {
            'n': len(spot),
            'max_abs_diff': max([max(s['du'], s['dw'], s['du10']) for s in spot] or [0.0]),
        }

        # --- cos(SZA) 天文性(myj 有 COSZEN) ---
        if scheme == 'myj':
            diffs, autocorr = [], []
            for p in c_files[::max(1, len(c_files) // 12)][:12]:
                with np.load(p) as d:
                    if 'c_coszen_wrf' not in d:
                        continue
                    a = d['c_coszen']
                    b = np.maximum(d['c_coszen_wrf'], 0.0)  # WRF COSZEN 夜间为负,按 0 截断
                    dif = a - b
                    diffs.append(float(np.abs(dif).max()))
                    if float(dif.std()) > 1e-4:
                        dc = dif - dif.mean()
                        autocorr.append(float(np.mean(dc[1:, :] * dc[:-1, :])
                                              / (dif.var() + 1e-12)))
            rep['coszen_check'] = {
                'max_abs_diff_vs_wrf': max(diffs) if diffs else None,
                'mean_lag1_autocorr_of_diff': float(np.mean(autocorr)) if autocorr else None,
                'note': '差值若为空间平滑场(lag1 自相关接近 1),说明 WRF COSZEN 是纯天文量;'
                        '我们两个方案统一采用自算天文值',
            }
    # --- 防泄漏断言(细端 keys 已在上面强断言) ---
    rep['leakage_assert'] = {
        'fine_keys_only_targets': sorted(FINE_KEYS),
        'coarse_keys': sorted(COARSE_KEYS),
        'note': '细端 npz 仅含 U/V/W/U10/V10(监督目标);输入侧的 d04 静态场全部来自 '
                'statics.npz(PHB 导出几何 + 土地数据),不含 d04 逐时刻量',
    }
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print("写出: " + args.json_out)


if __name__ == "__main__":
    main()
