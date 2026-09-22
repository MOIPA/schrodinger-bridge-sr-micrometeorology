# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · T0.6 数据划分

按天气过程把 2020-07 切成 N_BLOCKS(默认 10)个连续块:对逐日边界层/近地面指标
(近地风速、UST、PBLH、HFX、不稳定占比)做 1D 最优分段(最小化段内平方和);
再在块级分配 训练/验证/测试 ≈ 70/10/20,约束:留出块互不相邻。
输出 split.json(块表 + 各 split 的逐小时时间戳)+ 可读块表(供确认后冻结)。

运行(pytorch-gpu 环境,需 21 号粗端抽取已完成):
  python scripts/outline/30_blocks_and_split.py
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline_common import OUT_COARSE, OUT_FINE, ensure_dir

N_BLOCKS = 10
RATIOS = (0.70, 0.10, 0.20)  # train/valid/test
IND_NAMES = ['ws_low', 'ust', 'pblh', 'hfx', 'unstable_frac']


def parse_stamp(name):
    m = re.search(r'_(\d{8}T\d{6})\.npz$', name)
    return datetime.strptime(m.group(1), '%Y%m%dT%H%M%S') if m else None


def daily_indicators(coarse_dir, scheme, limit=None):
    """逐日指标(粗端,域平均)。"""
    files = sorted(glob.glob(os.path.join(coarse_dir, "c_{}_*.npz".format(scheme))))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit("no coarse files in " + coarse_dir)
    per_day = {}
    for f in files:
        dt = parse_stamp(os.path.basename(f))
        with np.load(f) as d:
            u0 = 0.5 * (d['c_u'][0][:, :-1] + d['c_u'][0][:, 1:])   # -> 质量点
            v0 = 0.5 * (d['c_v'][0][:-1, :] + d['c_v'][0][1:, :])
            spd = np.sqrt(u0 ** 2 + v0 ** 2)
            row = [float(np.mean(spd)), float(np.mean(d['c_ust'])),
                   float(np.mean(d['c_pblh'])), float(np.mean(d['c_hfx'])),
                   float(np.mean(d['c_rmol'] < 0))]
        per_day.setdefault(dt.date(), []).append(row)
    days = sorted(per_day)
    feats = np.array([np.mean(per_day[d], axis=0) for d in days])
    return days, feats


def dp_segment(feats, k):
    """1D 最优分段(最小化段内平方和),返回 k 个 [i, j) 区间。"""
    n, nf = feats.shape
    x = (feats - feats.mean(0)) / (feats.std(0) + 1e-12)
    p = np.concatenate([np.zeros((1, nf)), np.cumsum(x, axis=0)], axis=0)
    q = np.concatenate([np.zeros((1, nf)), np.cumsum(x ** 2, axis=0)], axis=0)

    def cost(i, j):
        m = j - i
        s = p[j] - p[i]
        return float(((q[j] - q[i]) - s * s / m).sum())

    inf = 1e18
    d = np.full((k + 1, n + 1), inf)
    back = np.zeros((k + 1, n + 1), dtype=int)
    d[0, 0] = 0.0
    for kk in range(1, k + 1):
        for j in range(kk, n + 1):
            for i in range(kk - 1, j):
                c = d[kk - 1, i] + cost(i, j)
                if c < d[kk, j]:
                    d[kk, j] = c
                    back[kk, j] = i
    segs = []
    j = n
    for kk in range(k, 0, -1):
        i = back[kk, j]
        segs.append((i, j))
        j = i
    return list(reversed(segs))


def choose_assignment(lengths):
    """从 k 个块中选 1 个 valid + 2 个 test,最小化比例偏差;留出块互不相邻。"""
    k = len(lengths)
    total = float(sum(lengths))
    best = None
    for v in range(k):
        for t1 in range(k):
            for t2 in range(t1 + 1, k):
                if len({v, t1, t2}) < 3:
                    continue
                held = sorted([v, t1, t2])
                if held[1] - held[0] < 2 or held[2] - held[1] < 2:
                    continue  # 留出块互不相邻(且不共用边界)
                n_v = lengths[v]
                n_t = lengths[t1] + lengths[t2]
                n_tr = total - n_v - n_t
                dev = (abs(n_tr / total - RATIOS[0]) + abs(n_v / total - RATIOS[1])
                       + abs(n_t / total - RATIOS[2]))
                spread = min(held[1] - held[0], held[2] - held[1])
                key = (round(dev, 6), -spread, v, t1, t2)
                if best is None or key < best[0]:
                    best = (key, v, t1, t2)
    return best[1], best[2], best[3]


def main():
    parser = argparse.ArgumentParser(description="阶段0 T0.6 块划分")
    parser.add_argument("--coarse_dir", default=OUT_COARSE)
    parser.add_argument("--fine_dir", default=OUT_FINE)
    parser.add_argument("--scheme", default="myj", help="用哪个方案的粗端指标分块")
    parser.add_argument("--n_blocks", type=int, default=N_BLOCKS)
    parser.add_argument("--json_out", default=None)
    parser.add_argument("--no_freeze", action="store_true", help="只打印块表不写 json")
    args = parser.parse_args()

    out_json = args.json_out or os.path.join(
        os.path.dirname(os.path.normpath(args.coarse_dir)),
        "prepare_npz_outline_static", "split.json")

    days, feats = daily_indicators(args.coarse_dir, args.scheme)
    print("天数 {}: {} .. {}".format(len(days), days[0], days[-1]))
    segs = dp_segment(feats, args.n_blocks)
    print("\n=== 天气块(最优分段)===")
    print("块 | 日期范围 | 天 | " + " | ".join(IND_NAMES))
    for bi, (i, j) in enumerate(segs):
        m = feats[i:j].mean(axis=0)
        print("{:2d} | {}~{} | {:2d} | ".format(bi, days[i], days[j - 1], j - i)
              + " | ".join("{:.3g}".format(v) for v in m))

    lengths = [j - i for i, j in segs]
    v, t1, t2 = choose_assignment(lengths)
    labels = ['train'] * len(segs)
    labels[v] = 'valid'
    labels[t1] = 'test'
    labels[t2] = 'test'
    print("\n分配: valid=块{} test=块{},{}".format(v, t1, t2))
    n_tr = sum(lengths[i] for i in range(len(segs)) if labels[i] == 'train')
    n_v = lengths[v]
    n_t = lengths[t1] + lengths[t2]
    tot = sum(lengths)
    print("天数: train={} ({:.1%}) valid={} ({:.1%}) test={} ({:.1%})".format(
        n_tr, n_tr / tot, n_v, n_v / tot, n_t, n_t / tot))

    blocks = []
    for bi, (i, j) in enumerate(segs):
        blocks.append({
            'idx': bi, 'start': str(days[i]), 'end': str(days[j - 1]),
            'n_days': j - i, 'split': labels[bi],
            'indicators': {n: round(float(x), 4)
                           for n, x in zip(IND_NAMES, feats[i:j].mean(axis=0))},
        })

    # 逐小时时间戳(共享,来自粗端文件)
    cfiles = sorted(glob.glob(os.path.join(args.coarse_dir, "c_{}_*.npz".format(args.scheme))))
    stamps = sorted(parse_stamp(os.path.basename(f)) for f in cfiles)
    fmt = '%Y-%m-%dT%H:%M:%S'
    hours = {'train': [], 'valid': [], 'test': []}
    for dt in stamps:
        d = str(dt.date())
        for b in blocks:
            if b['start'] <= d <= b['end']:
                hours[b['split']].append(dt.strftime(fmt))
                break
    fine_files = len(glob.glob(os.path.join(args.fine_dir, "f_*_*.npz")))

    split = {
        'generated': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'method': '逐日指标(粗端 {} 方案)最优分段 -> 块级 70/10/20,留出块不相邻'.format(args.scheme),
        'indicator_scheme': args.scheme,
        'ratios_target': list(RATIOS),
        'ratios_actual': {'train': n_tr / tot, 'valid': n_v / tot, 'test': n_t / tot},
        'blocks': blocks,
        'hours': hours,
        'counts': {'coarse_hours': {k: len(v) for k, v in hours.items()},
                   'fine_files_total': fine_files},
        'note': '时间戳在两端/两方案间共享;split 以时间戳(UTC)为准,冻结后不再改动',
    }
    if not args.no_freeze:
        ensure_dir(os.path.dirname(out_json))
        with open(out_json, 'w') as f:
            json.dump(split, f, indent=2, ensure_ascii=False)
        print("\n写出: " + out_json)
    print("\n小时数: " + json.dumps(split['counts']['coarse_hours']))


if __name__ == "__main__":
    main()
