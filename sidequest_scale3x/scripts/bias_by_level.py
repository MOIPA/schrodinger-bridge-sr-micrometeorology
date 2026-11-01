#!/usr/bin/env python3
"""按层拆开"粗端 − 细端"的风速偏差,判断它是不是高度错位造成的。

背景:`外传-1b` 里 9→3 这一档出现 **+0.316 m/s** 的整体偏强(其余三档都在 ±0.12 内)。
两种可能:
  (a) 水平/域效应 —— 9 km 域覆盖更大地形,重网格到 3 km 子域后整体偏强 → 偏差大致**与层无关**;
  (b) 垂直错位   —— 比较的是两套不同高度的层 → 偏差**随层单调变化**(风速随高度单调增)。

判定:看 per-level bias 的剖面形状;同时打印两边的层平均风速与相关系数。

用法(服务器 wind3d 环境,需 torch:CanvasStatics 依赖):
  python sidequest_scale3x/scripts/bias_by_level.py --stage 9to3 --n_times 4 --nlev 20
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import gap_real as G  # noqa: E402
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='9to3', choices=sorted(G.STAGES))
    ap.add_argument('--n_times', type=int, default=4)
    ap.add_argument('--nlev', type=int, default=20)
    ap.add_argument('--scheme', default='myj')
    ap.add_argument('--out_json', default='')
    args = ap.parse_args()

    cfg = G.STAGES[args.stage]
    st = CanvasStatics(os.path.join(REPO, cfg['statics']))
    fine = {G.stamp_of(p): p for p in glob.glob(os.path.join(REPO, cfg['fine'], 'f_*.npz'))}
    coarse = {G.stamp_of(p): p for p in glob.glob(os.path.join(REPO, cfg['coarse'], 'c_*.npz'))}
    stamps = sorted(set(fine) & set(coarse))
    step = max(1, len(stamps) // args.n_times)
    stamps = stamps[::step][:args.n_times]
    print('[{}] 用 {} 个时次,{} 层'.format(args.stage, len(stamps), args.nlev))

    rows = []
    for lv in range(args.nlev):
        sf, sc, dif, corrs = [], [], [], []
        for s in stamps:
            with np.load(coarse[s]) as g:
                cu = np.asarray(g['c_u'][:args.nlev], dtype=np.float32)
                cv = np.asarray(g['c_v'][:args.nlev], dtype=np.float32)
            with np.load(fine[s]) as g:
                fu = np.asarray(g['f_u'][:args.nlev], dtype=np.float32)
                fv = np.asarray(g['f_v'][:args.nlev], dtype=np.float32)
            spc = G.destagger_speed(st.regrid_field(cu, 'u'), st.regrid_field(cv, 'v'))
            spf = G.destagger_speed(fu, fv)
            f = spf[lv].ravel()
            c = spc[lv].ravel()
            sf.append(float(np.mean(f)))
            sc.append(float(np.mean(c)))
            dif.append(float(np.mean(f - c)))
            if np.std(f) > 0 and np.std(c) > 0:
                corrs.append(float(np.corrcoef(f, c)[0, 1]))
        rows.append({'level': lv,
                     'fine_mean': float(np.mean(sf)),
                     'coarse_mean': float(np.mean(sc)),
                     'bias_f_minus_c': float(np.mean(dif)),
                     'corr': float(np.mean(corrs)) if corrs else None})

    print('\n| L | fine | coarse | bias(f−c) | corr |')
    print('|---|---|---|---|---|')
    for r in rows:
        print('| %d | %.3f | %.3f | %+.3f | %s |' % (
            r['level'], r['fine_mean'], r['coarse_mean'], r['bias_f_minus_c'],
            '—' if r['corr'] is None else '%.3f' % r['corr']))

    b = np.array([r['bias_f_minus_c'] for r in rows])
    lv = np.arange(len(b))
    slope = float(np.polyfit(lv, b, 1)[0]) if len(b) > 2 else 0.0
    print('\n平均 bias %+.3f m/s;随层斜率 %+.4f m/s/层;首末层差 %+.3f' % (b.mean(), slope, b[-1] - b[0]))
    verdict = ('更像垂直错位(偏差随层单调变化)' if abs(slope) * len(b) > 0.15 else
               '更像水平/域效应(偏差与层近似无关)')
    print('[判读] ' + verdict)

    if args.out_json:
        json.dump({'stage': args.stage, 'rows': rows, 'mean_bias': float(b.mean()),
                   'slope_per_level': slope, 'verdict': verdict},
                  open(args.out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('[json] ' + args.out_json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
