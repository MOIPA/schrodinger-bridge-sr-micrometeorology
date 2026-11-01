#!/usr/bin/env python3
"""零样本跨尺度迁移:保住原任务增益的百分之多少?

定义(全部可追溯):
  gain(tag)      = (1 − RMSE_tag / RMSE_y0) × 100     y0 取自该 tag 自己 perhour.npz 的 acc_y0
  retained       = gain(xfer) / gain(indomain) × 100   "迁移保住了原任务的百分之多少"
  RMSE 口径      = AGL ≤ 500 m 前 9 层,Σacc_agl[...,1]/Σacc_agl[...,0] 开方

注意:retained 不是"准确率保持率"。若原任务增益 35%、迁移后 18%,retained ≈ 51%,
但迁移模型的绝对误差仍比原任务高很多 —— 报告里两个数都要写。

用法:
  python sidequest_scale3x/scripts/xfer_retention.py \
      --dir sidequest_scale3x/results \
      --pairs xfer9to3_by27to9_reg:sq9to3_reg,xfer27to9_by3to1_reg:sq27to9_reg \
      --out_md sidequest_scale3x/results/xfer_retention.md
"""
import argparse
import json
import os
import sys

import numpy as np


def pooled(z, levels):
    a = z['acc_agl'][:, levels, :, :]
    den = float(np.sum(a[..., 0]))
    return float(np.sqrt(np.sum(a[..., 1]) / den)) if den > 0 else None


def stat(path):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    lv = slice(0, 9)
    m = pooled(z, lv)
    y0 = pooled({'acc_agl': z['acc_y0']}, lv)
    return {'main': m, 'y0': y0,
            'gain': (1 - m / y0) * 100 if m and y0 else None,
            'hours': int(z['acc_agl'].shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='sidequest_scale3x/results')
    ap.add_argument('--pairs', default='xfer9to3_by27to9_reg:sq9to3_reg,xfer27to9_by3to1_reg:sq27to9_reg')
    ap.add_argument('--out_md', default='sidequest_scale3x/results/xfer_retention.md')
    ap.add_argument('--out_json', default='sidequest_scale3x/results/xfer_retention.json')
    args = ap.parse_args()

    rows = []
    for pair in [p for p in args.pairs.split(',') if p]:
        xfer, indo = pair.split(':')
        sx = stat(os.path.join(args.dir, '%s_perhour.npz' % xfer))
        si = stat(os.path.join(args.dir, '%s_perhour.npz' % indo))
        if sx is None:
            print('[警告] 缺 %s_perhour.npz' % xfer)
        if si is None:
            print('[警告] 缺 %s_perhour.npz' % indo)
        row = {'xfer': xfer, 'indomain': indo,
               'xfer_main': None if sx is None else sx['main'],
               'indomain_main': None if si is None else si['main'],
               'y0': None if sx is None else sx['y0'],
               'xfer_gain': None if sx is None else sx['gain'],
               'indomain_gain': None if si is None else si['gain']}
        row['retained'] = (None if (row['xfer_gain'] is None or not row['indomain_gain'])
                           else row['xfer_gain'] / row['indomain_gain'] * 100)
        rows.append(row)

    def f(x, n=4):
        return '—' if x is None else '%.*f' % (n, x)

    md = ['# 零样本跨尺度迁移:增益保持率', '',
          '> 由 `sidequest_scale3x/scripts/xfer_retention.py` 生成,请勿手改。',
          '> retained = 迁移后增益 / 同任务自训增益 × 100;**它不是准确率保持率**。', '',
          '| 迁移配置(权重来源) | 迁移 RMSE | 自训 RMSE | 插值 y0 | 迁移增益 | 自训增益 | retained |',
          '|---|---|---|---|---|---|---|']
    for r in rows:
        md.append('| %s | %s | %s | %s | %s | %s | **%s** |' % (
            r['xfer'], f(r['xfer_main']), f(r['indomain_main']), f(r['y0']),
            '—' if r['xfer_gain'] is None else '%+.1f%%' % r['xfer_gain'],
            '—' if r['indomain_gain'] is None else '%+.1f%%' % r['indomain_gain'],
            '—' if r['retained'] is None else '%.0f%%' % r['retained']))
    md += ['', '读法:retained 越接近 100% 说明该尺度上的映射越"通用";越接近 0 说明算子几乎完全尺度专用。',
           '两个方向都要看 —— 从粗尺度权重迁到细尺度、与反方向,若差很多,说明存在"只会放大、不会收敛"的不对称。']
    os.makedirs(os.path.dirname(args.out_md) or '.', exist_ok=True)
    open(args.out_md, 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('[md] %s' % args.out_md)
    if args.out_json:
        json.dump({'rows': rows}, open(args.out_json, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print('[json] %s' % args.out_json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
