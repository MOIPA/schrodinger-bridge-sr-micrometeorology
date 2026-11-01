#!/usr/bin/env python3
"""外传(三尺度 3×)表格与图的**唯一**生成入口。

为什么要这个脚本:外传-3 的第一版数字是手抄自不同运行的 json,后来发现三档的 split 各不相同导致
整篇作废(见 docs/外传-4 草稿 §2)。统一划分重训之后,报告里的每一个数都必须由一条命令重新生成,
不许再手抄 —— 这是 CLAUDE.md「报告数字可追溯红线」的要求。

用法(纯 numpy,不占 GPU):
  python sidequest_scale3x/scripts/scale3_tables.py \
      --dir sidequest_scale3x/results \
      --tags sq27to9_reg,sq9to3_reg,sq3to1_reg \
      --labels "27→9,9→3,3→1" \
      --gap_json sidequest_scale3x/results/gap_real.json \
      --gap_keys 27to9,9to3,3to1 \
      --log_dir logs \
      --out_md sidequest_scale3x/results/scale3_tables.md \
      --out_json sidequest_scale3x/results/scale3_tables.json \
      --fig docs/图/外传_三尺度增益.png

列定义(全部可追溯):
  主指标 RMSE = sqrt(Σ acc_agl[...,1] / Σ acc_agl[...,0]),层 = AGL ≤ 500 m 的前 9 层(与阶段 3/4 同口径)
  y0 RMSE    = 同上,但用 acc_y0(插值基线,同一批时次)
  增益       = (1 − RMSE/y0)·100
  切变 RMSE  = sqrt(Σ acc_sh[...,1] / Σ acc_sh[...,0]),文件里有这个键才算
  best epoch = 从 logs/<tag>.log 末尾的 "Best epoch: N, best_loss: X" 读
  R² / 缺口  = 从 gap_real.json 的同名列读(真粗端 + 项目重网格权重)
缺哪个输入就打 [警告] 并留空,不编数。
"""
import argparse
import json
import os
import re
import sys

import numpy as np


def pooled(arr, levels):
    a = arr[:, levels, :, :]
    num = np.sum(a[..., 1])
    den = np.sum(a[..., 0])
    if den <= 0:
        return None
    return float(np.sqrt(num / den))


def read_npz(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def best_epoch(log_path):
    if not os.path.exists(log_path):
        return None
    txt = open(log_path, 'rb').read().decode('utf-8', 'replace')
    m = re.findall(r'Best epoch:\s*(\d+),\s*best_loss:\s*([0-9.eE+-]+)', txt)
    if not m:
        return None
    return int(m[-1][0]), float(m[-1][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='sidequest_scale3x/results')
    ap.add_argument('--tags', default='sq27to9_reg,sq9to3_reg,sq3to1_reg')
    ap.add_argument('--labels', default='27→9,9→3,3→1')
    ap.add_argument('--gap_json', default='sidequest_scale3x/results/gap_real.json')
    ap.add_argument('--gap_keys', default='27to9,9to3,3to1')
    ap.add_argument('--log_dir', default='logs')
    ap.add_argument('--out_md', default='sidequest_scale3x/results/scale3_tables.md')
    ap.add_argument('--out_json', default='sidequest_scale3x/results/scale3_tables.json')
    ap.add_argument('--fig', default='')
    args = ap.parse_args()

    tags = [t for t in args.tags.split(',') if t]
    labels = [s for s in args.labels.split(',')]
    gap_keys = [s for s in args.gap_keys.split(',')]
    if len(labels) != len(tags):
        print('[警告] --labels 数量与 --tags 不一致,改用 tag 当标签')
        labels = tags

    gap = {}
    if os.path.exists(args.gap_json):
        d = json.load(open(args.gap_json))
        for row in d.get('rows', []):
            gap[row['stage']] = row
    else:
        print('[警告] 缺 %s,缺口列留空' % args.gap_json)

    rows = []
    for i, tag in enumerate(tags):
        per = os.path.join(args.dir, '%s_perhour.npz' % tag)
        z = read_npz(per)
        row = {'tag': tag, 'label': labels[i]}
        if z is None:
            print('[警告] 缺 %s' % per)
            row.update(main=None, y0=None, gain=None, shear=None, hours=None)
        else:
            main = pooled(z['acc_agl'], slice(0, 9))
            y0 = pooled(z['acc_y0'], slice(0, 9))
            row['main'] = main
            row['y0'] = y0
            row['gain'] = None if (main is None or y0 in (None, 0)) else (1 - main / y0) * 100
            row['hours'] = int(z['acc_agl'].shape[0])
            row['shear'] = pooled(z['acc_sh'], np.arange(z['acc_sh'].shape[1])) \
                if 'acc_sh' in z.files else None
        be = best_epoch(os.path.join(args.log_dir, '%s.log' % tag))
        row['best_epoch'], row['best_loss'] = (be if be else (None, None))
        g = gap.get(gap_keys[i]) if i < len(gap_keys) else None
        row['r2'] = None if not g else g.get('r2')
        row['gap_var_frac'] = None if not g else g.get('gap_var_frac')
        row['resid_rmse'] = None if not g else g.get('resid_rmse')
        rows.append(row)

    def f(x, n=4):
        return '—' if x is None else ('%.*f' % (n, x))

    def pct(x):
        return '—' if x is None else '%+.1f%%' % x

    md = ['# 外传 · 三尺度 3× 对比(自动生成)', '',
          '> 由 `sidequest_scale3x/scripts/scale3_tables.py` 生成,请勿手改。',
          '> 口径:统一划分 test 2020-07-16T00–2020-07-31T23,主指标 = AGL ≤ 500 m 层平均风矢量 RMSE(池化)。',
          '', '| 尺度段 | 小时 | 主指标 RMSE | 插值 y0 | 相对增益 | 切变 | 真粗端 R² | 缺口方差占比 | best epoch |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append('| %s | %s | %s | %s | **%s** | %s | %s | %s | %s |' % (
            r['label'], '—' if r['hours'] is None else r['hours'],
            f(r['main']), f(r['y0']), pct(r['gain']), f(r['shear']),
            f(r['r2'], 3),
            '—' if r['gap_var_frac'] is None else '%.1f%%' % (r['gap_var_frac'] * 100),
            '—' if r['best_epoch'] is None else '%d (%.5f)' % (r['best_epoch'], r['best_loss'])))
    md.append('')
    md.append('注:绝对 RMSE **不可跨尺度直接比较**(细端真值不同、域方差差 2 倍以上);'
              '跨尺度只比「相对增益」这一列。')
    os.makedirs(os.path.dirname(args.out_md) or '.', exist_ok=True)
    open(args.out_md, 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('[md] %s' % args.out_md)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        json.dump({'rows': rows, 'tags': tags, 'dir': args.dir},
                  open(args.out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('[json] %s' % args.out_json)

    print('\n'.join(md[1:]))

    if args.fig and all(r['gap_var_frac'] is not None and r['gain'] is not None for r in rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        xs = [r['gap_var_frac'] * 100 for r in rows]
        ys = [r['gain'] for r in rows]
        fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=160)
        ax.plot(xs, ys, 'o-', color='#1f6feb', lw=2, ms=7)
        for r, x, y in zip(rows, xs, ys):
            ax.annotate(r['label'], (x, y), textcoords='offset points', xytext=(6, 6), fontsize=10)
        ax.set_xlabel('missing variance fraction of the 3x gap (%)', fontsize=10)
        ax.set_ylabel('RMSE gain over interpolation (%)', fontsize=10)
        ax.set_title('Same 3x ratio, different base scale', fontsize=11)
        ax.grid(alpha=.3)
        os.makedirs(os.path.dirname(args.fig) or '.', exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.fig)
        print('[fig] %s' % args.fig)
    return 0


if __name__ == '__main__':
    sys.exit(main())
