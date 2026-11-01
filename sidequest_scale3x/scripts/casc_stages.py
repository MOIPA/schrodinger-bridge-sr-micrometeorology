#!/usr/bin/env python3
"""级联超分的逐级诊断:每一级"拿到的粗端" vs "该级域内自训"的差。

为什么单独需要它:级联 27→9→3→1 的终点指标只说明"最后好不好",不能说明**误差在哪一级被放大**。
本脚本把三级摆成一张表,并显式给出两条边界(先定后看,写在外传-4 §3.1):

  上界(物理最优)  = 第 3 级看到**真** 3 km 场时的误差 = sq3to1 域内自训
  参照(直连)      = 一步 27→1

判定:
  X < 上界 − 容差            → 有泄漏嫌疑,先查数据再庆祝(不允许直接下结论)
  上界 ≤ X ≤ 直连             → 分步不亏;X 越靠近上界,级联越值得
  X > 直连                    → 分步比一步还差,给出代价
用法(纯 numpy):
  python sidequest_scale3x/scripts/casc_stages.py \
      --dir sidequest_scale3x/results,results/phase4 \
      --stages sq27to9_reg_casc,sq9to3_casc_casc,sq3to1_casc_casc \
      --indomain sq27to9_reg,sq9to3_reg,sq3to1_reg \
      --direct p4_27to1_reg --out_md sidequest_scale3x/results/casc_stages.md
"""
import argparse
import json
import os
import sys

import numpy as np


def pooled(arr, levels, col=(0, 1)):
    a = arr[:, levels, :, :]
    den = float(np.sum(a[..., col[0]]))
    if den <= 0:
        return None
    return float(np.sqrt(np.sum(a[..., col[1]]) / den))


def load(path):
    return np.load(path, allow_pickle=False) if os.path.exists(path) else None


def find(dirs, tag):
    for d in dirs:
        p = os.path.join(d, '%s_perhour.npz' % tag)
        if os.path.exists(p):
            return p
    return None


def stats(z):
    if z is None:
        return {}
    lv = slice(0, 9)
    m = pooled(z['acc_agl'], lv)
    y0 = pooled(z['acc_y0'], lv)
    bias = float(np.sum(z['acc_agl'][:, lv, :, 3]) / max(np.sum(z['acc_agl'][:, lv, :, 0]), 1e-9))
    sh = pooled(z['acc_sh'], np.arange(z['acc_sh'].shape[1])) if 'acc_sh' in z.files else None
    return {'main': m, 'y0': y0, 'gain': (1 - m / y0) * 100 if m and y0 else None,
            'bias': bias, 'shear': sh, 'hours': int(z['acc_agl'].shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='sidequest_scale3x/results,results/phase4')
    ap.add_argument('--stages', default='sq27to9_reg_casc,sq9to3_casc_casc,sq3to1_casc_casc')
    ap.add_argument('--indomain', default='sq27to9_reg,sq9to3_reg,sq3to1_reg')
    ap.add_argument('--direct', default='p4_27to1_reg')
    ap.add_argument('--tol', type=float, default=0.02, help='容差(相对于上界),低于上界-容差判为泄漏嫌疑')
    ap.add_argument('--out_md', default='sidequest_scale3x/results/casc_stages.md')
    ap.add_argument('--out_json', default='sidequest_scale3x/results/casc_stages.json')
    args = ap.parse_args()

    dirs = [d for d in args.dir.split(',') if d]
    stages = [t for t in args.stages.split(',') if t]
    indo = [t for t in args.indomain.split(',') if t]

    rows = []
    for i, tag in enumerate(stages):
        p = find(dirs, tag)
        s = stats(load(p)) if p else {}
        s['tag'] = tag
        s['stage'] = i + 1
        ip = find(dirs, indo[i]) if i < len(indo) else None
        s['indomain_tag'] = indo[i] if i < len(indo) else ''
        s['indomain'] = stats(load(ip)) if ip else {}
        if not p:
            print('[警告] 缺 %s_perhour.npz' % tag)
        rows.append(s)

    direct_p = find(dirs, args.direct)
    direct = stats(load(direct_p)) if direct_p else {}
    direct['tag'] = args.direct
    if not direct_p:
        print('[警告] 缺 %s_perhour.npz(直连参照)' % args.direct)

    md = ['# 级联逐级诊断', '',
          '> 由 `sidequest_scale3x/scripts/casc_stages.py` 生成,请勿手改。',
          '> 主指标 = AGL ≤ 500 m 层平均风矢量 RMSE(池化);bias = ΣΔspeed/Σn。', '',
          '| 级 | 产物 | 主 RMSE | 插值 y0 | 增益 | 风速偏差 | 该级域内自训 | 域内 RMSE | 与域内差距 |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        def f(x, n=4):
            return '—' if x is None else '%.*f' % (n, x)
        g = '—' if r.get('gain') is None else '%+.1f%%' % r['gain']
        ig = r['indomain'].get('main')
        d = ('—' if (r.get('main') is None or ig is None) else '%+.1f%%' % ((r['main'] / ig - 1) * 100))
        md.append('| %d | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['stage'], r['tag'], f(r.get('main')), f(r.get('y0')), g,
            f(r.get('bias'), 3), r['indomain_tag'], f(ig), d))

    md += ['', '## 两端边界(判定用,先定后看)', '']
    ub = rows[-1]['indomain'].get('main') if rows else None
    fin = rows[-1].get('main') if rows else None
    md.append('| 情形 | 主 RMSE | 说明 |')
    md.append('|---|---|---|')
    md.append('| 上界:第 3 级看**真** 3 km 场 | %s | 即 3→1 域内自训;级联不可能比这更好 |' %
              ('—' if ub is None else '%.4f' % ub))
    md.append('| 直连:一步 27→1 | %s | 级联要打赢的对手 |' %
              ('—' if direct.get('main') is None else '%.4f' % direct['main']))
    md.append('| 实际:级联终点 | %s | %s |' % (
        '—' if fin is None else '%.4f' % fin, verdict(ub, fin, direct.get('main'), args.tol)))
    md.append('')

    os.makedirs(os.path.dirname(args.out_md) or '.', exist_ok=True)
    open(args.out_md, 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('[md] %s' % args.out_md)

    json.dump({'stages': rows, 'direct': direct, 'bounds': {'oracle': ub, 'final': fin}},
              open(args.out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[json] %s' % args.out_json)
    return 0


def verdict(ub, fin, direct, tol):
    if fin is None or ub is None:
        return '待填'
    if fin < ub - tol:
        return '⚠️ 低于物理上界,先查泄漏(数据/软链/时次),不许直接下结论'
    if direct is None:
        return '直连缺失,无法判定'
    if fin <= direct:
        return '分步不亏(且低于直连)'
    return '分步比一步更差,需给代价定价'


if __name__ == '__main__':
    sys.exit(main())
