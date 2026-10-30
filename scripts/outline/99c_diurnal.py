#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""99c_diurnal.py — 阶段 3 三臂的**日变化**与**逐日**误差剖面(纯 numpy,不需要 GPU)。

为什么做这个:
  能源侧最关心的不是"平均有多准",而是"一天里哪几个小时不准"。
  主指标(A GL 10–500 m 矢量 RMSE)是 144 小时池化的一个数,会把日变化抹平;
  perhour.npz 里有逐小时的误差累加量,足够复原 24 小时剖面与 6 天逐日剖面。

口径(与 84/99 完全一致,数字可追溯):
  - 逐小时误差:取 acc_agl[h, ml, si_all, 1](se2)与 [...,0](n),ml = 主指标层索引(AGL 10–500 m 共 9 层);
  - 按小时聚合的池化 RMSE = sqrt(Σse2 / Σn)(不是逐小时 RMSE 的算术平均);
  - 参照组 acc_y0 = 粗端输入插值(相当于"不做超分"),用同一口径算。

用法:
  python scripts/outline/99c_diurnal.py --perhour_dir results/phase3_arch \
      --base_tag p2_l1r2_lr2e4_ens16 --tags p3_arch_reg p3_arch_swin \
      --out_json results/phase3_arch/phase3_diurnal.json --out_md results/phase3_arch/phase3_diurnal.md \
      --fig docs/图/阶段3_日变化.png
"""
import argparse
import json
import os
import numpy as np

AGL_TARGETS = [10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
MAIN_MAX = 500          # 主指标:A GL 10–500 m
SI_ALL = 0              # STRATA[0] == 'all'


def main_idx():
    """主指标层索引:AGL <= 500 m 的层(10,30,50,70,100,150,200,300,500)= 9 层。"""
    return [i for i, z in enumerate(AGL_TARGETS) if z <= MAIN_MAX]


def hourly_rmse(npz, ml, which='agl'):
    """(T,) 逐小时矢量 RMSE;which='y0' 时取粗端输入。"""
    acc = npz['acc_agl'] if which == 'agl' else npz['acc_y0']
    se2 = acc[:, ml, SI_ALL, 1].sum(axis=1)
    n = acc[:, ml, SI_ALL, 0].sum(axis=1)
    return np.sqrt(se2 / n), se2, n


def pooled(se2, n):
    return float(np.sqrt(se2.sum() / n.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perhour_dir', required=True)
    ap.add_argument('--base_tag', default='p2_l1r2_lr2e4_ens16')
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--out_md', required=True)
    ap.add_argument('--fig', default='')
    ap.add_argument('--n_boot', type=int, default=2000)
    a = ap.parse_args()

    ml = main_idx()
    arms = [a.base_tag] + [t for t in a.tags if t != a.base_tag]

    # ---------- 读数据 ----------
    d = {}
    for t in arms:
        p = os.path.join(a.perhour_dir, f'{t}_perhour.npz')
        if not os.path.isfile(p):
            raise SystemExit(f'缺 {p}')
        z = np.load(p, allow_pickle=True)
        hours = [str(h) for h in z['hours']]
        rm, se2, n = hourly_rmse(z, ml, 'agl')
        rm0, se20, n0 = hourly_rmse(z, ml, 'y0')
        d[t] = dict(hours=hours, rm=rm, se2=se2, n=n, rm0=rm0, se20=se20, n0=n0)
        print(f'{t:26s} 池化 {pooled(se2, n):.4f}  粗端 y0 {pooled(se20, n0):.4f}  n/h {int(n[0])}')

    # ---------- 日变化(按地方时小时聚合)----------
    hod = np.array([int(h[11:13]) for h in d[arms[0]]['hours']])
    diurnal = {}
    for t in arms:
        se2, n, rm = d[t]['se2'], d[t]['n'], d[t]['rm']
        se20, n0, rm0 = d[t]['se20'], d[t]['n0'], d[t]['rm0']
        diurnal[t] = {
            'rmse': [pooled(se2[hod == h], n[hod == h]) for h in range(24)],
            'rmse_y0': [pooled(se20[hod == h], n0[hod == h]) for h in range(24)],
            'n_days': [int((hod == h).sum()) for h in range(24)],
        }
        diurnal[t]['rmse_lines'] = [float(x) for x in rm]  # 逐小时原始,供复核

    # ---------- 逐日剖面 ----------
    day = np.array([h[:10] for h in d[arms[0]]['hours']])
    days = sorted(set(day))
    perday = {}
    for t in arms:
        perday[t] = []
        for dd in days:
            m = day == dd
            perday[t].append({
                'day': dd,
                'n': int(m.sum()),
                'rmse': pooled(d[t]['se2'][m], d[t]['n'][m]),
                'rmse_y0': pooled(d[t]['se20'][m], d[t]['n0'][m]),
            })

    base = arms[0]
    # ---------- 逐日配对差(相对基准臂)----------
    # 注意:单日只有 24 个小时,这里的 bootstrap 是对"小时"做 iid 重采样,
    # CI 会偏窄、只用于判断符号是否稳定,不能当作严格的显著性结论。
    rng = np.random.default_rng(0)
    perday_delta = {}
    for t in arms[1:]:
        rows = []
        for dd in days:
            m = day == dd
            idx = np.where(m)[0]
            d1 = np.sqrt(d[t]['se2'][idx].sum() / d[t]['n'][idx].sum())
            d0 = float(np.sqrt(d[base]['se2'][idx].sum() / d[base]['n'][idx].sum()))
            boot = np.empty(a.n_boot)
            for b in range(a.n_boot):
                sel = rng.choice(idx, len(idx), replace=True)
                boot[b] = (np.sqrt(d[t]['se2'][sel].sum() / d[t]['n'][sel].sum())
                           - np.sqrt(d[base]['se2'][sel].sum() / d[base]['n'][sel].sum()))
            rows.append({'day': dd, 'n_hours': int(m.sum()), 'delta': d1 - d0,
                         'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         'p_delta_gt0': float((boot > 0).mean())})
        perday_delta[t] = rows
    # 变率代理:逐小时基准臂 RMSE 的标准差(衡量该日"难"是难在系统性还是难在起伏)
    day_volatility = {dd: float(np.std(d[base]['rm'][day == dd])) for dd in days}

    # ---------- 派生统计 ----------
    b = np.array(diurnal[base]['rmse'])
    worst_h = int(np.argmax(b))
    best_h = int(np.argmin(b))
    out = {
        'perhour_dir': a.perhour_dir,
        'base_tag': a.base_tag,
        'arms': arms,
        'main_levels_idx': ml,
        'main_agl': [AGL_TARGETS[i] for i in ml],
        'pooled_rmse': {t: float(pooled(d[t]['se2'], d[t]['n'])) for t in arms},
        'pooled_rmse_y0': {t: float(pooled(d[t]['se20'], d[t]['n0'])) for t in arms},
        'diurnal': diurnal,
        'perday': perday,
        'base_worst_hour': worst_h,
        'base_best_hour': best_h,
        'perday_delta_vs_base': perday_delta,
        'base_day_volatility': day_volatility,
        'bootstrap_note': 'per-day paired bootstrap is iid over hours (n=24), crude: use sign only',
    }

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---------- Markdown ----------
    L = []
    L.append('# 阶段 3 三臂的日变化与逐日剖面\n')
    L.append('口径:AGL %s m 共 %d 层、矢量 RMSE、池化 sqrt(Σse2/Σn);' %
             ('/'.join(str(x) for x in out['main_agl']), len(ml)))
    L.append('小时聚合把 6 天里同一地方时的样本并在一起算(不是逐小时 RMSE 的算术平均)。\n')
    L.append('| 地方时 | ' + ' | '.join(f'`{t}`' for t in arms) + ' | 粗端 y0(基准臂) |')
    L.append('|---' * (len(arms) + 2) + '|')
    for h in range(24):
        row = [f'{h:02d}'] + [f"{diurnal[t]['rmse'][h]:.4f}" for t in arms]
        row.append(f"{diurnal[base]['rmse_y0'][h]:.4f}")
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('## 逐日(06–21 共 %d 天)\n' % len(days))
    L.append('| 日期 | ' + ' | '.join(f'`{t}`' for t in arms) + ' | 粗端 y0 |')
    L.append('|---' * (len(arms) + 2) + '|')
    for i, dd in enumerate(days):
        row = [dd] + [f"{perday[t][i]['rmse']:.4f}" for t in arms]
        row.append(f"{perday[base][i]['rmse_y0']:.4f}")
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('### 逐日配对差(相对基准臂;单日仅 24 小时,iid bootstrap,只取符号)')
    L.append('')
    L.append('| 日期 | 臂 | Δ (m/s) | 95% CI | P(Δ>0) |')
    L.append('|---|---|---|---|---|')
    for t, rows in perday_delta.items():
        for r in rows:
            L.append('| %s | `%s` | %+.4f | [%+.4f, %+.4f] | %.2f |' %
                     (r['day'], t, r['delta'], r['ci95'][0], r['ci95'][1], r['p_delta_gt0']))
    L.append('')
    with open(a.out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    # ---------- 图 ----------
    if a.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        xs = np.arange(24)
        style = {base: dict(color='#888888', ls='--', lw=1.4),
                 'p3_arch_reg': dict(color='#d1495b', lw=2.0),
                 'p3_arch_swin': dict(color='#2a6f97', lw=2.0)}
        for t in arms:
            st = style.get(t, dict(lw=1.6))
            ax.plot(xs, diurnal[t]['rmse'], label=t, **st)
        ax.plot(xs, diurnal[base]['rmse_y0'], color='#bbbbbb', lw=1.2, ls=':',
                label='coarse input (y0)')
        ax.set_xlabel('local hour of day')
        ax.set_ylabel('RMSE (m/s), AGL 10-500 m')
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=.3, lw=.6)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.fig), exist_ok=True)
        fig.savefig(a.fig)
        print('图 ->', a.fig)

    print('json ->', a.out_json)
    print('md   ->', a.out_md)
    print('基准臂最差时次 %02d 时 %.4f / 最好 %02d 时 %.4f' %
          (worst_h, b[worst_h], best_h, b[best_h]))


if __name__ == '__main__':
    main()
