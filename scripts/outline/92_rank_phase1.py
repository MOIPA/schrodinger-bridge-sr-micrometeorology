# -*- coding: utf-8 -*-
"""阶段 1 排序:汇总各 run 的逐小时累加量 -> 主指标 + 配对移动块 bootstrap 显著性。

主指标 = AGL 10–500 m 层平均风矢量 RMSE(与阶段 3 T3.5 的主指标一致)。
每个 run 与 base 做配对 bootstrap(同一重采样小时集,块长默认 24 h;另报 12 h 稳健性)。

用法(需要 numpy,仓库根目录):
  python scripts/outline/92_rank_phase1.py --results_dir results/phase1 \
      --base_tag base [--block 24] [--n_boot 2000] [--tags a,b,c]
"""
import argparse
import glob
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agl_eval_common import (MAIN_LEVELS, STRATA, TARGET_AGL, main_levels_idx,  # noqa: E402
                             paired_delta_ci, pooled_rmse)


def load_runs(results_dir, only_tags=None):
    runs = {}
    for npz_path in sorted(glob.glob(os.path.join(results_dir, "*_perhour.npz"))):
        tag = os.path.basename(npz_path).replace("_perhour.npz", "")
        if only_tags and tag not in only_tags:
            continue
        with np.load(npz_path) as d:
            run = {k: d[k] for k in d.keys()}
        json_path = npz_path.replace("_perhour.npz", "_summary.json")
        run['summary'] = json.load(open(json_path)) if os.path.exists(json_path) else {}
        runs[tag] = run
    return runs


def align_hours(runs):
    """各 run 小时集必须一致(同一 split 同一排序);返回公共小时索引与参考 hours。"""
    ref = None
    for tag, r in runs.items():
        h = [str(x) for x in r['hours']]
        if ref is None:
            ref, ref_tag = h, tag
        elif h != ref:
            raise SystemExit("{} 与 {} 的小时集不一致,不能配对 bootstrap".format(tag, ref_tag))
    return ref


def main():
    ap = argparse.ArgumentParser(description="阶段 1 消融排序 + 配对 bootstrap")
    ap.add_argument("--results_dir", default="results/phase1")
    ap.add_argument("--base_tag", default="base")
    ap.add_argument("--block", type=int, default=24, help="移动块长度(小时);另报 12 h")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--tags", default=None, help="只评估这些 tag(逗号分隔)")
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    only = set(args.tags.split(',')) if args.tags else None
    runs = load_runs(args.results_dir, only)
    assert runs, "{} 下没有 *_perhour.npz".format(args.results_dir)
    assert args.base_tag in runs, "缺 base run( {} )".format(args.base_tag)
    hours = align_hours(runs)
    main_idx = main_levels_idx()
    print("run 数 {} / 小时数 {}".format(len(runs), len(hours)))

    base = runs[args.base_tag]['acc_agl']
    pool_base, _ = pooled_rmse(base, 'all', main_idx)
    rows = []
    for tag, r in sorted(runs.items()):
        acc = r['acc_agl']
        m, n = pooled_rmse(acc, 'all', main_idx)
        if tag == args.base_tag:
            rows.append({'tag': tag, 'main_rmse_vec': m, 'delta': 0.0, 'ci_lo': 0.0,
                         'ci_hi': 0.0, 'significant': False, 'n': n})
            continue
        d, lo, hi = paired_delta_ci(acc, base, 'all', main_idx, block=args.block,
                                    n_boot=args.n_boot, seed=0)
        d12, lo12, hi12 = paired_delta_ci(acc, base, 'all', main_idx, block=12,
                                          n_boot=args.n_boot, seed=0)
        # 点估计的 Δ 直接用两个汇总 RMSE 之差(不受累加量维数差异影响)
        d = m - pool_base
        rows.append({'tag': tag, 'main_rmse_vec': m, 'delta': d, 'ci_lo': lo, 'ci_hi': hi,
                     'significant': bool(lo > 0 or hi < 0),
                     'delta_block12': d12, 'ci_lo_block12': lo12, 'ci_hi_block12': hi12,
                     'n': n})
    # 免费基线:y0(粗端重网格),与 base 同口径
    if 'acc_y0' in runs[args.base_tag]:
        y0 = runs[args.base_tag]['acc_y0']
        m, n = pooled_rmse(y0, 'all', main_idx)
        d, lo, hi = paired_delta_ci(y0, base, 'all', main_idx, block=args.block,
                                    n_boot=args.n_boot, seed=0)
        rows.append({'tag': 'baseline_y0_regrid', 'main_rmse_vec': m, 'delta': d,
                     'ci_lo': lo, 'ci_hi': hi, 'significant': bool(lo > 0 or hi < 0),
                     'n': n})

    rows.sort(key=lambda r: r['main_rmse_vec'])
    # 逐层(主分层 all):RMSE 与该 run 相对 base 的 Δ(不做 bootstrap,供报告看廓线)
    per_level = []
    for tag, r in sorted(runs.items()):
        acc = r['acc_agl']
        for li, h in enumerate(TARGET_AGL):
            m, _ = pooled_rmse(acc, 'all', [li])
            mb, _ = pooled_rmse(base, 'all', [li])
            per_level.append({'tag': tag, 'agl_m': float(h), 'rmse': m, 'delta': m - mb})
    strata_rows = []
    for tag, r in sorted(runs.items()):
        d = {'tag': tag}
        for s in STRATA:
            d[s] = pooled_rmse(r['acc_agl'], s, main_idx)[0]
        strata_rows.append(d)

    prefix = args.out_prefix or os.path.join(args.results_dir, "ranking")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(prefix + ".csv", 'w') as f:
        cols = ['tag', 'main_rmse_vec', 'delta', 'ci_lo', 'ci_hi', 'significant', 'n']
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join([r['tag']] + ['{:.6f}'.format(r[c]) for c in cols[1:-1]]
                             + [str(r['n'])]) + '\n')
    with open(prefix + "_perlevel.csv", 'w') as f:
        f.write('tag,agl_m,rmse,delta_vs_base\n')
        for r in per_level:
            f.write('{},{:.0f},{:.6f},{:.6f}\n'.format(r['tag'], r['agl_m'], r['rmse'],
                                                      r['delta']))
    with open(prefix + "_strata.csv", 'w') as f:
        f.write(','.join(['tag'] + STRATA) + '\n')
        for r in strata_rows:
            f.write(','.join([r['tag']] + ['{:.6f}'.format(r[s]) for s in STRATA]) + '\n')

    lines = []
    lines.append("| run | 主指标 10–500 m 矢量 RMSE (m/s) | Δ vs base | 95% CI | 显著 |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        if r['tag'] == args.base_tag:
            lines.append("| {} (基准) | {:.4f} | — | — | — |".format(
                r['tag'], r['main_rmse_vec']))
            continue
        ci = "—" if r['ci_lo'] != r['ci_lo'] else "[{:+.4f}, {:+.4f}]".format(  # nan!=nan
            r['ci_lo'], r['ci_hi'])
        sig = "—" if r['ci_lo'] != r['ci_lo'] else ("是" if r['significant'] else "否")
        lines.append("| {} | {:.4f} | {:+.4f} | {} | {} |".format(
            r['tag'], r['main_rmse_vec'], r['delta'], ci, sig))
    md = "\n".join(lines) + "\n"
    with open(prefix + ".md", 'w') as f:
        f.write(md)
    print(md)
    print("写出 {}.csv / _perlevel.csv / _strata.csv / .md".format(prefix))


if __name__ == "__main__":
    main()
