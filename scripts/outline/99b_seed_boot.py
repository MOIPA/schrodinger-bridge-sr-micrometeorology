# -*- coding: utf-8 -*-
"""99b_seed_boot.py — 种子稳健性的联合块 bootstrap:Δ 之间差多少才算真差?

背景:阶段 3 的 `p3_joint` 有三个种子(主种子 / s2 / s3)。单看各自的 Δ 与 CI 只能回答
"这一种子是否优于基线",回答不了更重要的问题:
  **三个种子之间的差异,是真的有差别,还是同一份抽样噪声的不同表现?**

做法:对三个种子**同时**做同一套移动块 bootstrap(同一重采样小时集),得到每个种子
的 Δ_i,再直接构造 Δ_i − Δ_j 与种子均值 Δ̄ 的抽样分布。这样:
  - Δ_i 的 CI 与 99 号结果一致(配对口径相同);
  - Δ_i − Δ_j 的 CI 跨 0 ⇒ 两种子的差别在抽样噪声之内,不能声称"这个种子更好";
  - Δ̄ 的 CI 是"这套配置平均能拿多少"的诚实区间(种子作为固定效应,不做多层模型)。

用法(只需 numpy,仓库根目录):
  python scripts/outline/99b_seed_boot.py --perhour_dir results/phase3 \
      --base_tag p2_l1r2_lr2e4 --tags p3_joint,p3_joint_s2,p3_joint_s3 \
      --block 24 --n_boot 2000
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agl_eval_common import STRATA, main_levels_idx, pooled_rmse  # noqa: E402


def load_acc(path, key='acc_agl'):
    d = np.load(path, allow_pickle=True)
    return d[key] if key in d.files else None


def iter_blocks(n_hours, block, n_boot, seed=0):
    """移动块 bootstrap:返回 n_boot 个长度 n_hours 的小时索引(可重复)。"""
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n_hours - block + 1, 1)
    n_blocks = int(np.ceil(n_hours / block))
    for _ in range(n_boot):
        pick = rng.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in pick])[:n_hours]
        yield idx


def rmse_on(acc, idx, levels_idx, si):
    """在给定小时索引上池化 RMSE(只用样到的小时,重复样到的按重复计)。"""
    s = acc[idx][..., levels_idx, si, 1].sum()
    n = acc[idx][..., levels_idx, si, 0].sum()
    return float(np.sqrt(s / max(n, 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--perhour_dir', default='results/phase3')
    ap.add_argument('--base_tag', default='p2_l1r2_lr2e4')
    ap.add_argument('--tags', required=True, help='逗号分隔,同配置的多个种子')
    ap.add_argument('--block', type=int, default=24)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--metric', default='main', choices=['main', 'shear'],
                    help='main = AGL 10–500 m 矢量 RMSE;shear = 相邻主层切变误差')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    si = STRATA.index('all')
    key = 'acc_agl' if args.metric == 'main' else 'acc_sh'
    base_lv = main_levels_idx()

    base = load_acc(os.path.join(args.perhour_dir, f'{args.base_tag}_perhour.npz'), key)
    if base is None:
        raise SystemExit(f'基线缺 {key}: {args.base_tag}')
    lv = base_lv if args.metric == 'main' else list(range(int(base.shape[1])))
    accs = {}
    for t in tags:
        a = load_acc(os.path.join(args.perhour_dir, f'{t}_perhour.npz'), key)
        if a is None:
            raise SystemExit(f'{t} 缺 {key}')
        if a.shape[0] != base.shape[0]:
            raise SystemExit(f'{t} 小时数 {a.shape[0]} != 基线 {base.shape[0]},无法配对')
        accs[t] = a

    n_hours = int(base.shape[0])
    point = {t: rmse_on(accs[t], np.arange(n_hours), lv, si) for t in tags}
    base_pt = rmse_on(base, np.arange(n_hours), lv, si)

    deltas = {t: [] for t in tags}
    pairs = {}
    mean_d = []
    for i, idx in enumerate(iter_blocks(n_hours, args.block, args.n_boot)):
        b = rmse_on(base, idx, lv, si)
        d = {}
        for t in tags:
            d[t] = rmse_on(accs[t], idx, lv, si) - b
            deltas[t].append(d[t])
        mean_d.append(float(np.mean([d[t] for t in tags])))
        if i == 0:
            for a in range(len(tags)):
                for c in range(a + 1, len(tags)):
                    pairs[(tags[a], tags[c])] = []
        for a in range(len(tags)):
            for c in range(a + 1, len(tags)):
                pairs[(tags[a], tags[c])].append(d[tags[a]] - d[tags[c]])

    def ci(x):
        x = np.asarray(x, float)
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    print(f'=== 种子稳健性(块长 {args.block} h,{args.n_boot} 次;基线 {args.base_tag} = {base_pt:.4f}) ===')
    print(f'{"run":18s} {"RMSE":>8s} {"Δ":>9s} {"Δ 95% CI":>22s}  显著')
    for t in tags:
        lo, hi = ci(deltas[t])
        sig = '是' if (lo > 0 or hi < 0) else '否'
        print(f'{t:18s} {point[t]:8.4f} {point[t]-base_pt:+9.4f}  [{lo:+.4f}, {hi:+.4f}]   {sig}')

    lo, hi = ci(mean_d)
    print(f'\n种子均值 Δ = {np.mean(mean_d):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  '
          f'(把三个种子当同一配置的三个实现,平均能拿到的改进)')

    print('\n=== 种子间差别是否超过抽样噪声(Δ_i − Δ_j) ===')
    for (a, c), v in pairs.items():
        lo, hi = ci(v)
        verdict = '跨 0:分不出' if (lo <= 0 <= hi) else '不跨 0:可分辨'
        print(f'{a} − {c}: 点估计 {np.mean(v):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {verdict}')

    if args.out_json:
        out = {
            'base_tag': args.base_tag, 'base_rmse': base_pt, 'block': args.block,
            'n_boot': args.n_boot,
            'runs': {t: {'rmse': point[t], 'delta': point[t] - base_pt,
                         'delta_ci': ci(deltas[t])} for t in tags},
            'seed_mean_delta': {'point': float(np.mean(mean_d)), 'ci': ci(mean_d)},
            'pairwise': {f'{a}__vs__{c}': {'point': float(np.mean(v)), 'ci': ci(v)}
                         for (a, c), v in pairs.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'\n写出 {args.out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
