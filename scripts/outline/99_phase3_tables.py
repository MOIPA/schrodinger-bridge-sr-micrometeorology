# -*- coding: utf-8 -*-
"""阶段 3 表:90 号逐小时累加量 -> 主指标(AGL 10–500 m 矢量 RMSE)与切变误差 + 配对 bootstrap。

主指标 = pooled_rmse(acc_agl, 'all', main_levels_idx());
切变 = pooled_rmse(acc_sh, 'all', 全部相邻层对)(口径见 agl_eval_common.accum_hour_shear);
Δ 与 95% CI = 相对 --base_tag 的配对移动块 bootstrap(同一重采样小时集,块长默认 24 h 与 12 h
各一套)。另出逐层廓线(11 层矢量 RMSE 与 rmse_w)供报告;完整数值落在 json。
缺文件/缺键(如阶段 2 的 run 没有 acc_sh)一律标 N/A("—"/null),不抛异常。

用法(只需 numpy,仓库根目录):
  python scripts/outline/99_phase3_tables.py --perhour_dir results/phase3 \
      --tags p3_agl,p3_joint,p3_agllw --base_tag p2_l1r2_lr2e4
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agl_eval_common import (SHEAR_PAIRS, TARGET_AGL, main_levels_idx,  # noqa: E402
                             paired_delta_ci, pooled_rmse)

ND_MAIN = 4    # 主指标小数位
ND_SHEAR = 5   # 切变小数位(量级 ~1e-3–1e-1 m/s per m)


def load_perhour(path):
    """读 <tag>_perhour.npz;文件不存在返回 None。"""
    if not os.path.exists(path):
        return None
    with np.load(path) as d:
        return {k: d[k] for k in d.keys()}


def hours_of(run):
    if run is None or 'hours' not in run:
        return None
    return [str(x) for x in run['hours']]


def paired_ok(run_a, run_b, acc_a, acc_b):
    """配对 bootstrap 前提:两侧都有逐小时轴且小时集一致,否则 Δ 只有点估计。"""
    if acc_a is None or acc_b is None or acc_a.ndim < 4 or acc_b.ndim < 4:
        return False
    if acc_a.shape[0] != acc_b.shape[0]:
        return False
    ha, hb = hours_of(run_a), hours_of(run_b)
    return not (ha is not None and hb is not None and ha != hb)


def _clean(x):
    """numpy 标量 -> 有限 float;None/nan/inf 一律 None(落 json null)。"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt(x, nd=ND_MAIN, signed=False):
    x = _clean(x)
    if x is None:
        return "—"
    fmt = "{{:+.{0}f}}".format(nd) if signed else "{{:.{0}f}}".format(nd)
    return fmt.format(x)


def _ci_cell(c, nd=ND_MAIN):
    if not c or c.get('lo') is None or c.get('hi') is None:
        return "—"
    fmt = "{{:+.{0}f}}".format(nd)
    return "[" + fmt.format(c['lo']) + ", " + fmt.format(c['hi']) + "]"


def _sig_cell(c):
    if not c or c.get('significant') is None:
        return "—"
    return "是" if c['significant'] else "否"


def analyze(run):
    """单 run 的自身指标(不依赖基线):主指标、切变、逐层廓线。"""
    e = {'available': run is not None, 'n_hours': None,
         'main_rmse_vec': None, 'n_cells_main': None,
         'shear_rmse_vec': None, 'n_cells_shear': None, 'shear_n_pairs': None,
         'per_level': []}
    if run is None:
        return e
    acc = run.get('acc_agl')
    if acc is not None and acc.ndim >= 3:
        m, n = pooled_rmse(acc, 'all', main_levels_idx())
        e['main_rmse_vec'], e['n_cells_main'] = _clean(m), _clean(n)
        for li, h in enumerate(TARGET_AGL):
            mv, _ = pooled_rmse(acc, 'all', [li])
            mw, _ = pooled_rmse(acc, 'all', [li], comp='w')
            e['per_level'].append({'agl_m': float(h), 'rmse_vec': _clean(mv),
                                   'rmse_w': _clean(mw)})
    sh = run.get('acc_sh')
    if sh is not None and sh.ndim >= 3:
        n_pairs = int(sh.shape[-3])
        if n_pairs != len(SHEAR_PAIRS):
            print("[警告] acc_sh 层对数 {} != {};按实际层对池化".format(
                n_pairs, len(SHEAR_PAIRS)))
        m, n = pooled_rmse(sh, 'all', list(range(n_pairs)))
        e['shear_rmse_vec'], e['n_cells_shear'] = _clean(m), _clean(n)
        e['shear_n_pairs'] = n_pairs
    if 'hours' in run:
        e['n_hours'] = int(len(run['hours']))
    elif acc is not None and acc.ndim == 4:
        e['n_hours'] = int(acc.shape[0])
    return e


def _pair_block(tag, base_tag, run, base_run, acc, base_acc, levels_idx, blocks, n_boot):
    """各 block 的配对 Δ 与 95% CI;不可配对(小时轴缺失/不一致)时全 None。"""
    if acc is None or base_acc is None:
        return None
    out = {}
    ok = paired_ok(run, base_run, acc, base_acc)
    if not ok:
        print("[警告] {} 与 {} 的小时集/小时轴不可配对;CI 记 —".format(tag, base_tag))
    for blk in blocks:
        if not ok:
            out[str(blk)] = {'delta': None, 'lo': None, 'hi': None, 'significant': None}
            continue
        d0, lo, hi = paired_delta_ci(acc, base_acc, 'all', levels_idx,
                                     block=blk, n_boot=n_boot, seed=0)
        lo, hi = _clean(lo), _clean(hi)
        sig = None if (lo is None or hi is None) else bool(lo > 0 or hi < 0)
        out[str(blk)] = {'delta': _clean(d0), 'lo': lo, 'hi': hi, 'significant': sig}
    return out


def _row(e, is_base, blocks):
    """主表一行:主 RMSE + Δ + 各 block CI/显著 + 切变 RMSE + Δ 切变 + 各 block CI/显著。"""
    cells = [e['tag'] + (" (基准)" if is_base else "")]
    cells.append(_fmt(e['main_rmse_vec'], ND_MAIN))
    cells.append(_fmt(None if is_base else e.get('delta_vs_base'), ND_MAIN, signed=True))
    for blk in blocks:
        c = (e.get('ci') or {}).get(str(blk))
        cells.append(_ci_cell(c, ND_MAIN))
        cells.append(_sig_cell(c))
    cells.append(_fmt(e['shear_rmse_vec'], ND_SHEAR))
    cells.append(_fmt(None if is_base else e.get('shear_delta_vs_base'),
                      ND_SHEAR, signed=True))
    for blk in blocks:
        c = (e.get('shear_ci') or {}).get(str(blk))
        cells.append(_ci_cell(c, ND_SHEAR))
        cells.append(_sig_cell(c))
    return "| " + " | ".join(cells) + " |"


def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)


def main():
    ap = argparse.ArgumentParser(description="阶段 3 表(主指标 + 切变 + 配对 bootstrap)")
    ap.add_argument("--perhour_dir", default="results/phase3")
    ap.add_argument("--tags", default="p3_agl,p3_joint,p3_agllw",
                    help="逗号分隔(基准单独一行,不必重复)")
    ap.add_argument("--base_tag", default="p2_l1r2_lr2e4")
    ap.add_argument("--out_md", default=None, help="缺省 <perhour_dir>/phase3_tables.md")
    ap.add_argument("--out_json", default=None, help="缺省 <perhour_dir>/phase3_tables.json")
    ap.add_argument("--block", default="24,12",
                    help="移动块长(小时),逗号分隔;默认 24 与 12 同时出")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    perhour_dir = args.perhour_dir
    out_md = args.out_md or os.path.join(perhour_dir, "phase3_tables.md")
    out_json = args.out_json or os.path.join(perhour_dir, "phase3_tables.json")
    try:
        blocks = [int(b) for b in str(args.block).split(',') if str(b).strip()]
    except ValueError:
        blocks = []
    if not blocks:
        blocks = [24, 12]
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    load_tags = [args.base_tag] + [t for t in tags if t != args.base_tag]

    runs = {}
    for t in load_tags:
        p = os.path.join(perhour_dir, "{}_perhour.npz".format(t))
        runs[t] = load_perhour(p)
        if runs[t] is None:
            print("[警告] 缺 {};该行全标 —".format(p))

    entries = {}
    for t in load_tags:
        e = analyze(runs[t])
        e['tag'] = t
        e['file'] = os.path.join(perhour_dir, "{}_perhour.npz".format(t))
        entries[t] = e

    base_run = runs[args.base_tag]
    base_e = entries[args.base_tag]
    main_idx = main_levels_idx()
    base_acc = base_run.get('acc_agl') if base_run is not None else None
    base_sh = base_run.get('acc_sh') if base_run is not None else None

    for t in load_tags:
        e = entries[t]
        if t == args.base_tag:
            e['delta_vs_base'] = None
            e['ci'] = None
            e['shear_delta_vs_base'] = None
            e['shear_ci'] = None
            continue
        run = runs[t]
        acc = run.get('acc_agl') if run is not None else None
        sh = run.get('acc_sh') if run is not None else None
        if acc is not None and base_acc is not None \
                and e['main_rmse_vec'] is not None and base_e['main_rmse_vec'] is not None:
            e['delta_vs_base'] = _clean(e['main_rmse_vec'] - base_e['main_rmse_vec'])
        else:
            e['delta_vs_base'] = None
        e['ci'] = _pair_block(t, args.base_tag, run, base_run, acc, base_acc, main_idx,
                              blocks, args.n_boot) if e['delta_vs_base'] is not None else None
        if sh is not None and base_sh is not None \
                and e['shear_rmse_vec'] is not None and base_e['shear_rmse_vec'] is not None:
            e['shear_delta_vs_base'] = _clean(e['shear_rmse_vec'] - base_e['shear_rmse_vec'])
            sh_idx = list(range(int(sh.shape[-3])))
        else:
            e['shear_delta_vs_base'] = None
            sh_idx = None
        e['shear_ci'] = _pair_block(t, args.base_tag, run, base_run, sh, base_sh, sh_idx,
                                    blocks, args.n_boot) \
            if e['shear_delta_vs_base'] is not None else None

    def _key(t):
        m = entries[t].get('main_rmse_vec')
        return (0, m) if m is not None else (1, 0.0)

    order = [args.base_tag] + sorted([t for t in tags if t != args.base_tag], key=_key)

    lines = []
    hdr = ["run", "主 RMSE (m/s)", "Δ RMSE"]
    for blk in blocks:
        hdr += ["CI {} h".format(blk), "显著"]
    hdr += ["切变 RMSE (m/s per m)", "Δ 切变"]
    for blk in blocks:
        hdr += ["CI {} h".format(blk), "显著"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for t in order:
        lines.append(_row(entries[t], t == args.base_tag, blocks))

    lv = ["| run | AGL (m) | 矢量 RMSE (m/s) | Δ vs base | rmse_w (m/s) |",
          "|---|---|---|---|---|"]
    base_pl = base_e['per_level']
    for t in order:
        e = entries[t]
        if not e['per_level']:
            continue
        for i, pl in enumerate(e['per_level']):
            delta = None
            if t != args.base_tag and i < len(base_pl) \
                    and base_pl[i]['rmse_vec'] is not None and pl['rmse_vec'] is not None:
                delta = pl['rmse_vec'] - base_pl[i]['rmse_vec']
            lv.append("| {} | {:.0f} | {} | {} | {} |".format(
                t if i == 0 else "", pl['agl_m'], _fmt(pl['rmse_vec']),
                _fmt(delta, signed=True), _fmt(pl['rmse_w'])))

    nh = ", ".join("{}={}".format(t, entries[t]['n_hours']) for t in order
                   if entries[t]['n_hours'] is not None)
    notes = [
        "",
        "口径:主指标 = AGL 10–500 m 层平均风矢量 RMSE(池化;误差 = 预测 − 真值);"
        "切变 = 相邻主层 10–30 … 300–500 m 共 {} 对的矢量切变 ΔV/Δz 之差(预测 − 真值)的"
        "池化 RMSE,单位 m/s per m,分层掩码取每对上层;Δ 与 95% CI = 相对 {} 的配对移动块 "
        "bootstrap(同一重采样小时集,块长 {} h)。".format(
            len(SHEAR_PAIRS), args.base_tag, ", ".join(str(b) for b in blocks)),
        "缺文件/缺键(如未存 acc_sh 的 run)或小时集不可配对时记 —(json null),不抛异常"
        + (";小时数: " + nh if nh else ""),
    ]
    md = "\n".join(lines + [""] + lv + notes) + "\n"

    _ensure_dir(out_md)
    with open(out_md, 'w') as f:
        f.write(md)
    result = {
        'perhour_dir': perhour_dir,
        'base_tag': args.base_tag,
        'blocks': blocks,
        'n_boot': int(args.n_boot),
        'shear_unit': 'm/s per m',
        'runs': {t: entries[t] for t in order},
    }
    _ensure_dir(out_json)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(md)
    print("写出 {} / {}".format(out_md, out_json))


if __name__ == "__main__":
    main()
