# -*- coding: utf-8 -*-
"""阶段 2 净效应表:92 的排序(逐点误差 + 显著性)与 94 的三类诊断(散度/能谱/极值)汇总成
可直接贴进阶段报告的 markdown 表(results/phase2/phase2_net_effects.md)。

行 = run(基线 r_t14_noenc_cos 第一行);列:
  ① 主指标 RMSE 与 Δ(含 95% CI 与显著性,来自 92 的 ranking csv)
  ② 散度残差 P95 与 frac(|D|>τ)(相对基线变化%)——94 的 divergence_scalar
  ③ 高频段(半径 > --high_k cycles/pixel)平均 log 功率变化(等效功率比 % 与 ΔlogP)
  ④ 全局极值 max/min 差(全 split 时间×空间)
缺数据的格子写 "—"(例如只跑了部分 tag 的诊断)。

用法(只需 numpy,仓库根目录):
  python scripts/outline/97_phase2_tables.py --results_dir results/phase2 \
      --base_tag r_t14_noenc_cos --diag_tags p2_div_mid,p2_l2
不传 --diag_tags 时自动扫描 results_dir 下全部 {tag}_diag.json。
"""
import argparse
import glob
import json
import math
import os

import numpy as np


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_ranking(rank_csv):
    """读 92 的 ranking csv -> {tag: {...}};文件不存在返回 {}。"""
    rows = {}
    if not rank_csv or not os.path.exists(rank_csv):
        return rows
    with open(rank_csv) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != len(header):
                continue
            r = dict(zip(header, parts))
            rows[r['tag']] = {
                'main_rmse_vec': _to_float(r.get('main_rmse_vec')),
                'delta': _to_float(r.get('delta')),
                'ci_lo': _to_float(r.get('ci_lo')),
                'ci_hi': _to_float(r.get('ci_hi')),
                'significant': str(r.get('significant', '')).lower() in ('true', '1'),
            }
    return rows


def summary_rmse(results_dir, tag):
    """92 的 ranking 缺失时的兜底:从 90 的 {tag}_summary.json 读主指标。"""
    p = os.path.join(results_dir, "{}_summary.json".format(tag))
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return _to_float(json.load(f).get('main_rmse_vec'))


def load_diag(results_dir, tag):
    p = os.path.join(results_dir, "{}_diag.json".format(tag))
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_spec(results_dir, tag):
    p = os.path.join(results_dir, "{}_diag.npz".format(tag))
    if not os.path.exists(p):
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.keys()}


def div_scalar(diag):
    """(p95_abs_mean, frac_gt_tau_mean);94 已算好标量,缺时由逐层表平均兜底。"""
    if diag is None:
        return None, None
    sc = diag.get('divergence_scalar', {}).get('pred')
    if sc:
        return sc.get('p95_abs_mean'), sc.get('frac_gt_tau_mean')
    d = diag.get('divergence', {}).get('pred')
    if not d:
        return None, None
    return _to_float(np.mean(d['p95_abs'])), _to_float(np.mean(d['frac_gt_tau']))


def ext_global(diag):
    """(global_max_all, global_min_all)。"""
    if diag is None:
        return None, None
    e = diag.get('extremes', {}).get('pred')
    if not e:
        return None, None
    return e.get('global_max_all'), e.get('global_min_all')


def high_band_logp(spec, high_k, source='pred'):
    """高频段(箱中心半径 > high_k cycles/pixel)平均 log 功率,层与 u/v 再平均。"""
    if spec is None:
        return None
    edges = np.asarray(spec['bin_edges'], dtype=np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])
    m = centers > float(high_k)
    if not m.any():
        return None
    s = np.asarray(spec['spec_{}'.format(source)])
    return float(s[:, :, m].mean())


def fmt_delta_ci(rk, base_rmse):
    """主指标 Δ 与 95% CI 单元格。"""
    if rk is None or rk['main_rmse_vec'] is None:
        return "—", "—"
    d = rk['delta'] if rk['delta'] is not None else None
    if d is None and base_rmse is not None:
        d = rk['main_rmse_vec'] - base_rmse
    if d is None:
        return "—", "—"
    if rk['ci_lo'] is None or rk['ci_hi'] is None or \
            rk['ci_lo'] != rk['ci_lo'] or rk['ci_hi'] != rk['ci_hi']:   # nan
        return "{:+.4f}".format(d), "—"
    return "{:+.4f} [{:+.4f}, {:+.4f}]".format(d, rk['ci_lo'], rk['ci_hi']), \
        ("是" if rk['significant'] else "否")


def fmt_pct(cur, base):
    if cur is None or base is None or base == 0:
        return "—"
    return "{:+.1f}%".format(100.0 * (cur - base) / base)


def fmt_diff(cur, base, nd=3):
    if cur is None or base is None:
        return "—"
    return "{:+.{nd}f}".format(cur - base, nd=nd)


def main():
    ap = argparse.ArgumentParser(description="阶段 2 净效应表(92 排序 + 94 诊断)")
    ap.add_argument("--results_dir", default="results/phase2")
    ap.add_argument("--base_tag", default="r_t14_noenc_cos")
    ap.add_argument("--diag_tags", default=None, help="逗号分隔;缺省自动扫描 *_diag.json")
    ap.add_argument("--rank_csv", default=None, help="缺省 <results_dir>/ranking_phase2.csv")
    ap.add_argument("--out_md", default=None, help="缺省 <results_dir>/phase2_net_effects.md")
    ap.add_argument("--high_k", type=float, default=0.25, help="高频段阈值(cycles/pixel)")
    args = ap.parse_args()

    results_dir = args.results_dir
    rank_csv = args.rank_csv or os.path.join(results_dir, "ranking_phase2.csv")
    out_md = args.out_md or os.path.join(results_dir, "phase2_net_effects.md")

    ranking = load_ranking(rank_csv)
    if not ranking:
        print("[警告] 没读到 92 排序 {};主指标退回 summary.json,Δ 用点估计".format(rank_csv))

    if args.diag_tags:
        tags = [t for t in args.diag_tags.split(',') if t]
    else:
        tags = [os.path.basename(p)[:-len('_diag.json')]
                for p in sorted(glob.glob(os.path.join(results_dir, "*_diag.json")))]
    tags = [t for t in tags if t != args.base_tag]
    base_rmse = (ranking.get(args.base_tag) or {}).get('main_rmse_vec')
    if base_rmse is None:
        base_rmse = summary_rmse(results_dir, args.base_tag)

    # 行顺序:按主指标升序(基线固定第一行);无数据的排最后
    def _key(t):
        m = (ranking.get(t) or {}).get('main_rmse_vec')
        if m is None:
            m = summary_rmse(results_dir, t)
        return (0, m) if m is not None else (1, 0.0)
    tags.sort(key=_key)

    base_diag = load_diag(results_dir, args.base_tag)
    base_spec = load_spec(results_dir, args.base_tag)
    base_div_p95, base_div_frac = div_scalar(base_diag)
    base_high = high_band_logp(base_spec, args.high_k)
    base_gmax, base_gmin = ext_global(base_diag)

    if base_diag is None:
        print("[警告] 缺基线诊断 {}_diag.json;散度/谱/极值列会大量为空".format(args.base_tag))

    lines = []
    lines.append("| run | 主指标 RMSE (m/s) | Δ RMSE [95% CI] | 显著 | 散度 P95 (kg m⁻³ s⁻¹) "
                 "| Δ% | frac(\\|D\\|>τ) | Δ% | 高频 logP Δ% | 全局 max Δ (m/s) | 全局 min Δ (m/s) |")
    lines.append("|" + "---|" * 11)

    def row(tag, diag, spec, rk):
        m = (rk or {}).get('main_rmse_vec')
        if m is None:
            m = summary_rmse(results_dir, tag)
        rmse_cell = "—" if m is None else "{:.4f}".format(m)
        if tag == args.base_tag:
            dcell, sig = "—", "—"
        else:
            dcell, sig = fmt_delta_ci(rk, base_rmse)
        p95, frac = div_scalar(diag)
        cells = [
            tag + (" (基准)" if tag == args.base_tag else ""),
            rmse_cell, dcell, sig,
            "—" if p95 is None else "{:.3e}".format(p95),
            "—" if tag == args.base_tag else fmt_pct(p95, base_div_p95),
            "—" if frac is None else "{:.4f}".format(frac),
            "—" if tag == args.base_tag else fmt_pct(frac, base_div_frac),
        ]
        hi = high_band_logp(spec, args.high_k)
        if tag == args.base_tag or hi is None or base_high is None:
            cells.append("—")
        else:
            dlog = hi - base_high
            cells.append("{:+.1f}% (ΔlogP {:+.2f})".format((math.exp(dlog) - 1.0) * 100.0, dlog))
        gmax, gmin = ext_global(diag)
        if tag == args.base_tag:
            cells.append("—")
            cells.append("—")
        else:
            cells.append(fmt_diff(gmax, base_gmax))
            cells.append(fmt_diff(gmin, base_gmin))
        return "| " + " | ".join(cells) + " |"

    lines.append(row(args.base_tag, base_diag, base_spec, ranking.get(args.base_tag)))
    for t in tags:
        lines.append(row(t, load_diag(results_dir, t), load_spec(results_dir, t), ranking.get(t)))

    n_frames = {}
    if base_diag is not None:
        n_frames[args.base_tag] = base_diag.get('n_frames')
    for t in tags:
        d = load_diag(results_dir, t)
        if d is not None:
            n_frames[t] = d.get('n_frames')

    notes = [
        "",
        "口径:主指标 = AGL 10–500 m 层平均风矢量 RMSE(92,配对移动块 bootstrap);"
        "散度 = 全 23 层 |∇·(ρu)| 的逐层统计对层取平均(P95 / frac 为超过 τ_k 的格点占比),"
        "Δ% = 相对基线;能谱 = 去交错质量点上逐层 u/v 径向 log 谱(帧平均),"
        "高频段 = 箱中心半径 > {} cycles/pixel,Δ% 由平均 log 功率差换算等效功率比;".format(args.high_k),
        "极值 = 全 split 时间×空间全局 max/min 之差(在最低 10 层与 10 m 共 11 个风速场中取全局);"
        "y0(粗端重网格免费基线)与真值口径同 94 号,逐层表见各 {tag}_diag.json。",
        "诊断帧数: " + ", ".join("{}={}".format(k, v) for k, v in n_frames.items()),
    ]
    md = "\n".join(lines + notes) + "\n"
    with open(out_md, 'w') as f:
        f.write(md)
    print(md)
    print("写出 {}".format(out_md))


if __name__ == "__main__":
    main()
