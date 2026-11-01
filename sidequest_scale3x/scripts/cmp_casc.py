# -*- coding: utf-8 -*-
"""cmp_casc.py — 级联 vs 一步直连:同一批天气、同一口径、带配对块 bootstrap 的对比。

回答的问题(外传-4 的核心):
  1) 级联 27→9→3→1 的终点误差,与一步 27→1 直连比,谁低?
  2) 每走一级,误差怎么涨?(27→9 一级 / 加 9→3 / 再加 3→1)
  3) 级联的终点,与"用真 3 km 粗端自训的 3→1 模型"比,差多少?——这是级联的**代价**。

口径与 summary3 / 阶段 3 一致:主指标 = AGL 10–500 m 矢量 RMSE,
`acc_agl` 的层级维按 `agl_eval_common.main_levels_idx()`,分层取 `STRATA.index('all')`。

配对:所有 tag 在同一次块抽样里同时重算(同一小时集),因此 Δ 的 CI 是配对的,
不会把"天气样本不同"混进"模型不同"。

用法(纯 numpy,仓库根目录):
  python sidequest_scale3x/scripts/cmp_casc.py \
      --dir sidequest_scale3x/results \
      --ref p4_27to1_reg \
      --tags p4_27to1_reg,sq3to1_casc,sq3to1_reg \
      --block 24 --n_boot 2000 \
      --out_md sidequest_scale3x/docs/_cmp_casc.md
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 口径来源:`scripts/outline/agl_eval_common.py`。那个模块会连带 import torch(经
# src.dl_data.wind_canvas_statics),而本工具是纯 numpy 后处理,不该为此要求 torch;
# 所以优先用它的定义,拿不到时退回同一组常量,并在两者都可用时断言一致。
AGL = [10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
_STRATA_FALLBACK = ['all', 'day', 'night', 'urban', 'rural', 'weak', 'mid', 'strong',
                    'pbl_low', 'pbl_ent', 'pbl_high']
_SRC = os.path.join(REPO, "scripts", "outline")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)
try:
    from agl_eval_common import STRATA, main_levels_idx  # noqa: E402
except Exception:  # 典型原因:本机没装 torch
    STRATA = _STRATA_FALLBACK
    main_levels_idx = lambda: [i for i, z in enumerate(AGL) if z <= 500]
STRATA_LIST = list(STRATA)
SI = STRATA_LIST.index("all")


def load_tag(dirs, tag):
    """在若干目录里找 <tag>_perhour.npz;返回 None 表示都没有。"""
    for d in dirs:
        path = os.path.join(d, "{}_perhour.npz".format(tag))
        if os.path.exists(path):
            break
    else:
        return None
    z = np.load(path, allow_pickle=True)
    out = {"hours": z["hours"] if "hours" in z.files else None,
           "acc": z["acc_agl"],
           "y0": z["acc_y0"] if "acc_y0" in z.files else None,
           "sh": z["acc_sh"] if "acc_sh" in z.files else None}
    return out


def rmse_on(acc, idx, lv):
    se = acc[idx][..., lv, SI, 1].sum()
    n = acc[idx][..., lv, SI, 0].sum()
    return float(np.sqrt(se / max(n, 1.0)))


def iter_blocks(n_hours, block, n_boot, seed=0):
    """移动块 bootstrap(与 99/99b 同口径):块起点均匀、块数 ceil(n/block)、截到 n。"""
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n_hours - block + 1, 1)
    if len(starts) == 0:
        starts = np.arange(0, max(n_hours - block, 0) + 1, 1)
    n_blocks = int(np.ceil(n_hours / block))
    for _ in range(n_boot):
        pick = rng.choice(starts, size=n_blocks, replace=True)
        yield np.concatenate([np.arange(s, s + block) for s in pick])[:n_hours]


def ci(vals, lo=2.5, hi=97.5):
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sidequest_scale3x/results",
                    help="可逗号分隔多个目录(如级联与外传在 results/,直连在 results/phase4)")
    ap.add_argument("--ref", required=True, help="参照 tag(通常是 p4_27to1_reg 一步直连)")
    ap.add_argument("--tags", required=True, help="逗号分隔;第一行应为参照本身")
    ap.add_argument("--block", type=int, default=24)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--per_day", action="store_true", help="追加逐日 RMSE 表")
    ap.add_argument("--warn_stale", action="store_true",
                    help="数据来自 _preunify(旧划分),只做功能自检时打开")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.ref not in tags:      # 参照不在 --tags 里也自动补上,免得手误
        tags.insert(0, args.ref)
    dirs = [d.strip() for d in args.dir.split(",") if d.strip()]
    lv = main_levels_idx()
    data, missing = {}, []
    for t in tags:
        z = load_tag(dirs, t)
        if z is None:
            missing.append(t)
            continue
        data[t] = z
    if args.ref not in data:
        raise SystemExit("参照 {} 没有 perhour.npz(缺: {})".format(args.ref, missing))
    if missing:
        print("[警告] 缺 {} 的 perhour.npz,表里标 —".format(", ".join(missing)))

    n_hours = sorted({len(v["acc"]) for v in data.values()})
    if len(n_hours) > 1:
        print("[警告] 各 tag 小时数不一致 {};按最短的截断".format(n_hours))
    n = min(n_hours)
    for t in list(data):
        data[t]["acc"] = data[t]["acc"][:n]
        if data[t]["y0"] is not None:
            data[t]["y0"] = data[t]["y0"][:n]

    base = {t: rmse_on(data[t]["acc"], np.arange(n), lv) for t in data}
    y0 = {t: (rmse_on(data[t]["y0"], np.arange(n), lv) if data[t]["y0"] is not None else None)
          for t in data}

    deltas = {t: [] for t in data}
    for idx in iter_blocks(n, args.block, args.n_boot):
        b = {t: rmse_on(data[t]["acc"], idx, lv) for t in data}
        for t in data:
            deltas[t].append(b[t] - b[args.ref])

    rows = []
    for t in tags:
        if t not in data:
            rows.append({"tag": t, "n_hours": None})
            continue
        dv = np.asarray(deltas[t])
        rows.append({
            "tag": t,
            "n_hours": int(n),
            "rmse": base[t],
            "delta_vs_ref": base[t] - base[args.ref],
            "ci": ci(dv),
            "significant": bool(ci(dv)[0] > 0 or ci(dv)[1] < 0),
            "y0_rmse": y0[t],
            "gain_over_y0_pct": (1.0 - base[t] / y0[t]) * 100.0 if y0[t] else None,
        })

    lines = []
    if args.warn_stale:
        lines.append("> ⚠️ 本表由 `_preunify`(统一划分前的旧产物)生成,仅用于验证工具链,"
                     "**数字不可引用**。\n")
    lines.append("| tag | 小时数 | 主指标 RMSE | Δ vs {} | 95% CI(配对块 bootstrap) | 显著 | 插值 y0 | 相对插值增益 |".format(args.ref))
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("n_hours") is None:
            lines.append("| {} | — | — | — | — | — | — | — |".format(r["tag"]))
            continue
        lines.append("| {} | {} | {:.4f} | {:+.4f} | [{:+.4f}, {:+.4f}] | {} | {} | {} |".format(
            r["tag"], r["n_hours"], r["rmse"], r["delta_vs_ref"],
            r["ci"][0], r["ci"][1], "是" if r["significant"] else "否",
            "{:.4f}".format(r["y0_rmse"]) if r["y0_rmse"] else "—",
            "{:+.1f}%".format(r["gain_over_y0_pct"]) if r["gain_over_y0_pct"] is not None else "—"))
    md = "\n".join(lines)
    print(md)

    day_table = None
    if args.per_day:
        # 逐日只看点估计(不做 bootstrap):用于看级联/直连在强风日与平静日的相对表现。
        ref_hours = None
        for t in data:
            if data[t]["hours"] is not None:
                ref_hours = [str(h) for h in data[t]["hours"][:n]]
                break
        if ref_hours is None:
            print("\n[警告] npz 里没有 hours 键,跳过逐日表")
        else:
            days = sorted({h[:10] for h in ref_hours})
            dlines = ["", "| 日期 | " + " | ".join(t for t in tags if t in data) + " |",
                      "|---" * (1 + len([t for t in tags if t in data])) + "|"]
            day_table = {}
            for day in days:
                idx = np.array([i for i, h in enumerate(ref_hours) if h[:10] == day])
                if len(idx) == 0:
                    continue
                cells, row = [], {}
                for t in tags:
                    if t not in data:
                        continue
                    v = rmse_on(data[t]["acc"], idx, lv)
                    row[t] = v
                    cells.append("{:.4f}".format(v))
                day_table[day] = row
                dlines.append("| {} ({}) | {} |".format(day, len(idx), " | ".join(cells)))
            print("\n".join(dlines))
            md = md + "\n".join(dlines)

    # 自洽守卫:级联第 1 级(带 --dump_fields 重跑)与它本来的 in-domain 结果必须一致,
    # 不一致说明推理不确定(或 dump 改变了采样),那么后面各级的数不能信。
    for t in list(data):
        if t.endswith("_casc"):
            twin = t[: -len("_casc")]
            if twin in data:
                d_ = abs(base[t] - base[twin])
                print("\n[自洽] {} vs {}:Δ={:.2e} {}".format(
                    t, twin, d_, "✓" if d_ < 1e-6 else "⚠️ 不一致,级联数不可信"))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"ref": args.ref, "block": args.block, "n_boot": args.n_boot,
                       "n_hours": int(n), "rows": rows, "per_day": day_table},
                       f, ensure_ascii=False, indent=2)
        print("\n[json] {}".format(args.out_json))
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print("[md] {}".format(args.out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
