#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""99d_vertical.py —— 垂直增益剖面:超分相对粗端输入(y0)的收益随高度如何衰减

只读 results/*_perhour.npz 的累加量(se²、count、bias 分子),纯 numpy,无需 GPU。

用法:
  python3 scripts/outline/99d_vertical.py \
    --dir results/phase3_arch --tags p2_l1r2_lr2e4_ens16 p3_arch_reg p3_arch_swin \
    --det_dir results/phase2 --det_tag p2_l1r2_lr2e4 \
    --out_json results/phase3_arch/phase3_vertical.json \
    --out_md results/phase3_arch/phase3_vertical.md \
    --fig docs/图/阶段3_垂直增益.png

口径(每个数都能追到表达式,见 CLAUDE.md 红线):
  RMSE(lev)  = sqrt( acc_agl[:, lev, si, 1].sum() / acc_agl[:, lev, si, 0].sum() )
  bias(lev)  = acc_agl[:, lev, si, 3].sum() / acc_agl[:, lev, si, 0].sum()
  y0 RMSE    = 同上,用 acc_y0[:, lev, si, :](粗端输入,不做超分)
  gain(lev)  = (1 - RMSE_model(lev) / RMSE_y0(lev)) * 100 %
"""
import argparse
import json
import os

import numpy as np

AGL = [10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]


def load(path):
    return np.load(path, allow_pickle=True)


def level_stats(arr, lev, si=0, col=1):
    """arr: (hours, levels, states, 6) -> (rmse, bias, n_hours_with_samples)

    累加量列(col 见 agl_eval_common.accum_hour):
      0=n, 1=Σ(u²+v²) 矢量 se², 2=Σ|e| 矢量 MAE, 3=ΣΔspeed 风速偏差,
      4=Σ|Δdir| 风向弧度和, 5=Σw²
    col=5 时返回 w 的 RMSE(此时第二项无意义,填 nan)。
    """
    a = arr[:, lev, si, :]
    n = a[:, 0].sum()
    if n <= 0:
        return float("nan"), float("nan"), 0
    rmse = float(np.sqrt(a[:, col].sum() / n))
    bias = float(a[:, 3].sum() / n) if col == 1 else float("nan")
    return rmse, bias, int((a[:, 0] > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="perhour 目录(集合/回归臂)")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--det_dir", default=None, help="确定性基线的 perhour 目录")
    ap.add_argument("--det_tag", default=None)
    ap.add_argument("--si", type=int, default=0, help="状态索引:0=all, 1..3=风速分箱, 4..6=PBL 分层")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    order = ([args.det_tag] if args.det_tag else []) + list(args.tags)
    data, y0_rmse, y0_bias = {}, None, None
    for tag in args.tags:
        d = load(os.path.join(args.dir, f"{tag}_perhour.npz"))
        r, b, nh = zip(*[level_stats(d["acc_agl"], i, args.si) for i in range(len(AGL))])
        data[tag] = {"rmse": list(r), "bias": list(b), "n_hours": list(nh)}
        if y0_rmse is None:
            r0, b0, _ = zip(*[level_stats(d["acc_y0"], i, args.si) for i in range(len(AGL))])
            y0_rmse, y0_bias = list(r0), list(b0)

    if args.det_tag:
        d = load(os.path.join(args.det_dir, f"{args.det_tag}_perhour.npz"))
        r, b, nh = zip(*[level_stats(d["acc_agl"], i, args.si) for i in range(len(AGL))])
        data[args.det_tag] = {"rmse": list(r), "bias": list(b), "n_hours": list(nh)}

    for tag, dd in data.items():
        dd["gain_vs_y0"] = [100.0 * (1.0 - m / y0) if y0 > 0 else float("nan")
                            for m, y0 in zip(dd["rmse"], y0_rmse)]

    # 第三维:w 的 RMSE 剖面(累加量 col=5)
    wdata, y0_w = {}, None
    for tag in order:
        if args.det_tag and tag == args.det_tag:
            src = os.path.join(args.det_dir, f"{tag}_perhour.npz")
        else:
            src = os.path.join(args.dir, f"{tag}_perhour.npz")
        d = load(src)
        wdata[tag] = [level_stats(d["acc_agl"], i, args.si, col=5)[0] for i in range(len(AGL))]
        if y0_w is None:
            y0_w = [level_stats(d["acc_y0"], i, args.si, col=5)[0] for i in range(len(AGL))]
    for tag in wdata:
        wdata[tag] = {"rmse": wdata[tag],
                      "gain_vs_y0": [100.0 * (1.0 - m / y) if y > 0 else float("nan")
                                     for m, y in zip(wdata[tag], y0_w)]}

    out = {"agl": AGL, "y0_rmse": y0_rmse, "y0_bias": y0_bias, "arms": data,
           "y0_w_rmse": y0_w, "w_arms": wdata,
           "si": args.si, "note": "RMSE 由 perhour 累加量 sqrt(sum se2 / sum n) 得到"}
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    lines = [f"# 垂直增益剖面(si={args.si})", "",
             "| AGL (m) | 粗端 y0 | " + " | ".join(order) + " |",
             "|---|---|" + "---|" * len(order)]
    for i, z in enumerate(AGL):
        cells = [f"{y0_rmse[i]:.3f}"]
        for t in order:
            cells.append(f"{data[t]['rmse'][i]:.3f} ({data[t]['gain_vs_y0'][i]:+.1f}%)")
        lines.append(f"| {z} | " + " | ".join(cells) + " |")
    lines += ["", "括号内为相对粗端输入的增益 (1 − RMSE/RMSE_y0)。", "",
              "| AGL (m) | 粗端 y0 (w) | " + " | ".join(order) + " |",
              "|---|---|" + "---|" * len(order)]
    for i, z in enumerate(AGL):
        cells = [f"{y0_w[i]:.4f}"]
        for t in order:
            cells.append(f"{wdata[t]['rmse'][i]:.4f} ({wdata[t]['gain_vs_y0'][i]:+.1f}%)")
        lines.append(f"| {z} | " + " | ".join(cells) + " |")
    lines += ["", "下半表是垂直速度 w 的 RMSE(累加量 col=5,单位 m/s),括号内同样是对 y0 的增益。", ""]
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"wrote {args.out_json} / {args.out_md}")
    for t in order:
        g = data[t]["gain_vs_y0"]
        print(f"{t:24s} gain 10m {g[0]:+6.1f}%  100m {g[4]:+6.1f}%  "
              f"500m {g[8]:+6.1f}%  1000m {g[10]:+6.1f}%")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        style = {"p2_l1r2_lr2e4": ("SI deterministic (N=1)", "o", "#2a9d2a", "-."),
                 "p3_arch_reg": ("regression (ours)", "s", "#c62828", "-"),
                 "p2_l1r2_lr2e4_ens16": ("SI ensemble N=16", "^", "#7a7a7a", "--"),
                 "p3_arch_swin": ("SwinIR (19M)", "d", "#1f5fa9", "-")}
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        ypos = np.arange(len(AGL))  # 等距层索引,避免低层被挤压
        for t in order:
            lab, mk, col, ls = style.get(t, (t, "o", None, "-"))
            ax.plot(data[t]["gain_vs_y0"], ypos, marker=mk, color=col, ls=ls, label=lab, lw=1.8)
        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels([str(z) for z in AGL])
        ax.set_ylim(-0.4, len(AGL) - 0.4)
        ax.set_xlabel("RMSE gain over coarse input y0 (%)")
        ax.set_ylabel("height AGL (m)")
        ax.set_title("Where does super-resolution actually help?")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        fig.savefig(args.fig, dpi=160)
        print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
