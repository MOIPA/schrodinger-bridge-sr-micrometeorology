# -*- coding: utf-8 -*-
"""生成 2026-10-09 组会汇报配图(阶段 0-3),2026-10-14 美化改版。

- 图内文字全部英文,风格对齐 组会汇报/模范图片/(fig_common 设计系统):
  见 scripts/figstyle.py(NJU 紫主色/字号 token/白底淡化轴线)
- 个例图(fig_case*)加真实经纬度坐标轴 + 海岸线 + 地名,并带华南区位小图
- 本地运行:/opt/anaconda3/envs/d2l/bin/python scripts/plot_report_10_09.py
- 输出:组会汇报/2026-10-09/fig_*.png(文件名与报告引用保持一致)
"""
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "组会汇报", "2026-10-09")
os.makedirs(OUT, exist_ok=True)

ARMS = [("p2_l1r2_lr2e4", "Mode-level (baseline)", fs.GREY),
        ("p3_agl", "AGL-only", fs.R4[1]),
        ("p3_agllw", "Mode + low-level weighting", fs.R4[2]),
        ("p3_joint", "Mode + joint supervision", fs.R4[3])]


def save(fig, name):
    fs.save(fig, os.path.join(OUT, name))


# ============ 图 1:阶段 0 四档位动能谱 ============
def fig_spectra():
    d = json.load(open(os.path.join(ROOT, "results/outline/truth_diagnostics.json")))
    sp = d["spectra"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, tag in enumerate(["d01", "d02", "d03", "d04"]):
        s = sp[tag]
        k = np.array(s["k"])
        P = np.array(s["P"])
        m = (k > 0) & (P > 0)
        ax.loglog(k[m], P[m], "-", color=fs.R4[i], lw=1.7,
                  label="{} - effective resolution ~{:.0f} km".format(
                      tag, s["wavelength_7dx_km"]))
        if tag == "d04":
            k7 = s["k_7dx"]
            ax.axvline(k7, color=fs.GREY_LIGHT, ls=(0, (4, 3)), lw=0.9, zorder=0)
            ax.text(k7 * 1.05, float(np.max(P)) * 0.45, "7dx cutoff (d04)",
                    fontsize=fs.DS["annot"], color=fs.GREY, va="top")
    ax.set_xlabel("Wavenumber k (dimensionless)")
    ax.set_ylabel("Kinetic-energy spectrum P(k)")
    ax.set_title("Kinetic-energy spectra of the four model domains (2020-07-15 12Z)")
    ax.grid(True, which="both")
    ax.legend(loc="lower left")
    save(fig, "fig_p0_spectra.png")


# ============ 图 2:阶段 0 数据划分时间线 ============
def fig_split():
    d = json.load(open(os.path.join(ROOT, "results/outline/split.json")))
    blocks = d["blocks"]
    cmap = {"train": fs.GREY_LIGHT, "valid": fs.R4[1], "test": fs.PURPLE}
    days = {"train": 0, "valid": 0, "test": 0}
    fig, ax = plt.subplots(figsize=(7.0, 1.75))
    for b in blocks:
        d0 = int(b["start"][-2:])
        d1 = int(b["end"][-2:])
        days[b["split"]] += d1 - d0 + 1
        ax.barh(0, d1 - d0 + 1, left=d0 - 0.5, height=0.52,
                color=cmap[b["split"]], edgecolor="white", lw=1.2)
        ax.text((d0 + d1) / 2.0, 0, str(b["idx"]), ha="center", va="center",
                color="white" if b["split"] != "train" else fs.INK2,
                fontsize=fs.DS["value"])
    tot = sum(days.values())
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(-0.38, 0.38)
    ax.set_yticks([])
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Day of July 2020")
    ax.set_title("Weather-regime block split - 10 blocks, "
                 "train {:d}% / valid {:d}% / test {:d}%".format(
                     round(100 * days["train"] / tot),
                     round(100 * days["valid"] / tot),
                     round(100 * days["test"] / tot)), pad=26)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k], lw=0)
               for k in ["train", "valid", "test"]]
    ax.legend(handles,
              ["Train - {} d".format(days["train"]),
               "Validation - {} d".format(days["valid"]),
               "Test - {} d".format(days["test"])],
              ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines["left"].set_visible(False)
    save(fig, "fig_p0_split.png")


# ============ 图 3:阶段 1 输入消融(残差配置)============
def fig_p1_ablation():
    label_map = {
        "r_t14_noenc_cos": "cos(SZA) instead of time encoding",
        "r_t15_z0": "+ fine-grid log z0",
        "t16_residual": "baseline (residual output, V*)",
        "r_t13_theta": "+ theta profile",
        "r_t13_most": "+ MOST group",
        "r_t12_zagldiff": "+ per-level height-difference field",
        "r_t15_z0_urban": "+ urban fraction",
        "r_t15_z0_urban_wv": "+ water / vegetation fractions",
        "r_t17_coords": "+ geographic coordinates",
        "r_t12_hgtdiff": "+ 2-D terrain difference",
        "r_t13_w": "+ coarse W",
        "r_t13_ph": "+ hourly PH",
        "baseline_y0_regrid": "y0 coarse regrid (reference)",
        "r_t14_cos": "+ cos(SZA), keep time encoding",
        "baseline_bicubic": "bicubic interpolation (reference)",
        "r_t13_mostflux": "+ fluxes group",
    }
    ref_tags = {"baseline_y0_regrid", "baseline_bicubic"}
    rows = []
    with open(os.path.join(ROOT, "results/phase1/ranking_phase1r.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    configs = [r for r in rows if r["tag"] not in ref_tags]
    refs = [r for r in rows if r["tag"] in ref_tags]
    # y 轴向上,y=0 在最底部;rows[0] 画在最底部 -> 列表末尾 = 视觉顶部。
    # configs 按 delta 降序 -> 视觉自上而下 = 最优到最差;refs 置列表末尾
    # -> 参照组固定显示在顶部,与消融配置用分隔线隔开。
    configs.sort(key=lambda r: float(r["delta"]), reverse=True)
    refs.sort(key=lambda r: float(r["delta"]), reverse=True)
    rows = configs + refs

    labs, deltas, los, his, sigs, tags = [], [], [], [], [], []
    for r in rows:
        tags.append(r["tag"])
        labs.append(label_map.get(r["tag"], r["tag"]))
        deltas.append(float(r["delta"]))
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        los.append(lo if lo == lo else deltas[-1])
        his.append(hi if hi == hi else deltas[-1])
        sigs.append(float(r["significant"]) > 0.5)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(labs))
    for i, tag in enumerate(tags):
        if tag == "t16_residual":
            c, mk, ms = fs.PURPLE, "D", 5.5
        elif tag in ref_tags:
            c, mk, ms = fs.REF, "s", 4.5
        else:
            c = fs.ORANGE if sigs[i] else fs.GREY
            mk, ms = "o", 4.5
        ax.errorbar(deltas[i], y[i],
                    xerr=[[deltas[i] - los[i]], [his[i] - deltas[i]]],
                    fmt=mk, color=c, ecolor=c, elinewidth=1.1,
                    capsize=2.5, ms=ms, zorder=3)
    # 参照组(视觉顶部)与消融配置组的分隔线
    if ref_tags and len(rows) > len(ref_tags):
        ax.axhline(len(rows) - len(ref_tags) - 0.5, color=fs.GRID, lw=0.9,
                   zorder=1)
    ax.axvline(0, color=fs.REF, lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=fs.DS["tick"])
    ax.set_ylim(-0.7, len(labs) - 0.3)
    ax.set_xlabel("Change in wind-vector RMSE vs baseline (m/s) - "
                  "95% paired-block bootstrap CI")
    ax.set_title("Phase 1 - input-variable ablation (13 residual configs; "
                 "baseline 0.963 m/s)", loc="left")
    ax.grid(True, axis="x")
    ax.legend(handles=[
        Line2D([], [], marker="D", ls="", color=fs.PURPLE,
               label="baseline V* (adopted)"),
        Line2D([], [], marker="o", ls="", color=fs.ORANGE,
               label="significantly worse"),
        Line2D([], [], marker="o", ls="", color=fs.GREY,
               label="no significant change"),
        Line2D([], [], marker="s", ls="", color=fs.REF,
               label="reference baselines"),
    ], loc="lower left", fontsize=fs.DS["annot"])
    save(fig, "fig_p1_ablation.png")


# ============ 图 4:阶段 1 直接输出 vs 残差输出 ============
def fig_p1_direct_residual():
    groups = ["Baseline", "+MOST", "+coarse W", "+fluxes", "+theta profile"]
    direct = [1.311, 1.947, 1.946, 1.761, 1.139]
    residual = [0.963, 0.973, 1.084, 1.206, 0.972]
    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    b1 = ax.bar(x - w / 2, direct, w, color=fs.GREY_LIGHT,
                label="Direct 1-km output")
    b2 = ax.bar(x + w / 2, residual, w, color=fs.PURPLE,
                label="Residual output (1-km field minus regridded coarse)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.03,
                    "{:.2f}".format(r.get_height()), ha="center",
                    fontsize=fs.DS["annot"], color=fs.INK2)
    ax.text(0, direct[0] + 0.16, "-27%", ha="center", fontsize=fs.DS["value"],
            color=fs.PURPLE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Wind-vector RMSE (m/s)")
    ax.set_ylim(0, max(direct) * 1.26)
    ax.set_title("Residual vs direct output (five configurations)", loc="left")
    ax.legend(loc="upper center", ncol=2)
    ax.grid(True, axis="y")
    save(fig, "fig_p1_direct_residual.png")


# ============ 图 5:阶段 2 损失与约束对比 ============
def fig_p2_losses():
    items = [("Extreme-value loss", 15.7), ("Divergence constraint (warm-up)", 8.6),
             ("L2 data term", 6.1), ("L1 + spectral loss", 5.2),
             ("Vorticity structure matching", 4.2)]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], 0.55, color=fs.ORANGE)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.25, i, "+{:.1f}%".format(v), va="center",
                fontsize=fs.DS["annot"], color=fs.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in items], fontsize=fs.DS["tick"])
    ax.invert_yaxis()
    ax.axvline(0, color=fs.PURPLE, lw=1.4, zorder=1)
    ax.text(0.18, -0.45, "pure-L1 baseline", color=fs.PURPLE,
            fontsize=fs.DS["annot"], ha="left", va="center")
    ax.set_xlim(0, 18)
    ax.set_xlabel("Accuracy loss vs pure-L1 baseline (%)")
    ax.set_title("Phase 2 - loss terms & physics constraints "
                 "(ordered by harm, worst on top)", loc="left")
    ax.grid(True, axis="x")
    save(fig, "fig_p2_losses.png")


# ============ 图 6:阶段 2 散度残差廓线 ============
def fig_p2_divergence():
    base = json.load(open(os.path.join(ROOT, "results/phase2/r_t14_noenc_cos_diag.json")))
    dv = json.load(open(os.path.join(ROOT, "results/phase2/p2_div_mid_warm_diag.json")))
    tau = np.array(base["phys_div_tau"])
    lv = np.arange(len(tau))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    series = [
        ("d04 truth", base["divergence"]["truth"], fs.REF, "-", 1.7),
        ("y0 (coarse regrid)", base["divergence"]["y0"], fs.GREY_LIGHT, (0, (5, 2)), 1.5),
        ("Baseline (pure L1)", base["divergence"]["pred"], fs.PURPLE, "-", 1.7),
        ("Divergence-constrained", dv["divergence"]["pred"], fs.ORANGE, "-", 1.7),
    ]
    for lab, d, c, ls, lw in series:
        ax.plot(lv, np.array(d["p95_abs"]) / tau, ls=ls, color=c, lw=lw,
                marker="o", ms=3, label=lab)
    ax.axhline(1.0, color=fs.GREY, ls=(0, (4, 3)), lw=0.9, zorder=0)
    ax.text(len(tau) - 1.2, 1.08, "scaled threshold tau", color=fs.GREY,
            fontsize=fs.DS["annot"], ha="right", va="bottom")
    ax.set_xlabel("Training-layer index (0 = lowest)")
    ax.set_ylabel("Divergence residual P95 / tau")
    ax.set_ylim(0, 3.2)
    ax.set_title("Phase 2 - divergence residual by layer "
                 "(test set, 144 frames)", loc="left")
    ax.grid(True)
    ax.legend(loc="upper right", ncol=2)
    save(fig, "fig_p2_divergence.png")


# ============ 图 7:阶段 3 逐层误差廓线 ============
def fig_p3_levels():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axvspan(10, 100, color=fs.TINT, zorder=0)
    ax.text(13, 0.985, "10-100 m", transform=ax.get_xaxis_transform(),
            fontsize=fs.DS["annot"], color=fs.GREY, ha="left", va="top")
    for tag, lab, c in ARMS:
        d = json.load(open(os.path.join(ROOT,
                                        "results/phase3/{}_summary.json".format(tag))))
        levels = d["agl_targets"]
        rmse = d["rmse_vec_all_per_level"]
        ax.plot(levels, rmse, "-o", color=c, lw=1.7, ms=3.8, label=lab)
    ax.set_xscale("log")
    ticks = [10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Height above ground (m, log scale)")
    ax.set_ylabel("Wind-vector RMSE (m/s)")
    ax.set_title("Phase 3 - error profile by height (test set, 144 h, myj)",
                 loc="left")
    ax.grid(True, which="both")
    ax.legend(loc="best")
    save(fig, "fig_p3_levels.png")


# ============ 图 8:阶段 3 主指标与切变对比 ============
def fig_p3_bars():
    arms = ["Mode-level\n(baseline)", "AGL-only",
            "Mode +\nlow-level\nweighting", "Mode +\njoint\nsupervision"]
    main = [0.9574, 0.9383, 0.8904, 0.8697]
    shear = [0.00885, 0.00798, 0.00741, 0.00704]
    main_pct = ["—", "-2.0%", "-7.0%", "-9.2%"]
    shear_pct = ["—", "-9.9%", "-16.4%", "-20.5%"]
    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax, vals, pcts, title, ylab in [
        (axes[0], main, main_pct, "(a) Primary metric - mean RMSE, 10-500 m", "RMSE (m/s)"),
        (axes[1], shear, shear_pct, "(b) Inter-level shear error", "Shear RMSE (m/s)"),
    ]:
        ax.bar(x, vals, 0.55, color=[c for _, _, c in ARMS])
        for i, (xx, p) in enumerate(zip(x, pcts)):
            ax.text(xx, vals[i] * 1.012, p, ha="center",
                    fontsize=fs.DS["value"] if i == 3 else fs.DS["annot"],
                    color=fs.INK if i > 0 else fs.GREY,
                    fontweight="bold" if i == 3 else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels(arms, fontsize=fs.DS["annot"])
        ax.set_ylim(0, max(vals) * 1.15)
        ax.set_title(title, loc="left", fontsize=fs.DS["title"])
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y")
    fig.suptitle("Phase 3 - supervision-space comparison "
                 "(single seed; shear improvement robust across 3 seeds)",
                 fontsize=10.5, color=fs.INK)
    fig.subplots_adjust(wspace=0.32, top=0.86)
    save(fig, "fig_p3_bars.png")


# ============ 图 9:阶段 3 空间分组 ============
def fig_p3_spatial():
    d = json.load(open(os.path.join(ROOT, "results/phase3/spatial_summary.json")))
    groups = ["steep_city", "steep_rural", "flat_city", "flat_rural"]
    glab = ["Steep +\nurban", "Steep +\nrural", "Flat +\nurban", "Flat +\nrural"]
    base = {g["tag"]: g["group_rmse_vec"] for g in d}
    x = np.arange(len(groups))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, (tag, lab, c) in enumerate(ARMS):
        vals = [base[tag][g] for g in groups]
        off = (i - 1.5) * w
        bars = ax.bar(x + off, vals, w, color=c, label=lab)
        if tag != "p2_l1r2_lr2e4":
            for xi, (r, g) in enumerate(zip(bars, groups)):
                basev = base["p2_l1r2_lr2e4"][g]
                pct = (base[tag][g] - basev) / basev * 100
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                        "{:+.0f}%".format(pct), ha="center",
                        fontsize=7, color=fs.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(glab, fontsize=fs.DS["tick"])
    ax.set_ylabel("Low-level (10-100 m) wind-vector RMSE (m/s)")
    ax.set_ylim(0, max(v for g in groups for v in
                       [base[t][g] for t, _, _ in ARMS]) * 1.18)
    ax.set_title("Phase 3 - error by terrain / land-cover group "
                 "(labels: change vs baseline)", loc="left")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, axis="y")
    save(fig, "fig_p3_spatial.png")


# ============ 图 10:个例风场对比(带经纬度底图)============
def fig_cases(case=0, agl=100.0):
    """个例四联图:真值 / 粗场 / 基线 / 联合监督 的风速场。

    每个面板 = 真实经纬度坐标轴 + 海岸线(水陆覆盖 contour)+ 地形等高线;
    真值面板加地名,粗场面板嵌华南区位小图(含研究域方框)。
    依赖 results/report_10_09/cases.npz 与 results/outline/statics.npz。
    """
    path = os.path.join(ROOT, "results/report_10_09/cases.npz")
    if not os.path.exists(path):
        print("skip fig_cases: missing", path)
        return
    d = np.load(path)
    agls = list(d["agl"])
    li = agls.index(agl)
    srcs = [("truth", "WRF d04 truth (1 km)"),
            ("y0", "Coarse y0 (9 km, regridded)"),
            ("pred_base", "Baseline - mode-level"),
            ("pred_joint", "Joint supervision")]
    fields = {}
    for key, _ in srcs:
        u = d["case{}_{}_u".format(case, key)][li]
        v = d["case{}_{}_v".format(case, key)][li]
        fields[key] = np.sqrt(u ** 2 + v ** 2)
    vmin = np.floor(min(f.min() for f in fields.values()) * 2.0) / 2.0
    vmax = np.ceil(max(f.max() for f in fields.values()) * 2.0) / 2.0
    stamp = str(d["case{}_stamp".format(case)][0])
    m = re.search(r"(\d{8})T?(\d{4})", stamp)
    stamp_lab = ("{}-{}-{} {}:{} UTC".format(
        m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:8],
        m.group(2)[:2], m.group(2)[2:]) if m else stamp)

    st = np.load(os.path.join(ROOT, "results/outline/statics.npz"))
    xlat, xlon = st["xlat_fine"], st["xlong_fine"]
    water, hgt = st["water_fine"], st["hgt_fine"]

    fig = plt.figure(figsize=(7.3, 6.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045])
    axs = np.empty((2, 2), dtype=object)
    axs[0, 0] = fig.add_subplot(gs[0, 0])
    axs[0, 1] = fig.add_subplot(gs[0, 1])
    axs[1, 0] = fig.add_subplot(gs[1, 0])
    axs[1, 1] = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])

    asp = 1.0 / np.cos(np.deg2rad(float(np.mean(xlat))))
    xt = [113.6, 114.0, 114.4]
    yt = [22.2, 22.4, 22.6, 22.8]
    cmap = fs.seq_purple()
    im = None
    for j, (ax, (key, lab)) in enumerate(zip(axs.ravel(), srcs)):
        im = ax.pcolormesh(xlon, xlat, fields[key], cmap=cmap, vmin=vmin,
                           vmax=vmax, shading="nearest", rasterized=True)
        ax.contour(xlon, xlat, hgt, levels=[50, 200, 400],
                   colors=[fs.GREY_LIGHT], linewidths=0.5, zorder=3)
        ax.contour(xlon, xlat, water, levels=[0.5], colors=[fs.COAST],
                   linewidths=0.8, zorder=4)
        ax.set_xlim(float(xlon.min()), float(xlon.max()))
        ax.set_ylim(float(xlat.min()), float(xlat.max()))
        ax.set_aspect(asp)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.grid(True)
        ax.set_title("({}) {} - mean {:.2f} m/s".format(
            "abcd"[j], lab, fields[key].mean()),
            loc="left", fontsize=8.6)
    for ax in (axs[0, 0], axs[0, 1]):
        ax.tick_params(labelbottom=False)
    for ax in (axs[0, 1], axs[1, 1]):
        ax.tick_params(labelleft=False)
    for ax in (axs[0, 0], axs[1, 0]):
        ax.set_yticklabels(["{:.1f}°N".format(t) for t in yt])
    for ax in (axs[1, 0], axs[1, 1]):
        ax.set_xticklabels(["{:.1f}°E".format(t) for t in xt])

    # 地名(仅真值面板;白描边保证深色场上可读)
    a = axs[0, 0]
    for lon, lat, name, dx, ha in [
        (114.06, 22.55, "Shenzhen", 0.03, "left"),
        (114.17, 22.30, "Hong Kong", 0.03, "left"),
        (113.54, 22.19, "Macau", 0.03, "left"),
    ]:
        a.text(lon + dx, lat, name, fontsize=6.8, color=fs.INK, ha=ha,
               va="center", zorder=6, path_effects=fs.halo())
    a.text(113.70, 22.40, "Pearl River\nestuary", fontsize=6.8, style="italic",
           color=fs.INK2, ha="center", va="center", zorder=6, linespacing=1.1,
           path_effects=fs.halo())
    a.text(114.44, 22.50, "Mirs Bay", fontsize=6.8, style="italic",
           color=fs.INK2, ha="center", va="center", zorder=6,
           path_effects=fs.halo())

    # 华南区位小图(嵌在粗场面板左下角的海域空白处)
    # (粗网格坐标可分离误差 ~2e-5°,用 1D 代表坐标避免 pcolormesh 非单调告警)
    axi = axs[0, 1].inset_axes([0.02, 0.02, 0.52, 0.46])
    lat_c = st["xlat_coarse"][:, 0]
    lon_c = st["xlong_coarse"][0, :]
    wc = st["water_coarse"]
    axi.pcolormesh(lon_c, lat_c, (wc < 0.5).astype(float),
                   cmap=fs.land_sea_cmap(), vmin=0, vmax=1,
                   shading="nearest", rasterized=True)
    axi.contour(lon_c, lat_c, wc, levels=[0.5], colors=["#5A5A5A"],
                linewidths=0.5)
    ring_lon = np.concatenate([xlon[0, :], xlon[:, -1], xlon[-1, ::-1],
                               xlon[::-1, 0]])
    ring_lat = np.concatenate([xlat[0, :], xlat[:, -1], xlat[-1, ::-1],
                               xlat[::-1, 0]])
    axi.plot(ring_lon, ring_lat, color=fs.PURPLE, lw=1.3, zorder=5,
             path_effects=fs.halo(2.6))
    axi.plot([113.26], [23.13], marker="o", ms=1.8, color=fs.ORANGE,
             mec="white", mew=0.4, zorder=6)
    axi.text(112.95, 23.13, "Guangzhou", fontsize=6.2, color=fs.INK,
             ha="right", va="center", zorder=6, path_effects=fs.halo(1.6))
    axi.text(109.85, 19.35, "Hainan", fontsize=6.2, color=fs.INK,
             ha="center", va="center", zorder=6, path_effects=fs.halo(1.6))
    axi.text(116.0, 19.2, "South China Sea", fontsize=6.2, style="italic",
             color=fs.INK2, ha="center", va="center", zorder=6,
             path_effects=fs.halo(1.6))
    axi.annotate("study area", xy=(114.1, 22.5), xytext=(117.2, 20.8),
                 fontsize=6.2, color=fs.INK, ha="center", va="center",
                 zorder=7, path_effects=fs.halo(1.6),
                 arrowprops=dict(arrowstyle="-", color=fs.INK2, lw=0.6,
                                 shrinkA=1, shrinkB=2))
    axi.set_xlim(float(lon_c.min()), float(lon_c.max()))
    axi.set_ylim(float(lat_c.min()), float(lat_c.max()))
    axi.set_aspect(1.0 / np.cos(np.deg2rad(22.0)))
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("#B9B9B9")
        sp.set_linewidth(0.6)
    axi.set_facecolor("white")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Wind speed (m/s)", fontsize=8.5)
    cb.ax.tick_params(labelsize=fs.DS["tick"])
    cb.outline.set_visible(False)
    fig.suptitle("Strong-wind case {} - {} m AGL".format(stamp_lab, int(agl)),
                 fontsize=11, color=fs.INK)
    save(fig, "fig_case{}_{:.0f}m.png".format(case, agl))


if __name__ == "__main__":
    fig_spectra()
    fig_split()
    fig_p1_ablation()
    fig_p1_direct_residual()
    fig_p2_losses()
    fig_p2_divergence()
    fig_p3_levels()
    fig_p3_bars()
    fig_p3_spatial()
    fig_cases(0, 100.0)
    print("all figures done")
