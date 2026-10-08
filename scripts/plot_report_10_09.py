# -*- coding: utf-8 -*-
"""生成 2026-10-09 组会汇报配图(阶段 0-3)。
本地运行: python scripts/plot_report_10_09.py
输出: 组会汇报/2026-10-09/fig_*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "FZLanTingHeiS-R-GB"
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "组会汇报", "2026-10-09")
os.makedirs(OUT, exist_ok=True)

# 统一配色
C_TRAIN = "#4C72B0"
C_VALID = "#DD8452"
C_TEST = "#C44E52"
COLORS4 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ============ 图 1:阶段 0 四档位动能谱 ============
def fig_spectra():
    d = json.load(open(os.path.join(ROOT, "results/outline/truth_diagnostics.json")))
    sp = d["spectra"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, tag in enumerate(["d01", "d02", "d03", "d04"]):
        s = sp[tag]
        k = np.array(s["k"]); P = np.array(s["P"])
        m = (k > 0) & (P > 0)
        ax.loglog(k[m], P[m], "-", color=COLORS4[i], lw=1.8, label=f"{tag} ({s['wavelength_7dx_km']:.0f} km 有效分辨率)")
        k7 = s["k_7dx"]
        if i == 3:
            ax.axvline(k7, color="gray", ls="--", lw=1)
            ax.text(k7, P.max() * 0.5, "7Δx", color="gray", fontsize=9, ha="left", va="top")
    ax.set_xlabel("波数 k (无量纲,对数)")
    ax.set_ylabel("动能谱 P(k)(对数)")
    ax.set_title("四档位动能谱与有效分辨率(2020-07-15T12Z)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    save(fig, "fig_p0_spectra.png")


# ============ 图 2:阶段 0 数据划分时间线 ============
def fig_split():
    d = json.load(open(os.path.join(ROOT, "results/outline/split.json")))
    blocks = d["blocks"]
    cmap = {"train": C_TRAIN, "valid": C_VALID, "test": C_TEST}
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for b in blocks:
        d0 = int(b["start"][-2:])
        d1 = int(b["end"][-2:])
        y = 0
        ax.barh(y, d1 - d0 + 1, left=d0 - 0.5, height=0.55,
                color=cmap[b["split"]], edgecolor="white", lw=1)
        ax.text((d0 + d1) / 2, 0, str(b["idx"]), ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(-0.6, 0.8)
    ax.set_yticks([])
    ax.set_xlabel("2020 年 7 月(日)")
    ax.set_title("按天气过程的块级数据划分(10 块:训练 71% / 验证 10% / 测试 19%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in ["train", "valid", "test"]]
    ax.legend(handles, ["训练", "验证", "测试"], ncol=3, loc="upper right", fontsize=9, framealpha=0.9)
    save(fig, "fig_p0_split.png")


# ============ 图 3:阶段 1 输入消融(残差配置)============
def fig_p1_ablation():
    import csv
    rows = []
    with open(os.path.join(ROOT, "results/phase1/ranking_phase1r.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows = rows[::-1]  # 让最优在上
    label_map = {
        "r_t14_noenc_cos": "cos(SZA) 替代时间编码",
        "r_t15_z0": "+ 细端 log z0",
        "t16_residual": "基准(残差输出)",
        "r_t13_theta": "+ θ 廓线",
        "r_t13_most": "+ MOST 组",
        "r_t12_zagldiff": "+ 逐层高度差场",
        "r_t15_z0_urban": "+ urban 分数",
        "r_t15_z0_urban_wv": "+ water/veg 分数",
        "r_t17_coords": "+ 地理坐标",
        "r_t12_hgtdiff": "+ 二维地形差",
        "r_t13_w": "+ 粗端 W",
        "r_t13_ph": "+ 逐时 PH",
        "baseline_y0_regrid": "y0 粗端重网格(参照)",
        "r_t14_cos": "+ cos SZA(保留时间编码)",
        "baseline_bicubic": "双三次插值(参照)",
        "r_t13_mostflux": "+ 强度组",
    }
    labs, deltas, los, his, sigs = [], [], [], [], []
    for r in rows:
        tag = r["tag"]
        labs.append(label_map.get(tag, tag))
        deltas.append(float(r["delta"]))
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        los.append(lo if lo == lo else deltas[-1])
        his.append(hi if hi == hi else deltas[-1])
        sigs.append(float(r["significant"]) > 0.5)
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(labs))
    for i in range(len(labs)):
        c = "#C44E52" if sigs[i] else "#55A868"
        ax.errorbar(deltas[i], y[i],
                    xerr=[[deltas[i] - los[i]], [his[i] - deltas[i]]],
                    fmt="o", color=c, ecolor=c, capsize=3, ms=6)
    ax.axvline(0, color="black", lw=0.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#55A868", label="与基准无显著差异"),
        Line2D([], [], marker="o", ls="", color="#C44E52", label="显著变差"),
    ], fontsize=9, loc="lower right")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel("相对基准的主指标变化 ΔRMSE (m/s),误差棒为 95% 置信区间")
    ax.set_title("阶段 1 输入变量消融(13 个残差配置,基准 0.963 m/s)")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "fig_p1_ablation.png")


# ============ 图 4:阶段 1 直接输出 vs 残差输出 ============
def fig_p1_direct_residual():
    groups = ["基准", "+MOST 组", "+粗端 W", "+强度组", "+θ 廓线"]
    direct = [1.311, 1.947, 1.946, 1.761, 1.139]
    residual = [0.963, 0.973, 1.084, 1.206, 0.972]
    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, direct, w, color="#C0C0C0", label="直接输出 1 km 场")
    b2 = ax.bar(x + w / 2, residual, w, color=C_TRAIN, label="残差输出(1 km 场 - 插值粗场)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                    f"{r.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("主指标 RMSE (m/s)")
    ax.set_title("输出形式对比:残差输出在全部配置下大幅更优")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig_p1_direct_residual.png")


# ============ 图 5:阶段 2 损失与约束对比 ============
def fig_p2_losses():
    items = [("极值损失", 15.7), ("散度约束(预热)", 8.6), ("L2 数据项", 6.1),
             ("L1 + 谱损失", 5.2), ("涡度结构匹配", 4.2)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], color="#C44E52", alpha=0.85)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.2, i, f"+{v}%", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in items])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("相对纯 L1 基线的主指标劣化幅度 (%)")
    ax.set_title("阶段 2 损失与物理约束对比(全部以精度为代价,愈往上危害愈大)")
    ax.set_xlim(0, 18)
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "fig_p2_losses.png")


# ============ 图 6:阶段 3 逐层误差廓线 ============
def fig_p3_levels():
    arms = [("p2_l1r2_lr2e4", "模式层监督(基线)", COLORS4[0]),
            ("p3_agl", "纯高度层监督", COLORS4[1]),
            ("p3_agllw", "模式层 + 等效低层加权", COLORS4[2]),
            ("p3_joint", "模式层 + 联合监督", COLORS4[3])]
    fig, ax = plt.subplots(figsize=(7, 5))
    for tag, lab, c in arms:
        d = json.load(open(os.path.join(ROOT, f"results/phase3/{tag}_summary.json")))
        levels = d["agl_targets"]
        rmse = d["rmse_vec_all_per_level"]
        ax.plot(levels, rmse, "-o", color=c, lw=1.8, ms=4, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000])
    ax.set_xticklabels(["10", "30", "50", "70", "100", "150", "200", "300", "500", "700", "1000"], fontsize=9)
    ax.set_xlabel("离地高度 (m,对数轴)")
    ax.set_ylabel("风矢量 RMSE (m/s)")
    ax.set_title("阶段 3 逐层误差廓线(测试集 144 小时,myj)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    save(fig, "fig_p3_levels.png")


# ============ 图 7:阶段 3 主指标与切变对比 ============
def fig_p3_bars():
    arms = ["模式层监督\n(基线)", "纯高度层\n监督", "模式层 + 等效\n低层加权", "模式层 +\n联合监督"]
    main = [0.9574, 0.9383, 0.8904, 0.8697]
    shear = [0.00885, 0.00798, 0.00741, 0.00704]
    main_pct = ["—", "-2.0%", "-7.0%", "-9.2%"]
    shear_pct = ["—", "-9.9%", "-16.4%", "-20.5%"]
    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, vals, pcts, title, ylab in [
        (axes[0], main, main_pct, "主指标(10-500 m 平均风矢量 RMSE)", "RMSE (m/s)"),
        (axes[1], shear, shear_pct, "层间切变误差", "切变 RMSE (m/s)"),
    ]:
        bars = ax.bar(x, vals, 0.55, color=[COLORS4[0]] + ["#C8C8C8"] * 3)
        bars[3].set_color(COLORS4[3])
        bars[2].set_color(COLORS4[2])
        bars[1].set_color(COLORS4[1])
        for i, (r, p) in enumerate(zip(bars, pcts)):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() * 1.01,
                    p, ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.12)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("阶段 3 监督空间对比(单种子口径;切变三种子稳健)", fontsize=11)
    save(fig, "fig_p3_bars.png")


# ============ 图 8:阶段 2 散度残差廓线 ============
def fig_p2_divergence():
    base = json.load(open(os.path.join(ROOT, "results/phase2/r_t14_noenc_cos_diag.json")))
    dv = json.load(open(os.path.join(ROOT, "results/phase2/p2_div_mid_warm_diag.json")))
    tau = np.array(base["phys_div_tau"])
    lv = np.arange(len(tau))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    series = [
        ("truth", "d04 真值", base["divergence"]["truth"], "black"),
        ("y0", "粗端重网格(y0)", base["divergence"]["y0"], "#999999"),
        ("base", "基线预测(纯 L1)", base["divergence"]["pred"], COLORS4[0]),
        ("div", "散度约束预测", dv["divergence"]["pred"], COLORS4[3]),
    ]
    for _, lab, d, c in series:
        ax.plot(lv, np.array(d["p95_abs"]) / tau, "-o", color=c, lw=1.6, ms=3, label=lab)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.text(len(tau) - 1, 1.05, "定标阈值 tau", color="gray", fontsize=8, ha="right")
    ax.set_xlabel("训练层序号(0 为最低层)")
    ax.set_ylabel("散度残差 P95 / tau")
    ax.set_title("阶段 2 散度残差逐层廓线(测试集 144 帧)")
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    save(fig, "fig_p2_divergence.png")


# ============ 图 9:阶段 3 空间分组 ============
def fig_p3_spatial():
    d = json.load(open(os.path.join(ROOT, "results/phase3/spatial_summary.json")))
    groups = ["steep_city", "steep_rural", "flat_city", "flat_rural"]
    glab = ["陡地形+城市", "陡地形+郊区", "平坦+城市", "平坦+郊区"]
    arm_lab = {"p2_l1r2_lr2e4": "模式层监督(基线)", "p3_agl": "纯高度层监督",
               "p3_agllw": "低层加权", "p3_joint": "联合监督"}
    base = {g["tag"]: g["group_rmse_vec"] for g in d}
    x = np.arange(len(groups))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = ["p2_l1r2_lr2e4", "p3_agl", "p3_agllw", "p3_joint"]
    for i, tag in enumerate(order):
        vals = [base[tag][g] for g in groups]
        off = (i - 1.5) * w
        bars = ax.bar(x + off, vals, w, color=COLORS4[i], label=arm_lab[tag])
        if i > 0:
            for xi, (r, g) in enumerate(zip(bars, groups)):
                basev = base["p2_l1r2_lr2e4"][g]
                pct = (base[tag][g] - basev) / basev * 100
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                        f"{pct:+.0f}%", ha="center", fontsize=7.5, color=COLORS4[i])
    ax.set_xticks(x); ax.set_xticklabels(glab)
    ax.set_ylabel("低层 10-100 m 风矢量 RMSE (m/s)")
    ax.set_title("阶段 3 空间分组误差(标注为相对基线的变化)")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig_p3_spatial.png")


# ============ 图 10:个例风场对比(需 65 号快照回传后运行)============
def fig_cases(case=0, agl=100.0):
    """个例四联图:真值 / 粗场 / 基线 / 联合监督 的风速场。
    依赖 results/report_10_09/cases.npz(ops/queue/65_case_snapshots.sh 回传)。"""
    path = os.path.join(ROOT, "results/report_10_09/cases.npz")
    if not os.path.exists(path):
        print("skip fig_cases: 缺", path)
        return
    d = np.load(path)
    agls = list(d["agl"])
    li = agls.index(agl)
    srcs = [("truth", "d04 真值"), ("y0", "粗端重网格(y0)"),
            ("pred_base", "基线预测"), ("pred_joint", "联合监督预测")]
    fields = {}
    for key, _ in srcs:
        u = d[f"case{case}_{key}_u"][li]
        v = d[f"case{case}_{key}_v"][li]
        fields[key] = np.sqrt(u ** 2 + v ** 2)
    vmax = max(float(f.max()) for f in fields.values())
    stamp = str(d[f"case{case}_stamp"][0])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, lab) in zip(axes.ravel(), srcs):
        im = ax.imshow(fields[key], vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"{lab}  域均值 {fields[key].mean():.2f} m/s", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="风速 (m/s)")
    fig.suptitle("个例 {}({} m 高度)风场对比".format(stamp, agl), fontsize=12)
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
    fig_cases()          # 快照回传后自动生效,缺文件时跳过
    print("all figures done")
