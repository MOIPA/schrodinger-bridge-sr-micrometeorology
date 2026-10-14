# -*- coding: utf-8 -*-
"""报告图设计系统(2026-10-14 v1;对齐 组会汇报/模范图片/ 的 fig_common 体系)。

- 调色板:NJU 紫 #6E0F6D 主色 / #D89000 强调(须带可见标签)/
  灰阶 #7A7A7A(参照)、#4A4A4A(轴线)、#D9D9D9(网格)/
  淡紫底 #E8D7E8、#F2EEF2(带底、区位底)
- 字号 token DS 与模范图一致:轴 9 / 刻度 8 / 注释 7.5 / 数值 8 / 面板 10
- 图内文字全部英文(2026-10-14 用户口径);报告正文仍为中文
- 配色已过 dataviz 校验:紫色序数梯 R4 全过;PURPLE/ORANGE 对 CVD ΔE 40;
  橙对白底对比 2.66:1 -> 橙色承载数据处必须带图例文字或直标(relief 规则)

用法:
    import figstyle as fs
    fs.apply()
    ...
    fs.save(fig, "xxx.png")
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
import matplotlib.patheffects as _pe  # noqa: E402
import numpy as np  # noqa: E402

# ---- 调色板 -----------------------------------------------------------------
PURPLE = "#6E0F6D"        # 主色:本文方法 / 被采纳配置 / 关键实体
PURPLE_TINT = "#E8D7E8"   # 主色淡底(置信带等)
ORANGE = "#D89000"        # 强调:被检视的变体 / 显著变差(须带标签)
GREY = "#7A7A7A"          # 参照 / 探索性
GREY_LIGHT = "#ADADAD"    # 次浅灰(被弃方案、平坦区)
REF = "#4A4A4A"           # 参考线 / 轴线
GRID = "#D9D9D9"          # 网格
TINT = "#F2EEF2"          # 极浅底带(高亮区)
INK = "#2B2B2B"           # 主文字
INK2 = "#52514E"          # 次级文字
COAST = "#3F3F3F"         # 海岸线

# 紫色序数梯(浅->深,= 监督集成度递增 / 分辨率递增类序数语义)
R4 = ("#C9A9CE", "#A66FB0", "#8A4D96", "#6E0F6D")

# ---- 字号 token(pt,与模范图一致)-------------------------------------------
DS = {"axis": 9, "tick": 8, "annot": 7.5, "value": 8, "panel": 10,
      "title": 9.5, "legend": 8}

_FONT_CANDIDATES = ("Helvetica Neue", "Arial", "DejaVu Sans")


def apply():
    """全局 rcParams。每张图开始前调用一次。"""
    avail = {f.name for f in _fm.fontManager.ttflist}
    fams = [n for n in _FONT_CANDIDATES if n in avail] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fams,
        "font.size": DS["annot"],
        "axes.titlesize": DS["title"],
        "axes.titlecolor": INK,
        "axes.labelsize": DS["axis"],
        "axes.labelcolor": INK2,
        "axes.edgecolor": REF,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.labelsize": DS["tick"],
        "ytick.labelsize": DS["tick"],
        "xtick.color": REF,
        "ytick.color": REF,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": DS["legend"],
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 300,
    })


def save(fig, path):
    """统一导出:png 300dpi、紧凑留白,然后关闭。"""
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", path)


def seq_purple():
    """单色序数/连续量色带(风速等幅度场):近白 -> 深紫。"""
    base = plt.get_cmap("Purples")
    return LinearSegmentedColormap.from_list(
        "seq_purple", base(np.linspace(0.03, 0.94, 256)))


def land_sea_cmap():
    """区位图水陆二值色:海=白、陆=淡紫底。"""
    return ListedColormap(["#FFFFFF", "#EFE9F0"])


def halo(lw=2.2):
    """白描边文字效果(深色场上的地名/注记)。"""
    return [_pe.withStroke(linewidth=lw, foreground="white")]


def pct(v):
    """带符号百分数(一位小数;0 附近用占位符)。"""
    return "—" if v is None else "{:+.1f}%".format(v)
