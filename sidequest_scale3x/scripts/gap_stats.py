# -*- coding: utf-8 -*-
"""
外传 · 三尺度 3× 对比 —— "灰度区间"量化脚本

老师说法:"每个阶段的灰度区间不一样"。本脚本把这句定性话变成四个数。

做法(不需要读粗端文件,纯几何自洽):
  对每个 3× 嵌套段(27→9、9→3、3→1),取**细端**(d02/d03/d04)的风场,
  用 3×3 块平均构造"粗端该看到的样子" C,再最近邻放大回 C_up,
  残差 R = F − C_up 就是"3 倍放大要凭空长出来的东西"。

对每个量输出(逐层 + 汇总):
  1. sigma_fine / sigma_coarse   细节能长多少
  2. R2 = 1 - var(R)/var(F)      粗端能解释细端的比例(越小=任务越难)
  3. kurtosis(R)、R 的 1-99% 区间   残差分布的尖峰与"灰度区间"
  4. 功率谱斜率 beta(F) 与 beta(R)  场的空间粗糙度

用法(服务器 base anaconda python,需 netCDF4):
  python sidequest_scale3x/scripts/gap_stats.py \
      --pairs d02:d03:d04 --n_times 6 --nlev 20 \
      --out_json sidequest_scale3x/results/gap_stats.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from netCDF4 import Dataset

WRF_BASE = "/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen"
SCHEME_DIRS = {"myj": "meso_202007_myj", "ysu": "meso_202007_ysu"}


def domain_files(scheme, dom):
    return sorted(glob.glob(os.path.join(WRF_BASE, SCHEME_DIRS[scheme], "wrfout_%s_*" % dom)))


def read_times(nc):
    from netCDF4 import num2date
    tv = nc.variables["Times"]
    return [num2date(tv[i], tv.units, getattr(tv, "calendar", "standard")) for i in range(tv.shape[0])]


def mass_winds(nc, nlev):
    """把 U/V 从 C 网格面推到质量点(内部),取最低 nlev 层。"""
    u = np.asarray(nc.variables["U"][:, :nlev], dtype=np.float64)
    v = np.asarray(nc.variables["V"][:, :nlev], dtype=np.float64)
    # U: (t, lev, ny, nx+1) -> 质量点 x 方向取相邻两界面平均(去掉最外一列)
    u_m = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
    # V: (t, lev, ny+1, nx) -> y 方向平均
    v_m = 0.5 * (v[:, :, :-1, :] + v[:, :, 1:, :])
    return u_m, v_m


def block_mean(f, k=3):
    """(...，ny, nx) -> (..., ny//k, nx//k),要求整除。"""
    *lead, ny, nx = f.shape
    ny2, nx2 = ny // k, nx // k
    f = f[..., :ny2 * k, :nx2 * k]
    return f.reshape(*lead, ny2, k, nx2, k).mean(axis=(-3, -1))


def radial_psd(f):
    """2D 功率谱 -> 径向平均;返回 (k, S) 用于拟合斜率。"""
    f = f - f.mean()
    ny, nx = f.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    F = np.fft.fftshift(np.fft.fft2(f * win))
    P = np.abs(F) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(nx))[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    nb = min(ny, nx) // 2
    edges = np.linspace(0, 0.5, nb + 1)
    which = np.digitize(kr.ravel(), edges) - 1
    S = np.zeros(nb)
    kk = np.zeros(nb)
    for i in range(nb):
        m = which == i
        if m.any():
            S[i] = P.ravel()[m].mean()
            kk[i] = kr.ravel()[m].mean()
    good = (kk > 1.0 / min(ny, nx) * 3) & (S > 0)
    return kk[good], S[good]


def fit_slope(k, S, kmin=None, kmax=0.25):
    # k 的单位是 cycles/pixel;kmin 默认取 3 个格点波长(1/3),kmax 取 Nyquist 的一半
    if kmin is None:
        kmin = 1.0 / 32.0
    m = (k >= kmin) & (k <= kmax) & (S > 0)
    if m.sum() < 4:
        return float("nan")
    return float(-np.polyfit(np.log(k[m]), np.log(S[m]), 1)[0])


def kurtosis(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    s = x.std()
    if s == 0:
        return float("nan")
    return float(((x - x.mean()) ** 4).mean() / s ** 4 - 3.0)


def analyze_pair(scheme, fine_dom, n_times, nlev, k=3):
    files = domain_files(scheme, fine_dom)
    if not files:
        raise SystemExit("找不到 %s 的文件" % fine_dom)
    idx = np.linspace(0, len(files) - 1, n_times).astype(int)
    per_lev = {key: [] for key in
               ["sigma_fine", "sigma_coarse", "sigma_res", "r2", "kurt_res",
                "p1_99_fine", "p1_99_res", "beta_fine", "beta_res"]}
    for i in idx:
        with Dataset(files[i]) as nc:
            u, v = mass_winds(nc, nlev)
            # 取一个够大的内窗做谱,避免边界
            ny, nx = u.shape[1], u.shape[2]
            hy, hx = min(256, ny // 2 * 2), min(256, nx // 2 * 2)
            y0, x0 = ny // 2 - hy // 2, nx // 2 - hx // 2
            u = u[:, y0:y0 + hy, x0:x0 + hx]
            v = v[:, y0:y0 + hy, x0:x0 + hx]
            spd_all = np.sqrt(u ** 2 + v ** 2)                  # (t, lev, ny, nx)
        for ti in range(spd_all.shape[0]):
          spd = spd_all[ti]                                      # (lev, ny, nx)
          c_all = block_mean(spd, k)                             # (lev, ny2, nx2)
          c_up_all = np.repeat(np.repeat(c_all, k, axis=1), k, axis=2)
          for L in range(spd.shape[0]):
            f = spd[L][:c_up_all.shape[1], :c_up_all.shape[2]]   # 裁到 3 的整数倍
            c_up = c_up_all[L]
            r = f - c_up
            sf, sc, sr = f.std(), c_all[L].std(), r.std()
            per_lev["sigma_fine"].append(float(sf))
            per_lev["sigma_coarse"].append(float(sc))
            per_lev["sigma_res"].append(float(sr))
            per_lev["r2"].append(float(1.0 - (sr ** 2) / (sf ** 2)) if sf > 0 else float("nan"))
            per_lev["kurt_res"].append(kurtosis(r))
            per_lev["p1_99_fine"].append(float(np.percentile(f, 99) - np.percentile(f, 1)))
            per_lev["p1_99_res"].append(float(np.percentile(r, 99) - np.percentile(r, 1)))
            kf, Sf = radial_psd(f)
            kr, Sr = radial_psd(r)
            per_lev["beta_fine"].append(fit_slope(kf, Sf))
            per_lev["beta_res"].append(fit_slope(kr, Sr))
    out = {}
    for key, vals in per_lev.items():
        a = np.asarray(vals, dtype=np.float64).reshape(-1, nlev)
        out[key] = np.nanmean(a, axis=0).tolist()
    out["levels"] = list(range(nlev))
    out["n_times"] = n_times
    out["fine_domain"] = fine_dom
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="myj", choices=["myj", "ysu"])
    ap.add_argument("--pairs", default="d02,d03,d04",
                    help="逗号分隔的细端域,每段 = 该域与它上一级粗域之比(3×)")
    ap.add_argument("--n_times", type=int, default=6)
    ap.add_argument("--nlev", type=int, default=20, help="最低 nlev 层(默认 20 层≈1 km 以下)")
    ap.add_argument("--out_json", default="gap_stats.json")
    args = ap.parse_args()

    result = {}
    for dom in args.pairs.split(","):
        dom = dom.strip()
        print(">>> gap %s ..." % dom, flush=True)
        result[dom] = analyze_pair(args.scheme, dom, args.n_times, args.nlev)
        r = result[dom]
        print("    R2 %.3f | sf/sc %.2f | kurt_res %.2f | beta_fine %.2f"
              % (np.nanmean(r["r2"]),
                 np.nanmean(np.asarray(r["sigma_fine"]) / np.asarray(r["sigma_coarse"])),
                 np.nanmean(r["kurt_res"]), np.nanmean(r["beta_fine"])), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
