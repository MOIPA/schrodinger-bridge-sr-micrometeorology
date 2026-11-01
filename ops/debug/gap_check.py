#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gap_check.py — 核查 gap_real 的重网格是否"配得准"。

担心的事:如果 statics 的重网格索引把粗端映射到了错误位置,粗端插值后会和细端异常相似
(R2 虚高、σ比≈1),于是"3→1 的缺口比 9→3 还小"这种反直觉结论可能是假的。

判别办法(同一时次、同一层):
  * corr(F, C_up) —— 粗端插值后与细端的相关系数;
  * 对照:把 C_up **循环平移** 若干个格点后再算 corr(配错位置时,平移会显著提升相关);
  * var(C_up)/var(F) —— 插值是否把方差抹平。
正常配准时 corr 高、平移后下降;配错时相反。

用法: python ops/debug/gap_check.py --stages 27to9,9to3,3to1,27to1
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402
from sidequest_scale3x.scripts.gap_real import STAGES, destagger_speed, stamp_of  # noqa: E402


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="27to9,9to3,3to1,27to1")
    ap.add_argument("--lev", type=int, default=0, help="取第几层(0 = 最低层)")
    ap.add_argument("--scheme", default="myj")
    args = ap.parse_args()

    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        cfg = STAGES[name]
        st = CanvasStatics(os.path.join(REPO, cfg["statics"]))
        fine = {stamp_of(p): p for p in glob.glob(os.path.join(REPO, cfg["fine"], "f_*.npz"))}
        coarse = {stamp_of(p): p for p in glob.glob(os.path.join(REPO, cfg["coarse"], "c_*.npz"))}
        stamps = sorted(set(fine) & set(coarse))[:3]
        if not stamps:
            print("[跳过] {} 无共同时次".format(name))
            continue
        print("== {} ==".format(name))
        for s in stamps:
            with np.load(coarse[s]) as g:
                cu = np.asarray(g["c_u"][args.lev:args.lev + 1], dtype=np.float32)
                cv = np.asarray(g["c_v"][args.lev:args.lev + 1], dtype=np.float32)
            with np.load(fine[s]) as g:
                fu = np.asarray(g["f_u"][args.lev:args.lev + 1], dtype=np.float32)
                fv = np.asarray(g["f_v"][args.lev:args.lev + 1], dtype=np.float32)
            spd_f = destagger_speed(fu, fv)[0]
            spd_c = destagger_speed(st.regrid_field(cu, "u"), st.regrid_field(cv, "v"))[0]
            base = corr(spd_f, spd_c)
            shifts = []
            for dy, dx in ((1, 0), (0, 1), (3, 0), (0, 3), (5, 5), (10, 10)):
                shifts.append(corr(spd_f, np.roll(np.roll(spd_c, dy, 0), dx, 1)))
            print("  {} corr={:.4f} varC/varF={:.3f} 平移corr={} ".format(
                s, base, float(np.var(spd_c) / np.var(spd_f)),
                " ".join("{:.3f}".format(v) for v in shifts)))


if __name__ == "__main__":
    main()
