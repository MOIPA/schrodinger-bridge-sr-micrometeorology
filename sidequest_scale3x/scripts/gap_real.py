# -*- coding: utf-8 -*-
"""gap_real.py — 用**真实粗端**量化每一级超分的"缺口",替代块平均代理。

外传-1 用的是"细端场 3×3 块平均当作粗端该看到的样子",它便宜、但不等于真实的嵌套粗端
(块平均是各向同性的抽样,WRF 嵌套粗端是另一套物理场 + 双线性重网格)。本脚本改用
项目自己的重网格权重(statics.npz 的 regrid_idx_/regrid_w_)把**真粗端 npz** 搬到细端,
再和细端真值比:

  F      = 细端风速(项目抽取的 fine npz)
  C_up   = 真粗端风速经 statics 重网格到细端之后再算的风速
  R      = F − C_up
  R2     = 1 − var(R)/var(F)          "粗端能解释细端的比例"
  ratio  = std(F)/std(C_up)           "细节能长多少"
  kurt   = kurtosis(R)                "缺口是重尾还是近高斯"

四个尺度对都能算:27→9、9→3、3→1,以及阶段4 的 **27→1**(statics_d01)。

用法(服务器 wind3d 环境,需 torch):
  python sidequest_scale3x/scripts/gap_real.py --n_times 8 --nlev 23 \
      --out_json sidequest_scale3x/results/gap_real.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402

STAGES = {
    "27to9": dict(statics="prepare_npz_outline_static_sq27to9",
                  coarse="prepare_npz_outline_coarse_d01",
                  fine="prepare_npz_outline_fine_d02", nlev_coarse=40),
    "9to3": dict(statics="prepare_npz_outline_static_sq9to3",
                 coarse="prepare_npz_outline_coarse",
                 fine="prepare_npz_outline_fine_d03", nlev_coarse=40),
    "3to1": dict(statics="prepare_npz_outline_static_sq3to1",
                 coarse="prepare_npz_outline_coarse_d03",
                 fine="prepare_npz_outline_fine", nlev_coarse=40),
    # 阶段 4:一步 27→1(9 倍跨度),statics 由 10_extract_statics --coarse_file d01 生成
    "27to1": dict(statics="prepare_npz_outline_static_d01",
                  coarse="prepare_npz_outline_coarse_d01",
                  fine="prepare_npz_outline_fine", nlev_coarse=40),
}


def stamp_of(path):
    base = os.path.basename(path)
    return base.split("_", 2)[-1][:-4]


def destagger_speed(u, v):
    um = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vm = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    return np.sqrt(um ** 2 + vm ** 2)


def one_stage(name, cfg, n_times, nlev, scheme="myj"):
    st = CanvasStatics(os.path.join(REPO, cfg["statics"]))
    fine_files = {stamp_of(p): p for p in glob.glob(os.path.join(REPO, cfg["fine"], "f_*.npz"))}
    coarse_files = {stamp_of(p): p for p in glob.glob(os.path.join(REPO, cfg["coarse"], "c_*.npz"))}
    stamps = sorted(set(fine_files) & set(coarse_files))
    if not stamps:
        raise SystemExit("[{}] 细端与粗端没有共同的时次".format(name))
    step = max(1, len(stamps) // n_times)
    stamps = stamps[::step][:n_times]

    acc_f, acc_c, acc_r = [], [], []
    for s in stamps:
        with np.load(coarse_files[s]) as g:
            cu = np.asarray(g["c_u"][:nlev], dtype=np.float32)
            cv = np.asarray(g["c_v"][:nlev], dtype=np.float32)
        with np.load(fine_files[s]) as g:
            fu = np.asarray(g["f_u"][:nlev], dtype=np.float32)
            fv = np.asarray(g["f_v"][:nlev], dtype=np.float32)
        # 真粗端 -> 细端原生交错位置 -> 质量点 -> 风速
        cu_up = st.regrid_field(cu, "u")
        cv_up = st.regrid_field(cv, "v")
        spd_c = destagger_speed(cu_up, cv_up)
        spd_f = destagger_speed(fu, fv)
        acc_f.append(spd_f.ravel())
        acc_c.append(spd_c.ravel())
        acc_r.append((spd_f - spd_c).ravel())

    f = np.concatenate(acc_f)
    c = np.concatenate(acc_c)
    r = f - c
    var_f = float(np.var(f))
    r2 = 1.0 - float(np.var(r)) / var_f if var_f > 0 else float("nan")
    ratio = float(np.std(f) / np.std(c)) if np.std(c) > 0 else float("nan")
    m = float(np.mean(r))
    sd = float(np.std(r))
    kurt = float(np.mean(((r - m) / sd) ** 4)) if sd > 0 else float("nan")
    return {
        "stage": name,
        "n_times": len(stamps),
        "nlev": nlev,
        "n_points": int(f.size),
        "r2": r2,
        "std_ratio": ratio,
        "resid_mean": m,
        "resid_std": sd,
        "resid_kurtosis": kurt,
        "fine_mean_speed": float(np.mean(f)),
        "coarse_mean_speed": float(np.mean(c)),
        "stamps": stamps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="27to9,9to3,3to1,27to1")
    ap.add_argument("--n_times", type=int, default=8)
    ap.add_argument("--nlev", type=int, default=23, help="最低 nlev 层(与训练的目标层一致)")
    ap.add_argument("--scheme", default="myj")
    ap.add_argument("--out_json", default="sidequest_scale3x/results/gap_real.json")
    args = ap.parse_args()

    rows = []
    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if name not in STAGES:
            print("[跳过] 未知尺度对 {}".format(name))
            continue
        cfg = STAGES[name]
        if not os.path.isdir(os.path.join(REPO, cfg["statics"])):
            print("[跳过] {} 缺 statics:{}".format(name, cfg["statics"]))
            continue
        row = one_stage(name, cfg, args.n_times, args.nlev, args.scheme)
        rows.append(row)
        print("{:<6s} n={:3d} 时次 R2={:.3f} σ细/σ粗={:.3f} 残差均值={:+.3f} "
              "残差σ={:.3f} 峰度={:.2f}".format(
                  name, row["n_times"], row["r2"], row["std_ratio"],
                  row["resid_mean"], row["resid_std"], row["resid_kurtosis"]))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "scheme": args.scheme,
                       "nlev": args.nlev}, f, ensure_ascii=False, indent=2)
        print("[json] {}".format(args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
