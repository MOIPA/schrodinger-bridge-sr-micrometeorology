# -*- coding: utf-8 -*-
"""外传 · 任意域的细端抽取(原生 Arakawa C 网格)

与主仓 scripts/outline/20_extract_fine.py 的区别:形状不再写死 d04(99,120),
从文件自身推断,因此可用于 d02(27→9 组的细端)与 d03(9→3 组的细端)。

存储口径与主仓一致:质量层/U/V 取 index 0..39,W 取界面 0..40。
W 在 WRF 中只在垂直方向交错(W: (t, nz+1, ny, nx)),水平与质量点相同。

用法:
  python extract_fine.py --domain d02 --scheme both --workers 4 \
      --out_dir prepare_npz_outline_fine_d02
"""
import argparse
import multiprocessing
import os
import sys

import numpy as np
from netCDF4 import Dataset

# 复用主仓已验证的 Times 解析(WRF 的 Times 变量属性口径与常规不同)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "outline"))
from outline_common import read_times, stamp  # noqa: E402

N_MASS = 40
N_IFACE = 41
WRF_BASE = "/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen"
SCHEME_DIRS = {"myj": "meso_202007_myj", "ysu": "meso_202007_ysu"}


def domain_files(scheme, dom):
    import glob
    return sorted(glob.glob(os.path.join(WRF_BASE, SCHEME_DIRS[scheme], "wrfout_" + dom + "_*")))


def infer_shapes(nc):
    th = "THM" if "THM" in nc.variables else "T"
    ny, nx = nc.variables[th].shape[-2], nc.variables[th].shape[-1]
    return {
        "u": (N_MASS, ny, nx + 1),
        "v": (N_MASS, ny + 1, nx),
        "w": (N_IFACE, ny, nx),
        "u10": (ny, nx),
        "v10": (ny, nx),
    }


def process_file(task):
    path, out_dir, scheme, stride = task
    created = skipped = 0
    with Dataset(path) as nc:
        times = read_times(nc)
        shapes = infer_shapes(nc)
        u = np.asarray(nc.variables["U"][:, :N_MASS], dtype=np.float32)
        v = np.asarray(nc.variables["V"][:, :N_MASS], dtype=np.float32)
        w = np.asarray(nc.variables["W"][:, :N_IFACE], dtype=np.float32)
        u10 = np.asarray(nc.variables["U10"][:], dtype=np.float32)
        v10 = np.asarray(nc.variables["V10"][:], dtype=np.float32)
    assert u.shape[1:] == shapes["u"], "U shape %s vs %s" % (u.shape[1:], shapes["u"])
    assert v.shape[1:] == shapes["v"], "V shape %s vs %s" % (v.shape[1:], shapes["v"])
    assert w.shape[1:] == shapes["w"], "W shape %s vs %s" % (w.shape[1:], shapes["w"])
    assert u10.shape[1:] == shapes["u10"], "U10 shape %s" % (u10.shape[1:],)
    for t, dt in enumerate(times):
        if stride > 1 and (t % stride):
            continue
        out_path = os.path.join(out_dir, "f_{}_{}.npz".format(scheme, stamp(dt)))
        if os.path.exists(out_path):
            skipped += 1
            continue
        np.savez_compressed(out_path, f_u=u[t], f_v=v[t], f_w=w[t], f_u10=u10[t], f_v10=v10[t])
        created += 1
    return created, skipped


def main():
    ap = argparse.ArgumentParser(description="外传:任意域细端抽取")
    ap.add_argument("--domain", default="d04")
    ap.add_argument("--scheme", default="both", choices=["myj", "ysu", "both"])
    ap.add_argument("--limit", type=int, default=0, help="每个 scheme 只处理前 N 个文件")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1, help="时间抽样步长(1=全取)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    schemes = ["myj", "ysu"] if args.scheme == "both" else [args.scheme]
    out_dir = args.out_dir or "prepare_npz_outline_fine_%s" % args.domain
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for sc in schemes:
        files = domain_files(sc, args.domain)
        if args.limit:
            files = files[:args.limit]
        tasks += [(f, out_dir, sc, args.stride) for f in files]
    print("domain=%s schemes=%s files=%d out=%s" % (args.domain, schemes, len(tasks), out_dir), flush=True)

    total_c = total_s = 0
    with multiprocessing.Pool(args.workers) as pool:
        for c, s in pool.imap_unordered(process_file, tasks):
            total_c += c
            total_s += s
    print("done: created %d, skipped %d" % (total_c, total_s), flush=True)


if __name__ == "__main__":
    main()
