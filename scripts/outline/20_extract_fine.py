# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · T0.2 细端(d04,1 km)抽取

原生 Arakawa C 网格,不插质量点;存储层:质量层/U/V 取 index 0..39(z_agl < 2 km),
W 取界面 0..40。每 10 分钟一个 npz:
  f_{scheme}_{YYYYMMDDTHHMMSS}.npz
  keys: f_u (40,99,121), f_v (40,100,120), f_w (41,99,120),
        f_u10 (99,120), f_v10 (99,120)

运行(pytorch-gpu 环境):
  python scripts/outline/20_extract_fine.py --scheme both --workers 4
"""
import argparse
import multiprocessing
import os
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline_common import (N_IFACE, N_MASS, OUT_FINE, SCHEMES,
                            domain_files, ensure_dir, read_times, stamp)

SHAPE_U = (N_MASS, 99, 121)
SHAPE_V = (N_MASS, 100, 120)
SHAPE_W = (N_IFACE, 99, 120)


def process_file(task):
    path, out_dir, scheme = task
    created = 0
    skipped = 0
    with Dataset(path) as nc:
        times = read_times(nc)
        u = np.asarray(nc.variables['U'][:, :N_MASS], dtype=np.float32)
        v = np.asarray(nc.variables['V'][:, :N_MASS], dtype=np.float32)
        w = np.asarray(nc.variables['W'][:, :N_IFACE], dtype=np.float32)
        u10 = np.asarray(nc.variables['U10'][:], dtype=np.float32)
        v10 = np.asarray(nc.variables['V10'][:], dtype=np.float32)
    assert u.shape[1:] == SHAPE_U, "U shape {}".format(u.shape)
    assert v.shape[1:] == SHAPE_V, "V shape {}".format(v.shape)
    assert w.shape[1:] == SHAPE_W, "W shape {}".format(w.shape)
    assert u10.shape[1:] == (99, 120), "U10 shape {}".format(u10.shape)
    for t, dt in enumerate(times):
        out_path = os.path.join(out_dir, "f_{}_{}.npz".format(scheme, stamp(dt)))
        if os.path.exists(out_path):
            skipped += 1
            continue
        np.savez_compressed(out_path, f_u=u[t], f_v=v[t], f_w=w[t],
                            f_u10=u10[t], f_v10=v10[t])
        created += 1
    return created, skipped


def main():
    parser = argparse.ArgumentParser(description="d04 细端抽取(原生 C 网格)")
    parser.add_argument("--scheme", default="both", choices=["myj", "ysu", "both"])
    parser.add_argument("--limit", type=int, default=0, help="每个 scheme 只处理前 N 个文件")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out_dir", default=OUT_FINE)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    schemes = SCHEMES if args.scheme == "both" else [args.scheme]
    for scheme in schemes:
        files = domain_files(scheme, "d04")
        if args.limit > 0:
            files = files[:args.limit]
        print("scheme {}: {} files -> {}".format(scheme, len(files), args.out_dir))
        tasks = [(f, args.out_dir, scheme) for f in files]
        n_created = 0
        n_skipped = 0
        if args.workers > 1 and len(tasks) > 1:
            pool = multiprocessing.Pool(args.workers)
            for i, (c, s) in enumerate(pool.imap_unordered(process_file, tasks, chunksize=1)):
                n_created += c
                n_skipped += s
                if (i + 1) % 50 == 0:
                    print("  {}/{}".format(i + 1, len(tasks)))
            pool.close()
            pool.join()
        else:
            for i, t in enumerate(tasks):
                c, s = process_file(t)
                n_created += c
                n_skipped += s
                if (i + 1) % 50 == 0:
                    print("  {}/{}".format(i + 1, len(tasks)))
        print("scheme {}: created {} skipped {}".format(scheme, n_created, n_skipped))
    print("done")


if __name__ == "__main__":
    main()
