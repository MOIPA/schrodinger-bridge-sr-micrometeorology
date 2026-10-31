# -*- coding: utf-8 -*-
"""外传 · 三段 3× 的数据体检:各域网格尺寸、已抽取目录的体积。

用法:/fs00/software/anaconda/3/bin/python sidequest_scale3x/scripts/probe_dims.py
"""
import glob
import os

from netCDF4 import Dataset

WRF_BASE = "/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_myj"
TARGET_VARS = ["U", "V", "W", "U10", "V10", "PHB", "HGT"]


def dir_size(path):
    total = 0
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
            n += 1
    return total, n


def main():
    for dom in ["d01", "d02", "d03", "d04"]:
        files = sorted(glob.glob(os.path.join(WRF_BASE, "wrfout_%s_*" % dom)))
        if not files:
            print("%s: 无文件" % dom)
            continue
        with Dataset(files[0]) as nc:
            shapes = {}
            for v in TARGET_VARS:
                if v in nc.variables:
                    shapes[v] = nc.variables[v].shape
            nlev = shapes.get("U", (0,))[1] if "U" in shapes else 0
        # 单时次 npz 估算:40 层存储口径
        u = shapes.get("U", (0, 0, 0, 0))
        ny, nx = u[2], u[3] - 1
        est = (40 * ny * (nx + 1) + 40 * (ny + 1) * nx + 41 * ny * nx) * 4 / 1e6
        print("%s  文件数 %d  nlev=%d  mass=(%d,%d)  单时次 npz≈%.0f MB  全月≈%.1f GB"
              % (dom, len(files), nlev, ny, nx, est, est * 744 / 1000))
        print("      shapes:", shapes)

    print("\n--- 已抽取目录 ---")
    for d in sorted(glob.glob("prepare_npz_*")):
        sz, n = dir_size(d)
        print("  %-42s %8.2f GB  %6d files" % (d, sz / 1e9, n))


if __name__ == "__main__":
    main()
