#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SZ npz 全量统计:biases/scales/昼夜样本数(用于训练配置)。

说明:prepare_wind_data_3d_sz.py 的统计只覆盖本次 created 的样本,
需要全量重算时用本脚本。

用法(登录节点,纯 CPU,约 10-15 分钟):
  /fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/compute_stats_sz.py \
      --data_dir prepare_npz_wind_3d_sz
"""
from __future__ import print_function

import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SZ npz 全量统计")
    parser.add_argument("--data_dir", default="prepare_npz_wind_3d_sz")
    args = parser.parse_args()

    npz_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    print("文件数: {}".format(len(npz_paths)))
    if not npz_paths:
        print("没有 npz 文件,退出。")
        return

    with np.load(npz_paths[0]) as data:
        keys = sorted(data.keys())

    sums = {k: 0.0 for k in keys}
    sq_sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    n_day = 0
    n_night = 0
    n_other = 0

    for i, npz_path in enumerate(npz_paths):
        with np.load(npz_path) as data:
            for k in keys:
                v = data[k].astype(np.float64)
                sums[k] += np.nansum(v)
                sq_sums[k] += np.nansum(v * v)
                counts[k] += np.isfinite(v).sum()
            sw = float(np.mean(data["swdown"]))
            if sw > 50.0:
                n_day += 1
            elif sw < 5.0:
                n_night += 1
            else:
                n_other += 1
        if (i + 1) % 1000 == 0:
            print("进度: {}/{}".format(i + 1, len(npz_paths)))

    print("")
    print("--- 数据集统计结果 (用于更新 config) ---")
    print("biases:")
    for k in keys:
        print("    {}: {:.6f}".format(k, sums[k] / max(counts[k], 1)))
    print("scales:")
    for k in keys:
        mean = sums[k] / max(counts[k], 1)
        var = sq_sums[k] / max(counts[k], 1) - mean * mean
        std = np.sqrt(max(var, 0.0))
        if std == 0.0:
            std = 1.0
            print("    {}: {:.6f}  # WARNING: std=0".format(k, std))
        else:
            print("    {}: {:.6f}".format(k, std))
    print("")
    print("昼夜样本数: day={} night={} twilight/other={}".format(
        n_day, n_night, n_other))
    print("完成。")


if __name__ == "__main__":
    main()
