# -*- coding: utf-8 -*-
"""
d03 数据集全量统计:遍历 prepare_npz_wind_3d_sz_d03/ 全部 1488 个 npz
(不是"本次进程生成的"列表——修复 created 列表不完整导致的统计偏差),
计算 biases/scales 与昼夜样本数,完整打印。

用法(服务器,登录节点即可):
  /fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/compute_stats_d03.py
"""
import glob
import os

import numpy as np

NPZ_DIR = "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d_sz_d03"


def main():
    paths = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    print("npz 总数: {}".format(len(paths)))
    if not paths:
        return

    with np.load(paths[0]) as data:
        keys = sorted(data.keys())
    print("变量数: {}".format(len(keys)))

    sums = {k: 0.0 for k in keys}
    sq_sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    n_day = n_night = n_other = 0

    for i, p in enumerate(paths):
        with np.load(p) as data:
            for k in keys:
                v = data[k].astype(np.float64)
                sums[k] += np.nansum(v)
                sq_sums[k] += np.nansum(v * v)
                counts[k] += np.isfinite(v).sum()
            sw = float(np.mean(data['swdown']))
            if sw > 50.0:
                n_day += 1
            elif sw < 5.0:
                n_night += 1
            else:
                n_other += 1
        if (i + 1) % 200 == 0:
            print("  进度: {}/{}".format(i + 1, len(paths)))

    print("\n--- biases (全量 {}) ---".format(len(paths)))
    for k in keys:
        print("    {}: {:.6f}".format(k, sums[k] / max(counts[k], 1)))
    print("\n--- scales (全量 {}) ---".format(len(paths)))
    for k in keys:
        mean = sums[k] / max(counts[k], 1)
        var = sq_sums[k] / max(counts[k], 1) - mean * mean
        std = np.sqrt(max(var, 0.0))
        if std == 0.0:
            std = 1.0
            print("    {}: {:.6f}  # WARNING std=0".format(k, std))
        else:
            print("    {}: {:.6f}".format(k, std))

    print("\n昼夜样本数: day={} night={} twilight/other={}".format(
        n_day, n_night, n_other))


if __name__ == "__main__":
    main()
