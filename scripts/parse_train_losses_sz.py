# -*- coding: utf-8 -*-
"""
解析深圳训练日志中的 loss 曲线,输出 results/sz_train_losses.json。

在服务器上运行(wind3d 或默认 python 均可,纯标准库):
  python scripts/parse_train_losses_sz.py
"""
import glob
import json
import os
import re

PATTERN = re.compile(r"train error: avg loss = ([\d.]+)")

LOG_TO_MODEL = {
    "sz_baseline": "baseline",
    "sz_lrcond": "lrcond",
    "sz_day": "day",
    "sz_night": "night",
    "sz_pinn": "pinn",
    "sz_d03base": "d03_baseline",
    "sz_d03lrcond": "d03_lrcond",
    "sz_d03pinn": "d03_pinn",
    "sz_d03lrpin": "d03_lrcond_pinn",
}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    out = {}
    for log_path in sorted(glob.glob(os.path.join(BASE, "logs", "sz_*.out"))):
        name = os.path.basename(log_path).split("_")[0:2]
        key = "_".join(name)
        model = LOG_TO_MODEL.get(key)
        if model is None:
            continue
        losses = []
        with open(log_path) as f:
            for line in f:
                m = PATTERN.search(line)
                if m:
                    losses.append(float(m.group(1)))
        if losses:
            out[model] = losses
            print("{}: {} epochs, last={:.4f}".format(model, len(losses), losses[-1]))

    out_path = os.path.join(BASE, "results", "sz_train_losses.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved {}".format(out_path))


if __name__ == "__main__":
    main()
