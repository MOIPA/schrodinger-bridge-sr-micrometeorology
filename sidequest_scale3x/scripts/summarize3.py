# -*- coding: utf-8 -*-
"""外传·三尺度 3× 的评估汇总(纯 CPU,读 perhour npz)。

每个 tag 输出:
  - 主指标 RMSE(AGL 10–500 m 池化,与阶段 3 同口径)
  - 插值基线 y0 的同一指标
  - 相对插值增益 (%)  ← "普通插值法对比"就是这一列
  - 切变 RMSE(8 对相邻主层,若 npz 有 acc_sh)
  - 逐层增益(AGL 11 层)

用法:
  python sidequest_scale3x/scripts/summarize3.py --dir sidequest_scale3x/results
"""
import argparse
import glob
import json
import os

import numpy as np

AGL = [10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
MAIN = [i for i, z in enumerate(AGL) if z <= 500]


def pooled(arr, levels):
    a = arr[..., levels, :, :]
    n = a[..., 0].sum()
    se = a[..., 1].sum()
    return float(np.sqrt(se / max(n, 1)))


def shear_rmse(acc_sh):
    if acc_sh is None or acc_sh.size == 0:
        return None
    se = acc_sh[..., 1].sum()
    n = acc_sh[..., 0].sum()
    return float(np.sqrt(se / max(n, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sidequest_scale3x/results")
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--md_out", default=None)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*_perhour.npz"))):
        tag = os.path.basename(path)[: -len("_perhour.npz")]
        d = np.load(path)
        acc = d["acc_agl"]
        y0 = d["acc_y0"] if "acc_y0" in d else None
        sh = d["acc_sh"] if "acc_sh" in d else None
        row = {
            "tag": tag,
            "n_hours": int(len(d["hours"])),
            "main_rmse": pooled(acc, MAIN),
            "y0_rmse": pooled(y0, MAIN) if y0 is not None else None,
            "shear": shear_rmse(sh),
            "per_level": {
                str(AGL[i]): {"rmse": pooled(acc, [i]),
                              "y0": pooled(y0, [i]) if y0 is not None else None}
                for i in range(len(AGL))
            },
        }
        if row["y0_rmse"]:
            row["gain_pct"] = (1.0 - row["main_rmse"] / row["y0_rmse"]) * 100.0
        else:
            row["gain_pct"] = None
        rows.append(row)

    lines = ["| 模型 | 小时数 | 主指标 RMSE | 插值 y0 RMSE | 相对插值增益 | 切变 RMSE |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {} | {} | {:.4f} | {} | {} | {} |".format(
            r["tag"], r["n_hours"], r["main_rmse"],
            "{:.4f}".format(r["y0_rmse"]) if r["y0_rmse"] else "—",
            "{:+.1f}%".format(r["gain_pct"]) if r["gain_pct"] is not None else "—",
            "{:.5f}".format(r["shear"]) if r["shear"] else "—"))
    md = "\n".join(lines)
    print(md)

    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md + "\n")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
