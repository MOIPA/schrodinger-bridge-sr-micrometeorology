#!/usr/bin/env python3
"""级联超分用配置生成:把某一级配置的 coarse_dir 换掉,其余不动。

产出 `configs/<cfg_root>/config_wind_canvas_<tag>.yml`,配合同名 checkpoint 软链即可直接
喂给 `scripts/outline/84_ensemble_eval.py --tags <tag>`。

用法:
  python3 sidequest_scale3x/scripts/gen_cascade_cfg.py \
      --src configs/phase3_arch/config_wind_canvas_sq9to3_reg.yml \
      --tag sq9to3_casc --coarse_dir <绝对路径>
"""
import argparse
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--coarse_dir", required=True)
    ap.add_argument("--out_dir", default="sidequest_scale3x/configs/phase3_arch")
    args = ap.parse_args()

    s = open(args.src).read()
    new, n = re.subn(r"(?m)^(\s*coarse_dir:\s*).*$",
                     lambda m: m.group(1) + args.coarse_dir, s)
    if n != 1:
        raise SystemExit("coarse_dir 替换失败(找到 {} 处)".format(n))
    out = os.path.join(args.out_dir, "config_wind_canvas_{}.yml".format(args.tag))
    os.makedirs(args.out_dir, exist_ok=True)
    open(out, "w").write("# 级联超分:" + args.tag + "(仅改 coarse_dir)\n" + new)
    print(out)


if __name__ == "__main__":
    main()
