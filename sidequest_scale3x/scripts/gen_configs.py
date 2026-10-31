# -*- coding: utf-8 -*-
"""外传 · 三组训练配置生成(27→9 / 9→3 / 3→1)

从阶段 3 回归臂基座派生,只改数据路径、画布尺寸与数据版本号,
保证"架构/超参逐字相同",否则无法把差异归因到尺度。

用法:
  python sidequest_scale3x/scripts/gen_configs.py [--dry_run]
"""
import argparse
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "configs/深圳/phase3_arch/config_wind_canvas_p3_arch_reg.yml")
OUT_DIR = os.path.join(REPO, "sidequest_scale3x/configs/phase3_arch")

# 组 -> (tag, 细端域, 画布形状, coarse_dir, statics_dir, dl_data_ver)
GROUPS = {
    "sq27to9_reg": dict(
        tag="sq27to9_reg", fine="d02", hr_shape=[120, 150],
        coarse_dir="prepare_npz_outline_coarse_d01",
        statics_dir="prepare_npz_outline_static_sq27to9",
        ver="sq27to9"),
    "sq9to3_reg": dict(
        tag="sq9to3_reg", fine="d03", hr_shape=[120, 150],
        coarse_dir="prepare_npz_outline_coarse",
        statics_dir="prepare_npz_outline_static_sq9to3",
        ver="sq9to3"),
    "sq3to1_reg": dict(
        tag="sq3to1_reg", fine="d04", hr_shape=[99, 120],
        coarse_dir="prepare_npz_outline_coarse_d03",
        statics_dir="prepare_npz_outline_static_sq3to1",
        ver="sq3to1"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    with open(BASE) as f:
        base = yaml.safe_load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, g in GROUPS.items():
        cfg = yaml.safe_load(yaml.safe_dump(base))  # 深拷贝
        cfg["data"]["coarse_dir"] = os.path.join(REPO, g["coarse_dir"])
        cfg["data"]["statics_dir"] = os.path.join(REPO, g["statics_dir"])
        cfg["data"]["split_manifest"] = os.path.join(REPO, g["statics_dir"], "split.json")
        cfg["data"]["normalize_json"] = None
        cfg["data"]["hr_data_shape"] = list(g["hr_shape"])
        cfg["loader"]["dl_data_ver"] = g["ver"]

        path = os.path.join(OUT_DIR, "config_wind_canvas_%s.yml" % tag)
        header = (
            "# 外传 · 三尺度 3× 对比 —— %s\n"
            "# 基座: configs/深圳/phase3_arch/config_wind_canvas_p3_arch_reg.yml(逐键相同)\n"
            "# 变更: data.coarse_dir / statics_dir / split_manifest / hr_data_shape + loader.dl_data_ver\n"
            "# 目的: 固定架构与超参, 只变尺度, 使差异可归因于'灰度区间'\n" % tag)
        if args.dry_run:
            print("[DRY] %s  coarse=%s fine=%s hr=%s ver=%s"
                  % (path, g["coarse_dir"], g["fine"], g["hr_shape"], g["ver"]))
            continue
        with open(path, "w") as f:
            f.write(header)
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print("wrote", path)


if __name__ == "__main__":
    main()
