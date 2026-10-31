# -*- coding: utf-8 -*-
"""阶段 4 (导师大纲) · T4.1 跨分辨率档位配置生成

把阶段 3 的胜出配置(回归臂 p3_arch_reg)复制成两个新档位:
  p4_27to1_reg  : 粗端 d01(27 km) -> d04(1 km)
  p4_3to1_reg   : 粗端 d03(3 km)  -> d04(1 km)
只改 data.coarse_dir / data.statics_dir / data.split_manifest 三行,
model / si / train / loader 逐键与基座相同 —— 这样"技巧-跨度曲线"比的是跨度,不是配方。

前提(服务器上先跑 ops/queue/65_phase4_prep.sh):
  prepare_npz_outline_coarse_d01/  prepare_npz_outline_static_d01/
  prepare_npz_outline_coarse_d03/  prepare_npz_outline_static_d03/

用法:
  python scripts/outline/41_gen_phase4_configs.py                 # 生成到 configs/深圳/phase4
  python scripts/outline/41_gen_phase4_configs.py --dry_run       # 只打印将写入的内容摘要
"""
import argparse
import os
import sys

SERVER_ROOT = "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology"

# tag -> (域, 显示名)
ARMS = {
    "p4_27to1_reg": ("d01", "27 km -> 1 km"),
    "p4_3to1_reg": ("d03", "3 km -> 1 km"),
}


def patch_paths(text, arm):
    dom = ARMS[arm][0]
    coarse = "{}/prepare_npz_outline_coarse_{}".format(SERVER_ROOT, dom)
    static = "{}/prepare_npz_outline_static_{}".format(SERVER_ROOT, dom)
    lines = []
    hit = {"coarse_dir": 0, "statics_dir": 0, "split_manifest": 0}
    for ln in text.splitlines():
        key = ln.strip().split(":")[0] if ":" in ln else ""
        if key == "coarse_dir":
            ln = "  coarse_dir: " + coarse
            hit["coarse_dir"] += 1
        elif key == "statics_dir":
            ln = "  statics_dir: " + static
            hit["statics_dir"] += 1
        elif key == "split_manifest":
            ln = "  split_manifest: " + static + "/split.json"
            hit["split_manifest"] += 1
        lines.append(ln)
    for k, v in hit.items():
        if v != 1:
            raise SystemExit("基座里 {} 出现 {} 次(应为 1 次),拒绝生成".format(k, v))
    return "\n".join(lines) + "\n"


def main():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="生成阶段 4 跨分辨率配置")
    parser.add_argument("--base", default=os.path.join(
        here, "configs/深圳/phase3_arch/config_wind_canvas_p3_arch_reg.yml"))
    parser.add_argument("--out_dir", default=os.path.join(here, "configs/深圳/phase4"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--data_root", default=SERVER_ROOT,
                        help="服务器数据根(含 prepare_npz_outline_*)")
    args = parser.parse_args()

    if not os.path.isfile(args.base):
        raise SystemExit("找不到基座配置: " + args.base)
    base_text = open(args.base, encoding="utf-8").read()
    os.makedirs(args.out_dir, exist_ok=True)

    globals()["SERVER_ROOT"] = args.data_root.rstrip("/")

    for arm, (dom, label) in ARMS.items():
        text = patch_paths(base_text, arm)
        head = (
            "# {} — 阶段 4 T4.1 跨分辨率({})\n"
            "# 基座: {} (阶段 3 回归臂,73 层/23 层/同容量)\n"
            "# 变更: 仅 data.coarse_dir / data.statics_dir / data.split_manifest 三行 -> {}\n"
            "# 依赖: ops/queue/65_phase4_prep.sh 已在该机生成 prepare_npz_outline_*_{}\n"
        ).format(arm, label, os.path.relpath(args.base, here), dom, dom)
        out = os.path.join(args.out_dir, "config_wind_canvas_{}.yml".format(arm))
        if args.dry_run:
            print("[dry] {} -> {}(coarse={})".format(arm, out, dom))
            continue
        with open(out, "w", encoding="utf-8") as f:
            f.write(head + text)
        print("wrote {}".format(out))


if __name__ == "__main__":
    main()
