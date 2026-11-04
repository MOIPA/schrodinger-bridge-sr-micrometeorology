# -*- coding: utf-8 -*-
"""阶段 4 T4.3:27→1 与 3→1 上的缩减版输入消融配置生成器。

大纲 T4.3 的两个待检假设
  ① 辅助变量的边际价值随降尺度跨度增大而增大 —— 阶段 1/3 在 9→1 上发现
     加 θ 廓线 / PH / MOST / 通量 组都是负收益;若在 27→1 上它们变正收益,
     说明"跨度越大、辅助变量越值钱"。
  ② 粗端 W 的价值随输入分辨率变化(预期 3 km 有用、27 km 无用)—— 阶段 1 在
     9→1 上加粗端 W 是 +0.121 的负收益;需要 27 km 与 3 km 两端的证据。

四个臂(基准臂已训好,不重复提交)
  p4_27to1_reg    基准 V*(+ 已有,1.0909 / +28.1%)      ← 参照,不重训
  p4_27to1_aux    V* + coarse_theta + coarse_ph + coarse_most + coarse_flux   → 假设 ①
  p4_27to1_w      V* + coarse_w                                              → 假设 ②(27 km 端)
  sq3to1_w        V* + coarse_w(3 km 粗端)                                  → 假设 ②(3 km 端)

通道数(必须与 in_channel 一致,否则建网/加载会当场报错):
  V* = coarse_wind_uv 46 + coarse_zagl 23 + coarse_logz0 1 + fine_geom 24 + coszen 1 = 95
  aux 组 = theta 23 + ph 24 + most 3 + flux 3 = 53
  coarse_w = 24(WL = range(0, 24))
  in_channel = 95 + 通道数 + 72(state)

用法:
    python scripts/outline/42_gen_phase4_ablate_configs.py            # 写到 configs/深圳/phase4/
    python scripts/outline/42_gen_phase4_ablate_configs.py --dry_run  # 只打印差异
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG_DIR = os.path.join(REPO, "configs", "深圳")

BASE_27 = os.path.join(CFG_DIR, "phase4", "config_wind_canvas_p4_27to1_reg.yml")
BASE_3 = os.path.join(REPO, "sidequest_scale3x", "configs", "phase3_arch",
                      "config_wind_canvas_sq3to1_reg.yml")

V_STAR = ["coarse_wind_uv", "coarse_zagl", "coarse_logz0", "fine_geom", "coszen"]
AUX = ["coarse_theta", "coarse_ph", "coarse_most", "coarse_flux"]
W_GROUP = ["coarse_w"]

N_CHAN = {"coarse_wind_uv": 46, "coarse_zagl": 23, "coarse_logz0": 1, "fine_geom": 24,
          "coszen": 1, "time_enc": 4, "coarse_w": 24, "coarse_theta": 23,
          "coarse_ph": 24, "coarse_most": 3, "coarse_flux": 3,
          "fine_logz0": 1, "fine_urban": 1, "fine_waterveg": 2,
          "geom_diff_zagl": 23, "geom_diff_hgt": 1, "coords": 2}
N_STATE = 72

ARMS = [
    ("p4_27to1_aux", BASE_27, V_STAR + AUX, "假设①:27→1 上补辅助变量组"),
    ("p4_27to1_w", BASE_27, V_STAR + W_GROUP, "假设②:27→1 上加粗端 W"),
    ("sq3to1_w", BASE_3, V_STAR + W_GROUP, "假设②:3→1 上加粗端 W"),
]


def replace_group_block(text):
    """把 input_groups 列表整段替换(保留其余文本与缩进风格)。"""
    pat = re.compile(r"^(\s*)input_groups:\n(?:\1\s*- .*\n)+", re.M)

    def sub(groups):
        def fn(m):
            ind = m.group(1)
            return ind + "input_groups:\n" + "".join("{}- {}\n".format(ind, g)
                                                      for g in groups)
        return fn
    return pat, sub


def set_groups(text, groups):
    pat, sub = replace_group_block(text)
    new, n = pat.subn(sub(groups), text)
    if n != 1:
        raise SystemExit("input_groups 块匹配 {} 处(应为 1)".format(n))
    return new


def set_scalar(text, key, value):
    pat = re.compile(r"^(\s*){}(\s*:).*$".format(re.escape(key)), re.M)
    new, n = pat.subn(lambda m: "{}{}{} {}".format(m.group(1), key, m.group(2), value), text)
    if n != 1:
        raise SystemExit("键 {} 匹配 {} 处(应为 1)".format(key, n))
    return new


def in_channel_for(groups):
    return sum(N_CHAN[g] for g in groups) + N_STATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out_dir", default=os.path.join(CFG_DIR, "phase4"))
    ap.add_argument("--out_dir_sq", default=os.path.join(REPO, "sidequest_scale3x",
                                                          "configs", "phase3_arch"))
    args = ap.parse_args()

    for tag, base, groups, why in ARMS:
        with open(base) as f:
            text = f.read()
        ic = in_channel_for(groups)
        out = set_groups(text, groups)
        out = set_scalar(out, "in_channel", ic)
        header = ("# {}\n# 由 scripts/outline/42_gen_phase4_ablate_configs.py 生成"
                  "(基准 {};input_groups 与 in_channel 同步改动)\n").format(why, os.path.basename(base))
        if args.dry_run:
            print("[dry] {}  in_channel={}  groups={}".format(tag, ic, groups))
            continue
        out_dir = args.out_dir if tag.startswith("p4_") else args.out_dir_sq
        path = os.path.join(out_dir, "config_wind_canvas_{}.yml".format(tag))
        with open(path, "w") as f:
            f.write("# " + header.lstrip("# ") + out)
        print("[ok] {}  in_channel={}  -> {}".format(tag, ic, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
