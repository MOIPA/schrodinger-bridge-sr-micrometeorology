# -*- coding: utf-8 -*-
"""配置校验:逐个 load_config,断言通道数与通道函数一致,并校验阶段 2 物理参数(秒级)。

通道口径:model.out_channel == len(build_target_channel_names(target_levels, include_w)),
          model.in_channel == out_channel + len(build_input_channel_names(input_groups, ...))。
phys 口径(阶段 2,字段缺省时跳过):
  si.phys_scale 存在     -> 长度 == model.out_channel(通道序同 y);
  si.phys_dz / phys_div_tau 存在 -> 长度 == len(data.target_levels);
  si.extreme_levels 存在 -> 全部落在 [0, len(target_levels));
  si.divergence_weight>0 -> phys_scale / phys_dz / phys_div_tau 三项齐全且 phys_dx>0。

运行(pytorch-gpu 或 wind3d 环境,仓库根目录):
  python scripts/outline/81_config_channel_check.py                     # 默认查阶段 1
  python scripts/outline/81_config_channel_check.py --config_dir "configs/深圳/phase1r"
  python scripts/outline/81_config_channel_check.py --config_dir "configs/深圳/phase2"
  python scripts/outline/81_config_channel_check.py --config_dir <dir> --glob "*.yml"
"""
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_data.dataset_wind_canvas import (  # noqa: E402
    build_input_channel_names,
    build_target_channel_names,
)

EXPERIMENT = "ExperimentSchrodingerBridgeWindCanvas"
DEFAULT_DIR = os.path.join(ROOT, "configs", "深圳", "phase1")


def check_phys(cfg, name):
    """阶段 2 物理参数静态校验;返回人类可读摘要(无物理项时返回 '')。"""
    si = cfg.si
    n_lev = len(cfg.data.target_levels)
    tags = []

    def need(cond, msg):
        assert cond, "{}: {}".format(name, msg)

    if si.phys_scale is not None:
        need(len(si.phys_scale) == cfg.model.out_channel,
             "phys_scale 长度 {} != out_channel {}".format(
                 len(si.phys_scale), cfg.model.out_channel))
        tags.append('scale={}'.format(len(si.phys_scale)))
    if si.phys_dz is not None:
        need(len(si.phys_dz) == n_lev,
             "phys_dz 长度 {} != len(target_levels) {}".format(
                 len(si.phys_dz), n_lev))
        tags.append('dz={}'.format(len(si.phys_dz)))
    if si.phys_div_tau is not None:
        need(len(si.phys_div_tau) == n_lev,
             "phys_div_tau 长度 {} != len(target_levels) {}".format(
                 len(si.phys_div_tau), n_lev))
        need(all(t > 0 for t in si.phys_div_tau),
             "phys_div_tau 存在非正值(hinge 阈值必须>0)")
        tags.append('tau={}'.format(len(si.phys_div_tau)))
    if si.extreme_levels is not None:
        lv = [int(v) for v in si.extreme_levels]
        need(len(lv) > 0 and all(0 <= v < n_lev for v in lv),
             "extreme_levels {} 非法(需非空且落在 [0,{}))".format(lv, n_lev))
        tags.append('ext_lv={}'.format(len(lv)))
    if si.divergence_weight > 0:
        need(si.phys_scale is not None and si.phys_dz is not None
             and si.phys_div_tau is not None,
             "divergence_weight>0 时 phys_scale/phys_dz/phys_div_tau 必须齐全")
        need(si.phys_dx > 0, "divergence_weight>0 时 phys_dx 必须>0")
        tags.append('div={:g}'.format(si.divergence_weight))
    if si.vorticity_weight > 0:
        need(si.phys_scale is not None, "vorticity_weight>0 时需要 phys_scale")
        tags.append('vort={:g}'.format(si.vorticity_weight))
    if si.spectral_weight > 0:
        need(si.phys_scale is not None, "spectral_weight>0 时需要 phys_scale")
        tags.append('spec={:g}'.format(si.spectral_weight))
    if si.extreme_weight > 0:
        need(si.phys_scale is not None, "extreme_weight>0 时需要 phys_scale")
        tags.append('ext={:g}'.format(si.extreme_weight))
    return ' '.join(tags)


def main():
    ap = argparse.ArgumentParser(description="配置通道/物理参数一致性校验")
    ap.add_argument("--config_dir", default=DEFAULT_DIR,
                    help="配置目录(相对仓库根或绝对路径),默认 configs/深圳/phase1")
    ap.add_argument("--glob", default="*.yml", help="目录内文件名通配,默认 *.yml")
    args = ap.parse_args()

    conf_dir = args.config_dir if os.path.isabs(args.config_dir) \
        else os.path.join(ROOT, args.config_dir)
    paths = sorted(glob.glob(os.path.join(conf_dir, args.glob)))
    assert paths, "没有找到配置: {} (先跑 85/86 生成)".format(
        os.path.join(conf_dir, args.glob))
    for p in paths:
        name = os.path.basename(p)
        cfg = load_config(EXPERIMENT, p)
        n_out = len(build_target_channel_names(cfg.data.target_levels, cfg.data.include_w))
        n_cond = len(build_input_channel_names(cfg.data.input_groups,
                                               cfg.data.target_levels, cfg.data.include_w))
        assert cfg.model.out_channel == n_out, name + " out_channel 不一致"
        assert cfg.model.in_channel == n_out + n_cond, name + " in_channel 不一致"
        phys = check_phys(cfg, name)
        print("{:<44} out={:>3} cond={:>3} in={:>3}  {}".format(
            name, n_out, n_cond, n_out + n_cond, phys))
    print("CHANNEL CHECK OK ({} 个配置, {})".format(len(paths), conf_dir))


if __name__ == "__main__":
    main()
