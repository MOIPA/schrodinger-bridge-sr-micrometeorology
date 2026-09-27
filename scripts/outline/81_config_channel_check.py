# -*- coding: utf-8 -*-
"""阶段 1 配置通道校验:逐个 load_config,断言 model.in/out_channel 与通道函数一致(秒级)。

运行(pytorch-gpu 或 wind3d 环境,仓库根目录,需先跑 85 生成配置):
  python scripts/outline/81_config_channel_check.py
"""
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
PATTERN = os.path.join(ROOT, "configs", "深圳", "phase1", "*.yml")


def main():
    paths = sorted(glob.glob(PATTERN))
    assert paths, "没有找到阶段 1 配置(先跑 85_gen_phase1_configs.py)"
    for p in paths:
        cfg = load_config(EXPERIMENT, p)
        n_out = len(build_target_channel_names(cfg.data.target_levels, cfg.data.include_w))
        n_cond = len(build_input_channel_names(cfg.data.input_groups,
                                               cfg.data.target_levels, cfg.data.include_w))
        assert cfg.model.out_channel == n_out, os.path.basename(p) + " out_channel 不一致"
        assert cfg.model.in_channel == n_out + n_cond, os.path.basename(p) + " in_channel 不一致"
        print("{:<42} out={:>3} cond={:>3} in={:>3}".format(
            os.path.basename(p), n_out, n_cond, n_out + n_cond))
    print("CHANNEL CHECK OK ({} 个配置)".format(len(paths)))


if __name__ == "__main__":
    main()
