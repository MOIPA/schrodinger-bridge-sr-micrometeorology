# -*- coding: utf-8 -*-
"""
阶段 0 canvas 管线冒烟:分块划分 -> Dataset -> 一个 batch -> UNet 前向 + SI 损失。

用法(pytorch-gpu 环境,在仓库根目录):
  python scripts/smoke_wind_canvas.py --config_path configs/深圳/config_wind_canvas_smoke.yml \
      [--device cpu] [--backward]
"""
import argparse
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_data.dataloader import make_dataloaders_and_samplers  # noqa: E402
from src.dl_model.model_maker import make_model  # noqa: E402
from src.dl_model.si_follmer.si_follmer_framework import (  # noqa: E402
    StochasticInterpolantFollmer,
)
from src.utils.random_seed_helper import set_seeds  # noqa: E402

EXPERIMENT = "ExperimentSchrodingerBridgeWindCanvas"


def main():
    parser = argparse.ArgumentParser(description="canvas 管线冒烟")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    config = load_config(EXPERIMENT, args.config_path)
    device = torch.device(args.device)
    set_seeds(config.train.seed)

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT_DIR,
        loader_config=config.loader,
        dataset_config=config.data,
        world_size=None,
        rank=None,
        train_valid_test_kinds=["train", "valid", "test"],
    )
    ds = dict_loaders["train"].dataset
    in_names = ds.input_channel_names()
    out_names = ds.target_channel_names()
    print("数据集大小: train={} valid={} test={}".format(
        len(dict_loaders["train"].dataset), len(dict_loaders["valid"].dataset),
        len(dict_loaders["test"].dataset)))
    print("输入通道 {} (config {}) / 输出通道 {} (config {})".format(
        len(in_names), config.model.in_channel, len(out_names), config.model.out_channel))
    assert len(in_names) == config.model.in_channel, "输入通道数与 config.model.in_channel 不一致"
    assert len(out_names) == config.model.out_channel, "输出通道数与 config.model.out_channel 不一致"

    batch = next(iter(dict_loaders["train"]))
    x, y, y0 = batch["x"].to(device), batch["y"].to(device), batch["y0"].to(device)
    print("batch: x={} y={} y0={}".format(tuple(x.shape), tuple(y.shape), tuple(y0.shape)))
    assert torch.isfinite(x).all(), "x 含非有限值"
    assert torch.isfinite(y).all(), "y 含非有限值"
    print("统计: y 均值 {:.4f} 标准差 {:.4f};y-y0 RMSE {:.4f}".format(
        float(y.mean()), float(y.std()), float(((y - y0) ** 2).mean().sqrt())))

    net = make_model(config.model).to(device)
    si = StochasticInterpolantFollmer(config=config.si, neural_net=net)
    loss = si(y0, y, x)
    print("SI 损失(L2, 无物理约束): {:.6f}".format(float(loss)))
    assert torch.isfinite(loss), "损失非有限值"
    if args.backward:
        loss.backward()
        gnorm = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
        print("反向传播完成,梯度范数和 {:.4f}".format(gnorm))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
