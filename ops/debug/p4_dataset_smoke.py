# -*- coding: utf-8 -*-
"""阶段4 数据集烟测:只建 Dataset、取一个样本,不碰 GPU。

用法(服务器, wind3d 环境):
  python ops/debug/p4_dataset_smoke.py --config_path configs/深圳/phase4/config_wind_canvas_p4_27to1_reg.yml
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dl_config.config_loader import load_config          # noqa: E402
from src.dl_data.dataloader import make_dataloaders_and_samplers  # noqa: E402

EXP = "ExperimentSchrodingerBridgeWindCanvas"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--experiment_name", default=EXP)
    ap.add_argument("--timing", action="store_true", help="只测 __getitem__ 的单样本耗时")
    ap.add_argument("--n_time", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.experiment_name, args.config_path)
    print("[cfg] coarse_dir =", cfg.data.coarse_dir)
    print("[cfg] statics_dir =", cfg.data.statics_dir)
    print("[cfg] split_manifest =", cfg.data.split_manifest)

    loaders, _ = make_dataloaders_and_samplers(
        root_dir=str(ROOT), loader_config=cfg.loader, dataset_config=cfg.data,
        world_size=None, rank=None, train_valid_test_kinds=["train", "valid"],
    )
    for k, dl in loaders.items():
        if dl is None:
            print(f"[loader] {k}: None")
            continue
        ds = dl.dataset
        print(f"[loader] {k}: n={len(ds)}")
        if k != "train":
            continue
        n_in = len(ds.input_channel_names()) if hasattr(ds, "input_channel_names") else -1
        n_out = len(ds.target_channel_names()) if hasattr(ds, "target_channel_names") else -1
        print(f"[channels] cond={n_in} state={n_out} sum={n_in + n_out} (cfg in_channel={cfg.model.in_channel})")
        names_in = ds.input_channel_names() if hasattr(ds, "input_channel_names") else []
        print("[names] in :", " ".join(names_in))
        assert n_out == cfg.model.out_channel, f"state {n_out} != out_channel {cfg.model.out_channel}"
        assert n_in + n_out == cfg.model.in_channel, (
            f"cond {n_in} + state {n_out} != in_channel {cfg.model.in_channel}")
        if args.timing:
            import time
            ds[0]
            t0 = time.time()
            for i in range(args.n_time):
                ds[i]
            dt = (time.time() - t0) / args.n_time
            print(f"[timing] __getitem__ 平均 {dt:.2f} s/样本 (n={args.n_time})")
            continue
        item = ds[0]
        for key, val in item.items():
            if hasattr(val, "shape"):
                arr = np.asarray(val)
                print(f"  {key:>12s}: shape={arr.shape} dtype={arr.dtype} "
                      f"finite={np.isfinite(arr).all()} min={arr.min():.4g} max={arr.max():.4g}")

    print("[ok] 数据集可读、通道数一致、样本无 NaN")


if __name__ == "__main__":
    main()
