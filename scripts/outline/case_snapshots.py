# -*- coding: utf-8 -*-
"""导出个例风场快照(供 10-09 组会汇报配图,本地绘图用)。

选帧:测试集(myj)内按 10 m 风速域平均取 top-N 强风个例。
来源:truth(d04 真值)、y0(粗端重网格)、pred_base(p2_l1r2_lr2e4)、pred_joint(p3_joint)。
场:指定 AGL 层(默认 100/300 m)的 u/v/w,形状 (n_agl, 99, 120)。
输出:results/report_10_09/cases.npz(npz 压缩,约数 MB)+ stdout 摘要。

推理路径与 98_phase3_spatial.py 完全一致(90 的 build_si/pad16/split_denorm,
agl_eval_common 的 destagger/agl_fields),仅把"累计统计"换成"存场快照"。

用法(仓库根目录,wind3d 环境):
  python scripts/outline/case_snapshots.py --device cuda:0
  # 可选: --n_cases 3 --agl 100 300 --out_dir results/report_10_09
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agl_eval_common import (TARGET_AGL, agl_fields, destagger_canvas,  # noqa: E402
                             load_norm_sigma, load_tables)
from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_data.dataloader import make_dataloaders_and_samplers  # noqa: E402
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402
from src.utils.random_seed_helper import set_seeds  # noqa: E402


def _load_90():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "90_agl_eval_phase1.py")
    spec = importlib.util.spec_from_file_location("agl_eval_90", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV90 = _load_90()
EXPERIMENT = EV90.EXPERIMENT

# 两个对比模型(基线 vs 联合监督);tag -> (config, checkpoint)
TAGS = ["p2_l1r2_lr2e4", "p3_joint"]
SRC_KEYS = ["pred_base", "pred_joint"]  # 与 TAGS 对应


def tag_paths(tag):
    phase = "phase2" if tag.startswith("p2_") else "phase3"
    return (os.path.join("configs/深圳", phase, "config_wind_canvas_{}.yml".format(tag)),
            os.path.join("data", "DL_result", EXPERIMENT,
                         "config_wind_canvas_{}".format(tag), "checkpoint.pth"))


def build_loader(config, split):
    """与 90/98 相同的取数路径:全画布 (100,121),确定性裁剪。"""
    config.data.hr_data_shape = [100, 121]
    config.data.hr_cropped_shape = [100, 121]
    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=config.loader, dataset_config=config.data,
        world_size=None, rank=None, train_valid_test_kinds=[split])
    return dict_loaders[split]


def agl_indices(levels):
    out = []
    for h in levels:
        w = np.where(TARGET_AGL == float(h))[0]
        if len(w) == 0:
            raise SystemExit("AGL {} m 不在 TARGET_AGL 目标层里".format(h))
        out.append(int(w[0]))
    return out


def to_agl_fields(tensor_state, levels, sigma, tables):
    """标准化状态张量 (72,100,121) -> AGL 场 (u,v,w) 各 (11,99,120)。"""
    u, v, w, u10, v10 = EV90.split_denorm(tensor_state, levels, sigma)
    m = destagger_canvas(u.numpy(), v.numpy(), w.numpy(), u10.numpy(), v10.numpy())
    return agl_fields(m[0], m[1], m[2], m[3], m[4], tables)


def main():
    ap = argparse.ArgumentParser(description="导出个例快照(汇报配图用)")
    ap.add_argument("--n_cases", type=int, default=3, help="导出几个强风个例")
    ap.add_argument("--agl", type=float, nargs="+", default=[100.0, 300.0])
    ap.add_argument("--out_dir", type=str, default="results/report_10_09")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- 基座配置与数据 ----
    cfg_base, ck_base = tag_paths(TAGS[0])
    config = load_config(EXPERIMENT, cfg_base)
    set_seeds(config.train.seed)
    loader = build_loader(config, args.split)
    ds = loader.dataset
    n = len(ds)
    print("测试集 {} 帧(myj, {} )".format(n, config.data.scheme))

    levels = list(config.data.target_levels)
    statics = CanvasStatics(config.data.statics_dir)
    tables = load_tables(statics, levels)
    sigma = load_norm_sigma(
        os.path.join(config.data.statics_dir, "normalize_config.json"),
        config.data.scheme, levels)
    sel = agl_indices(args.agl)

    # ---- Pass 1:按 10 m 风速域平均选强风个例(不推理,仅读数据) ----
    metrics = np.zeros(n, dtype=np.float64)
    stamps = []
    for i in range(n):
        s = ds[i]
        u10 = float(s["y"][-2].abs().mean())  # 标准化空间,仅用于排序
        v10 = float(s["y"][-1].abs().mean())
        metrics[i] = (u10 ** 2 + v10 ** 2) ** 0.5
        stamps.append(os.path.basename(ds.ps[i]))
    order = np.argsort(metrics)[::-1][:args.n_cases]
    print("选中测试帧(top {} 近地风速):".format(args.n_cases))
    for k, idx in enumerate(order):
        print("  case{}: idx={} stamp={} metric={:.3f}".format(k, idx, stamps[idx], metrics[idx]))

    # ---- Pass 2:加载两个模型,逐帧推理并存场 ----
    models = {}
    for tag, key in zip(TAGS, SRC_KEYS):
        cfg_p, ck_p = tag_paths(tag)
        cfg_t = load_config(EXPERIMENT, cfg_p)
        models[key] = EV90.build_si(cfg_t, ck_p, device, use_ema=False)
        print("模型已加载: {} ({})".format(tag, key))

    out = {"agl": np.array(args.agl, dtype=np.float64)}
    for k, idx in enumerate(order):
        s = ds[int(idx)]
        x = s["x"].unsqueeze(0).to(device)
        y0 = s["y0"].unsqueeze(0).to(device)
        y = s["y"]
        fields = {"truth": to_agl_fields(y, levels, sigma, tables),
                  "y0": to_agl_fields(s["y0"], levels, sigma, tables)}
        for key in SRC_KEYS:
            with torch.no_grad():
                pred, _ = models[key].sample_y1_bare_diffusion(
                    y0=EV90.pad16(y0), y_cond=EV90.pad16(x), add_noise=False)
            pred = pred[:, :, :100, :121][0].detach().cpu()
            fields[key] = to_agl_fields(pred, levels, sigma, tables)
        for src, (au, av, aw) in fields.items():
            for comp, arr in zip(["u", "v", "w"], (au, av, aw)):
                out["case{}_{}_{}".format(k, src, comp)] = arr[sel].astype(np.float32)
        out["case{}_stamp".format(k)] = np.array([stamps[int(idx)]])
        print("  case{} 完成(来源: {})".format(k, list(fields.keys())))

    path = os.path.join(args.out_dir, "cases.npz")
    np.savez_compressed(path, **out)
    print("已保存 {}".format(path))
    print("来源键: truth / y0 / pred_base / pred_joint;AGL 层: {}".format(args.agl))


if __name__ == "__main__":
    main()
