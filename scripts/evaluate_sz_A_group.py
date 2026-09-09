# -*- coding: utf-8 -*-
"""
A 组评估实验(sr-exp-design skill 的 A1-A6,全部纯评估不训练)。

mode=diag:    A1 功率谱(输出谱曲线 JSON) + A2 逐层误差 + A3 强风分箱 + A5 跨方案分组
mode=shuffle: A4 条件变量空间打乱消融(8 个条件各一次额外推理)
mode=moments: A6 跨域归一化矩匹配消融(d03 输入改用 d04 侧统计量)

用法(服务器 wind3d env):
  python scripts/evaluate_sz_A_group.py --mode diag    --model baseline --results_dir results/A_group
  python scripts/evaluate_sz_A_group.py --mode shuffle --model baseline --results_dir results/A_group
  python scripts/evaluate_sz_A_group.py --mode moments --model baseline --results_dir results/A_group

评估口径与 evaluate_sz_experiments.py 完全一致(同 config、同 test 划分、同指标函数)。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.model_maker import make_model
from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer
from scripts.evaluate_sz_experiments import compute_metrics, summarize_by_component

EXPERIMENT_NAME = "ExperimentSchrodingerBridge3dWind"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_SUBDIR = "深圳"

LEVELS = ["ml0", "ml1", "ml2", "ml3", "ml5", "ml10"]
COMPONENTS = ["U", "V", "W"]
COND_NAMES = ["t2", "z", "lu", "tsk", "hfx", "lh", "psfc", "pblh"]
WIND_BINS = [(0, 5), (5, 10), (10, 1e9)]
# 显示名(报告/结果 json) -> 技术名(配置/checkpoint 文件名)
MODEL_NAME_MAP = {"baseline": "baseline", "allLR": "lrcond", "phys": "pinn"}


def load_model_and_loader(cfg_path, eval_filter="all"):
    config = load_config(EXPERIMENT_NAME, cfg_path)
    config.data.day_night_filter = eval_filter
    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT_DIR, loader_config=config.loader,
        dataset_config=config.data, world_size=None, rank=None,
        train_valid_test_kinds=["test"])
    loader = dict_loaders["test"]
    return config, loader


def build_model(config, device, checkpoint_dir):
    """加载模型。checkpoint_dir 如 config_wind_3d_sz_baseline(注意:config 无 _path 属性)。"""
    model = make_model(config.model)
    ckpt = torch.load(os.path.join(
        ROOT_DIR, "data", "DL_result", "ExperimentSchrodingerBridge3dWind",
        checkpoint_dir, "checkpoint.pth"),
        map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return StochasticInterpolantFollmer(config=config.si, neural_net=model)


def predict_all(loader, si_follmer, device):
    """一次推理,返回 y0/y/y_est 全量数组与每样本文件路径(loader 顺序=dataset.ps 顺序)。"""
    y0s, ys, preds = [], [], []
    for batch in loader:
        y_cond = batch["x"].to(device)
        y0 = batch["y0"].to(device)
        with torch.no_grad():
            y_est, _ = si_follmer.sample_y1_bare_diffusion(y0=y0, y_cond=y_cond)
        y0s.append(y0.cpu().numpy())
        ys.append(batch["y"].numpy())
        preds.append(y_est.cpu().numpy())
    paths = loader.dataset.ps
    return (np.concatenate(y0s, 0), np.concatenate(ys, 0),
            np.concatenate(preds, 0), paths)


def component_mask(name):
    for i, comp in enumerate(COMPONENTS):
        if "_{}_".format(comp.lower()) in name:
            return i
    return -1


def level_of(name):
    for lv in LEVELS:
        if name.endswith(lv):
            return lv
    return None


def per_level_metrics(pred, target, names):
    """A2: 逐垂直层 RMSE/SSIM(对该层 U/V/W 平均)。"""
    out = {}
    for lv in LEVELS:
        idx = [i for i, n in enumerate(names) if level_of(n) == lv]
        if not idx:
            continue
        rmse, mae, ssim, corr, bias = compute_metrics(pred[:, idx], target[:, idx])
        out[lv] = {"rmse": float(rmse.mean()), "mae": float(mae.mean()),
                   "ssim": float(ssim.mean()), "corr": float(corr.mean())}
    return out


def radial_spectra(fields, n_bins=48):
    """A1: 单通道场集 -> 径向平均功率谱 [n_bins]。"""
    n, h, w = fields.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2).ravel()
    bins = np.linspace(0, r.max(), n_bins + 1)
    spec = np.zeros(n_bins)
    for f in fields:
        p = np.abs(np.fft.fftshift(np.fft.fft2(f - f.mean()))) ** 2
        pr = p.ravel()
        for b in range(n_bins):
            m = (r >= bins[b]) & (r < bins[b + 1])
            if m.any():
                spec[b] += pr[m].mean()
    spec /= max(n, 1)
    return 0.5 * (bins[:-1] + bins[1:]), np.log10(np.maximum(spec, 1e-20))


def run_diag(config, loader, si_follmer, device, results_dir, tag):
    names = config.data.target_variable_names
    y0, y, pred, paths = predict_all(loader, si_follmer, device)
    n = len(y)
    print("[diag] n={}".format(n))

    # 整体 + 逐层(A2)
    out = {"tag": tag, "n": n}
    rmse, mae, ssim, corr, bias = compute_metrics(pred, y)
    per_ch = {"rmse": rmse, "mae": mae, "ssim": ssim, "corr": corr, "bias": bias}
    out["overall"] = summarize_by_component(per_ch, names)
    out["by_level"] = per_level_metrics(pred, y, names)
    out["by_level_interp"] = per_level_metrics(y0, y, names)

    # 跨方案分组(A5): 按文件名前缀 myj_/ysu_
    scheme_idx = {"myj": [], "ysu": []}
    for i, p in enumerate(paths):
        base = os.path.basename(p)
        if base.startswith("myj_"):
            scheme_idx["myj"].append(i)
        elif base.startswith("ysu_"):
            scheme_idx["ysu"].append(i)
    out["by_scheme"] = {}
    for sc, idx in scheme_idx.items():
        if idx:
            m = compute_metrics(pred[idx], y[idx])
            out["by_scheme"][sc] = summarize_by_component(
                {"rmse": m[0], "mae": m[1], "ssim": m[2], "corr": m[3], "bias": m[4]}, names)
            out["by_scheme"][sc]["n"] = len(idx)

    # 强风分箱(A3): ml0 水平风速(反标准化)的样本均值分箱
    u_i = names.index("hr_u_ml0")
    v_i = names.index("hr_v_ml0")
    bu, su = config.data.biases["hr_u_ml0"], config.data.scales["hr_u_ml0"]
    bv, sv = config.data.biases["hr_v_ml0"], config.data.scales["hr_v_ml0"]
    speed = np.sqrt((y[:, u_i] * su + bu) ** 2 + (y[:, v_i] * sv + bv) ** 2)
    speed_mean = speed.reshape(n, -1).mean(axis=1)
    out["wind_bins"] = {}
    for lo, hi in WIND_BINS:
        idx = np.where((speed_mean >= lo) & (speed_mean < hi))[0]
        if len(idx) == 0:
            continue
        m = compute_metrics(pred[idx], y[idx])
        label = "{}-{}".format(int(lo), "inf" if hi > 100 else int(hi))
        out["wind_bins"][label] = summarize_by_component(
            {"rmse": m[0], "mae": m[1], "ssim": m[2], "corr": m[3], "bias": m[4]}, names)
        out["wind_bins"][label]["n"] = int(len(idx))
        out["wind_bins"][label]["speed_mean"] = float(speed_mean[idx].mean())

    # 功率谱(A1): U/V/W 各层平均后,真值/模型/插值 径向谱(取前 300 样本)
    sub = min(n, 300)
    out["spectra"] = {}
    for comp in ["U", "V", "W"]:
        ci = [i for i, nm in enumerate(names) if component_mask(nm) == COMPONENTS.index(comp)]
        yy = y[:sub][:, ci].mean(axis=1)
        pp = pred[:sub][:, ci].mean(axis=1)
        ii = y0[:sub][:, ci].mean(axis=1)
        k, s_true = radial_spectra(yy)
        _, s_pred = radial_spectra(pp)
        _, s_interp = radial_spectra(ii)
        out["spectra"][comp] = {"k": k.tolist(), "truth": s_true.tolist(),
                                "model": s_pred.tolist(), "interp": s_interp.tolist()}

    path = os.path.join(results_dir, "A_group_{}_diag.json".format(tag))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved {}".format(path))


def run_shuffle(config, loader, si_follmer, device, results_dir, tag):
    """A4: 逐个打乱 8 个条件变量的空间结构,报 RMSE 增量。"""
    names = config.data.target_variable_names
    inputs = config.data.input_variable_names
    # 条件通道在 x 中的位置 = input_variable_names 里非 lr_ 风场的通道
    cond_slots = []
    for c, nm in enumerate(inputs):
        if nm in COND_NAMES:
            cond_slots.append(c)
    assert len(cond_slots) == len(COND_NAMES), (cond_slots, inputs)

    rng = np.random.RandomState(0)
    all_delta = {}

    def run_pass(shuffle_slot=None):
        """一次全量推理;shuffle_slot 不为 None 时打乱该条件通道。返回 (pred, y)。"""
        preds, ys = [], []
        for batch in loader:
            x = batch["x"].clone()
            if shuffle_slot is not None:
                b, c, h, w = x.shape
                perm = rng.permutation(h * w)
                x[:, shuffle_slot] = x[:, shuffle_slot].reshape(b, h * w)[
                    :, perm].reshape(b, h, w)
            y0 = batch["y0"].to(device)
            with torch.no_grad():
                y_est, _ = si_follmer.sample_y1_bare_diffusion(
                    y0=y0, y_cond=x.to(device))
            preds.append(y_est.cpu().numpy())
            ys.append(batch["y"].numpy())
        return np.concatenate(preds, 0), np.concatenate(ys, 0)

    # 干净基准(不打乱)
    pred, y = run_pass()
    rmse, *_ = compute_metrics(pred, y)
    base_rmse = float(rmse.mean())
    print("[shuffle] clean RMSE {:.4f} (基准)".format(base_rmse))

    # 逐个条件打乱
    for ci, slot in enumerate(cond_slots):
        cond = COND_NAMES[ci]
        pred, y = run_pass(shuffle_slot=slot)
        rmse, *_ = compute_metrics(pred, y)
        ov_rmse = float(rmse.mean())
        all_delta[cond] = {"rmse": ov_rmse, "delta": ov_rmse - base_rmse}
        print("[shuffle] {:>5s} RMSE {:.4f}  delta {:+.4f}".format(
            cond, ov_rmse, ov_rmse - base_rmse))

    out = {"tag": tag, "base_rmse": base_rmse, "shuffle": all_delta}
    path = os.path.join(results_dir, "A_group_{}_shuffle.json".format(tag))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved {}".format(path))


def run_moments(device, results_dir, tag="baseline"):
    """A6: d04 训练的 baseline 在 d03 上评估,LR 输入统计量用 d04 侧(d03 侧为官方口径)。"""
    d03_cfg_path = os.path.join(ROOT_DIR, "configs", CONFIG_SUBDIR,
                                "config_wind_3d_sz_d03_baseline.yml")
    d04_cfg_path = os.path.join(ROOT_DIR, "configs", CONFIG_SUBDIR,
                                "config_wind_3d_sz_baseline.yml")
    config = load_config(EXPERIMENT_NAME, d03_cfg_path)
    config.data.day_night_filter = "all"
    ref = load_config(EXPERIMENT_NAME, d04_cfg_path)

    # 把 18 个 lr_ 风场通道的 bias/scale 换成 d04 侧统计量
    # (条件变量 t2/z/... 在 d03 配置里本就是 d04 1km 版本+d04 统计,目标用 d04 统计——均不动)
    for nm in config.data.input_variable_names:
        if nm.startswith("lr_") and nm in ref.data.biases:
            config.data.biases[nm] = ref.data.biases[nm]
            config.data.scales[nm] = ref.data.scales[nm]

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT_DIR, loader_config=config.loader,
        dataset_config=config.data, world_size=None, rank=None,
        train_valid_test_kinds=["test"])
    loader = dict_loaders["test"]
    # 评估的是 d04 训练的 baseline 模型(配置是 d03 口径,checkpoint 用 d04 baseline)
    si = build_model(config, device, checkpoint_dir="config_wind_3d_sz_baseline")
    names = config.data.target_variable_names
    y0, y, pred, paths = predict_all(loader, si, device)
    rmse, mae, ssim, corr, bias = compute_metrics(pred, y)
    out = {"tag": tag, "n": len(y), "stats": "d04-input-stats",
           "overall": summarize_by_component(
               {"rmse": rmse, "mae": mae, "ssim": ssim, "corr": corr, "bias": bias}, names)}
    path = os.path.join(results_dir, "A_group_{}_moments.json".format(tag))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved {}".format(path))
    print("[moments] Overall RMSE {:.4f} (官方 d03 口径 0.2410)".format(
        out["overall"]["Overall"]["rmse"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diag", "shuffle", "moments"], required=True)
    parser.add_argument("--model", default="baseline",
                        help="模型名 baseline/lrcond/phys (d04 侧)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--results_dir", default=os.path.join(ROOT_DIR, "results", "A_group"))
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode == "moments":
        run_moments(args.device, args.results_dir, tag=args.model)
        return

    tech = MODEL_NAME_MAP[args.model]
    cfg_name = "config_wind_3d_sz_{}.yml".format(tech)
    cfg_path = os.path.join(ROOT_DIR, "configs", CONFIG_SUBDIR, cfg_name)
    print("[step1] loading dataset/loader")
    config, loader = load_model_and_loader(cfg_path)
    print("[step2] loader ready, n={}; building model".format(len(loader.dataset)))
    si = build_model(config, args.device,
                     checkpoint_dir="config_wind_3d_sz_{}".format(tech))
    print("[step3] model ready")
    tag = args.model
    if args.mode == "diag":
        run_diag(config, loader, si, args.device, args.results_dir, tag)
    else:
        run_shuffle(config, loader, si, args.device, args.results_dir, tag)


if __name__ == "__main__":
    main()
