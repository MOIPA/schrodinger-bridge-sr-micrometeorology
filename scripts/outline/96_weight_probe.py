# -*- coding: utf-8 -*-
"""阶段 2 权重探针:用 V* 基线 checkpoint 在 valid 上量出四个物理项的原始量级,
给出"各物理项加权后 ≈ data 项 10%"的权重建议(定标后回填 86 的 WEIGHTS 重生成)。

做法(不训练、不反传,torch.no_grad):
  - 复用 90 的模型/数据/checkpoint 加载路径(build_si / make_dataloaders_and_samplers);
  - 每个 batch 调 si.forward(y0, y1, y_cond, rho=batch['rho'], return_parts=True),
    累计 data/div/vort/spectral/extreme 的批均值(分项原始值与权重无关);
  - 另按 SIFollmer._canvas_physics_raw 的估计步骤,用同一次 (timestep, noise)
    单独算 |D|/τ,报 hinge 激活率 fraction(|D|>τ) 与 P95(|D|/τ)(min_t 掩码内),
    取批均值(另附 pooled 值供复核)。
  - 窗口沿用配置自带的训练裁剪(96×112),与训练同口径,不做 pad16。

运行(需 torch + 数据 + checkpoint,仓库根目录):
  python scripts/outline/96_weight_probe.py \
      --config_path "configs/深圳/phase2/config_wind_canvas_p2_div_mid.yml" \
      --checkpoint <r_t14_noenc_cos>/checkpoint.pth \
      --split valid --max_batches 20 --device cuda:0 --out_json results/phase2/probe_weights.json
本地无 GPU 可用 --device cpu(样本量小时也能跑)。
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# physics_canvas 只用 numpy/torch,模块级导入不影响 --help
from src.dl_model.si_follmer.physics_canvas import (  # noqa: E402
    divergence_rho_u, estimate_residual, infer_n_levels, split_canvas_state)

EXPERIMENT = "ExperimentSchrodingerBridgeWindCanvas"
PARTS = ("data", "div", "vort", "spectral", "extreme")


def hinge_stats(si, y0, y1, y_cond, rho, seed):
    """照 _canvas_physics_raw 的步骤单独算散度比 |D|/τ(仅 min_t 掩码内样本)。

    与同 seed 的 forward 用同一次 (timestep, noise),即统计的正是 forward 里那组估计。
    返回 (fraction(|D|>τ), P95(|D|/τ), n_cell, n_over) 或 None(rho 缺失/掩码全 False)。
    """
    torch.manual_seed(seed)
    b = y0.shape[0]
    if si.c.residual_output:
        y0e, y1e = torch.zeros_like(y0), y1 - y0
    else:
        y0e, y1e = y0, y1
    timestep, t = si._sample_timestep(b)
    noise = torch.randn_like(y0e)
    yt = si._sample_yt(y0=y0e, y1=y1e, noise=noise, timestep=timestep)
    b_est = si.net(yt=yt, y_cond=y_cond, gamma=t)
    d_b = torch.gather(si.dot_beta, dim=-1, index=timestep)[:, None, None, None]
    r_hat, mask = estimate_residual(b_est, d_b, min_t=si.c.phys_min_t)
    y_hat = y0 + r_hat
    n_levels = infer_n_levels(y_hat.shape[1])
    dev, dt = y_hat.device, y_hat.dtype
    ps = torch.as_tensor(si.c.phys_scale, dtype=dt, device=dev).view(1, -1, 1, 1)
    u, v, w, _, _ = split_canvas_state(y_hat * ps, n_levels)
    dz = torch.as_tensor(si.c.phys_dz, dtype=dt, device=dev)
    tau = torch.as_tensor(si.c.phys_div_tau, dtype=dt, device=dev).view(1, n_levels, 1, 1)
    div = divergence_rho_u(u, v, w, rho, si.c.phys_dx, dz)
    if not bool(mask.any()):
        return None
    ratio = (div.abs() / tau)[mask].detach().cpu().numpy().ravel()
    if ratio.size == 0:
        return None
    return (float((ratio > 1.0).mean()), float(np.percentile(ratio, 95)),
            int(ratio.size), int((ratio > 1.0).sum()))


def main():
    ap = argparse.ArgumentParser(description="阶段 2 物理项权重探针(不训练)")
    ap.add_argument("--config_path", required=True, help="任一 p2 配置(提供 si 物理参数)")
    ap.add_argument("--checkpoint", required=True, help="基线 r_t14_noenc_cos 的 checkpoint.pth")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--max_batches", type=int, default=20, help="最多扫多少个 batch(0=全部)")
    ap.add_argument("--out_json", default="results/phase2/probe_weights.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", default="model", choices=["model", "ema"])
    args = ap.parse_args()

    # src 的导入放在 main 内:本文件只在服务器 torch 环境实跑,但这样 --help 在
    # 任何环境(含缺 sklearn / py<3.10 的本地环境)都可用。
    _agl90 = importlib.import_module("90_agl_eval_phase1")   # 复用 build_si,不复制
    from src.dl_config.config_loader import load_config
    from src.dl_data.dataloader import make_dataloaders_and_samplers
    from src.utils.random_seed_helper import set_seeds

    config = load_config(EXPERIMENT, args.config_path)
    device = torch.device(args.device)
    set_seeds(config.train.seed)
    si = _agl90.build_si(config, args.checkpoint, device, args.weights == "ema")
    si.eval()
    for name in ("phys_scale", "phys_dz", "phys_div_tau"):
        if getattr(si.c, name) is None:
            raise SystemExit("配置缺少 si.{}:探针需要完整物理参数,请用 86 生成的 p2 配置"
                             .format(name))

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=config.loader, dataset_config=config.data,
        world_size=None, rank=None, train_valid_test_kinds=[args.split])
    loader = dict_loaders[args.split]
    print("探针 {}: {} 帧,每 batch {} 帧,最多 {} 个 batch".format(
        args.split, len(loader.dataset), config.loader.batch_size, args.max_batches))

    acc = dict((k, 0.0) for k in PARTS)
    n_batch, n_frames, n_cell, n_over = 0, 0, 0, 0
    frac_list, p95_list = [], []
    seed0 = int(config.train.seed) % 100000
    for i, batch in enumerate(loader):
        if args.max_batches > 0 and i >= args.max_batches:
            break
        y0 = batch["y0"].to(device)
        y1 = batch["y"].to(device)
        y_cond = batch["x"].to(device)
        rho = batch.get("rho")
        rho = rho.to(device) if rho is not None else None
        h, w = y0.shape[-2:]
        assert h % 16 == 0 and w % 16 == 0, \
            "窗口 {}x{} 不是 16 的倍数(UNet 要求);探针按配置的训练裁剪窗口跑".format(h, w)
        torch.manual_seed(seed0 + i)
        with torch.no_grad():
            out = si(y0=y0, y1=y1, y_cond=y_cond, rho=rho, return_parts=True)
            for k in PARTS:
                if out[k] is not None:
                    acc[k] += float(out[k])
            if rho is not None:
                hs = hinge_stats(si, y0, y1, y_cond, rho, seed0 + i)
                if hs is not None:
                    frac_list.append(hs[0])
                    p95_list.append(hs[1])
                    n_cell += hs[2]
                    n_over += hs[3]
        n_batch += 1
        n_frames += int(y0.shape[0])
    if n_batch == 0:
        raise SystemExit("{} 加载器为空,没有帧可探针".format(args.split))

    mean = dict((k, acc[k] / n_batch) for k in PARTS)
    data_m = mean["data"]

    def suggest(raw):
        return (0.10 * data_m / raw) if (raw is not None and raw > 0) else None

    suggested = {}
    div_mid = suggest(mean["div"])
    if div_mid is not None:
        suggested["div_mid"] = div_mid
        suggested["div_lo"] = div_mid / 3.0
        suggested["div_hi"] = div_mid * 3.0
    suggested["spectral"] = suggest(mean["spectral"])
    suggested["extreme"] = suggest(mean["extreme"])
    suggested["vorticity"] = suggest(mean["vort"])

    hinge_frac = float(np.mean(frac_list)) if frac_list else None
    p95_ratio = float(np.mean(p95_list)) if p95_list else None
    hinge_frac_pooled = (float(n_over) / n_cell) if n_cell else None
    out = {
        "config_path": args.config_path, "checkpoint": args.checkpoint,
        "split": args.split, "ckpt_weights": args.weights, "n_batches": n_batch,
        "n_frames": n_frames, "target_ratio": 0.10,
        "parts_raw_mean": mean,
        "hinge_frac": hinge_frac,           # 批均值
        "p95_ratio": p95_ratio,             # 批均值
        "hinge_frac_pooled": hinge_frac_pooled,   # 全部统计体元汇总
        "hinge_n_cell": n_cell,
        "suggested_weights": suggested,
    }

    print("\n{:<10} {:>14} {:>16}".format("项", "原始均值", "建议权重(data×10%)"))
    key_map = {"div": "div_mid", "vort": "vorticity"}   # 项名 -> suggested_weights 键名
    for k in PARTS:
        w = None if k == "data" else suggested.get(key_map.get(k, k))
        print("{:<10} {:>14.6e} {:>16}".format(
            k, mean[k], ("-" if w is None else "{:.4e}".format(w))))
    if div_mid is not None:
        print("div 三档: lo={:.4e}  mid={:.4e}  hi={:.4e}".format(
            suggested["div_lo"], suggested["div_mid"], suggested["div_hi"]))
    if hinge_frac is not None:
        print("散度 hinge: fraction(|D|>τ)={:.4f}(pooled {:.4f}), P95(|D|/τ)={:.3f}, "
              "统计体元 {}".format(hinge_frac, hinge_frac_pooled, p95_ratio, n_cell))
    else:
        print("散度 hinge: 未统计(batch 缺 rho 或 min_t 掩码全 False)")

    out_json = args.out_json if os.path.isabs(args.out_json) \
        else os.path.join(ROOT, args.out_json)
    out["out_json"] = out_json
    if not os.path.isdir(os.path.dirname(out_json)):
        os.makedirs(os.path.dirname(out_json))
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\n写出 {}".format(out_json))
    print("PROBE OK")


if __name__ == "__main__":
    main()
