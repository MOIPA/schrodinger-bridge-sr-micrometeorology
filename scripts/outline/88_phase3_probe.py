# -*- coding: utf-8 -*-
"""阶段 3 探针:用基座 p2_l1r2_lr2e4 的 checkpoint 在 valid 上量出 AGL 项与数据项量级,
给出 87 生成 p3_joint / p3_agllw 所需的两个数:

  lambda_joint = mean(data 项) / mean(AGL 项)  -> p3_joint 的 si.agl_weight(两空间量级对齐);
  channel_weights_72                           -> p3_agllw 的低层等效通道加权(AGL 算子对
                                                  每个通道的梯度占比,归一到均值 1);
  lambda_joint_third = lambda_joint/3          -> 弱档备选(量级约为 data 项的 1/3)。

做法(不训练、不反传,torch.no_grad;不动磁盘 yml,只在内存里改):
  - 复用 96/60d 的加载路径:config = config_wind_canvas_p2_l1r2_lr2e4.yml,
    checkpoint = data/DL_result/<EXP>/config_wind_canvas_p2_l1r2_lr2e4/checkpoint.pth;
  - 内存中 data.return_agl_tables=true、si.agl_weight=1.0(基座无 si.phys_scale 时补上:
    p2_div_mid.yml -> normalize_config.json,与 60d/86 的口径一致);
  - valid 上 si.forward(..., rho=rho, agl=agl, return_parts=True),累计 data/agl 批均值;
  - agl 三分量:同 seed 复放同一次 (timestep, noise) 取 b_est/b_true,再用 60 号公共
    numpy 算子 + 窗口 destagger 复算 |au|/|av|/|aw| 均值(顺带校验 parts["agl"]);
  - 抽查:数据集表(place+crop 前的 build_agl_tables_native 输出)与
    agl_eval_common.load_tables 逐元素相等(打印 bool)。

运行(服务器 torch 环境,仓库根目录):
  python scripts/outline/88_phase3_probe.py --device cuda:0 --max_batches 20
  python scripts/outline/88_phase3_probe.py --device cpu --max_batches 2   # 小样本冒烟
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EXPERIMENT = "ExperimentSchrodingerBridgeWindCanvas"
BASE_CONFIG = os.path.join("configs", "深圳", "phase2",
                           "config_wind_canvas_p2_l1r2_lr2e4.yml")
PHYS_FALLBACK = os.path.join("configs", "深圳", "phase2",
                             "config_wind_canvas_p2_div_mid.yml")
CKPT_REL = os.path.join("data", "DL_result", EXPERIMENT,
                        "config_wind_canvas_p2_l1r2_lr2e4", "checkpoint.pth")


def _load_numpy_agl_interp():
    """importlib 载入 60_agl_operator.py 的 numpy agl_interp(评估侧唯一实现)。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "60_agl_operator.py")
    spec = importlib.util.spec_from_file_location("agl_operator_60_probe", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agl_interp


def load_phys_params(config):
    """探针需要的物理参数(基座没有):rho 非空 + return_parts=True 时,框架
    _canvas_physics_raw 会要求 phys_scale/phys_dz/phys_div_tau 三项齐全(与 96 同款)。

    来源:config.si -> p2_div_mid.yml(60d 的物理参数来源,已入库);phys_scale 还可
    回退按 86 口径从 normalize_config.json 现算。返回 (dict, 来源说明)。
    """
    import yaml
    L = list(config.data.target_levels)
    n_out = int(config.model.out_channel)
    keys = ("phys_scale", "phys_dz", "phys_div_tau")
    got, srcs = {}, []
    for k in keys:
        v = getattr(config.si, k, None)
        if v is not None:
            got[k] = [float(x) for x in v]
            srcs.append("{}=config".format(k))
    need = [k for k in keys if k not in got]
    if need and os.path.isfile(os.path.join(ROOT, PHYS_FALLBACK)):
        with open(os.path.join(ROOT, PHYS_FALLBACK)) as f:
            si_fb = yaml.safe_load(f).get("si") or {}
        for k in need:
            v = si_fb.get(k)
            if isinstance(v, list):
                got[k] = [float(x) for x in v]
                srcs.append("{}=p2_div_mid".format(k))
    if "phys_scale" not in got:          # 最后回退:normalize_config.json(86 口径)
        npath = config.data.normalize_json or os.path.join(config.data.statics_dir,
                                                           "normalize_config.json")
        with open(npath) as f:
            fn = json.load(f)["fine"][config.data.scheme]
        got["phys_scale"] = ([float(v) for v in fn["u"]["sigma"][:len(L)]]
                             + [float(v) for v in fn["v"]["sigma"][:len(L)]]
                             + [float(v) for v in fn["w"]["sigma"][L[0]:L[-1] + 2]]
                             + [float(fn["u10"]["sigma"]), float(fn["v10"]["sigma"])])
        srcs.append("phys_scale=normalize_config")
    missing = [k for k in keys if k not in got]
    if missing:
        raise SystemExit("缺 {}:在 config.si 与 {} 都找不到;探针的 return_parts=True "
                         "需要完整物理参数".format("、".join(missing), PHYS_FALLBACK))
    if len(got["phys_scale"]) != n_out:
        raise SystemExit("phys_scale 长度 {} != out_channel {}".format(
            len(got["phys_scale"]), n_out))
    for k in ("phys_dz", "phys_div_tau"):
        if len(got[k]) != len(L):
            raise SystemExit("{} 长度 {} != len(target_levels) {}".format(
                k, len(got[k]), len(L)))
    return got, ", ".join(srcs)


def _window_destagger(u, v, w, u10, v10):
    """训练窗口口径 destagger(同 physics_canvas.destagger_canvas):(…,H-1,W-1)。"""
    um = 0.5 * (u[..., :, :-1] + u[..., :, 1:])[..., :-1, :]
    vm = 0.5 * (v[..., :-1, :] + v[..., 1:, :])[..., :, :-1]
    return um, vm, w[..., :-1, :-1], u10[..., :-1, :-1], v10[..., :-1, :-1]


def _agl_components(err_canvas, agl, np_interp, n_lev):
    """公共算子复算 AGL 三项:err_canvas (B,72,H,W) 物理单位,numpy 输入。"""
    au_l, av_l, aw_l = [], [], []
    for i in range(err_canvas.shape[0]):
        u, v, w, u10, v10 = (err_canvas[i, 0:n_lev], err_canvas[i, n_lev:2 * n_lev],
                             err_canvas[i, 2 * n_lev:3 * n_lev + 1],
                             err_canvas[i, 3 * n_lev + 1], err_canvas[i, 3 * n_lev + 2])
        um, vm, wm, u10m, v10m = _window_destagger(u, v, w, u10, v10)
        hh, ww = um.shape[-2], um.shape[-1]
        def tb(key):                       # 表可能在 GPU 上,统一回 CPU 再进 numpy 算子
            return agl[key][i][:, :hh, :ww].detach().cpu().numpy()
        im = tb("idx_m").astype(np.int64)
        wm_ = tb("w_m")
        ii = tb("idx_i").astype(np.int64)
        wi_ = tb("w_i")
        au_l.append(np_interp(um, im, wm_, field10=u10m))
        av_l.append(np_interp(vm, im, wm_, field10=v10m))
        aw_l.append(np_interp(wm, ii, wi_, field10=None))
    return (float(np.mean([np.abs(a).mean() for a in au_l])),
            float(np.mean([np.abs(a).mean() for a in av_l])),
            float(np.mean([np.abs(a).mean() for a in aw_l])))


def build_channel_weights(tables, n_lev=23, w_weight=0.5):
    """由 4 张 AGL 表算 72 通道等效权重,归一到均值 1(口径见模块 docstring)。

    通道序 u0..22, v23..45, w46..69, u10=70, v10=71:
      u/v(质量表): idx>=0 -> 级 kc 得 w、(kc+1) 得 1−w;
                   idx==−1 -> 级 0 得 w、u10 通道得 1−w;idx==−2 -> u10 通道得 1;
      w(界面表):   idx>=0 -> 级 kc 得 w、(kc+1) 得 1−w(界面表无 −1/−2);
      逐级累加后 ÷((2+agl_w_weight)·11·像素数),拼 72 后除以均值。
    """
    idx_m, w_m = tables["idx_m"], tables["w_m"]
    idx_i, w_i = tables["idx_i"], tables["w_i"]
    nt = idx_m.shape[0]
    nw = idx_i.shape[0]
    npix = float(idx_m.shape[1] * idx_m.shape[2])
    assert idx_i.shape[0] == nt and w_i.shape == idx_i.shape, "界面表应与质量表同形(目标层轴)"
    out = np.zeros(3 * n_lev + 3, dtype=np.float64)
    u10_ch = 3 * n_lev + 1
    v10_ch = 3 * n_lev + 2
    for uv_off, ten_ch in ((0, u10_ch), (n_lev, v10_ch)):
        for t in range(nt):
            k = idx_m[t].astype(np.int64)
            wt = w_m[t].astype(np.float64)
            m = k >= 0
            np.add.at(out[uv_off:uv_off + n_lev], k[m], wt[m])
            np.add.at(out[uv_off:uv_off + n_lev], k[m] + 1, 1.0 - wt[m])
            m1 = k == -1
            out[uv_off] += float(wt[m1].sum())             # 级 0 得 w
            out[ten_ch] += float((1.0 - wt[m1]).sum())     # u10/v10 得 1−w
            out[ten_ch] += float((k == -2).sum())          # u10/v10 得 1
    for t in range(nw):
        k = idx_i[t].astype(np.int64)
        wt = w_i[t].astype(np.float64)
        m = k >= 0
        np.add.at(out[2 * n_lev:2 * n_lev + n_lev + 1], k[m], wt[m])
        np.add.at(out[2 * n_lev:2 * n_lev + n_lev + 1], k[m] + 1, 1.0 - wt[m])
    out = out / ((2.0 + w_weight) * nt * npix)
    return out / out.mean()


def main():
    ap = argparse.ArgumentParser(description="阶段 3 探针(data/AGL 量级 + 通道加权)")
    ap.add_argument("--config_path", default=BASE_CONFIG)
    ap.add_argument("--checkpoint", default=CKPT_REL)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--max_batches", type=int, default=20, help="最多扫多少个 batch(0=全部)")
    ap.add_argument("--out_json", default="results/phase3/probe.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", default="model", choices=["model", "ema"])
    args = ap.parse_args()

    # src 的导入放在 main 内:--help 在任何环境都可用(96 同款)
    _agl90 = importlib.import_module("90_agl_eval_phase1")     # 复用 build_si
    import agl_eval_common as AEC
    from src.dl_config.config_loader import load_config
    from src.dl_data.dataloader import make_dataloaders_and_samplers
    from src.dl_data.dataset_wind_canvas import build_target_channel_names
    from src.dl_data.wind_canvas_statics import CanvasStatics, build_agl_tables_native
    from src.utils.random_seed_helper import set_seeds

    cfg_path = args.config_path if os.path.isabs(args.config_path) \
        else os.path.join(ROOT, args.config_path)
    ck_path = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(ROOT, args.checkpoint)
    if not os.path.isfile(ck_path):
        raise SystemExit("找不到 checkpoint {};先确认基座训练已完成(60d 的 ck_of 路径)".format(ck_path))
    config = load_config(EXPERIMENT, cfg_path)
    device = torch.device(args.device)
    set_seeds(config.train.seed)

    # ---- 内存改动:表开关 + AGL 权重 + (基座缺的)phys_scale;磁盘 yml 不动 ----
    config.data.return_agl_tables = True
    config.si.agl_weight = 1.0
    w_weight = float(config.si.agl_w_weight)
    phys, phys_src = load_phys_params(config)          # 探针 forward 用全套物理参数
    for k, v in phys.items():
        setattr(config.si, k, v)
    phys_scale = phys["phys_scale"]

    si = _agl90.build_si(config, ck_path, device, args.weights == "ema")
    si.eval()

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=config.loader, dataset_config=config.data,
        world_size=None, rank=None, train_valid_test_kinds=[args.split])
    loader = dict_loaders[args.split]
    ds = loader.dataset
    print("探针 {}: {} 帧,每 batch {} 帧,最多 {} 个 batch".format(
        args.split, len(ds), config.loader.batch_size, args.max_batches))
    print("物理参数: {} (phys_scale 长度 {})".format(phys_src, len(phys_scale)))

    # ---- 抽查:数据集表(place+crop 前)与评估侧 load_tables 逐元素相等 ----
    statics = CanvasStatics(config.data.statics_dir)
    levels = list(config.data.target_levels)
    raw = build_agl_tables_native(statics.d["zagl_mass_fine"],
                                  statics.d["zagl_iface_fine"], levels)
    tabs = AEC.load_tables(statics, levels)
    tables_eq = all(np.array_equal(raw[k], tabs[k]) for k in
                    ("idx_m", "w_m", "idx_i", "w_i"))
    print("数据集表(place+crop 前)== agl_eval_common.load_tables: {}".format(tables_eq))
    assert tables_eq, "数据集与评估侧的 AGL 表不一致(同参同源契约被破坏)"

    np_interp = _load_numpy_agl_interp()
    n_lev = len(levels)
    acc = {"data": 0.0, "agl": 0.0, "agl_u": 0.0, "agl_v": 0.0, "agl_w": 0.0}
    n_batch, n_frames = 0, 0
    recomp_rel = 0.0
    seed0 = int(config.train.seed) % 100000
    for i, batch in enumerate(loader):
        if args.max_batches > 0 and i >= args.max_batches:
            break
        y0 = batch["y0"].to(device)
        y1 = batch["y"].to(device)
        y_cond = batch["x"].to(device)
        rho = batch.get("rho")
        rho = rho.to(device) if rho is not None else None
        agl = {k: batch["agl_" + k].to(device) for k in ("idx_m", "w_m", "idx_i", "w_i")}
        torch.manual_seed(seed0 + i)
        with torch.no_grad():
            out = si(y0=y0, y1=y1, y_cond=y_cond, rho=rho, agl=agl, return_parts=True)
            for k in ("data", "agl"):
                v = out[k]
                if v is None or not bool(torch.isfinite(v)):
                    raise SystemExit("batch {} 的 {} 项为 None/NaN".format(i, k))
                acc[k] += float(v)
            # 三分量:同 seed 复放 (timestep, noise),公共算子复算
            torch.manual_seed(seed0 + i)
            y0r, y1r = y0, y1
            if si.c.residual_output:
                y1r, y0r = y1 - y0, torch.zeros_like(y0)
            timestep, t = si._sample_timestep(y0.shape[0])
            noise = torch.randn_like(y0r)
            yt = si._sample_yt(y0=y0r, y1=y1r, noise=noise, timestep=timestep)
            b_true = si._calc_b_true(y0=y0r, y1=y1r, noise=noise, timestep=timestep)
            b_est = si.net(yt=yt, y_cond=y_cond, gamma=t)
            ps = torch.as_tensor(phys_scale, dtype=b_est.dtype,
                                 device=b_est.device).view(1, -1, 1, 1)
            err_phys = ((b_est - b_true) * ps).cpu().numpy()
            u_m, v_m, w_m = _agl_components(err_phys, agl, np_interp, n_lev)
            recomp = (u_m + v_m + w_weight * w_m) / (2.0 + w_weight)
            rel = abs(recomp - float(out["agl"])) / max(abs(recomp), 1e-9)
            if rel > 1e-3:
                raise SystemExit("batch {}:parts['agl'] 与公共算子复算不一致(rel {:.2e}),"
                                 "复放的 (timestep, noise) 可能没对上".format(i, rel))
            recomp_rel = max(recomp_rel, rel)
            acc["agl_u"] += u_m
            acc["agl_v"] += v_m
            acc["agl_w"] += w_m
        n_batch += 1
        n_frames += int(y0.shape[0])
    if n_batch == 0:
        raise SystemExit("{} 加载器为空,没有帧可探针".format(args.split))

    mean = {k: acc[k] / n_batch for k in acc}
    lambda_joint = mean["data"] / mean["agl"]
    cw = build_channel_weights(tabs, n_lev=n_lev, w_weight=w_weight)
    assert abs(cw.mean() - 1.0) < 1e-12, "channel_weights_72 均值应为 1"
    names = build_target_channel_names(levels, config.data.include_w)

    out = {
        "data_mean": mean["data"],
        "agl_mean": mean["agl"],
        "agl_u": mean["agl_u"],
        "agl_v": mean["agl_v"],
        "agl_w": mean["agl_w"],
        "lambda_joint": lambda_joint,
        "lambda_joint_third": lambda_joint / 3.0,
        "channel_weights_72": [float(x) for x in cw],
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "config_path": cfg_path, "checkpoint": ck_path, "split": args.split,
            "ckpt_weights": args.weights, "n_batches": n_batch, "n_frames": n_frames,
            "agl_weight_used": 1.0, "agl_w_weight": w_weight,
            "phys_scale": phys_scale, "phys_scale_source": phys_src,
            "phys_dz_len": len(phys["phys_dz"]), "phys_div_tau_len": len(phys["phys_div_tau"]),
            "tables_match_load_tables": tables_eq,
            "agl_recomp_max_rel": recomp_rel,
            "note": "lambda_joint = data_mean/agl_mean;channel_weights_72 由 4 张原生表累加"
                    "(÷((2+w_w)·11·像素数) 后归一到均值 1)",
        },
    }
    out_json = args.out_json if os.path.isabs(args.out_json) \
        else os.path.join(ROOT, args.out_json)
    if not os.path.isdir(os.path.dirname(out_json)):
        os.makedirs(os.path.dirname(out_json))
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    top = np.argsort(cw)[::-1][:5]
    print("\n{:<10} {:>14}".format("项", "批均值"))
    for k in ("data_mean", "agl_mean", "agl_u", "agl_v", "agl_w", "lambda_joint"):
        print("{:<10} {:>14.6g}".format(k, out[k]))
    print("lambda_joint_third = {:.6g}".format(out["lambda_joint_third"]))
    print("agl 复算最大相对偏差 = {:.2e}(框架 vs 公共算子,U/V 系数 1、W 系数 {:g})".format(
        recomp_rel, w_weight))
    print("channel_weights_72: min={:.4f} mean={:.4f} max={:.4f};前 5 通道 {}".format(
        float(cw.min()), float(cw.mean()), float(cw.max()),
        ", ".join("{}={:.3f}".format(names[j] if j < len(names) else j, cw[j]) for j in top)))
    print("\n写出 {}".format(out_json))
    print("PROBE OK")


if __name__ == "__main__":
    main()
