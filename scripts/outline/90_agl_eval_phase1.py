# -*- coding: utf-8 -*-
"""阶段 1 模型评估:checkpoint -> 确定性 SI 采样 -> 反标准化 -> AGL 11 层 -> 逐小时指标。

- 采样用 add_noise=False(纯 ODE),消除采样噪声对 run 间排序的干扰;
- UNet 只支持 16 的倍数尺寸(96/112 即由此而来),评估时把 canvas (100,121) 复制
  填充到 (112,128) 再切回,不丢边界;
- 逐小时 × 每层 × 每分层存累加量(n/平方误差和/...),排名脚本 92 免重推理做 bootstrap;
- 顺带算免费基线 y0(粗端重网格),与模型同算子同口径。

用法(需要 torch 的环境,仓库根目录):
  python scripts/outline/90_agl_eval_phase1.py \
      --config_path configs/深圳/phase1/config_wind_canvas_p1_base.yml \
      --checkpoint /path/to/checkpoint.pth --split test \
      --out_dir results/phase1 --tag base
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agl_eval_common import (SHEAR_PAIRS, STRATA, TARGET_AGL, accum_hour,  # noqa: E402
                             accum_hour_shear, acc_metrics, agl_fields, build_masks,
                             destagger_canvas, load_norm_sigma, load_tables,
                             main_levels_idx, new_acc, new_acc_shear, pooled_rmse)
from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_data.dataloader import make_dataloaders_and_samplers  # noqa: E402
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402
from src.dl_model.model_maker import make_model  # noqa: E402
from src.dl_model.si_follmer.si_follmer_framework import (  # noqa: E402
    StochasticInterpolantFollmer,
)
from src.utils.random_seed_helper import set_seeds  # noqa: E402

EXPERIMENT = "ExperimentSchrodingerBridgeWindCanvas"
_STAMP = re.compile(r'_(\d{8}T\d{6})\.npz$')


def pad16(t, mult=16):
    """UNet 逐级 2 倍下采样要求各维是 16 的倍数;复制填充(与画布边缘复制一致)。"""
    h, w = t.shape[-2:]
    H = ((h + mult - 1) // mult) * mult
    W = ((w + mult - 1) // mult) * mult
    if H == h and W == w:
        return t
    return torch.nn.functional.pad(t, (0, W - w, 0, H - h), mode='replicate')


def build_si(config, ckpt_path, device, use_ema):
    model = make_model(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['ema_model_state_dict'] if (
            use_ema and ckpt.get('ema_model_state_dict') is not None) else ckpt['model_state_dict']
    model.load_state_dict(sd)
    model.eval()
    return StochasticInterpolantFollmer(config=config.si, neural_net=model)


def split_denorm(t, levels, sigma):
    """标准化状态张量 (n_ch,100,121) -> 物理 (u,v,w,u10,v10),u/v/w 在 canvas 上。"""
    n = len(levels)
    u = t[0:n] * sigma['u']
    v = t[n:2 * n] * sigma['v']
    w = t[2 * n:2 * n + sigma['w'].shape[0]] * sigma['w']
    u10 = t[-2] * sigma['u10']
    v10 = t[-1] * sigma['v10']
    return u, v, w, u10, v10


def main():
    ap = argparse.ArgumentParser(description="阶段 1 AGL 评估")
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--out_dir", default="results/phase1")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", default="model", choices=["model", "ema"])
    ap.add_argument("--max_frames", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    tag = args.tag or os.path.basename(args.config_path).replace("config_wind_canvas_p1_", "") \
        .replace(".yml", "")
    config = load_config(EXPERIMENT, args.config_path)
    device = torch.device(args.device)
    set_seeds(config.train.seed)
    # 评估不做随机裁剪:全画布(100,121);dataloader 会强制 hr_cropped_shape = hr_data_shape
    config.data.hr_data_shape = [100, 121]
    config.data.hr_cropped_shape = [100, 121]

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=config.loader, dataset_config=config.data,
        world_size=None, rank=None, train_valid_test_kinds=[args.split])
    loader = dict_loaders[args.split]
    ds = loader.dataset
    print("评估 {}: {} 帧(共 {} 帧)".format(args.split, len(ds),
                                               len(ds) if args.max_frames <= 0 else args.max_frames))

    levels = list(config.data.target_levels)
    statics = CanvasStatics(config.data.statics_dir)
    tables = load_tables(statics, levels)
    sigma = load_norm_sigma(os.path.join(config.data.statics_dir, "normalize_config.json"),
                            config.data.scheme, levels)
    urban = np.asarray(statics.d['urban_fine'], dtype=np.float32)
    mask_idx = main_levels_idx()

    si = build_si(config, args.checkpoint, device, args.weights == "ema")

    n_lev_agl = len(TARGET_AGL)
    hours, acc_model, acc_model_w = [], [], []
    accs, accs_y0 = [], []          # 逐小时累加量,供 92 号配对 bootstrap
    accs_sh = []                    # 切变累加量(模型预测 vs 真值),供阶段 3 表(99 号)

    done, cursor = 0, 0
    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y']
        y0 = batch['y0'].to(device)
        with torch.no_grad():
            pred, _ = si.sample_y1_bare_diffusion(
                y0=pad16(y0), y_cond=pad16(x), add_noise=False)
        pred = pred[:, :, :100, :121].detach().cpu()
        y0u = y0[:, :, :100, :121].detach().cpu()

        for k in range(pred.shape[0]):
            stamp = _STAMP.search(os.path.basename(ds.ps[cursor + k])).group(1)
            with np.load(os.path.join(config.data.coarse_dir,
                                      "c_{}_{}.npz".format(config.data.scheme, stamp))) as f:
                co = {kk: f[kk] for kk in f.keys()}
            coszen = statics.regrid_field(
                np.asarray(co['c_coszen'], dtype=np.float32)[None], 'mass')[0]
            pblh = statics.regrid_field(
                np.asarray(co['c_pblh'], dtype=np.float32)[None], 'mass')[0]

            # 真值/预测/y0 -> 物理 -> 原生质量点 -> AGL
            fields = {}
            for name, tensor in (('truth', y[k]), ('pred', pred[k]), ('y0', y0u[k])):
                u, v, w, u10, v10 = split_denorm(tensor, levels, sigma)
                m = destagger_canvas(u.numpy(), v.numpy(), w.numpy(),
                                     u10.numpy(), v10.numpy())
                fields[name] = (m, agl_fields(m[0], m[1], m[2], m[3], m[4], tables))
            masks = build_masks(fields['truth'][1][0], fields['truth'][1][1],
                                coszen, pblh, urban)
            acc_h, acc_h_y0 = new_acc(n_lev_agl), new_acc(n_lev_agl)
            accum_hour(acc_h, fields['pred'][1][0], fields['pred'][1][1], fields['pred'][1][2],
                       fields['truth'][1][0], fields['truth'][1][1], fields['truth'][1][2],
                       masks)
            accum_hour(acc_h_y0, fields['y0'][1][0], fields['y0'][1][1], fields['y0'][1][2],
                       fields['truth'][1][0], fields['truth'][1][1], fields['truth'][1][2],
                       masks)
            acc_h_sh = new_acc_shear(len(SHEAR_PAIRS))
            accum_hour_shear(acc_h_sh, fields['pred'][1][0], fields['pred'][1][1],
                             fields['truth'][1][0], fields['truth'][1][1], masks)
            accs.append(acc_h)
            accs_y0.append(acc_h_y0)
            accs_sh.append(acc_h_sh)
            # 模式层(原生质量点)逐层:U/V 在 23 个质量层、W 在 24 个界面层,分开累加
            tu, tv, tw = fields['truth'][0][0], fields['truth'][0][1], fields['truth'][0][2]
            pu, pv, pw = fields['pred'][0][0], fields['pred'][0][1], fields['pred'][0][2]
            ncells = float(tu.shape[1] * tu.shape[2])
            e2 = ((pu - tu) ** 2 + (pv - tv) ** 2).sum(axis=(1, 2))
            we2 = ((pw - tw) ** 2).sum(axis=(1, 2))
            acc_model.append(np.stack([np.full(e2.shape[0], ncells), e2], axis=-1))
            acc_model_w.append(np.stack([np.full(we2.shape[0], ncells), we2], axis=-1))
            hours.append(datetime.strptime(stamp, '%Y%m%dT%H%M%S').strftime('%Y-%m-%dT%H:%M:%S'))
        cursor += pred.shape[0]
        done += pred.shape[0]
        if done % 24 == 0:
            print("  ... {} 帧".format(done))
        if args.max_frames > 0 and done >= args.max_frames:
            break

    acc = np.stack(accs)            # (n_hours, 11, 11, 6)
    acc_y0 = np.stack(accs_y0)
    acc_sh = np.stack(accs_sh)      # (n_hours, 8, 11, 6):预测 vs 真值的矢量切变误差
    acc_tot = acc.sum(axis=0)
    metrics = acc_metrics(acc_tot)
    out = {
        'tag': tag, 'split': args.split, 'n_hours': len(hours),
        'config_path': args.config_path, 'checkpoint': args.checkpoint,
        'weights': args.weights, 'scheme': config.data.scheme,
        'target_levels': levels, 'agl_targets': TARGET_AGL.tolist(),
        'strata': STRATA, 'speed_bins': [3.0, 7.0],
        'main_rmse_vec': pooled_rmse(acc_tot, 'all', mask_idx)[0],
        'rmse_vec_all_per_level': metrics['rmse_vec'][:, STRATA.index('all')].tolist(),
        'rmse_w_all_per_level': metrics['rmse_w'][:, STRATA.index('all')].tolist(),
        'mae_vec_all_per_level': metrics['mae_vec'][:, STRATA.index('all')].tolist(),
        'speed_bias_all_per_level': metrics['speed_bias'][:, STRATA.index('all')].tolist(),
        'dir_err_all_per_level': metrics['dir_err_deg'][:, STRATA.index('all')].tolist(),
        'by_stratum_main': {s: pooled_rmse(acc_tot, s, mask_idx)[0] for s in STRATA},
        'n_cells_main': {s: pooled_rmse(acc_tot, s, mask_idx)[1] for s in STRATA},
        'acc_y0_note': 'y0(粗端重网格)的逐小时累加量存在 npz 的 acc_y0,由 92 号脚本聚合',
    }
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    npz_path = os.path.join(args.out_dir, "{}_perhour.npz".format(tag))
    np.savez_compressed(
        npz_path,
        hours=np.array(hours),
        acc_agl=acc,
        acc_model=np.stack(acc_model),
        acc_model_w=np.stack(acc_model_w),
        acc_y0=acc_y0,
        acc_sh=acc_sh,
    )
    json_path = os.path.join(args.out_dir, "{}_summary.json".format(tag))
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ('tag', 'n_hours', 'main_rmse_vec')},
                     ensure_ascii=False))
    print("写出 {} / {}".format(npz_path, json_path))


if __name__ == "__main__":
    main()
