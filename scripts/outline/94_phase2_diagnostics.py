# -*- coding: utf-8 -*-
"""阶段 2 诊断(逐 run):散度残差 / 能谱 / 极值,三类净效应同一套确定性采样。

- 采样/loader/反标准化与 90 号评估完全同源:90 的 pad16/build_si/split_denorm 直接
  import(90 文件名以数字开头,用 importlib 载入,与 agl_eval_common 载 60 的做法一致);
  评估口径同 90:dataloader 强制全画布 (100,121)、add_noise=False 纯 ODE、细端 σ 反标准化。
- 逐帧对 pred / truth(y1) / y0 三者同口径计算,以帧均值汇总:
  ① 散度残差:physics_canvas.divergence_rho_u 的可压缩 ∇·(ρu)
     (dx=si.phys_dx, dz=si.phys_dz, ρ=batch['rho'] 与 y 同窗口全画布);
     逐层 p95|D| / rms / frac(|D|>τ_k),τ=si.phys_div_tau(23 个 hinge 阈值);y0 也算(免费基线)。
  ② 能谱:去交错到质量点后 radial_log_spectra(逐层 u/v,32 箱),帧平均存 npz。
  ③ 极值:最低 10 层 + 10 m 的风速 max/min——每帧空间极值的帧间均值,以及
     整个 split 的时间×空间全局极值(大纲 T2.2 的"时间维"口径)。
- 输出 results/phase2/{tag}_diag.json(标量+逐层表)与 {tag}_diag.npz(谱数组,可关)。

用法(需要 torch 的环境,仓库根目录):
  python scripts/outline/94_phase2_diagnostics.py \
      --config_path configs/深圳/phase2/config_wind_canvas_p2_div_mid.yml \
      --checkpoint data/DL_result/ExperimentSchrodingerBridgeWindCanvas/config_wind_canvas_p2_div_mid/checkpoint.pth \
      --split test --tag p2_div_mid --out_dir results/phase2 --device cuda:0
smoke: 追加 --max_frames 2
阶段 1 基线配置没有 si.phys_*:用 --phys_from_config 指任意阶段 2 配置读取物理参数。
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agl_eval_common import load_norm_sigma  # noqa: E402
from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_data.dataloader import make_dataloaders_and_samplers  # noqa: E402
from src.dl_model.si_follmer import physics_canvas as pc  # noqa: E402
from src.utils.random_seed_helper import set_seeds  # noqa: E402

EXPERIMENT = "ExperimentSchrodingerBridgeWindCanvas"
SRC_NAMES = ('pred', 'truth', 'y0')


def _load_eval90():
    """载入 90_agl_eval_phase1.py(文件名以数字开头,只能 importlib;90 只定义函数不执行 main)。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '90_agl_eval_phase1.py')
    spec = importlib.util.spec_from_file_location('agl_eval_90', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EVAL90 = _load_eval90()


def resolve_phys(config, alt_config_path):
    """取 si.phys_dx / phys_dz / phys_div_tau;缺(阶段 1 基线配置)时从 alt 配置读。

    返回 (dx, dz_list, tau_list);仍缺则 SystemExit(提示先跑 86 生成阶段 2 配置)。
    """
    def _pick(cfg):
        si = cfg.si
        return float(si.phys_dx), si.phys_dz, si.phys_div_tau

    dx, dz, tau = _pick(config)
    if (dz is None or tau is None) and alt_config_path:
        dx_a, dz_a, tau_a = _pick(load_config(EXPERIMENT, alt_config_path))
        if dz is None:
            dx, dz = dx_a, dz_a
        if tau is None:
            tau = tau_a
    if dz is None or tau is None:
        raise SystemExit(
            "配置缺 si.phys_dz / si.phys_div_tau(阶段 1 配置没有,属正常);"
            "用 --phys_from_config <阶段 2 配置> 读取,或先跑 86 生成阶段 2 配置")
    return dx, list(dz), list(tau)


def _stats(div_abs_batch, tau):
    """(B,L,N) 的 |D| -> 逐帧逐层 (p95, rms, frac>tau),返回各 (B,L)。"""
    p95 = np.percentile(div_abs_batch, 95, axis=2)
    rms = np.sqrt((div_abs_batch ** 2).mean(axis=2))
    frac = (div_abs_batch > tau[None, :, None]).mean(axis=2)
    return p95, rms, frac


def main():
    ap = argparse.ArgumentParser(description="阶段 2 三类诊断(散度/能谱/极值)")
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--out_dir", default="results/phase2")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", default="model", choices=["model", "ema"])
    ap.add_argument("--max_frames", type=int, default=0, help="0 = 全部(冒烟用 2)")
    ap.add_argument("--save_spectra", default="true", choices=["true", "false"],
                    help="false 不算/不存谱数组(npz)")
    ap.add_argument("--nbins", type=int, default=32, help="谱的径向箱数")
    ap.add_argument("--phys_from_config", default=None,
                    help="本配置缺 si.phys_dz/phys_div_tau 时(阶段 1 基线)从该配置读")
    args = ap.parse_args()

    tag = args.tag or os.path.basename(args.config_path).replace("config_wind_canvas_", "") \
        .replace(".yml", "")
    config = load_config(EXPERIMENT, args.config_path)
    device = torch.device(args.device)
    set_seeds(config.train.seed)
    # 评估不做随机裁剪:全画布(100,121);与 90 号一致
    config.data.hr_data_shape = [100, 121]
    config.data.hr_cropped_shape = [100, 121]

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=config.loader, dataset_config=config.data,
        world_size=None, rank=None, train_valid_test_kinds=[args.split])
    loader = dict_loaders[args.split]
    ds = loader.dataset
    print("诊断 {}: {} 帧(共 {} 帧)".format(args.split, len(ds),
                                             len(ds) if args.max_frames <= 0 else args.max_frames))

    levels = list(config.data.target_levels)
    L = len(levels)
    sigma = load_norm_sigma(os.path.join(config.data.statics_dir, "normalize_config.json"),
                            config.data.scheme, levels)
    si = EVAL90.build_si(config, args.checkpoint, device, args.weights == "ema")

    dx, dz_list, tau_list = resolve_phys(config, args.phys_from_config)
    if len(dz_list) != L or len(tau_list) != L:
        raise SystemExit("si.phys_dz({}) / si.phys_div_tau({}) 长度应为 {} 层".format(
            len(dz_list), len(tau_list), L))
    dz_t = torch.as_tensor(dz_list, dtype=torch.float32, device=device)
    tau_np = np.asarray(tau_list, dtype=np.float64)
    ext_levels = list(range(min(10, L)))          # 最低 10 层;最后另加 10 m
    n_ext = len(ext_levels) + 1
    do_spec = args.save_spectra == "true"

    div_acc = {s: {'p95': np.zeros(L), 'rms': np.zeros(L), 'frac': np.zeros(L), 'n': 0}
               for s in SRC_NAMES}
    spec_acc = {s: np.zeros((L, 2, args.nbins)) for s in SRC_NAMES}
    ext_acc = {s: {'max': np.zeros(n_ext), 'min': np.zeros(n_ext),
                   'gmax': np.full(n_ext, -np.inf), 'gmin': np.full(n_ext, np.inf), 'n': 0}
               for s in SRC_NAMES}

    done = 0
    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y']
        y0 = batch['y0'].to(device)
        rho = batch['rho'].to(device)             # (B,L,100,121) 物理 kg/m³
        with torch.no_grad():
            pred, _ = si.sample_y1_bare_diffusion(
                y0=EVAL90.pad16(y0), y_cond=EVAL90.pad16(x), add_noise=False)
        pred = pred[:, :, :100, :121].detach().cpu()
        y0u = y0[:, :, :100, :121].detach().cpu()
        B = pred.shape[0]

        for s, t in (('pred', pred), ('truth', y), ('y0', y0u)):
            u, v, w, u10, v10 = EVAL90.split_denorm(t, levels, sigma)
            u, v, w = u.to(device), v.to(device), w.to(device)
            u10, v10 = u10.to(device), v10.to(device)

            # ① 散度残差 ∇·(ρu)(画布 C 网格,与训练损失同一算子)
            div = pc.divergence_rho_u(u, v, w, rho, dx, dz_t)          # (B,L,99,120)
            p95, rms, frac = _stats(div.abs().cpu().numpy().reshape(B, L, -1), tau_np)
            div_acc[s]['p95'] += p95.sum(axis=0)
            div_acc[s]['rms'] += rms.sum(axis=0)
            div_acc[s]['frac'] += frac.sum(axis=0)
            div_acc[s]['n'] += B

            # ② 能谱(去交错到质量点后逐层 u/v 径向 log 谱)
            if do_spec:
                um, vm, _ = pc.destagger_canvas(u, v, w)
                spec = pc.radial_log_spectra(um, vm, nbins=args.nbins)  # (B,L,2,nbins)
                spec_acc[s] += spec.cpu().numpy().sum(axis=0)

            # ③ 极值(最低 10 层 + 10 m 风速)
            spd = pc.windspeed_levels(u, v, u10, v10, ext_levels)      # (B,n_ext,99,120)
            smax = spd.amax(dim=(2, 3)).cpu().numpy()
            smin = spd.amin(dim=(2, 3)).cpu().numpy()
            ext_acc[s]['max'] += smax.sum(axis=0)
            ext_acc[s]['min'] += smin.sum(axis=0)
            ext_acc[s]['gmax'] = np.maximum(ext_acc[s]['gmax'], smax.max(axis=0))
            ext_acc[s]['gmin'] = np.minimum(ext_acc[s]['gmin'], smin.min(axis=0))
            ext_acc[s]['n'] += B

        done += B
        if done % 24 == 0:
            print("  ... {} 帧".format(done))
        if args.max_frames > 0 and done >= args.max_frames:
            break

    n = div_acc['pred']['n']
    if n == 0:
        raise SystemExit("一帧都没跑到,检查 --split / dataloader")
    low_idx = list(range(min(10, L)))
    div = {s: {'p95_abs': (div_acc[s]['p95'] / n).tolist(),
               'rms': (div_acc[s]['rms'] / n).tolist(),
               'frac_gt_tau': (div_acc[s]['frac'] / n).tolist()} for s in SRC_NAMES}
    div_scalar = {}
    for s in SRC_NAMES:
        div_scalar[s] = {
            'p95_abs_mean': float(np.mean(div[s]['p95_abs'])),
            'p95_abs_mean_low10': float(np.mean([div[s]['p95_abs'][i] for i in low_idx])),
            'frac_gt_tau_mean': float(np.mean(div[s]['frac_gt_tau'])),
            'frac_gt_tau_mean_low10': float(np.mean([div[s]['frac_gt_tau'][i] for i in low_idx])),
        }
    ext = {}
    for s in SRC_NAMES:
        m = ext_acc[s]['n']
        ext[s] = {
            'frame_mean_spatial_max': (ext_acc[s]['max'] / m).tolist(),
            'frame_mean_spatial_min': (ext_acc[s]['min'] / m).tolist(),
            'global_max': ext_acc[s]['gmax'].tolist(),      # 时间×空间全局(逐层)
            'global_min': ext_acc[s]['gmin'].tolist(),
            'global_max_all': float(ext_acc[s]['gmax'].max()),
            'global_min_all': float(ext_acc[s]['gmin'].min()),
        }

    # 高频段判定用的箱中心(半径 cycles/pixel);H'/W' 与谱算子一致(质量点 99×120)
    Hp, Wp = 99, 120
    edges = np.logspace(np.log10(1.0 / max(Hp, Wp)), np.log10(0.5), args.nbins + 1)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    npz_name = None
    if do_spec:
        npz_path = os.path.join(args.out_dir, "{}_diag.npz".format(tag))
        np.savez_compressed(npz_path, bin_edges=edges, nbins=args.nbins,
                            spec_pred=spec_acc['pred'] / n,
                            spec_truth=spec_acc['truth'] / n,
                            spec_y0=spec_acc['y0'] / n)
        npz_name = os.path.basename(npz_path)

    out = {
        'tag': tag, 'split': args.split, 'n_frames': n,
        'config_path': args.config_path, 'checkpoint': args.checkpoint,
        'weights': args.weights, 'n_levels': L,
        'dx_m': dx, 'dz_m': dz_list, 'phys_div_tau': tau_list,
        'phys_from_config': args.phys_from_config,
        'ext_levels': ext_levels + ['10m'],
        'divergence': div,
        'divergence_scalar': div_scalar,
        'extremes': ext,
        'spectra_npz': npz_name,
        'spec_bin_edges': edges.tolist(),
        'note': ('div=physics_canvas.divergence_rho_u 的可压缩散度 ∇·(ρu),逐层统计对帧取均值;'
                 'rms/p95 单位 kg m^-3 s^-1;spec=去交错质量点上逐层 u/v 径向 log 谱(帧平均,'
                 '箱中心>0.25 cycles/pixel 为高频段);ext 最后一项为 10 m;'
                 'global_max/min=全 split 时间×空间全局极值;y0 为粗端重网格免费基线'),
    }
    json_path = os.path.join(args.out_dir, "{}_diag.json".format(tag))
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print(json.dumps({
        'tag': tag, 'n_frames': n,
        'div_p95_mean': {s: round(div_scalar[s]['p95_abs_mean'], 8) for s in SRC_NAMES},
        'frac_gt_tau_mean': {s: round(div_scalar[s]['frac_gt_tau_mean'], 5) for s in SRC_NAMES},
        'global_max_all': {s: round(ext[s]['global_max_all'], 3) for s in SRC_NAMES},
        'global_min_all': {s: round(ext[s]['global_min_all'], 3) for s in SRC_NAMES},
    }, ensure_ascii=False))
    print("写出 {} / {}".format(json_path, npz_name or "(未存谱)"))


if __name__ == "__main__":
    main()
