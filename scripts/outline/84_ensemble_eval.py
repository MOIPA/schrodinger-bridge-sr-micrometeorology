# -*- coding: utf-8 -*-
"""阶段 3 架构对比集合评估(84):checkpoint -> 集合预测 -> 反标准化 -> AGL 11 层 -> 逐时指标。

与 90_agl_eval_phase1.py 同口径同链路(pad16 -> 模型 -> 裁回(100,121) -> split_denorm
-> destagger_canvas -> agl_fields -> accum_hour / accum_hour_shear -> 逐时 npz +
summary json;切变层对/分层掩码/主指标层完全一致)。差别只在"模型 -> 单帧预测":

  si       : 冻结 SI(config.model,make_model)采样 N 次
             sample_y1_bare_diffusion(add_noise=True)  -> 均值 -> AGL 链
             (90 用 add_noise=False 的确定性 ODE;集合评估要随机性,故 add_noise=True;
              每帧 RNG 由 --seed 与 batch 序号确定,同 seed 重跑可复现);
  reg      : RegressionModel 单次前向 == 均值(N=1,确定性);
  two_step : 冻结回归网(y1_reg = y0 + net_reg(y0,x,gamma=1)) + EDMCorrector Heun 采样
             N 次(seed 局部 generator,不扰动全局 RNG)-> 均值 -> AGL 链。

分布类指标(N>1,写 <tag><out_suffix>_ensemble.json):
  - spread        : 逐像素样本标准差(ddof=1)的池化均值,vec = sqrt(var_u+var_v);
  - spread-skill  : 每小时在(主指标层 x 像素)上按 spread 四分位分组,组内集合均值的
                    矢量 RMSE(池化),spread 大 -> RMSE 大 说明离散度能指示误差;
  - rank histogram: 真值风速的秩(样本中严格小于真值的个数,+0.5 计并列)-> counts;
  - 样本式 CRPS   : fair 估计(Hersbach 2000)于风速,逐层像素 x 小时池化均值;N=1 记 null。

产物(绝不写不带 suffix 的既有文件名;所有文件都带 <out_suffix>):
  {out_dir}/{tag}{out_suffix}_perhour.npz   键:acc_agl/acc_sh/acc_y0/hours 与 90 逐键同义
                                            (N>1 时另存 acc_spread_vec/acc_spread_w/acc_ss)
  {out_dir}/{tag}{out_suffix}_summary.json  与 90 同键(main_rmse_vec 等)+ mode/n_ensemble 等
  {out_dir}/{tag}{out_suffix}_ensemble.json N>1 时的分布类指标

用法(需要 torch 的环境,仓库根目录;tag 前缀决定配置目录:p2_ -> phase2,其余 -> phase3_arch):
  python scripts/outline/84_ensemble_eval.py --mode si --tags p3_arch_swin,p3_arch_unet_s2 \
      --n_ensemble 16 --out_dir results/phase3_arch --device cuda:0
  python scripts/outline/84_ensemble_eval.py --mode si --tags p2_l1r2_lr2e4 \
      --n_ensemble 16 --out_suffix _ens16 --out_dir results/phase3_arch --device cuda:0
  python scripts/outline/84_ensemble_eval.py --mode reg --tags p3_arch_reg,p3_arch_reg_s2 ...
  python scripts/outline/84_ensemble_eval.py --mode two_step --tags p3_arch_edm \
      --edm_tags p3_arch_edm --n_ensemble 16 ...
"""
import argparse
import importlib.util
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
from agl_eval_common import (SHEAR_PAIRS, STRATA, TARGET_AGL, acc_metrics,  # noqa: E402
                             accum_hour, accum_hour_shear, agl_fields, build_masks,
                             destagger_canvas, load_norm_sigma, load_tables,
                             main_levels_idx, new_acc, new_acc_shear, pooled_rmse)
from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402
from src.dl_model.edm_correction import EDMCorrector, EDMCorrectorConfig  # noqa: E402
from src.dl_model.regression_wrapper import RegressionModel  # noqa: E402
from src.utils.random_seed_helper import set_seeds  # noqa: E402

EXPERIMENT = "ExperimentSchrodingerBridgeWindCanvas"
_STAMP = re.compile(r'_(\d{8}T\d{6})\.npz$')

MODE_NOTE = {
    'si': 'SI 冻结采样 add_noise=True 采样 N 次取均值(随机;与 90 的确定性 ODE 不同口径)',
    'reg': '回归网单次前向(确定性,y1 = y0 + net(y0, x, gamma=1));N=1 无分布类指标',
    'two_step': '回归网 + EDM Heun 采样 N 次取均值(seed 局部 generator)',
}


def _load_90():
    """90_agl_eval_phase1.py 文件名以数字开头,importlib 载入以复用其 pad16/split_denorm/build_si。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '90_agl_eval_phase1.py')
    spec = importlib.util.spec_from_file_location('agl_eval_90', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EV90 = [None]


def ev90():
    """延迟载入 90(其模块级导入 dataloader -> sklearn;纯函数/合成自检无需该依赖)。"""
    if _EV90[0] is None:
        _EV90[0] = _load_90()
    return _EV90[0]


def torch_load_any(path, device):
    """torch.load 兼容新旧 torch(旧版无 weights_only 参数)。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# 预测器:统一返回 (N, B, C, Hp, Wp) 的 padding 后画布样本(N=1 表示确定性单样本)
# ---------------------------------------------------------------------------
def predictor_si(si, n_samples, base_seed):
    """SI:每帧循环 n_samples 次随机采样(add_noise=True),返回堆叠样本。"""
    def predict(batch_idx, y0_pad, x_pad):
        outs = []
        for i in range(int(n_samples)):
            torch.manual_seed(int(base_seed) + 100003 * batch_idx + i)
            y1, _ = si.sample_y1_bare_diffusion(
                y0=y0_pad, y_cond=x_pad, add_noise=True)
            outs.append(y1)
        return torch.stack(outs, dim=0)
    return predict


def predictor_reg(reg):
    """回归步:单次前向 = 预测本身(确定性)。"""
    def predict(batch_idx, y0_pad, x_pad):
        y1, _ = reg.sample_y1_bare_diffusion(y0=y0_pad, y_cond=x_pad)
        return y1.unsqueeze(0)
    return predict


def predictor_two_step(reg, corrector, n_samples, steps, base_seed):
    """两步法:冻结回归网产出 y1_reg,EDM 订正器采样 n_samples 次。"""
    def predict(batch_idx, y0_pad, x_pad):
        y1_reg, _ = reg.sample_y1_bare_diffusion(y0=y0_pad, y_cond=x_pad)
        return corrector.sample(
            y0=y0_pad, y1_reg=y1_reg, y_cond=x_pad,
            n_samples=int(n_samples), steps=int(steps),
            seed=int(base_seed) + 100003 * batch_idx)
    return predict


def tag_paths(tag, cfg_root="configs/深圳", ck_root=None):
    """tag -> (config, checkpoint);p2_ 前缀走 phase2,其余走 phase3_arch。"""
    phase = "phase2" if tag.startswith("p2_") else "phase3_arch"
    if ck_root is None:
        ck_root = os.path.join("data", "DL_result", EXPERIMENT)
    return (os.path.join(cfg_root, phase, "config_wind_canvas_{}.yml".format(tag)),
            os.path.join(ck_root, "config_wind_canvas_{}".format(tag), "checkpoint.pth"))


def load_reg_model(reg_cfg, device, ckpt_path, weights="model"):
    """建 RegressionModel 并按 SI 训练脚本的 checkpoint 口径加载(裸键/前缀键均可)。"""
    reg = RegressionModel(reg_cfg.model).to(device)
    ckpt = torch_load_any(ckpt_path, device)
    sd = ckpt
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['ema_model_state_dict'] if (
            weights == "ema" and ckpt.get('ema_model_state_dict') is not None
        ) else ckpt['model_state_dict']
    reg.load_state_dict(sd)
    reg.eval()
    return reg


def build_predictor(mode, config, device, ckpt_path, weights, n_ensemble, seed,
                    edm_ckpt_path=None, reg_weights="model", cfg_path=None):
    """按 mode 建预测器;ckpt_path 是被评模型的 checkpoint,two_step 另给 edm_ckpt_path。"""
    if mode == 'si':
        si = ev90().build_si(config, ckpt_path, device, weights == "ema")
        return predictor_si(si, n_ensemble, seed)

    if mode == 'reg':
        return predictor_reg(load_reg_model(config, device, ckpt_path, weights))

    if mode == 'two_step':
        ckpt = torch_load_any(ckpt_path, device)
        edm_raw = ckpt.get('edm_config') if isinstance(ckpt, dict) else None
        if not edm_raw:
            raise SystemExit(
                "{} 的 checkpoint 缺 'edm_config' 键,无法重建订正器;"
                "two_step 只支持 train_edm_correction.py 的 checkpoint".format(ckpt_path))
        edm_cfg = EDMCorrectorConfig(**edm_raw)
        if edm_cfg.sigma_data is None:
            raise SystemExit("edm_config.sigma_data 为空({});训练脚本会写入该值".format(
                ckpt_path))
        corrector = EDMCorrector(edm_cfg).to(device)
        sd = ckpt['ema_model_state_dict'] if (
            weights == "ema" and ckpt.get('ema_model_state_dict') is not None
        ) else ckpt['model_state_dict']
        corrector.net.load_state_dict(sd)
        corrector.eval()
        # 冻结回归网:优先用 edm checkpoint 里记录的训练期路径,退回 tag 拼接
        reg_info = (ckpt.get('reg') or {}) if isinstance(ckpt, dict) else {}
        reg_ckpt = reg_info.get('reg_checkpoint_path')
        reg_cfg_path = reg_info.get('reg_config_path')
        if not reg_ckpt or not os.path.isfile(str(reg_ckpt)):
            reg_tag = reg_info.get('reg_tag')
            if not reg_tag:
                raise SystemExit(
                    "{} 缺 reg 记录(reg_tag/reg_checkpoint_path),无法定位冻结回归网".format(
                        ckpt_path))
            reg_ckpt = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)),
                                    "config_wind_canvas_{}".format(reg_tag),
                                    "checkpoint.pth")
        if not reg_cfg_path or not os.path.isfile(str(reg_cfg_path)):
            reg_cfg_path = os.path.join(
                os.path.dirname(str(cfg_path or "")),
                "config_wind_canvas_{}.yml".format(reg_info.get('reg_tag', "")))
        if not os.path.isfile(str(reg_cfg_path)):
            raise SystemExit("找不到回归配置 {}".format(reg_cfg_path))
        reg_cfg = load_config(EXPERIMENT, reg_cfg_path)
        reg = load_reg_model(reg_cfg, device, reg_ckpt, reg_weights)
        reg.requires_grad_(False)
        return predictor_two_step(reg, corrector, n_ensemble, edm_cfg.steps, seed)

    raise ValueError("未知 mode: {}".format(mode))


# ---------------------------------------------------------------------------
# 分布类指标工具(纯 numpy;82 号自检直接 import 复用)
# ---------------------------------------------------------------------------
def crps_sample(samples, truth):
    """样本式 CRPS(fair 估计,Hersbach 2000),samples (N, ...),truth (...) 或 (1,...)。

    CRPS = mean_i|x_i - y| - (1/(N(N-1))) Σ_{i<j}|x_i - x_j|;要求 N>=2。
    """
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if truth.shape != samples.shape[1:]:
        truth = truth.reshape(samples.shape[1:])
    n = samples.shape[0]
    if n < 2:
        raise ValueError("crps_sample 需要 N>=2(N=1 无样本式 CRPS)")
    x = np.sort(samples, axis=0)
    term1 = np.abs(x - truth[None, ...]).mean(axis=0)
    k = np.arange(n, dtype=np.float64).reshape((n,) + (1,) * truth.ndim)
    pair_sum = (x * (2.0 * k + 1.0 - n)).sum(axis=0)   # Σ_{i<j}(x_j - x_i)(升序恒等式)
    return term1 - pair_sum / (n * (n - 1.0))


def rank_of_truth(samples, truth):
    """真值在 N 个样本中的秩:严格小于的个数 + 0.5 x 并列个数(连续场并列≈0)。"""
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if truth.shape != samples.shape[1:]:
        truth = truth.reshape(samples.shape[1:])
    return ((samples < truth[None, ...]).sum(axis=0)
            + 0.5 * (samples == truth[None, ...]).sum(axis=0))


def rank_histogram(ranks, n_samples, n_layers):
    """(n_layers, ny, nx) 的秩 -> (n_layers, N+1) counts(箱 i 覆盖 [i-0.5, i+0.5))。"""
    edges = np.arange(-0.5, n_samples + 1.5, 1.0)
    hist = np.stack([np.histogram(ranks[l], bins=edges)[0]
                     for l in range(n_layers)], axis=0)
    return hist.astype(np.int64)


# ---------------------------------------------------------------------------
# 单 tag 评估
# ---------------------------------------------------------------------------
def evaluate(tag, mode, config, loader, static_ctx, predictor, out_dir, out_suffix,
             n_ensemble, split, weights, max_frames=0, meta=None, verbose=True):
    """跑一个 tag:集合预测 -> 均值 -> 与 90 同链的逐时累加 + 分布类指标 -> 写产物。

    predictor(batch_idx, y0_pad, x_pad) -> (N, B, C, Hp, Wp) padding 空间样本。
    """
    EV90 = ev90()   # 延迟载入 90(pad16/split_denorm);此处起才需要 dataloader 依赖
    statics = static_ctx['statics']
    tables = static_ctx['tables']
    sigma = static_ctx['sigma']
    urban = static_ctx['urban']
    ds = loader.dataset
    device = static_ctx['device']
    levels = list(config.data.target_levels)
    n_lev_agl = len(TARGET_AGL)
    mask_idx = main_levels_idx()
    main_idx = mask_idx

    hours = []
    accs, accs_y0, accs_sh = [], [], []
    spread_vec_f, spread_w_f, ss_f, ss_edges_f, rank_f, crps_f = [], [], [], [], [], []
    done, cursor = 0, 0

    for bi, batch in enumerate(loader):
        x = batch['x'].to(device)
        y = batch['y']
        y0 = batch['y0'].to(device)
        with torch.no_grad():
            samples = predictor(bi, EV90.pad16(y0), EV90.pad16(x))
        # (N,B,C,Hp,Wp) -> 裁回画布 (N,B,C,100,121) 的 numpy
        samples = samples[:, :, :, :100, :121].detach().cpu().float().numpy()
        y0u = y0[:, :, :100, :121].detach().cpu()
        n_s = samples.shape[0]
        if n_s != n_ensemble:
            raise RuntimeError("预测器返回样本数 {} != n_ensemble {}".format(
                n_s, n_ensemble))

        for k in range(samples.shape[1]):
            stamp = _STAMP.search(os.path.basename(ds.ps[cursor + k])).group(1)
            with np.load(os.path.join(config.data.coarse_dir,
                                      "c_{}_{}.npz".format(config.data.scheme, stamp))) as f:
                co = {kk: f[kk] for kk in f.keys()}
            coszen = statics.regrid_field(
                np.asarray(co['c_coszen'], dtype=np.float32)[None], 'mass')[0]
            pblh = statics.regrid_field(
                np.asarray(co['c_pblh'], dtype=np.float32)[None], 'mass')[0]

            # 真值 / y0 / 各样本 -> 物理 -> 原生质量点 -> AGL
            fields = {}
            for name, tensor in (('truth', y[k]), ('y0', y0u[k])):
                u, v, w, u10, v10 = EV90.split_denorm(tensor, levels, sigma)
                m = destagger_canvas(u.numpy(), v.numpy(), w.numpy(),
                                     u10.numpy(), v10.numpy())
                fields[name] = agl_fields(m[0], m[1], m[2], m[3], m[4], tables)
            sample_agl = []                       # [(au, av, aw), ...] 每样本
            for i in range(n_s):
                t = torch.from_numpy(samples[i, k])
                u, v, w, u10, v10 = EV90.split_denorm(t, levels, sigma)
                m = destagger_canvas(u.numpy(), v.numpy(), w.numpy(),
                                     u10.numpy(), v10.numpy())
                sample_agl.append(agl_fields(m[0], m[1], m[2], m[3], m[4], tables))
            mean = [np.stack([sample_agl[i][c] for i in range(n_s)]).mean(axis=0)
                    for c in range(3)]

            masks = build_masks(fields['truth'][0], fields['truth'][1], coszen, pblh, urban)
            acc_h = new_acc(n_lev_agl)
            accum_hour(acc_h, mean[0], mean[1], mean[2],
                       fields['truth'][0], fields['truth'][1], fields['truth'][2], masks)
            acc_h_y0 = new_acc(n_lev_agl)
            accum_hour(acc_h_y0, fields['y0'][0], fields['y0'][1], fields['y0'][2],
                       fields['truth'][0], fields['truth'][1], fields['truth'][2], masks)
            acc_h_sh = new_acc_shear(len(SHEAR_PAIRS))
            accum_hour_shear(acc_h_sh, mean[0], mean[1],
                             fields['truth'][0], fields['truth'][1], masks)
            accs.append(acc_h)
            accs_y0.append(acc_h_y0)
            accs_sh.append(acc_h_sh)

            # ---- 分布类指标(N>1)----
            if n_s > 1:
                su = np.stack([sample_agl[i][0] for i in range(n_s)])   # (N,11,99,120)
                sv = np.stack([sample_agl[i][1] for i in range(n_s)])
                sw = np.stack([sample_agl[i][2] for i in range(n_s)])
                spread_vec = np.sqrt(su.var(axis=0, ddof=1) + sv.var(axis=0, ddof=1))
                spread_w = np.sqrt(sw.var(axis=0, ddof=1))
                spread_vec_f.append(spread_vec.sum(axis=(1, 2)))
                spread_w_f.append(spread_w.sum(axis=(1, 2)))
                # spread-skill:主指标层,按每小时 spread 四分位分组
                err2 = ((mean[0] - fields['truth'][0]) ** 2
                        + (mean[1] - fields['truth'][1]) ** 2)[main_idx]
                sp_main = spread_vec[main_idx]
                qs = np.quantile(sp_main, [0.25, 0.5, 0.75])
                bins = np.digitize(sp_main, qs)
                ss_h = np.zeros((4, 3), dtype=np.float64)   # [n, se2, spread_sum]
                ss_edges_f.append(qs)
                for b in range(4):
                    mb = (bins == b)
                    ss_h[b, 0] = float(mb.sum())
                    ss_h[b, 1] = float((err2 * mb).sum())
                    ss_h[b, 2] = float((sp_main * mb).sum())
                ss_f.append(ss_h)
                # rank histogram(风速)
                speed_s = np.sqrt(su ** 2 + sv ** 2)
                speed_t = np.sqrt(fields['truth'][0] ** 2 + fields['truth'][1] ** 2)
                rank_f.append(rank_histogram(rank_of_truth(speed_s, speed_t), n_s, n_lev_agl))
                # CRPS(风速,fair;逐层像素均值)
                crps_f.append(crps_sample(speed_s, speed_t).mean(axis=(1, 2)))
            hours.append(datetime.strptime(stamp, '%Y%m%dT%H%M%S').strftime(
                '%Y-%m-%dT%H:%M:%S'))
        cursor += samples.shape[1]
        done += samples.shape[1]
        if verbose and done % 24 < samples.shape[1]:
            print("  ... {} 帧".format(done))
        if max_frames > 0 and done >= max_frames:
            break

    if not hours:
        raise SystemExit("FAIL {}: 评估 0 帧".format(tag))

    acc = np.stack(accs)             # (T,11,11,6)
    acc_y0 = np.stack(accs_y0)
    acc_sh = np.stack(accs_sh)       # (T,8,11,6)
    acc_tot = acc.sum(axis=0)
    metrics = acc_metrics(acc_tot)
    t_frames = len(hours)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    stem = "{}{}".format(tag, out_suffix)
    npz_path = os.path.join(out_dir, "{}_perhour.npz".format(stem))
    npz_payload = dict(hours=np.array(hours), acc_agl=acc, acc_y0=acc_y0, acc_sh=acc_sh)
    if n_ensemble > 1:
        npz_payload['acc_spread_vec'] = np.stack(spread_vec_f)     # (T,11) 池化 spread 之和
        npz_payload['acc_spread_w'] = np.stack(spread_w_f)
        npz_payload['acc_ss'] = np.stack(ss_f)                     # (T,4,3):n/se2/spread_sum
    np.savez_compressed(npz_path, **npz_payload)

    out = {
        'tag': tag, 'split': split, 'n_hours': t_frames,
        'config_path': meta.get('config_path') if meta else None,
        'checkpoint': meta.get('checkpoint') if meta else None,
        'weights': weights, 'scheme': config.data.scheme,
        'target_levels': levels, 'agl_targets': TARGET_AGL.tolist(),
        'strata': STRATA, 'speed_bins': [3.0, 7.0],
        'mode': mode, 'n_ensemble': int(n_ensemble), 'out_suffix': out_suffix,
        'sample_protocol': MODE_NOTE[mode],
        'main_rmse_vec': pooled_rmse(acc_tot, 'all', mask_idx)[0],
        'rmse_vec_all_per_level': metrics['rmse_vec'][:, STRATA.index('all')].tolist(),
        'rmse_w_all_per_level': metrics['rmse_w'][:, STRATA.index('all')].tolist(),
        'mae_vec_all_per_level': metrics['mae_vec'][:, STRATA.index('all')].tolist(),
        'speed_bias_all_per_level': metrics['speed_bias'][:, STRATA.index('all')].tolist(),
        'dir_err_all_per_level': metrics['dir_err_deg'][:, STRATA.index('all')].tolist(),
        'by_stratum_main': {s: pooled_rmse(acc_tot, s, mask_idx)[0] for s in STRATA},
        'n_cells_main': {s: pooled_rmse(acc_tot, s, mask_idx)[1] for s in STRATA},
        'perhour_npz': npz_path,
        'acc_y0_note': 'y0(粗端重网格)的逐小时累加量存在 npz 的 acc_y0',
    }
    if meta:
        out.update({k: v for k, v in meta.items() if k not in out})
    summary_path = os.path.join(out_dir, "{}_summary.json".format(stem))
    with open(summary_path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    ens_path = None
    ens = None
    if n_ensemble > 1:
        rank_hist = np.stack(rank_f).sum(axis=0)                    # (11,N+1)
        n_rank = float(rank_hist.sum())
        rank_mean = (rank_hist * np.arange(n_ensemble + 1)).sum(axis=1) / np.maximum(
            rank_hist.sum(axis=1), 1)
        ss = np.stack(ss_f).sum(axis=0)                             # (4,3)
        ss_rmse = np.sqrt(ss[:, 1] / np.maximum(ss[:, 0], 1.0))
        ss_spread = ss[:, 2] / np.maximum(ss[:, 0], 1.0)
        q_edges = np.stack(ss_edges_f).mean(axis=0)                 # 各帧四分位阈值的均值
        bins_meta = [
            ('0-25%', None, float(q_edges[0])),
            ('25-50%', float(q_edges[0]), float(q_edges[1])),
            ('50-75%', float(q_edges[1]), float(q_edges[2])),
            ('75-100%', float(q_edges[2]), None),
        ]
        ens = {
            'tag': tag, 'mode': mode, 'n_ensemble': int(n_ensemble),
            'out_suffix': out_suffix, 'split': split, 'n_hours': t_frames,
            'agl_targets': TARGET_AGL.tolist(),
            'spread_vec_per_level': (np.stack(spread_vec_f).sum(axis=0) / t_frames).tolist(),
            'spread_w_per_level': (np.stack(spread_w_f).sum(axis=0) / t_frames).tolist(),
            'spread_note': '逐像素样本标准差(ddof=1)在 (99x120) 像素上的均值;'
                           'vec = sqrt(var_u+var_v),w 单独;跨小时均值',
            'spread_skill': {
                'levels': 'AGL 10-500 m(主指标层)',
                'bin_edges_quantile': [0.25, 0.5, 0.75],
                'bins': [
                    {'name': bins_meta[b][0], 'spread_lo': bins_meta[b][1],
                     'spread_hi': bins_meta[b][2],
                     'spread_mean': float(ss_spread[b]),
                     'rmse': float(ss_rmse[b]), 'n_cells': float(ss[b, 0])}
                    for b in range(4)
                ],
                'note': '每小时在(主指标层 x 99 x 120)像素上按该帧 spread 的四分位分组;'
                        'RMSE = 集合均值矢量误差的池化;n_cells = Σ 像素数(跨小时);'
                        'spread_lo/hi = 各帧四分位阈值的均值',
            },
            'rank_hist_per_level': rank_hist.tolist(),
            'rank_hist_main_pooled': rank_hist[main_idx].sum(axis=0).tolist(),
            'rank_expected_per_bin': n_rank / (n_ensemble + 1),
            'rank_mean_per_level': rank_mean.tolist(),
            'rank_note': '秩 = 样本风速中严格小于真值的个数(+0.5 计并列);'
                         '理想集合每箱期望 = 总计数/(N+1);rank_mean 期望 = N/2',
            'crps_speed_per_level': (np.stack(crps_f).mean(axis=0)).tolist(),
            'crps_speed_main': float(np.stack(crps_f).mean(axis=0)[main_idx].mean()),
            'crps_note': '样本式 CRPS(fair;Hersbach 2000)于风速,逐层像素 x 小时池化均值;'
                         'N=1 时为 null(无分布)',
        }
        ens_path = os.path.join(out_dir, "{}_ensemble.json".format(stem))
        with open(ens_path, 'w') as f:
            json.dump(ens, f, indent=1, ensure_ascii=False)
        out['ensemble_json'] = ens_path
        with open(summary_path, 'w') as f:
            json.dump(out, f, indent=1, ensure_ascii=False)

    print(json.dumps({'tag': tag, 'mode': mode, 'n_ensemble': int(n_ensemble),
                      'n_hours': t_frames, 'main_rmse_vec': out['main_rmse_vec']},
                     ensure_ascii=False))
    print("写出 {} / {}{}".format(
        npz_path, summary_path, (" / " + ens_path) if ens_path else ""))
    return out, ens


def build_loader(config, split):
    """与 90 相同的取数路径:全画布 (100,121),不做随机裁剪。

    dataloader 依赖 sklearn,延迟到这里导入,使 84 的纯函数(crps/rank/单帧链路)
    在精简环境(合成自检 82)可被 importlib 复用。
    """
    from src.dl_data.dataloader import make_dataloaders_and_samplers
    config.data.hr_data_shape = [100, 121]
    config.data.hr_cropped_shape = [100, 121]
    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=config.loader, dataset_config=config.data,
        world_size=None, rank=None, train_valid_test_kinds=[split])
    return dict_loaders[split]


def build_static_ctx(config, device):
    """statics + AGL 表 + 反标准化 sigma + 城郊掩码(与 90 同源)。"""
    levels = list(config.data.target_levels)
    statics = CanvasStatics(config.data.statics_dir)
    tables = load_tables(statics, levels)
    sigma = load_norm_sigma(os.path.join(config.data.statics_dir, "normalize_config.json"),
                            config.data.scheme, levels)
    urban = np.asarray(statics.d['urban_fine'], dtype=np.float32)
    return {'statics': statics, 'tables': tables, 'sigma': sigma, 'urban': urban,
            'device': device}


def main():
    ap = argparse.ArgumentParser(description="阶段 3 架构对比集合评估")
    ap.add_argument("--mode", required=True, choices=["si", "reg", "two_step"])
    ap.add_argument("--tags", required=True, help="逗号分隔的评估 tag")
    ap.add_argument("--edm_tags", default=None,
                    help="two_step 模式:与 tags 一一对应的 EDM checkpoint tag")
    ap.add_argument("--n_ensemble", type=int, default=16)
    ap.add_argument("--out_suffix", default="",
                    help="输出文件名后缀(默认空;参考臂重评用 _ens16,避免与既有产物混淆)")
    ap.add_argument("--out_dir", default="results/phase3_arch")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", default="model", choices=["model", "ema"])
    ap.add_argument("--reg_weights", default="model", choices=["model", "ema"])
    ap.add_argument("--seed", type=int, default=0, help="采样种子(每帧按 batch 序号偏移)")
    ap.add_argument("--cfg_root", default="configs/深圳")
    ap.add_argument("--ck_root", default=None)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    edm_tags = [t.strip() for t in (args.edm_tags or "").split(',') if t.strip()]
    if args.mode == "two_step":
        if not edm_tags:
            raise SystemExit("two_step 需要 --edm_tags(与 --tags 一一对应)")
        if len(edm_tags) != len(tags):
            raise SystemExit("--edm_tags 数量 {} != --tags 数量 {}".format(
                len(edm_tags), len(tags)))
    elif edm_tags:
        raise SystemExit("--edm_tags 只在 two_step 模式使用")
    if args.n_ensemble < 1:
        raise SystemExit("--n_ensemble 至少为 1")
    n_eff = args.n_ensemble
    if args.mode == "reg" and n_eff != 1:
        print("[说明] reg 模式为确定性单次前向,忽略 --n_ensemble={}(按 1 处理);"
              "分布类指标需要 si/two_step 的集合".format(n_eff))
        n_eff = 1

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    for ti, tag in enumerate(tags):
        cfg_path, ck_path = tag_paths(tag, args.cfg_root, args.ck_root)
        if args.mode == "two_step":
            edm_cfg_path, ck_path = tag_paths(edm_tags[ti], args.cfg_root, args.ck_root)
            if not os.path.isfile(cfg_path):     # 评测名与 edm 产物名不同时,配置按 edm tag 取
                cfg_path = edm_cfg_path
        if not os.path.isfile(cfg_path):
            print("[跳过] {} 缺配置 {}".format(tag, cfg_path))
            continue
        if not os.path.isfile(ck_path):
            print("[跳过] {} 缺 checkpoint {}".format(tag, ck_path))
            continue
        config = load_config(EXPERIMENT, cfg_path)
        set_seeds(config.train.seed)
        loader = build_loader(config, args.split)
        static_ctx = build_static_ctx(config, device)
        predictor = build_predictor(
            args.mode, config, device, ck_path, args.weights, n_eff,
            args.seed, cfg_path=cfg_path, reg_weights=args.reg_weights)
        print("=== {} (mode={}, N={}, {} 帧, {}) ===".format(
            tag, args.mode, n_eff, len(loader.dataset), args.split))
        evaluate(tag, args.mode, config, loader, static_ctx, predictor,
                 args.out_dir, args.out_suffix, n_eff, args.split,
                 args.weights, max_frames=args.max_frames,
                 meta={'config_path': cfg_path, 'checkpoint': ck_path,
                       'edm_tag': edm_tags[ti] if edm_tags else None})


if __name__ == "__main__":
    main()
