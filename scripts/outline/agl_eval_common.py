# -*- coding: utf-8 -*-
"""阶段 1 AGL 评估公共部分:AGL 表、原生场换算、分层掩码、指标累加器、移动块 bootstrap。

口径(检查点 A,`docs/大纲阶段0_协议冻结.md` §3):
  预测与真值共用同一个 AGL 插值算子(60 号脚本的 agl_interp);10 m 层由 U10/V10 通道承担。
  指标先在"每小时 × 每层 × 每分层"上累加(n/平方误差和/...),排名脚本 92 再聚合 + 配对 bootstrap。
"""
import importlib.util
import json
import os

import numpy as np
from src.dl_data.wind_canvas_statics import TARGET_AGL, build_agl_table

# TARGET_AGL 由 src.dl_data.wind_canvas_statics 提供(训练侧插值表与评估侧同参,单一来源)
STRATA = ['all', 'day', 'night', 'urban', 'rural', 'weak', 'mid', 'strong',
          'pbl_low', 'pbl_ent', 'pbl_high']
SPEED_BINS = (3.0, 7.0)   # 真值风速分箱:m/s;弱 <3 / 中 3–7 / 强 ≥7
PBL_RATIO = (0.8, 1.2)    # 相对 PBLH 位置:混合层内 / 夹卷层附近 / 混合层之上
MAIN_LEVELS = [10, 30, 50, 70, 100, 150, 200, 300, 500]   # 主指标 10–500 m 的 AGL 层
SHEAR_PAIRS = [(MAIN_LEVELS[k], MAIN_LEVELS[k + 1])
               for k in range(len(MAIN_LEVELS) - 1)]      # 相邻主层对,共 8 对:(10,30)…(300,500)
SHEAR_DZ = np.diff(np.asarray(MAIN_LEVELS, dtype=np.float64))   # 各对间距 Δz (m)
N_ACC = 6                 # 每 (层, 分层) 存 6 个累加量:n, se2, se1, dspeed, dir_abs, we2


def _load_agl_interp():
    """60_agl_operator.py 文件名以数字开头,用 importlib 载入,保证算子唯一来源。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '60_agl_operator.py')
    spec = importlib.util.spec_from_file_location('agl_operator_60', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agl_interp


agl_interp = _load_agl_interp()


def load_tables(stats, levels):
    """AGL 插值表(只覆盖训练用的层集合,避免 idx 指向未训练层)。"""
    zagl_m = np.asarray(stats.d['zagl_mass_fine'], dtype=np.float64)[levels]
    zagl_i = np.asarray(stats.d['zagl_iface_fine'],
                        dtype=np.float64)[levels[0]:levels[-1] + 2]
    idx_m, w_m = build_agl_table(zagl_m, TARGET_AGL, True)
    idx_i, w_i = build_agl_table(zagl_i, TARGET_AGL, False)
    return {'idx_m': idx_m, 'w_m': w_m, 'idx_i': idx_i, 'w_i': w_i,
            'zagl_mass': zagl_m, 'n_lev_model': len(levels)}


def load_norm_sigma(norm_json_path, scheme, levels):
    """细端各分量的物理 σ(用于把标准化预测/ y0 反标准化)。"""
    with open(norm_json_path) as f:
        norms = json.load(f)
    fn = norms['fine'][scheme]
    return {
        'u': np.asarray(fn['u']['sigma'], dtype=np.float32)[levels][:, None, None],
        'v': np.asarray(fn['v']['sigma'], dtype=np.float32)[levels][:, None, None],
        'w': np.asarray(fn['w']['sigma'], dtype=np.float32)[
            levels[0]:levels[-1] + 2][:, None, None],
        'u10': float(fn['u10']['sigma']),
        'v10': float(fn['v10']['sigma']),
    }


def destagger_native(u, v, w, u10, v10):
    """原生交错场 -> 质量点:U (n,99,121)、V (n,100,120)、W/10 m (n,99,120) -> (n,99,120)。"""
    um = 0.5 * (u[..., :, :-1] + u[..., :, 1:])
    vm = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
    return um, vm, w, u10, v10


def destagger_canvas(u_c, v_c, w_c, u10_c, v10_c):
    """canvas (…,100,121) -> 原生质量点 (…,99,120)。U/V 去交错,W/10 m 直接切片。"""
    return destagger_native(u_c[..., :99, :121], v_c[..., :100, :120],
                            w_c[..., :99, :120], u10_c[..., :99, :120],
                            v10_c[..., :99, :120])


def agl_fields(u, v, w, u10, v10, tables):
    """质量点 (n_lev,99,120) + 10 m (99,120) -> AGL 11 层 (11,99,120),物理单位。"""
    au = agl_interp(u, tables['idx_m'], tables['w_m'], field10=u10)
    av = agl_interp(v, tables['idx_m'], tables['w_m'], field10=v10)
    aw = agl_interp(w, tables['idx_i'], tables['w_i'], field10=None)
    return au, av, aw


def _b(mask, n_lev):
    return np.broadcast_to(np.asarray(mask, dtype=bool), (n_lev, 99, 120))


def build_masks(truth_u, truth_v, coszen, pblh, urban):
    """每小时的分层掩码;真值/环境量决定弱强风、相对 PBLH,静态量决定城郊。

    相对 PBLH 的"高度"用 AGL 目标层高度本身(10..1000 m),与真值/预测的层定义一致。
    """
    n_lev = truth_u.shape[0]
    assert n_lev == len(TARGET_AGL), "build_masks 需要 AGL 层的真值场"
    speed = np.sqrt(truth_u ** 2 + truth_v ** 2)
    lo, hi = SPEED_BINS
    ratio = TARGET_AGL[:, None, None] / np.maximum(pblh[None, :, :], 1.0)
    r_lo, r_hi = PBL_RATIO
    masks = {
        'all': _b(np.ones((1, 1, 1)), n_lev),
        'day': _b(coszen > 0.01, n_lev),
        'night': _b(coszen <= 0.01, n_lev),
        'urban': _b(urban > 0.3, n_lev),
        'rural': _b(urban <= 0.3, n_lev),
        'weak': speed < lo,
        'mid': (speed >= lo) & (speed < hi),
        'strong': speed >= hi,
        'pbl_low': ratio < r_lo,
        'pbl_ent': (ratio >= r_lo) & (ratio <= r_hi),
        'pbl_high': ratio > r_hi,
    }
    return masks


def new_acc(n_lev):
    return np.zeros((n_lev, len(STRATA), N_ACC), dtype=np.float64)


def new_acc_shear(n_pairs=8):
    """切变累加量:(相邻层对, 分层, N_ACC);默认 8 对 = 10–30 … 300–500 m。"""
    return np.zeros((n_pairs, len(STRATA), N_ACC), dtype=np.float64)


def accum_hour(acc, pred_u, pred_v, pred_w, true_u, true_v, true_w, masks):
    """把一小时的误差按 (层, 分层) 累加。pred/true 均为 AGL 物理场 (n_lev,99,120)。"""
    du = pred_u - true_u
    dv = pred_v - true_v
    e2 = du * du + dv * dv
    e1 = np.sqrt(e2)
    sp = np.sqrt(pred_u ** 2 + pred_v ** 2)
    st = np.sqrt(true_u ** 2 + true_v ** 2)
    ds = sp - st
    # 风向循环误差:两矢量夹角绝对值(弧度)
    dth = np.abs(np.arctan2(pred_u * true_v - pred_v * true_u, pred_u * true_u + pred_v * true_v))
    dw2 = (pred_w - true_w) ** 2
    for si, s in enumerate(STRATA):
        m = masks[s]
        acc[:, si, 0] += m.sum(axis=(1, 2))
        acc[:, si, 1] += (e2 * m).sum(axis=(1, 2))
        acc[:, si, 2] += (e1 * m).sum(axis=(1, 2))
        acc[:, si, 3] += (ds * m).sum(axis=(1, 2))
        acc[:, si, 4] += (dth * m).sum(axis=(1, 2))
        acc[:, si, 5] += (dw2 * m).sum(axis=(1, 2))
    return acc


def accum_hour_shear(acc, pred_u, pred_v, true_u, true_v, masks):
    """把一小时的矢量切变误差按 (相邻层对, 分层) 累加。

    口径:切变 = 矢量 ΔV/Δz(m/s per m),取相邻 MAIN_LEVELS 对共 8 对(10–30 … 300–500);
    误差 = 预测切变 − 真值切变,平方和放 col1(se2),col1 之外的列不用——池化 RMSE =
    sqrt(Σe²/Σn),与风矢量 RMSE 同口径,可直接复用 pooled_rmse / paired_delta_ci;
    分层掩码取每对**上层**层的掩码 masks[s][1:len(MAIN_LEVELS)](形如 (8,99,120)),
    即切变误差归到 30..500 m 各层。pred/true 均为 AGL 物理场 (len(TARGET_AGL),99,120)。
    """
    n_lev = len(MAIN_LEVELS)
    dz = SHEAR_DZ[:, None, None]                     # (n_pairs,1,1)
    su_p = (pred_u[1:n_lev] - pred_u[:n_lev - 1]) / dz
    sv_p = (pred_v[1:n_lev] - pred_v[:n_lev - 1]) / dz
    su_t = (true_u[1:n_lev] - true_u[:n_lev - 1]) / dz
    sv_t = (true_v[1:n_lev] - true_v[:n_lev - 1]) / dz
    e2 = (su_p - su_t) ** 2 + (sv_p - sv_t) ** 2
    for si, s in enumerate(STRATA):
        m = masks[s][1:n_lev]                        # 上层掩码 (n_pairs,99,120)
        acc[:, si, 0] += m.sum(axis=(1, 2))
        acc[:, si, 1] += (e2 * m).sum(axis=(1, 2))
    return acc


def acc_metrics(acc):
    """累加量 -> 指标 dict(逐层/逐分层);数量为 0 的格子返回 nan。"""
    n = acc[..., 0]
    safe = np.where(n > 0, n, np.nan)
    return {
        'n': n,
        'rmse_vec': np.sqrt(acc[..., 1] / safe),
        'mae_vec': acc[..., 2] / safe,
        'speed_bias': acc[..., 3] / safe,
        'dir_err_deg': np.degrees(acc[..., 4] / safe),
        'rmse_w': np.sqrt(acc[..., 5] / safe),
    }


def pooled_rmse(acc, stratum, levels_idx=None, comp='vec'):
    """跨小时池化 RMSE(comp='vec' 用 se2 列,'w' 用 we2 列)。

    兼容带小时轴 (T,L,S,6) 与不带小时轴的 (L,S,6) 累加量。
    """
    col = 1 if comp == 'vec' else 5
    si = STRATA.index(stratum)
    s = acc[..., :, si, col]
    n = acc[..., :, si, 0]
    if levels_idx is not None:
        s = s[..., levels_idx]
        n = n[..., levels_idx]
    n_tot = float(n.sum())
    return float(np.sqrt(s.sum() / max(n_tot, 1.0))), n_tot


def paired_delta_ci(acc_a, acc_b, stratum, levels_idx, block=24, n_boot=2000, seed=0,
                    comp='vec'):
    """配对移动块 bootstrap:Δ = RMSE(acc_a) − RMSE(acc_b),同一重采样小时集。

    返回 (delta, lo, hi)。块长 24 h(一天);测试集仅 6 天,故同时报块长 12 h 作稳健性。
    """
    col = 1 if comp == 'vec' else 5
    si = STRATA.index(stratum)
    if acc_a.ndim == 3 or acc_b.ndim == 3:
        # 只有汇总累加量(无逐小时轴):只能给点估计,CI 记 nan
        # 注意两侧维数可以不同(旧评估只存汇总),故用 ... 索引各自处理
        def _pool(acc):
            s2 = acc[..., levels_idx, si, col].sum()
            n = acc[..., levels_idx, si, 0].sum()
            return float(np.sqrt(s2 / max(n, 1.0)))
        return _pool(acc_a) - _pool(acc_b), float('nan'), float('nan')
    s2a = acc_a[:, levels_idx, si, col].sum(axis=-1)
    n_a = acc_a[:, levels_idx, si, 0].sum(axis=-1)
    s2b = acc_b[:, levels_idx, si, col].sum(axis=-1)
    n_b = acc_b[:, levels_idx, si, 0].sum(axis=-1)
    T = s2a.shape[0]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / float(block)))

    def _rmse(s2, nn, idx):
        tot = s2[idx].sum()
        cnt = nn[idx].sum()
        return np.sqrt(tot / max(cnt, 1.0))

    d0 = float(_rmse(s2a, n_a, np.arange(T)) - _rmse(s2b, n_b, np.arange(T)))
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        deltas[b] = _rmse(s2a, n_a, idx) - _rmse(s2b, n_b, idx)
    return d0, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main_levels_idx():
    """主指标(10–500 m)在 TARGET_AGL 里的下标。"""
    return [int(np.where(TARGET_AGL == h)[0][0]) for h in MAIN_LEVELS]
