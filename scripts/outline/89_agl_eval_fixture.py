# -*- coding: utf-8 -*-
"""阶段 1 评估链路合成自检(秒级,不读 44 GB 抽取结果,只需真实 statics.npz)。

照出三类接线错误(阶段 0 的教训:先合成自检再上全量):
  1) AGL 算子在"ln z 线性"廓线上应与解析值一致(表/层集合/10 m 通道接线);
  2) 累加器聚合出的 RMSE 应与暴力逐点计算一致(掩码/维度);
  3) 配对移动块 bootstrap:同分布 -> Δ≈0 且 CI 含 0;已知偏差 -> Δ 落在 CI 内。

运行(本地或服务器,只需 numpy):
  python scripts/outline/89_agl_eval_fixture.py --static_dir results/outline
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agl_eval_common import (N_ACC, STRATA, TARGET_AGL, accum_hour, agl_fields,  # noqa: E402
                             build_masks, load_tables, main_levels_idx, new_acc,
                             paired_delta_ci, pooled_rmse)
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402

LEVELS = list(range(23))
NY, NX = 99, 120


def check_agl_operator(statics, tables):
    """合成 field = a + b·ln z -> AGL 插值应等于解析值(idx>=0 的目标层)。"""
    a, b = 3.0, 2.0
    z_m = tables['zagl_mass']                      # (23,99,120)
    z_i = np.asarray(statics.d['zagl_iface_fine'], dtype=np.float64)[:len(LEVELS) + 1]
    u = a + b * np.log(np.maximum(z_m, 1e-3))
    w = a + b * np.log(np.maximum(z_i, 1e-3))
    u10 = np.full((NY, NX), a + b * np.log(10.0), dtype=np.float32)
    au, av, aw = agl_fields(u.astype(np.float32), u.astype(np.float32),
                            w.astype(np.float32), u10, u10, tables)
    err = []
    for t, h in enumerate(TARGET_AGL):
        analytic = a + b * np.log(h)
        if (tables['idx_m'][t] >= 0).any():
            m = tables['idx_m'][t] >= 0
            err.append(np.abs(au[t][m] - analytic).max())
        if (tables['idx_i'][t] >= 0).any():
            m = tables['idx_i'][t] >= 0
            err.append(np.abs(aw[t][m] - analytic).max())
    e = max(err)
    print('1) AGL 算子解析自检:最大偏差 {:.3e}'.format(e))
    assert e < 1e-3, 'AGL 插值接线错误'


def check_accumulator():
    """累加器 -> pooled_rmse 与暴力计算一致。"""
    rng = np.random.default_rng(0)
    n_h, n_lev = 5, len(TARGET_AGL)
    acc = new_acc(n_lev)
    brute = []
    for _ in range(n_h):
        pu = rng.normal(5, 2, (n_lev, NY, NX)).astype(np.float32)
        pv = rng.normal(2, 2, (n_lev, NY, NX)).astype(np.float32)
        pw = rng.normal(0, 0.5, (n_lev, NY, NX)).astype(np.float32)
        tu = rng.normal(5, 2, (n_lev, NY, NX)).astype(np.float32)
        tv = rng.normal(2, 2, (n_lev, NY, NX)).astype(np.float32)
        tw = rng.normal(0, 0.5, (n_lev, NY, NX)).astype(np.float32)
        masks = build_masks(tu, tv, np.full((NY, NX), 0.5, np.float32),
                            np.full((NY, NX), 800.0, np.float32),
                            rng.random((NY, NX)).astype(np.float32))
        accum_hour(acc, pu, pv, pw, tu, tv, tw, masks)
        brute.append(((pu - tu) ** 2 + (pv - tv) ** 2))
    idx = main_levels_idx()
    m, n = pooled_rmse(acc, 'all', idx)
    ref = np.sqrt(np.concatenate([e[idx].ravel() for e in brute]).mean())
    print('2) 累加器自检:pooled {:.6f} vs 暴力 {:.6f}(n={:.0f})'.format(m, ref, n))
    assert abs(m - ref) < 1e-6, '累加器/掩码维度不一致'


def check_bootstrap():
    """配对 bootstrap:同分布 -> 含 0;已知偏差 -> 显著。"""
    rng = np.random.default_rng(1)
    T, n_lev = 144, len(TARGET_AGL)
    si = STRATA.index('all')
    rng_scale = 0.3
    rmse_h = 0.5 * (1.0 + rng_scale * rng.standard_normal(T))
    n_cell = NY * NX
    acc_a = np.zeros((T, n_lev, len(STRATA), N_ACC))
    acc_a[:, :, si, 0] = n_cell
    acc_a[:, :, si, 1] = (rmse_h ** 2 * n_cell)[:, None]
    acc_b = acc_a.copy()
    d0, lo, hi = paired_delta_ci(acc_a, acc_b, 'all', main_levels_idx(), n_boot=500, seed=0)
    print('3a) bootstrap 同分布:Δ={:+.5f} CI=[{:+.5f},{:+.5f}]'.format(d0, lo, hi))
    assert abs(d0) < 1e-9 and lo <= 0 <= hi, 'bootstrap 同分布时 CI 应包含 0'
    worse = 0.55 * (1.0 + rng_scale * rng.standard_normal(T))
    acc_c = acc_a.copy()
    acc_c[:, :, si, 1] = (worse ** 2 * n_cell)[:, None]
    d1, lo1, hi1 = paired_delta_ci(acc_c, acc_a, 'all', main_levels_idx(), n_boot=500, seed=0)
    print('3b) bootstrap 已知偏差:Δ={:+.5f} CI=[{:+.5f},{:+.5f}]'.format(d1, lo1, hi1))
    assert d1 > 0 and lo1 > 0, '已知变差的 CI 应完全在 0 之上'


def check_mixed_dims():
    """4 维(逐小时)与 3 维(仅汇总)累加量混用时的 Δ 必须等于两个汇总 RMSE 之差。"""
    rng = np.random.default_rng(2)
    T, n_lev = 40, len(TARGET_AGL)
    si = STRATA.index('all')
    n_cell = NY * NX
    rmse_h = 0.5 * (1.0 + 0.2 * rng.standard_normal(T))
    acc4 = np.zeros((T, n_lev, len(STRATA), N_ACC))
    acc4[:, :, si, 0] = n_cell
    acc4[:, :, si, 1] = (rmse_h ** 2 * n_cell)[:, None]
    acc3 = acc4.sum(axis=0)                    # 旧评估:只有汇总
    d, lo, hi = paired_delta_ci(acc3, acc4, 'all', main_levels_idx(), n_boot=50, seed=0)
    print('4) 3维/4维混用:Δ={:+.6f} CI=({:s},{:s})'.format(d, str(lo), str(hi)))
    assert abs(d) < 1e-9, '同数据的 3 维/4 维累加量 Δ 应为 0'
    scale = acc4.copy()
    scale[:, :, si, 1] = (1.2 * rmse_h ** 2 * n_cell)[:, None]
    d2, _, _ = paired_delta_ci(scale.sum(axis=0), acc4, 'all', main_levels_idx(),
                              n_boot=50, seed=0)
    m_scale = np.sqrt(scale[..., si, 1].sum() / scale[..., si, 0].sum())
    m_base = np.sqrt(acc4[..., si, 1].sum() / acc4[..., si, 0].sum())
    print('   3维 vs 4维 已知偏差:Δ={:+.6f}(期望 {:+.6f})'.format(d2, m_scale - m_base))
    assert abs(d2 - (m_scale - m_base)) < 1e-9, '3 维/4 维混用的 Δ 计算有误'


def main():
    ap = argparse.ArgumentParser(description="阶段 1 评估链路合成自检")
    ap.add_argument("--static_dir", default="results/outline")
    args = ap.parse_args()
    statics = CanvasStatics(args.static_dir)
    global tables_glob
    tables_glob = load_tables(statics, LEVELS)
    check_agl_operator(statics, tables_glob)
    check_accumulator()
    check_bootstrap()
    check_mixed_dims()
    print("AGL EVAL FIXTURE OK")


if __name__ == "__main__":
    main()
