# -*- coding: utf-8 -*-
"""阶段 1 评估链路合成自检(秒级,不读 44 GB 抽取结果,只需真实 statics.npz)。

照出三类接线错误(阶段 0 的教训:先合成自检再上全量):
  1) AGL 算子在"ln z 线性"廓线上应与解析值一致(表/层集合/10 m 通道接线);
  2) 累加器聚合出的 RMSE 应与暴力逐点计算一致(掩码/维度);
  3) 配对移动块 bootstrap:同分布 -> Δ≈0 且 CI 含 0;已知偏差 -> Δ 落在 CI 内;
  5) 切变累加器(SHEAR_PAIRS/SHEAR_DZ/new_acc_shear/accum_hour_shear,阶段 3 T3.5)
     解析自检:true u=c·z -> 每对切变常数;pred 逐层加常数增量 d -> 每对误差 |d|/Δz;
     以及掩码取"上层"的接线。接口缺失时跳过并在输出里注明。

运行(本地或服务器,只需 numpy):
  python scripts/outline/89_agl_eval_fixture.py --static_dir results/outline
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import agl_eval_common as AEC  # noqa: E402  (切变接口按 hasattr 探测,缺失则跳过)
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


def check_shear_analytic():
    """切变累加解析自检(接口契约:8 对相邻主层,误差 = 预测切变 − 真值切变)。

    合成:11 层 AGL 高度 z = TARGET_AGL;true u = c·z、v = 0 -> 每对切变 su_t = c;
    pred 逐层加常数增量 d(第 k 层偏移 d·k)-> 每对切变误差 = |d|/Δz(该对间距)。
    "(层间均匀的)常数偏移 d" 在差分里相消,不产生切变误差(单独断言,固定口径)。
    """
    need = ("SHEAR_PAIRS", "SHEAR_DZ", "new_acc_shear", "accum_hour_shear")
    missing = [k for k in need if not hasattr(AEC, k)]
    if missing:
        print("5) 切变累加解析自检:SKIP(agl_eval_common 尚无 {})".format(",".join(missing)))
        return
    pairs = list(AEC.SHEAR_PAIRS)
    dz = np.asarray(AEC.SHEAR_DZ, dtype=np.float64)
    assert len(pairs) == 8, "SHEAR_PAIRS 应为 8 对(10–30 … 300–500),实际 {}".format(pairs)
    want_pairs = [(int(a), int(b)) for a, b in zip(AEC.MAIN_LEVELS[:-1], AEC.MAIN_LEVELS[1:])]
    assert [tuple(p) for p in pairs] == want_pairs, \
        "SHEAR_PAIRS {} != 相邻主层对 {}".format(pairs, want_pairs)
    assert np.allclose(dz, np.diff(np.asarray(AEC.MAIN_LEVELS, dtype=np.float64))), \
        "SHEAR_DZ 应等于 np.diff(MAIN_LEVELS)"
    acc = AEC.new_acc_shear()
    assert acc.shape == (8, len(STRATA), N_ACC), \
        "new_acc_shear() 形状 {} != (8,{},6)".format(acc.shape, len(STRATA))
    assert AEC.new_acc_shear(8).shape == acc.shape

    z = np.asarray(TARGET_AGL, dtype=np.float64)
    c, d = 2.5e-3, 0.4                      # m/s per m;m/s 每层
    tu = np.broadcast_to(c * z[:, None, None], (len(z), NY, NX)).astype(np.float32).copy()
    tv = np.zeros((len(z), NY, NX), np.float32)
    su_true = (tu[1:9].astype(np.float64) - tu[:8].astype(np.float64)) / dz[:, None, None]
    assert np.allclose(su_true, c, rtol=0, atol=1e-9), \
        "合成真值每对切变应为常数 c(float32 层值差分的舍入 {:.1e})".format(
            float(np.abs(su_true - c).max()))
    print("5) 切变累加解析自检(8 对相邻主层;true u=c·z,c={:g};逐层增量 d={:g})".format(c, d))

    masks_all = {s: np.ones((len(z), NY, NX), bool) for s in STRATA}
    sa = STRATA.index('all')

    # (1) pred == true:误差恒 0,每对计数 = 像素数
    acc0 = AEC.new_acc_shear()
    AEC.accum_hour_shear(acc0, tu, tv, tu, tv, masks_all)
    ok0 = bool(np.all(acc0[..., 1] == 0.0)) and bool(np.all(acc0[:, sa, 0] == NY * NX))
    print("   1) pred==true:se2 全 0(P={}),每对计数={}".format(ok0, int(acc0[0, sa, 0])))
    assert ok0, "pred==true 时切变误差应恒 0 且计数 = NY*NX"

    # (2) u 分量逐层常数增量 d -> 每对 RMSE = |d|/Δz
    off = (d * np.arange(len(z), dtype=np.float64))[:, None, None].astype(np.float32)
    acc1 = AEC.new_acc_shear()
    AEC.accum_hour_shear(acc1, tu + off, tv, tu, tv, masks_all)
    rmse_pair = np.sqrt(acc1[:, sa, 1] / acc1[:, sa, 0])
    want = np.abs(d) / dz
    print("   2) u 逐层增量:每对 RMSE {} vs 解析 |d|/Δz {}".format(
        np.array2string(rmse_pair, precision=4), np.array2string(want, precision=4)))
    assert np.allclose(rmse_pair, want, rtol=1e-5, atol=1e-12), \
        "每对切变误差应为 |d|/Δz(float32 合成输入,rtol 1e-5)"
    m, n = pooled_rmse(acc1, 'all')
    want_pool = np.abs(d) * np.sqrt(np.mean(1.0 / dz ** 2))
    print("   3) 池化 RMSE {:.8e} vs 解析 {:.8e}(n={:.0f},期望 {})".format(
        m, want_pool, n, 8 * NY * NX))
    assert abs(m - want_pool) <= 1e-6 * want_pool, "池化切变 RMSE 与解析值不符"
    assert n == 8 * NY * NX, "池化计数应为 8 对 × 像素数"

    # (4) v 分量单独:同 d 的逐层增量,每对误差与 u 相同(接线对称)
    acc2 = AEC.new_acc_shear()
    AEC.accum_hour_shear(acc2, tu, tv + off, tu, tv, masks_all)
    rv = np.sqrt(acc2[:, sa, 1] / acc2[:, sa, 0])
    print("   4) v 逐层增量:每对 RMSE 与 u 逐对相等:P={}".format(
        bool(np.allclose(rv, rmse_pair, rtol=1e-6, atol=1e-12))))
    assert np.allclose(rv, want, rtol=1e-5, atol=1e-12), "v 分量切变误差接线不对称"

    # (5) 层间均匀常数偏移 d 在差分中相消 -> 切变误差可忽略(仅 float32 舍入;口径固定)
    acc3 = AEC.new_acc_shear()
    AEC.accum_hour_shear(acc3, (tu + np.float32(d)).astype(np.float32), tv, tu, tv, masks_all)
    rs = np.sqrt(acc3[:, sa, 1] / acc3[:, sa, 0])
    print("   5) 层间均匀常数偏移 d:每对残余 RMSE 最大 {:.2e}(信号 {:.2e},仅舍入)".format(
        float(rs.max()), float(want.min())))
    assert float(rs.max()) < 1e-3 * float(want.min()), "均匀常数偏移不应产生切变误差"

    # (6) 掩码取上层:只有 night 层 5(150 m = pair 4 的上层)为 True -> 仅 pair 4 有计数
    masks_u = {s: np.zeros((len(z), NY, NX), bool) for s in STRATA}
    masks_u['night'][5] = True
    acc4 = AEC.new_acc_shear()
    AEC.accum_hour_shear(acc4, tu, tv, tu, tv, masks_u)
    n_pair = acc4[:, STRATA.index('night'), 0]
    ok_u = (n_pair[4] == NY * NX) and bool(np.all(n_pair[np.arange(8) != 4] == 0))
    print("   6) 掩码取上层(150 m 仅上层):各对计数 {} -> P={}".format(
        n_pair.astype(int).tolist(), ok_u))
    assert ok_u, "切变累加应取上层掩码 masks[s][1:9](150 m 归 pair 4)"


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
    check_shear_analytic()
    print("AGL EVAL FIXTURE OK")


if __name__ == "__main__":
    main()
