# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · T0.3/T0.4 统一 AGL 插值算子与往返试验

算子(预测与真值共用,权重由 1 km 静态 z_agl 构造):
  质量层(U/V): idx>=0 -> w*f[k]+(1-w)*f[k+1](ln z 线性);
              idx=-1 低于首层 -> w*f[0]+(1-w)*f_10m(10 m 通道锚定);
              idx=-2 目标 10 m -> 直接取 10 m 通道;不做外推
  界面层(W)  : 目标 10 m 由界面 0(z≈0)与界面 1 夹逼,天然无外推

往返试验(T0.4):d04 真值 模式层 -> AGL 11 层 -> 回插模式层,
报告往返误差量级(与降尺度误差对比);<1000 m 的层可回插,以上不参与。

运行(pytorch-gpu 环境,需 20 号细端抽取已完成):
  python scripts/outline/60_agl_operator.py
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline_common import OUT_FINE, OUT_STATIC

TARGET_AGL = np.array([10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000], dtype=np.float64)


def agl_interp(field, idx, w, field10=None):
    """field (nlev, ny, nx) -> (nt, ny, nx);idx/w 来自 statics 的 AGL 表。"""
    nlev, ny, nx = field.shape
    nt = idx.shape[0]
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    out = np.zeros((nt, ny, nx), dtype=np.float32)
    for t in range(nt):
        k = np.clip(idx[t], 0, nlev - 2)
        wt = w[t].astype(np.float32)
        val = wt * field[k, jj, ii] + (1.0 - wt) * field[k + 1, jj, ii]
        if field10 is not None:
            m10 = idx[t] == -1
            if m10.any():
                val = np.where(m10, wt * field[0, jj, ii] + (1.0 - wt) * field10[jj, ii], val)
            m2 = idx[t] == -2
            if m2.any():
                val = np.where(m2, field10[jj, ii], val)
        out[t] = val
    return out


def agl_interp_torch(field, idx, w, field10=None):
    """torch 版(阶段 3 可微插值层用),与 numpy 版数学等价。"""
    import torch
    nlev, ny, nx = field.shape
    nt = idx.shape[0]
    jj = torch.arange(ny).view(-1, 1)
    ii = torch.arange(nx).view(1, -1)
    outs = []
    for t in range(nt):
        k = idx[t].clamp(0, nlev - 2)
        wt = w[t]
        val = wt * field[k, jj, ii] + (1.0 - wt) * field[k + 1, jj, ii]
        if field10 is not None:
            val = torch.where(idx[t] == -1, wt * field[0] + (1.0 - wt) * field10, val)
            val = torch.where(idx[t] == -2, field10, val)
        outs.append(val)
    return torch.stack(outs, 0)


def back_to_model_levels(agl_vals, z_targets, z_agl_levels, max_h=1000.0):
    """AGL 11 层 -> 模式层(仅在 [z_targets[0], max_h] 内可回插;ln z 线性)。"""
    nlev = z_agl_levels.shape[0]
    lnz = np.log(np.maximum(z_agl_levels, 1e-3))
    lnt = np.log(z_targets)
    out = np.full_like(z_agl_levels, np.nan, dtype=np.float32)
    inside = (z_agl_levels >= z_targets[0]) & (z_agl_levels <= max_h)
    for k in range(nlev):
        # 找到夹住该层高度的两个目标层
        idxs = np.searchsorted(z_targets, z_agl_levels[k], side='right') - 1
        for t in range(agl_vals.shape[0] - 1):
            m = inside & (idxs == t)
            if not m.any():
                continue
            wt = (lnt[t + 1] - lnz[k]) / (lnt[t + 1] - lnt[t])
            out[k] = np.where(m, wt * agl_vals[t] + (1.0 - wt) * agl_vals[t + 1], out[k])
    return out


def main():
    parser = argparse.ArgumentParser(description="阶段0 T0.3/T0.4 AGL 算子和往返试验")
    parser.add_argument("--fine_dir", default=OUT_FINE)
    parser.add_argument("--static_dir", default=OUT_STATIC)
    parser.add_argument("--scheme", default="myj")
    parser.add_argument("--n_frames", type=int, default=24, help="往返试验用多少帧(整点)")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    statics = np.load(os.path.join(args.static_dir, "statics.npz"))
    idx_m = statics['agl_idx_mass']
    w_m = statics['agl_w_mass']
    idx_i = statics['agl_idx_iface']
    w_i = statics['agl_w_iface']
    zagl_mass = np.asarray(statics['zagl_mass_fine'], dtype=np.float64)   # (40, 99, 120)
    zagl_iface = np.asarray(statics['zagl_iface_fine'], dtype=np.float64)  # (41, 99, 120)

    files = sorted(glob.glob(os.path.join(args.fine_dir, "f_{}_*.npz".format(args.scheme))))
    files = [f for f in files if re.search(r'_(\d{8}T\d{4})00\.npz$', f)][:args.n_frames]
    if not files:
        raise SystemExit("没有找到整点细端文件")
    print("往返试验帧数: {} ({} .. {})".format(
        len(files), os.path.basename(files[0]), os.path.basename(files[-1])))

    err_u, err_v, err_w = [], [], []
    err_uv_at_agl = []
    for f in files:
        with np.load(f) as d:
            u, v, w, u10, v10 = d['f_u'], d['f_v'], d['f_w'], d['f_u10'], d['f_v10']
        # --- 质量层:U/V 往返 ---
        agl_u = agl_interp(u, idx_m, w_m, field10=u10)
        agl_v = agl_interp(v, idx_m, w_m, field10=v10)
        back_u = back_to_model_levels(agl_u, TARGET_AGL, zagl_mass)
        back_v = back_to_model_levels(agl_v, TARGET_AGL, zagl_mass)
        # AGL 空间上的往返(AGL -> 模式层 -> AGL)仅对模式层可覆盖的目标层有意义,
        # 这里报告模式层往返误差(核心证据),AGL 层残差另算:
        agl_u2 = agl_interp(np.where(np.isnan(back_u), u, back_u), idx_m, w_m, field10=u10)
        agl_v2 = agl_interp(np.where(np.isnan(back_v), v, back_v), idx_m, w_m, field10=v10)
        m = ~np.isnan(back_u)
        err_u.append(np.abs(back_u - u)[m])
        err_v.append(np.abs(back_v - v)[m])
        err_uv_at_agl.append(np.abs(agl_u2 - agl_u)[np.isfinite(agl_u2) & np.isfinite(agl_u)])
        # --- 界面层:W 往返(用界面表) ---
        agl_w = agl_interp(w, idx_i, w_i)
        back_w = back_to_model_levels(agl_w, TARGET_AGL, zagl_iface)
        mw = ~np.isnan(back_w)
        err_w.append(np.abs(back_w - w)[mw])

    def stats(chunks):
        a = np.concatenate([c.ravel() for c in chunks])
        return {'n': int(a.size), 'mean_abs': float(a.mean()),
                'rmse': float(np.sqrt((a ** 2).mean())), 'p95': float(np.percentile(a, 95)),
                'max': float(a.max())}

    rep = {
        'generated': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'scheme': args.scheme, 'n_frames': len(files),
        'u_roundtrip': stats(err_u), 'v_roundtrip': stats(err_v), 'w_roundtrip': stats(err_w),
        'uv_agl_residual': stats(err_uv_at_agl),
        'note': '回插仅对 z_agl ∈ [10, 1000] m 的模式层有定义(上界为目标层范围)',
    }
    # 与信号尺度对比:同帧 U 的逐层 σ
    with np.load(files[0]) as d:
        u0 = d['f_u']
    sig = float(np.sqrt((u0 ** 2).mean()))
    rep['reference'] = {
        'u_rms_single_frame': sig,
        'ratio_rmse_over_signal': rep['u_roundtrip']['rmse'] / max(sig, 1e-9),
        'model_rmse_note': '现有 3km->1km 模型标准化空间 RMSE ≈ 0.31(≈31% 信号尺度),'
                           '往返误差应远小于该量级',
    }
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
