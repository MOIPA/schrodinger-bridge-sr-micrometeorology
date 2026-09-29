# -*- coding: utf-8 -*-
"""阶段 1 非学习基线(T1.1①):粗端逐模式层水平双三次插值 -> AGL 11 层 -> 同口径指标。

- 水平双三次用 torch.grid_sample(align_corners=True, padding_mode='border');
  坐标由经纬度直接构造(两域同为 Mercator 规则网格:经度对应列、Mercator y 对应行);
- 不涉及垂直插值:粗端第 k 层 -> 细端第 k 层(四档 eta 配置一致,阶段 0 已核对);
- U/V 先各自去交错到质量点再插值,与模型评估(90 号,先切原生再去交错)aligned;
- 掩码/AGL 算子/累加器与 90 号完全共用(agl_eval_common)。

用法(需要 torch 与 numpy 的环境,仓库根目录):
  python scripts/outline/91_bicubic_baseline.py --split test --out_dir results/phase1
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
from agl_eval_common import (STRATA, TARGET_AGL, accum_hour, acc_metrics,  # noqa: E402
                             agl_fields, build_masks, destagger_native, load_tables,
                             main_levels_idx, new_acc, pooled_rmse)
from outline_common import OUT_COARSE, OUT_FINE, OUT_STATIC  # noqa: E402
from src.dl_data.block_split import split_paths_by_manifest  # noqa: E402
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402

_STAMP = re.compile(r'_(\d{8}T\d{6})\.npz$')


def mercator_y(lat_deg):
    return np.log(np.tan(np.pi / 4.0 + np.radians(lat_deg) / 2.0))


def build_coords(statics, levels):
    """细端质量点 -> 粗端质量网格的分数坐标(用于 grid_sample 的归一化坐标)。"""
    lat_c = np.asarray(statics.d['xlat_coarse'], dtype=np.float64)
    lon_c = np.asarray(statics.d['xlong_coarse'], dtype=np.float64)
    lat_f = np.asarray(statics.d['xlat_fine'], dtype=np.float64)
    lon_f = np.asarray(statics.d['xlong_fine'], dtype=np.float64)
    dlon = lon_c[0, 1] - lon_c[0, 0]
    dmy = mercator_y(lat_c[1, 0]) - mercator_y(lat_c[0, 0])
    col = (lon_f - lon_c[0, 0]) / dlon
    row = (mercator_y(lat_f) - mercator_y(lat_c[0, 0])) / dmy
    # 校验:坐标应落在粗网格范围内;粗节点经纬度应映回整数坐标
    assert abs(dlon) > 0 and abs(dmy) > 0, "粗网格经纬度退化"
    assert col.min() > -0.5 and col.max() < lon_c.shape[1] - 0.5, "列坐标越界"
    assert row.min() > -0.5 and row.max() < lat_c.shape[0] - 0.5, "行坐标越界"
    fi = [(0, 0), (0, 149), (119, 0), (119, 149)]
    for (j, i) in fi:
        c = (lon_c[j, i] - lon_c[0, 0]) / dlon
        r = (mercator_y(lat_c[j, i]) - mercator_y(lat_c[0, 0])) / dmy
        assert abs(c - i) < 0.02 and abs(r - j) < 0.02, \
            "坐标映射自检失败: 节点 ({},{}) -> ({:.3f},{:.3f})".format(j, i, r, c)
    return row.astype(np.float32), col.astype(np.float32)


def sample_bicubic(coarse_field, row, col):
    """coarse_field (C,120,150) -> (C,99,120),水平双三次(边界复制)。"""
    c, h, w = coarse_field.shape
    t = torch.from_numpy(np.ascontiguousarray(coarse_field, dtype=np.float32))[None]
    gy = 2.0 * torch.from_numpy(row) / (h - 1) - 1.0
    gx = 2.0 * torch.from_numpy(col) / (w - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None].to(t.device)
    out = torch.nn.functional.grid_sample(
        t.to(t.device), grid, mode='bicubic', padding_mode='border', align_corners=True)
    return out[0].cpu().numpy()


def self_check(row, col, ny_c, nx_c):
    """grid_sample 双三次复现"按索引线性"的场:用于确认坐标/轴序接线。

    阈值按"每格梯度 1.0"折算:偏差 <0.5 即小于半格错位,足以排除轴序/映射类错误;
    (torch 的 bicubic 内核本身有 ~5e-2 的实现残差,故不能要求浮点级一致)
    """
    jj, ii = np.meshgrid(np.arange(ny_c), np.arange(nx_c), indexing='ij')
    lin = (0.7 * ii + 0.3 * jj).astype(np.float32)
    got = sample_bicubic(lin[None], row, col)[0]
    ref = (0.7 * col + 0.3 * row).astype(np.float32)
    err = float(np.abs(got - ref).max())
    print("坐标自检:双三次复现线性场最大偏差 {:.3e}(阈值 0.5 格)".format(err))
    assert err < 0.5, "坐标映射或 grid_sample 轴序有误"


def main():
    ap = argparse.ArgumentParser(description="阶段 1 双三次插值基线")
    ap.add_argument("--fine_dir", default=OUT_FINE)
    ap.add_argument("--coarse_dir", default=OUT_COARSE)
    ap.add_argument("--static_dir", default=OUT_STATIC)
    ap.add_argument("--scheme", default="myj")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--levels", default="0-22")
    ap.add_argument("--out_dir", default="results/phase1")
    ap.add_argument("--tag", default="baseline_bicubic")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    a, b = args.levels.split('-')
    levels = list(range(int(a), int(b) + 1))
    WL = list(range(levels[0], levels[-1] + 2))

    statics = CanvasStatics(args.static_dir)
    tables = load_tables(statics, levels)
    urban = np.asarray(statics.d['urban_fine'], dtype=np.float32)
    row, col = build_coords(statics, levels)
    self_check(row, col, np.asarray(statics.d['xlat_coarse']).shape[0],
               np.asarray(statics.d['xlong_coarse']).shape[1])
    mask_idx = main_levels_idx()

    manifest_path = os.path.join(args.static_dir, "split.json")
    all_paths = sorted([p for p in os.listdir(args.fine_dir) if p.endswith('.npz')])
    all_paths = [os.path.join(args.fine_dir, p) for p in all_paths]
    dict_paths, unmatched = split_paths_by_manifest(all_paths, manifest_path)
    assert not unmatched, "有文件不在 split.json 清单中"
    tag_scheme = '_' + args.scheme + '_'
    files = [p for p in dict_paths[args.split] if tag_scheme in os.path.basename(p)
             and _STAMP.search(os.path.basename(p))
             and _STAMP.search(os.path.basename(p)).group(1)[11:13] == '00']
    files = sorted(files)
    if args.max_frames > 0:
        files = files[:args.max_frames]
    print("双三次基线评估 {}: {} 帧".format(args.split, len(files)))

    device = torch.device(args.device)
    acc = new_acc(len(TARGET_AGL))
    hours = []
    for n, fp in enumerate(files):
        stamp = _STAMP.search(os.path.basename(fp)).group(1)
        with np.load(fp) as f:
            tu_c, tv_c, tw_c = f['f_u'], f['f_v'], f['f_w']
            tu10, tv10 = f['f_u10'], f['f_v10']
        tu_m, tv_m = destagger_native(tu_c, tv_c, tw_c, tu10, tv10)[:2]
        truth = agl_fields(tu_m[levels], tv_m[levels],
                           np.asarray(tw_c[:len(WL)], dtype=np.float32), tu10, tv10, tables)

        with np.load(os.path.join(args.coarse_dir,
                                  "c_{}_{}.npz".format(args.scheme, stamp))) as f:
            co = {k: f[k] for k in f.keys()}
        pu_all, pv_all, pw_all = destagger_native(co['c_u'], co['c_v'], co['c_w'],
                                                  None, None)[:3]
        p10 = 0.5 * (co['c_u'][0, :, :-1] + co['c_u'][0, :, 1:])
        p10v = 0.5 * (co['c_v'][0, :-1, :] + co['c_v'][0, 1:, :])
        pred = agl_fields(
            sample_bicubic(pu_all[list(levels)], row, col),
            sample_bicubic(pv_all[list(levels)], row, col),
            sample_bicubic(pw_all[WL], row, col),
            sample_bicubic(p10[None], row, col)[0],
            sample_bicubic(p10v[None], row, col)[0],
            tables)

        coszen = statics.regrid_field(np.asarray(co['c_coszen'], dtype=np.float32)[None],
                                      'mass')[0]
        pblh = statics.regrid_field(np.asarray(co['c_pblh'], dtype=np.float32)[None],
                                    'mass')[0]
        masks = build_masks(truth[0], truth[1], coszen, pblh, urban)
        accum_hour(acc, pred[0], pred[1], pred[2], truth[0], truth[1], truth[2], masks)
        hours.append(datetime.strptime(stamp, '%Y%m%dT%H%M%S').strftime('%Y-%m-%dT%H:%M:%S'))
        if (n + 1) % 48 == 0:
            print("  ... {} 帧".format(n + 1))

    metrics = acc_metrics(acc)
    out = {
        'tag': args.tag, 'split': args.split, 'n_hours': len(hours),
        'scheme': args.scheme, 'target_levels': levels,
        'agl_targets': TARGET_AGL.tolist(), 'strata': STRATA,
        'method': 'horizontal bicubic (grid_sample, per mode layer, no vertical interpolation)',
        'main_rmse_vec': pooled_rmse(acc, 'all', mask_idx)[0],
        'rmse_vec_all_per_level': metrics['rmse_vec'][:, STRATA.index('all')].tolist(),
        'rmse_w_all_per_level': metrics['rmse_w'][:, STRATA.index('all')].tolist(),
        'mae_vec_all_per_level': metrics['mae_vec'][:, STRATA.index('all')].tolist(),
        'speed_bias_all_per_level': metrics['speed_bias'][:, STRATA.index('all')].tolist(),
        'dir_err_all_per_level': metrics['dir_err_deg'][:, STRATA.index('all')].tolist(),
        'by_stratum_main': {s: pooled_rmse(acc, s, mask_idx)[0] for s in STRATA},
    }
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    npz_path = os.path.join(args.out_dir, "{}_perhour.npz".format(args.tag))
    np.savez_compressed(npz_path, hours=np.array(hours), acc_agl=acc)
    json_path = os.path.join(args.out_dir, "{}_summary.json".format(args.tag))
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ('tag', 'n_hours', 'main_rmse_vec')},
                     ensure_ascii=False))
    print("写出 {} / {}".format(npz_path, json_path))


if __name__ == "__main__":
    main()
