# -*- coding: utf-8 -*-
"""阶段 3 T3.5 空间误差分析:逐像素平均 |误差| 图 + 地形/城郊分组池化 RMSE。

对给定 tag(默认 p2_l1r2_lr2e4,p3_agl,p3_joint)加载各自 checkpoint,复用 90 的
数据集与推理路径(pad16 + 确定性 SI 采样 add_noise=False),把预测/真值统一 AGL
插值后累计:
  1) 选定 AGL 层(默认 10/50/100/300 m)的逐像素平均 |Δu|/|Δv|/|Δw|,
     形状 (n_sel,99,120),可直接画空间误差图;
  2) 低层(默认 10–100 m 主指标子集)按 陡/平 x 城/郊 四组(外加边缘组)的
     池化 RMSE(矢量 sqrt(Σ(du²+dv²)/N) 与 W sqrt(Σdw²/N),口径同 90/92)。

分组口径(与任务契约一致):
  坡度 = |∇HGT|(HGT 取 statics 的 hgt_fine,有限差分,网格距 1 km);
  陡 = 坡度 > 全场上四分位(阈值与占比会在摘要里打印;若钝化为 0 会有 WARN);
  城 = urban_fine > 0.3、郊 = 其余(与 90 build_masks 的 urban 定义一致)。

产物:results/phase3/spatial_<tag>.npz(逐像素 MAE 图 + 分组 RMSE + 坡度/掩码);
stdout 打印 markdown 摘要(逐 tag 一节 + 跨 tag 对照表),供 63e 汇总。

用法(仓库根目录,需要 torch 的环境):
  python scripts/outline/98_phase3_spatial.py \
      --tags p2_l1r2_lr2e4 p3_agl p3_joint --split test \
      --out_dir results/phase3 --device cuda:0
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
from agl_eval_common import (TARGET_AGL, agl_fields, destagger_canvas,  # noqa: E402
                             load_norm_sigma, load_tables)
from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_data.dataloader import make_dataloaders_and_samplers  # noqa: E402
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402
from src.utils.random_seed_helper import set_seeds  # noqa: E402

DEFAULT_TAGS = ["p2_l1r2_lr2e4", "p3_agl", "p3_joint"]
DEFAULT_SEL_AGL = [10, 50, 100, 300]
DEFAULT_LOW_AGL = [10, 30, 50, 70, 100]   # 主指标(10–500 m)的低层子集
GRID_DX_M = 1000.0                        # 细端 d04 网格距(statics meta: dx_fine=1000 m)
URBAN_THRESHOLD = 0.3                     # 与 90 build_masks 相同


def _load_90():
    """90_agl_eval_phase1.py 文件名以数字开头,用 importlib 载入以复用其推理路径。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '90_agl_eval_phase1.py')
    spec = importlib.util.spec_from_file_location('agl_eval_90', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV90 = _load_90()
EXPERIMENT = EV90.EXPERIMENT


def tag_paths(tag, cfg_root="configs/深圳", ck_root=None):
    """tag -> (config, checkpoint)。阶段目录按 tag 前缀推断(p2_ -> phase2,p3_ -> phase3)。"""
    phase = "phase2" if tag.startswith("p2_") else "phase3"
    if ck_root is None:
        ck_root = os.path.join("data", "DL_result", EXPERIMENT)
    return (os.path.join(cfg_root, phase, "config_wind_canvas_{}.yml".format(tag)),
            os.path.join(ck_root, "config_wind_canvas_{}".format(tag), "checkpoint.pth"))


def terrain_groups(statics, q=0.75):
    """坡度(1 km 有限差分)/城郊分组掩码;返回 (groups, slope, q75, urban)。"""
    hgt = np.asarray(statics.d['hgt_fine'], dtype=np.float64)
    gy, gx = np.gradient(hgt, GRID_DX_M, GRID_DX_M)
    slope = np.sqrt(gy ** 2 + gx ** 2)
    q75 = float(np.percentile(slope, 100.0 * q))
    urban = np.asarray(statics.d['urban_fine'], dtype=np.float32) > URBAN_THRESHOLD
    steep = slope > q75
    flat = np.logical_not(steep)
    groups = {
        'steep_city': steep & urban, 'steep_rural': steep & np.logical_not(urban),
        'flat_city': flat & urban, 'flat_rural': flat & np.logical_not(urban),
        'steep': steep, 'flat': flat, 'city': urban, 'rural': np.logical_not(urban),
        'all': np.ones_like(steep),
    }
    return groups, slope, q75, urban


def agl_indices(levels):
    out = []
    for h in levels:
        w = np.where(TARGET_AGL == float(h))[0]
        if len(w) == 0:
            raise SystemExit("AGL {} m 不在 TARGET_AGL 目标层里".format(h))
        out.append(int(w[0]))
    return out


def build_loader(config, split):
    """与 90 相同的取数路径:全画布 (100,121),不做随机裁剪。"""
    config.data.hr_data_shape = [100, 121]
    config.data.hr_cropped_shape = [100, 121]
    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=config.loader, dataset_config=config.data,
        world_size=None, rank=None, train_valid_test_kinds=[split])
    return dict_loaders[split]


def run_tag(tag, args, device):
    """单 tag:推理 test 集 -> 累计空间 MAE 图与分组池化 RMSE -> 写 npz。"""
    cfg_path, ck_path = tag_paths(tag, args.cfg_root, args.ck_root)
    if not os.path.isfile(cfg_path):
        print("[跳过] {} 缺配置 {}".format(tag, cfg_path))
        return None
    if not os.path.isfile(ck_path):
        print("[跳过] {} 缺 checkpoint {}".format(tag, ck_path))
        return None

    config = load_config(EXPERIMENT, cfg_path)
    set_seeds(config.train.seed)   # 与 90 相同
    loader = build_loader(config, args.split)
    ds = loader.dataset
    print("{}: {} 帧({})".format(tag, len(ds), args.split))

    levels = list(config.data.target_levels)
    statics = CanvasStatics(config.data.statics_dir)
    tables = load_tables(statics, levels)
    sigma = load_norm_sigma(os.path.join(config.data.statics_dir, "normalize_config.json"),
                            config.data.scheme, levels)
    groups, slope, q75, urban = terrain_groups(statics)
    # 备选阈值(仅非零坡度格点的 q75;不进主表,只随 npz 与摘要附报,防"全场 q75=0"钝化)
    nz = slope[slope > 0.0]
    q75_nz = float(np.percentile(nz, 75.0)) if nz.size else 0.0
    steep_alt = slope > q75_nz
    if q75 <= 0.0:
        print("[WARN] {} 坡度上四分位 = 0(域内低平占比过高),陡组会退化为全部非零坡度格点;"
              "npz 里另存备选阈值 steep_alt(q75_nz={:.4f})".format(tag, q75_nz))

    sel_idx = agl_indices(args.sel_agl)
    low_idx = agl_indices(args.low_agl)
    n_sel, n_low = len(sel_idx), len(low_idx)
    ny, nx = 99, 120

    si = EV90.build_si(config, ck_path, device, args.weights == "ema")

    sum_abs_u = np.zeros((n_sel, ny, nx), dtype=np.float64)
    sum_abs_v = np.zeros((n_sel, ny, nx), dtype=np.float64)
    sum_abs_w = np.zeros((n_sel, ny, nx), dtype=np.float64)
    group_n = {k: 0.0 for k in groups}
    group_se2 = {k: 0.0 for k in groups}   # Σ(du²+dv²)
    group_we2 = {k: 0.0 for k in groups}   # Σdw²
    n_frames = 0

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y']
        y0 = batch['y0'].to(device)
        with torch.no_grad():
            pred, _ = si.sample_y1_bare_diffusion(
                y0=EV90.pad16(y0), y_cond=EV90.pad16(x), add_noise=False)
        pred = pred[:, :, :100, :121].detach().cpu()
        for k in range(pred.shape[0]):
            fields = {}
            for name, tensor in (('truth', y[k]), ('pred', pred[k])):
                u, v, w, u10, v10 = EV90.split_denorm(tensor, levels, sigma)
                m = destagger_canvas(u.numpy(), v.numpy(), w.numpy(),
                                     u10.numpy(), v10.numpy())
                fields[name] = agl_fields(m[0], m[1], m[2], m[3], m[4], tables)
            au_p, av_p, aw_p = fields['pred']
            au_t, av_t, aw_t = fields['truth']
            du_sel = (au_p[sel_idx] - au_t[sel_idx]).astype(np.float64)
            dv_sel = (av_p[sel_idx] - av_t[sel_idx]).astype(np.float64)
            dw_sel = (aw_p[sel_idx] - aw_t[sel_idx]).astype(np.float64)
            sum_abs_u += np.abs(du_sel)
            sum_abs_v += np.abs(dv_sel)
            sum_abs_w += np.abs(dw_sel)

            du_low = (au_p[low_idx] - au_t[low_idx]).astype(np.float64)
            dv_low = (av_p[low_idx] - av_t[low_idx]).astype(np.float64)
            dw_low = (aw_p[low_idx] - aw_t[low_idx]).astype(np.float64)
            e2_low = du_low ** 2 + dv_low ** 2
            w2_low = dw_low ** 2
            for name, m in groups.items():
                mb = np.broadcast_to(m[None, :, :], (n_low, ny, nx))
                group_n[name] += float(mb.sum())
                group_se2[name] += float((e2_low * mb).sum())
                group_we2[name] += float((w2_low * mb).sum())
            n_frames += 1
        if n_frames % 24 < pred.shape[0]:
            print("  ... {} 帧".format(n_frames))
        if args.max_frames > 0 and n_frames >= args.max_frames:
            break

    if n_frames == 0:
        print("FAIL {}: 评估 0 帧".format(tag))
        return None

    mae_u = (sum_abs_u / n_frames).astype(np.float32)
    mae_v = (sum_abs_v / n_frames).astype(np.float32)
    mae_w = (sum_abs_w / n_frames).astype(np.float32)
    group_rmse_vec = {k: float(np.sqrt(v / max(group_n[k], 1.0)))
                      for k, v in group_se2.items()}
    group_rmse_w = {k: float(np.sqrt(v / max(group_n[k], 1.0)))
                    for k, v in group_we2.items()}

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    npz_path = os.path.join(args.out_dir, "spatial_{}.npz".format(tag))
    np.savez_compressed(
        npz_path,
        tag=np.array(tag), split=np.array(args.split), n_frames=np.array(n_frames),
        config_path=np.array(cfg_path), checkpoint=np.array(ck_path),
        sel_agl=np.asarray(args.sel_agl, dtype=np.float64),
        low_agl=np.asarray(args.low_agl, dtype=np.float64),
        mae_abs_u=mae_u, mae_abs_v=mae_v, mae_abs_w=mae_w,
        slope=slope.astype(np.float32), steep_q75=np.array(q75),
        steep_alt=steep_alt, steep_alt_q75=np.array(q75_nz),
        urban_frac=urban.astype(np.float32),
        group_names=np.array(sorted(groups.keys())),
        group_rmse_vec=np.array([group_rmse_vec[k] for k in sorted(groups.keys())]),
        group_rmse_w=np.array([group_rmse_w[k] for k in sorted(groups.keys())]),
        group_n=np.array([group_n[k] for k in sorted(groups.keys())]),
        group_note=np.array("n = 掩码格点 x 低层数(10-100 m)x 帧数;"
                            "vec RMSE=sqrt(sum(du^2+dv^2)/n), w RMSE=sqrt(sum(dw^2)/n)"),
    )

    print("")
    print("## {} ({} 帧, {})".format(tag, n_frames, args.split))
    print("- config: `{}`".format(cfg_path))
    print("- 坡度 q75 = {:.4f}(陡格点占比 {:.1f}%,城格点占比 {:.1f}%)".format(
        q75, 100.0 * float(groups['steep'].mean()), 100.0 * float(urban.mean())))
    print("- 备选阈值(仅非零坡度 q75)= {:.4f}(陡占比 {:.1f}%;主表口径仍为全场上四分位)".format(
        q75_nz, 100.0 * float(steep_alt.mean())))
    print("")
    print("选定层空间平均 |误差| (m/s):")
    print("")
    print("| AGL (m) | mean|Δu| | mean|Δv| | mean|Δw| |")
    print("|---|---|---|---|")
    for i, h in enumerate(args.sel_agl):
        print("| {:g} | {:.4f} | {:.4f} | {:.4f} |".format(
            h, mae_u[i].mean(), mae_v[i].mean(), mae_w[i].mean()))
    print("")
    print("低层 {} m 分组池化 RMSE (m/s):".format(
        "-".join("{:g}".format(h) for h in args.low_agl)))
    print("")
    print("| 组 | vec RMSE | w RMSE | n(格点x层x帧) |")
    print("|---|---|---|---|")
    for k in sorted(groups.keys()):
        print("| {} | {:.4f} | {:.4f} | {:.3e} |".format(
            k, group_rmse_vec[k], group_rmse_w[k], group_n[k]))
    print("")
    print("写出 {}".format(npz_path))

    return {
        'tag': tag, 'n_frames': n_frames, 'npz': npz_path,
        'sel_agl': list(args.sel_agl),
        'sel_mean_abs': [[float(mae_u[i].mean()), float(mae_v[i].mean()),
                          float(mae_w[i].mean())] for i in range(n_sel)],
        'group_rmse_vec': group_rmse_vec, 'group_rmse_w': group_rmse_w,
        'group_n': group_n, 'steep_q75': q75,
        'steep_alt_q75': q75_nz,
        'steep_frac': float(groups['steep'].mean()),
        'steep_alt_frac': float(steep_alt.mean()),
        'urban_frac': float(urban.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="阶段 3 T3.5 空间误差分析")
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--out_dir", default="results/phase3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", default="model", choices=["model", "ema"])
    ap.add_argument("--sel_agl", nargs="+", type=float, default=DEFAULT_SEL_AGL,
                    help="逐像素 |误差| 图的 AGL 层(默认 10 50 100 300)")
    ap.add_argument("--low_agl", nargs="+", type=float, default=DEFAULT_LOW_AGL,
                    help="分组池化 RMSE 用的低层子集(默认 10 30 50 70 100)")
    ap.add_argument("--cfg_root", default="configs/深圳")
    ap.add_argument("--ck_root", default=None,
                    help="默认 data/DL_result/ExperimentSchrodingerBridgeWindCanvas")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    device = torch.device(args.device)
    results = []
    for tag in args.tags:
        r = run_tag(tag, args, device)
        if r is not None:
            results.append(r)

    if not results:
        print("FAIL: 没有任何 tag 产出(检查 --tags/--cfg_root 与 checkpoint)")
        sys.exit(1)

    print("")
    print("# 跨 tag 对照(低层分组 vec RMSE, m/s)")
    print("")
    cols = ['steep_city', 'steep_rural', 'flat_city', 'flat_rural', 'steep',
            'flat', 'city', 'rural', 'all']
    print("| tag | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for r in results:
        print("| {} | ".format(r['tag'])
              + " | ".join("{:.4f}".format(r['group_rmse_vec'][c]) for c in cols)
              + " |")
    print("")
    print("逐像素 MAE 图(npz 内 mae_abs_u/v/w,形状 (n_sel,99,120)): "
          + ", ".join(r['npz'] for r in results))
    summary_path = os.path.join(args.out_dir, "spatial_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("写出 {}".format(summary_path))


if __name__ == "__main__":
    main()
