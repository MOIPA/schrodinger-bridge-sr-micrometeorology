#!/usr/bin/env python3
"""把上一级超分预测的风场,组装成下一级"粗端输入"npz 目录(级联超分用)。

思路:级联 27→9→3→1 里,第 S 级模型的输出(9 km 风场)正好就是第 S+1 级
(9→3)粗端该看到的场。因此只要把预测的 u/v/w 填进下一级粗端 npz 的对应层,
其余 2D 场(psfc/t2/hfx/pblh/ust/rmol/coszen)按"从 27 km 一路带下来"的方式
用上一级粗端 2D 场重网格过来 —— 这样整条级联只用到 27 km 的信息,和一步
27→1 的信息量对等(否则级联会偷看中间尺度的真值地面场)。

用法:
  python3 sidequest_scale3x/scripts/synth_coarse.py \
      --pred_dir /tmp/casc_s1 --coarse_dir prepare_npz_outline_coarse \
      --td_from prepare_npz_outline_coarse_d01 --td_statics prepare_npz_outline_static_sq27to9 \
      --out_dir prepare_npz_outline_coarse_casc_s2
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.dl_data.wind_canvas_statics import CanvasStatics  # noqa: E402

TD2D = ['c_rmol', 'c_ust', 'c_pblh', 'c_hfx', 'c_t2', 'c_psfc', 'c_coszen', 'c_coszen_wrf']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="84 --dump_fields 的输出目录(d_*.npz)")
    ap.add_argument("--coarse_dir", required=True, help="下一级粗端 npz 目录(提供形状/键/时间列表)")
    ap.add_argument("--td_from", default=None, help="上一级粗端目录(提供 2D 场),缺省则沿用 coarse_dir 自己的 2D 场")
    ap.add_argument("--td_statics", default=None, help="把 td_from 的 2D 场重网格到本级网格所用的静态场目录")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scheme", default="myj")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个时次(调试用)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preds = sorted(glob.glob(os.path.join(args.pred_dir, "d_*.npz")))
    if not preds:
        raise SystemExit("没有找到 {}".format(os.path.join(args.pred_dir, "d_*.npz")))
    if args.limit:
        preds = preds[:args.limit]

    stat = CanvasStatics(args.td_statics) if (args.td_from and args.td_statics) else None
    if stat is not None:
        # 守卫:--td_from 必须是 --td_statics 的"源网格"那一级。
        # 级联里最常见的错法是把 d01(27 km)的地面场喂给 d02→d03 的 statics,
        # 报出来的却是一句看不懂的 reshape 错误,所以提前比对格点数。
        probe = sorted(glob.glob(os.path.join(args.td_from, "c_*.npz")))
        if not probe:
            raise SystemExit("--td_from 目录里没有 c_*.npz: {}".format(args.td_from))
        with np.load(probe[0]) as g:
            key = next((k for k in TD2D if k in g), None)
            if key is not None:
                got = int(np.prod(g[key].shape))
                want = int(np.prod(stat.fine_shapes['mass']))
                if got != want:
                    raise SystemExit(
                        "--td_from 的 2D 场有 {} 个格点({}),而 --td_statics 的源网格是 {}".format(
                            got, g[key].shape, stat.fine_shapes['mass'])
                        + "\n  级联里这一级的地面场应该来自上一级输出目录(如 prepare_npz_outline_coarse_casc_s2),"
                        + "而不是更粗的 d01。")
    n_done = 0
    for p in preds:
        stamp = os.path.basename(p)[2:-4]
        with np.load(p) as d:
            u, v, w = d['u'], d['v'], d['w']
            lev = d['levels']
            wl = d['wl']
        src = os.path.join(args.coarse_dir, "c_{}_{}.npz".format(args.scheme, stamp))
        if not os.path.exists(src):
            print("[跳过] 缺模板 {}".format(src))
            continue
        with np.load(src) as f:
            out = {k: f[k] for k in f.keys()}

        # 风场:用预测值替换对应层
        out['c_u'][lev] = u
        out['c_v'][lev] = v
        out['c_w'][wl] = w

        # 2D 场:从上一级粗端重网格过来(保持"只有 27 km 信息")
        if stat is not None:
            tdf = os.path.join(args.td_from, "c_{}_{}.npz".format(args.scheme, stamp))
            if not os.path.exists(tdf):
                print("[跳过] 缺上一级粗端 {}".format(tdf))
                continue
            with np.load(tdf) as g:
                for k in TD2D:
                    if k not in g or k not in out:
                        continue
                    out[k] = stat.regrid_field(
                        np.asarray(g[k], dtype=np.float32)[None], 'mass')[0]

        np.savez_compressed(os.path.join(args.out_dir,
                                         "c_{}_{}.npz".format(args.scheme, stamp)), **out)
        n_done += 1
        if n_done % 24 == 0:
            print("  已写 {} 个时次".format(n_done))
    print("完成:{} -> {} ({} 个时次)".format(args.pred_dir, args.out_dir, n_done))


if __name__ == "__main__":
    main()
