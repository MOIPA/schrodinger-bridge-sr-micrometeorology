#!/usr/bin/env python3
"""级联组装(dump -> synth_coarse)的 CPU 预检:不碰模型、不占 GPU,只验管道与形状。

为什么要有它:级联三级串跑,任何一级的粗端组装错了,后面两级都会白跑几十分钟 GPU。
这里用真实粗端模板造一个"假的上一级预测",走一遍 synth_coarse.py,再检查:
  1) 输出的键集合与模板一致;2) 风场形状与模板逐键一致;3) 写进去的层值确实等于输入;
  4) 2D 地面场存在且全 finite。

用法(登录节点即可,wind3d 环境):
  python ops/debug/cascade_smoke.py --stage 2          # 27->9 的输出喂给 9->3
  python ops/debug/cascade_smoke.py --stage 3          # 9->3 的输出喂给 3->1
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

STAGES = {
    2: dict(coarse="prepare_npz_outline_coarse",
            td_from="prepare_npz_outline_coarse_d01",
            td_statics="prepare_npz_outline_static_sq27to9",
            out="prepare_npz_outline_coarse_casc_s2"),
    3: dict(coarse="prepare_npz_outline_coarse_d03",
            td_from="prepare_npz_outline_coarse_d01",
            td_statics="prepare_npz_outline_static_sq9to3",
            out="prepare_npz_outline_coarse_casc_s3"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2, choices=[2, 3])
    ap.add_argument("--scheme", default="myj")
    ap.add_argument("--n_lev", type=int, default=23, help="上一级模型输出的层数(主层)")
    ap.add_argument("--stamp", default=None, help="默认取模板目录里第一个文件")
    ap.add_argument("--keep", action="store_true", help="保留临时目录便于排查")
    args = ap.parse_args()

    cfg = STAGES[args.stage]
    coarse_dir = os.path.join(REPO, cfg["coarse"])
    files = sorted(f for f in os.listdir(coarse_dir) if f.endswith(".npz"))
    if not files:
        raise SystemExit("模板目录为空: {}".format(coarse_dir))
    name = files[0] if args.stamp is None else "c_{}_{}.npz".format(args.scheme, args.stamp)
    stamp = name[len("c_{}_".format(args.scheme)):-len(".npz")]
    print("[smoke] stage={} stamp={} template={}".format(args.stage, stamp, name))

    with np.load(os.path.join(coarse_dir, name)) as f:
        tpl = {k: f[k] for k in f.keys()}
    n_lev_all = tpl["c_u"].shape[0]
    n_w_all = tpl["c_w"].shape[0]
    n_lev = min(args.n_lev, n_lev_all)
    n_w = min(n_lev + 1, n_w_all)
    print("[smoke] 模板 c_u={} c_v={} c_w={} | 取 lev={} wl={}".format(
        tpl["c_u"].shape, tpl["c_v"].shape, tpl["c_w"].shape, n_lev, n_w))

    tmp = tempfile.mkdtemp(prefix="casc_smoke_")
    pred_dir = os.path.join(tmp, "pred")
    out_dir = os.path.join(tmp, "out")
    os.makedirs(pred_dir, exist_ok=True)

    lev = np.arange(n_lev, dtype=np.int64)
    wl = np.arange(n_w, dtype=np.int64)
    u = tpl["c_u"][lev].astype(np.float32)
    v = tpl["c_v"][lev].astype(np.float32)
    w = tpl["c_w"][wl].astype(np.float32)
    np.savez_compressed(os.path.join(pred_dir, "d_{}.npz".format(stamp)),
                        u=u, v=v, w=w, levels=lev, wl=wl)

    cmd = [sys.executable, os.path.join(REPO, "sidequest_scale3x/scripts/synth_coarse.py"),
           "--pred_dir", pred_dir,
           "--coarse_dir", coarse_dir,
           "--td_from", os.path.join(REPO, cfg["td_from"]),
           "--td_statics", os.path.join(REPO, cfg["td_statics"]),
           "--out_dir", out_dir,
           "--scheme", args.scheme]
    print("[smoke] " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit("synth_coarse.py 退出码 {}".format(rc))

    out_file = os.path.join(out_dir, name)
    if not os.path.exists(out_file):
        raise SystemExit("没有产出 {}".format(out_file))
    with np.load(out_file) as g:
        out = {k: g[k] for k in g.keys()}

    problems = []
    missing = set(tpl) - set(out)
    if missing:
        problems.append("输出缺键: {}".format(sorted(missing)))
    for k in ("c_u", "c_v", "c_w"):
        if out[k].shape != tpl[k].shape:
            problems.append("{} 形状 {} != 模板 {}".format(k, out[k].shape, tpl[k].shape))
    if not problems:
        if not np.allclose(out["c_u"][lev], u, equal_nan=True):
            problems.append("c_u 写入的层值与输入不一致")
        if not np.allclose(out["c_w"][wl], w, equal_nan=True):
            problems.append("c_w 写入的层值与输入不一致")
    td2d = ["c_rmol", "c_ust", "c_pblh", "c_hfx", "c_t2", "c_psfc", "c_coszen", "c_coszen_wrf"]
    bad = [k for k in td2d if k in out and not np.all(np.isfinite(out[k]))]
    if bad:
        problems.append("2D 场有非有限值: {}".format(bad))
    print("[smoke] 2D 场与模板差异(重网格后应不同):")
    for k in td2d:
        if k in out and k in tpl:
            d = np.nanmax(np.abs(out[k] - tpl[k]))
            print("   {:14s} max|Δ| = {:.4g}".format(k, float(d)))

    if problems:
        print("[smoke] 失败:")
        for p in problems:
            print("  - " + p)
        print("[smoke] 临时目录保留: {}".format(tmp))
        raise SystemExit(1)
    print("[smoke] 通过:形状/写入值/2D 场都对得上")
    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
