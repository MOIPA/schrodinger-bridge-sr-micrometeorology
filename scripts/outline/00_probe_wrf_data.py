# -*- coding: utf-8 -*-
"""
阶段 0 (导师大纲) · T0.1 数据探查脚本

在服务器上运行,回答以下问题并输出 JSON 报告:
  1. case01_Shenzhen 目录结构(各 scheme 各 domain 文件数、时间覆盖与步长)
  2. wrfinput_d0X 是否存在(静态场 LANDUSEF/VEGFRA/ZNT 的最自洽来源)
  3. d01-d04 网格尺寸、垂直层数、eta 配置与参考大气常数(global attrs)一致性
  4. 由 PHB 计算的各档模式层离地高度 z_agl,统计 1/2 km 以下层数(定训练层集合 N)
  5. 新管线所需变量(含 XLAT_U/V 等交错坐标)在各域/各 scheme 的可用性清单
  6. WPS_GeoStatic 备选静态场来源的存在性

不写任何数据,只读。

运行(pytorch-gpu 环境有 netCDF4):
  /fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/outline/00_probe_wrf_data.py \
      --json_out results/outline/00_probe_wrf_data.json
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

WRF_BASE = "/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen"
SHARE_BASE = "/fsb/home/yutingwang/share"
GEOG_CANDIDATES = [
    os.path.join(SHARE_BASE, "WPS_GeoStatic"),
    os.path.join(WRF_BASE, "WPS_GeoStatic"),
    os.path.join(WRF_BASE, "..", "WPS_GeoStatic"),
    os.path.join(SHARE_BASE, "Data_WRFout", "WPS_GeoStatic"),
]
DOMAINS = ["d01", "d02", "d03", "d04"]
G = 9.81

ATTR_WHITELIST = [
    "TITLE", "START_DATE", "SIMULATION_START_DATE", "GRID_ID",
    "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
    "BOTTOM-TOP_GRID_DIMENSION", "DX", "DY", "CEN_LAT", "CEN_LON",
    "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON",
    "P_TOP", "BASE_PRES", "BASE_TEMP", "BASE_LAPSE", "NUM_LAND_CAT",
    "MMINLU", "ISWATER", "ISLAKE", "ISICE", "ISURBAN",
]

PROBE_VARS = [
    # 动力学 / 几何
    "U", "V", "W", "PH", "PHB", "T", "P", "PB", "QVAPOR", "MU", "MUB",
    "U10", "V10", "HGT", "ZNU", "ZNW", "ZS",
    # 坐标(交错位置,用于重网格配对)
    "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
    # 地表 / 边界层 / 强迫
    "T2", "PSFC", "TSK", "PBLH", "HFX", "LH", "RMOL", "UST",
    "ZNT", "LU_INDEX", "LANDUSEF", "VEGFRA", "COSZEN",
    "SWDOWN", "SWNORM", "GLW", "RAINC", "RAINNC",
]


def jsonable(o):
    """Recursively convert numpy types so json.dump works."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def read_times(ncfile):
    """Extract all timestamps in a wrfout file."""
    times = ncfile.variables['Times']
    out = []
    for i in range(times.shape[0]):
        chars = times[i]
        ts = b''.join([c if isinstance(c, bytes) else str(c).encode('utf-8')
                       for c in chars]).decode('utf-8').strip()
        out.append(datetime.strptime(ts, '%Y-%m-%d_%H:%M:%S'))
    return out


def iso(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%S')


def glob_domain_files(scheme_dir, dom):
    return sorted(glob.glob(os.path.join(scheme_dir, "wrfout_" + dom + "_*")))


def level_report(ncfile):
    """PHB/g - HGT 计算离地高度,给出各层统计与阈值计数。"""
    if 'PHB' not in ncfile.variables or 'HGT' not in ncfile.variables:
        return {"error": "PHB or HGT missing"}
    phb = np.array(ncfile.variables['PHB'][0], dtype=np.float64)   # (nz_stag, ny, nx)
    hgt = np.array(ncfile.variables['HGT'][0], dtype=np.float64)   # (ny, nx)
    z_if_agl = phb / G - hgt[None, :, :]
    z_mass_agl = 0.5 * (z_if_agl[:-1] + z_if_agl[1:])
    rep = {
        "n_levels_mass": int(z_mass_agl.shape[0]),
        "n_levels_iface": int(z_if_agl.shape[0]),
        "mass_mean": [round(float(x), 2) for x in z_mass_agl.mean(axis=(1, 2))],
        "mass_min": [round(float(x), 2) for x in z_mass_agl.min(axis=(1, 2))],
        "mass_max": [round(float(x), 2) for x in z_mass_agl.max(axis=(1, 2))],
        "iface_mean": [round(float(x), 2) for x in z_if_agl.mean(axis=(1, 2))],
    }
    mean_prof = z_mass_agl.mean(axis=(1, 2))
    for thr in (1000, 1500, 2000, 2500):
        rep["mass_below_" + str(thr) + "m"] = int((mean_prof < thr).sum())
    mean_if = z_if_agl.mean(axis=(1, 2))
    for thr in (1000, 2000):
        rep["iface_below_" + str(thr) + "m"] = int((mean_if < thr).sum())
    return rep


def probe_domain(files, dom):
    """Probe first (and last) file of a domain."""
    rep = {"files": {"count": len(files),
                     "first": os.path.basename(files[0]),
                     "last": os.path.basename(files[-1])}}
    with Dataset(files[0]) as nc:
        # global attrs (whitelist + full dump)
        attrs = {}
        attrs_all = {}
        for k in nc.ncattrs():
            v = jsonable(nc.getncattr(k))
            attrs_all[k] = str(v)[:200]
            if k in ATTR_WHITELIST:
                attrs[k] = v
        rep["attrs"] = attrs
        rep["attrs_all"] = attrs_all

        # times in first file + step inference
        t0 = read_times(nc)
        rep["times"] = {
            "n_per_file": len(t0),
            "first": iso(t0[0]),
            "last_in_file": iso(t0[-1]),
            "step_sec_in_file": int((t0[1] - t0[0]).total_seconds()) if len(t0) > 1 else None,
        }

        # variable availability and dims (dims exclude Time)
        present = [v for v in PROBE_VARS if v in nc.variables]
        dims = {}
        for v in present:
            dims[v] = [int(s) for s in nc.variables[v].shape[1:]]
        rep["vars_present"] = present
        rep["vars_missing"] = [v for v in PROBE_VARS if v not in nc.variables]
        rep["dims"] = dims

        # horizontal size / bbox
        lat = np.array(nc.variables['XLAT'][0]) if 'XLAT' in nc.variables else None
        lon = np.array(nc.variables['XLONG'][0]) if 'XLONG' in nc.variables else None
        if lat is not None:
            rep["bbox"] = {"lat": [round(float(lat.min()), 4), round(float(lat.max()), 4)],
                           "lon": [round(float(lon.min()), 4), round(float(lon.max()), 4)]}

        # AGL profile
        rep["agl"] = level_report(nc)

    if len(files) > 1:
        with Dataset(files[-1]) as nc:
            t1 = read_times(nc)
        dt_cross = (t1[0] - t0[-1]).total_seconds()
        rep["times"]["last"] = iso(t1[-1])
        rep["times"]["step_sec_across_file"] = int(dt_cross)

    return rep


def probe_geog():
    rep = {"candidates": GEOG_CANDIDATES, "found": None, "entries": []}
    for c in GEOG_CANDIDATES:
        if os.path.isdir(c):
            rep["found"] = c
            try:
                rep["entries"] = sorted(os.listdir(c))[:80]
            except OSError as e:
                rep["error"] = str(e)
            break
    return rep


def probe_wrfinput(scheme_dir):
    """Check for wrfinput files (static fields source)."""
    rep = {}
    for dom in DOMAINS:
        files = sorted(glob.glob(os.path.join(scheme_dir, "wrfinput_" + dom + "*")))
        if not files:
            rep[dom] = {"count": 0}
            continue
        item = {"count": len(files), "names": [os.path.basename(f) for f in files[:3]]}
        try:
            with Dataset(files[0]) as nc:
                item["vars_of_interest"] = [v for v in
                                            ["LANDUSEF", "VEGFRA", "ZNT", "HGT", "LU_INDEX",
                                             "PHB", "XLAT", "XLONG", "LANDMASK", "TSK", "TMN"]
                                            if v in nc.variables]
                item["dims"] = {v: [int(s) for s in nc.variables[v].shape]
                                for v in item["vars_of_interest"]}
        except Exception as e:  # noqa: BLE001 - report and continue
            item["error"] = str(e)
        rep[dom] = item
    return rep


def main():
    parser = argparse.ArgumentParser(description="阶段0 T0.1 数据探查")
    parser.add_argument("--base", default=WRF_BASE)
    parser.add_argument("--json_out", default="results/outline/00_probe_wrf_data.json")
    args = parser.parse_args()

    report = {
        "base": args.base,
        "probed_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "schemes": {},
        "eta_check": {},
        "wrfinput": {},
        "wps_geog": {},
    }

    if not os.path.isdir(args.base):
        print("ERROR: base dir not found: " + args.base)
        sys.exit(1)

    scheme_dirs = sorted([os.path.join(args.base, d) for d in os.listdir(args.base)
                          if os.path.isdir(os.path.join(args.base, d))])
    print("base: " + args.base)
    print("subdirs: " + ", ".join(os.path.basename(d) for d in scheme_dirs))

    znu_cache = {}   # (scheme, dom) -> ZNU array (if present)
    for scheme_dir in scheme_dirs:
        scheme = os.path.basename(scheme_dir)
        rep = {}
        for dom in DOMAINS:
            files = glob_domain_files(scheme_dir, dom)
            if not files:
                continue
            print("  probing {} / {} ({} files)".format(scheme, dom, len(files)))
            rep[dom] = probe_domain(files, dom)
            with Dataset(files[0]) as nc:
                if 'ZNU' in nc.variables:
                    znu_cache[(scheme, dom)] = np.array(nc.variables['ZNU'][0], dtype=np.float64)
        report["schemes"][scheme] = rep
        report["wrfinput"][scheme] = probe_wrfinput(scheme_dir)

    # eta configuration consistency (ZNU equality across domains / schemes)
    def _diff(a, b):
        if a is None or b is None:
            return None
        if a.shape != b.shape:
            return "shape mismatch: {} vs {}".format(a.shape, b.shape)
        return float(np.abs(a - b).max())

    ref_key = None
    for s in sorted(report["schemes"].keys()):
        if (s, "d04") in znu_cache:
            ref_key = (s, "d04")
            break
    ref = znu_cache.get(ref_key) if ref_key is not None else None
    for key, arr in znu_cache.items():
        report["eta_check"]["|".join(key)] = {
            "maxdiff_vs_first_scheme_d04": _diff(arr, ref),
            "n_levels": int(arr.shape[0]),
            "first_values": [round(float(x), 6) for x in arr[:5]],
        }

    report["wps_geog"] = probe_geog()

    # write JSON
    out_dir = os.path.dirname(args.json_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.json_out, 'w') as f:
        json.dump(jsonable(report), f, indent=2, ensure_ascii=False)
    print("\nJSON report written: " + args.json_out)

    # compact stdout summary
    for scheme, rep in report["schemes"].items():
        for dom, d in rep.items():
            t = d["times"]
            print("\n[{}/{}] files={} times/file={} step={}s  {} -> {}".format(
                scheme, dom, d["files"]["count"], t["n_per_file"],
                t.get("step_sec_in_file"), t.get("first"), t.get("last")))
            agl = d.get("agl", {})
            if "mass_mean" in agl:
                print("   mass grid={} nz={} nz_stag={}  below 1km={} below 2km={}".format(
                    d["dims"].get("T", [None, None, None])[1:],
                    agl["n_levels_mass"], agl["n_levels_iface"],
                    agl["mass_below_1000m"], agl["mass_below_2000m"]))
                print("   mass z_agl (first 25 levels, domain mean): " +
                      ", ".join(str(x) for x in agl["mass_mean"][:25]))
            if d["vars_missing"]:
                print("   MISSING vars: " + ", ".join(d["vars_missing"]))
    print("\nattrs (first scheme): " +
          json.dumps(report["schemes"][sorted(report["schemes"].keys())[0]]
                     .get("d04", {}).get("attrs", {}), ensure_ascii=False))
    print("\nwrfinput: " + json.dumps(report["wrfinput"], ensure_ascii=False)[:1500])
    print("\nwps_geog: " + json.dumps(report["wps_geog"], ensure_ascii=False)[:800])
    print("\neta_check: " + json.dumps(report["eta_check"], ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()
