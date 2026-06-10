#!/usr/bin/env python3
"""
检查 WRF 输出文件中所有可用变量
===============================
读取 WRF_DIR 下的 nc 文件，列出全部变量名、维度、属性，
保存到 check_wrf_vars.txt，方便确定能额外提取哪些变量。
"""

import os
import glob
from netCDF4 import Dataset

# ---- 配置 ----
WRF_DIR = "/public3/home/scg4074/gp_zhaokc/SimulationJuly/RESULT_UCM/SLUCM/"
OUTPUT = "check_wrf_vars.txt"

# ---- 主逻辑 ----
nc_files = sorted(glob.glob(os.path.join(WRF_DIR, "*.nc")))
if not nc_files:
    nc_files = sorted(glob.glob(os.path.join(WRF_DIR, "wrfout*")))
if not nc_files:
    print(f"ERROR: 未在 {WRF_DIR} 找到 nc 文件，请修改 WRF_DIR")
    exit(1)

sample_file = nc_files[0]
print(f"共找到 {len(nc_files)} 个 nc 文件")
print(f"检查文件: {sample_file}")

with Dataset(sample_file, 'r') as nc:
    variables = list(nc.variables.keys())
    dimensions = list(nc.dimensions.keys())
    global_attrs = {k: getattr(nc, k) for k in nc.ncattrs()}

with open(OUTPUT, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("WRF 输出变量检查报告\n")
    f.write(f"数据目录: {WRF_DIR}\n")
    f.write(f"文件数量: {len(nc_files)}\n")
    f.write(f"样本文件: {os.path.basename(sample_file)}\n")
    f.write("=" * 70 + "\n\n")

    # 维度
    f.write("--- 维度 ---\n")
    with Dataset(sample_file, 'r') as nc:
        for dim_name, dim_obj in nc.dimensions.items():
            f.write(f"  {dim_name}: size={len(dim_obj)}\n")

    # 全局属性
    f.write(f"\n--- 全局属性 ({len(global_attrs)} 个) ---\n")
    for k, v in sorted(global_attrs.items()):
        val_str = str(v)[:120]
        f.write(f"  {k}: {val_str}\n")

    # 变量详情
    f.write(f"\n--- 变量列表 ({len(variables)} 个) ---\n\n")
    with Dataset(sample_file, 'r') as nc:
        for i, var_name in enumerate(variables):
            var = nc.variables[var_name]
            dims = var.dimensions
            shape = var.shape
            dtype = var.dtype

            # 单位/描述
            units = getattr(var, 'units', '')
            desc = getattr(var, 'description', '')

            f.write(f"[{i+1}] {var_name}\n")
            f.write(f"    dims  : {dims}\n")
            f.write(f"    shape : {shape}\n")
            f.write(f"    dtype : {dtype}\n")
            if units:
                f.write(f"    units : {units}\n")
            if desc:
                f.write(f"    desc  : {desc}\n")

            # 如果有 stagger 属性
            stag = getattr(var, 'stagger', '')
            if stag:
                f.write(f"    stagger: {stag}\n")

            # 如果有 MemoryOrder 属性
            mem = getattr(var, 'MemoryOrder', '')
            if mem:
                f.write(f"    memory: {mem}\n")

            f.write("\n")

    # 分类汇总
    f.write("--- 变量分类汇总 ---\n\n")
    with Dataset(sample_file, 'r') as nc:
        # 4D 变量 (Time, bottom_top, south_north, west_east)
        vars_4d = [v for v in variables if len(nc.variables[v].shape) == 4]
        # 3D 变量 (Time, south_north, west_east)
        vars_3d = [v for v in variables if len(nc.variables[v].shape) == 3]
        # 2D 及其他
        vars_other = [v for v in variables if len(nc.variables[v].shape) not in (3, 4)]

        f.write(f"4D 变量 (Time, bottom_top, south_north, west_east) - {len(vars_4d)} 个:\n")
        for v in vars_4d:
            shape = nc.variables[v].shape
            f.write(f"  {v:<20s} shape={shape}\n")

        f.write(f"\n3D 变量 (Time, south_north, west_east) - {len(vars_3d)} 个:\n")
        for v in vars_3d:
            shape = nc.variables[v].shape
            units = getattr(nc.variables[v], 'units', '')
            desc = getattr(nc.variables[v], 'description', '')
            f.write(f"  {v:<20s} shape={shape}")
            if units:
                f.write(f"  units={units}")
            if desc:
                f.write(f"  desc={desc}")
            f.write("\n")

        f.write(f"\n其他 - {len(vars_other)} 个:\n")
        for v in vars_other:
            shape = nc.variables[v].shape
            f.write(f"  {v:<20s} shape={shape}\n")

print(f"\n结果已保存到 {OUTPUT}")
print(f"共 {len(variables)} 个变量")
