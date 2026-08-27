#!/bin/bash
# 46_check_level_heights.sh — 实测 ml0/1/2/3/5/10 六层的真实高度
# 用 WRF 输出的 PH/PHB(位势高度)直接计算,验证报告里"约 30 m 至 10 km"的说法
# 手机执行: bash ops/queue/46_check_level_heights.sh (约 1 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/46_level_heights.txt
: > "$OUT"

echo "===== 1. 定位 wrfout 文件 =====" >> "$OUT"
F=$(ls /fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_myj/wrfout_d04_* 2>/dev/null | head -1)
if [ -z "$F" ]; then
  F=$(find /fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen -name "wrfout_d04_*" 2>/dev/null | head -1)
fi
echo "文件: $F" >> "$OUT"
if [ -z "$F" ]; then
  echo "找不到 wrfout 文件,中止(检查 WRF_BASE 路径)" >> "$OUT"
  echo "===== 完成 =====" >> "$OUT"
  cat "$OUT"
  cd ops && git add . && git commit -m "result 46_check_level_heights" && git push
  exit 1
fi
export F

echo "" >> "$OUT"
echo "===== 2. 用 PH/PHB 计算各层实际高度(全网格平均) =====" >> "$OUT"
module load anaconda/3 2>/dev/null; source activate wind3d 2>/dev/null || conda activate wind3d 2>/dev/null
python - <<'PYEOF' >> "$OUT" 2>&1
import os
import numpy as np
import netCDF4 as nc

f = nc.Dataset(os.environ["F"])
g = 9.81
phb = f.variables["PHB"]  # (Time, bottom_top_stag, south_north, west_east), m2/s2
ph  = f.variables["PH"]
print("层数(bottom_top):", f.dimensions["bottom_top"].size)
print("网格:", f.dimensions["south_north"].size, "x", f.dimensions["west_east"].size)
print("时次:", f.dimensions["Time"].size)
# 界面层位势高度 -> 质量层高度(相邻界面平均)
z_iface = (phb[0] + ph[0]) / g          # (62, sn, we)
z_mass = 0.5 * (z_iface[:-1] + z_iface[1:])  # (61, sn, we)
print("")
print("目标 6 层:")
for idx in [0, 1, 2, 3, 5, 10]:
    z = z_mass[idx]
    print(f"  ml{idx:<2} (索引 {idx:>2}): 平均 {z.mean():8.1f} m | 范围 {z.min():8.1f} ~ {z.max():8.1f} m")
print("")
print("参考(10 km 在哪一层):")
for idx in [20, 30, 40, 45]:
    z = z_mass[idx]
    print(f"  索引 {idx:>2}: 平均 {z.mean():8.1f} m")
f.close()
PYEOF

echo "" >> "$OUT"
echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 46_check_level_heights" && git push
