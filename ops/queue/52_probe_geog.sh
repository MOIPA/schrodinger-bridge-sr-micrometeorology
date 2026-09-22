#!/bin/bash
# 52 · 探查 WPS_GeoStatic 地理数据(土地/植被格式)、share 根目录、namelist/VEGPARM 位置
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
OUT=ops/result/52_geog_probe.txt
G=/fsb/home/yutingwang/share/WPS_GeoStatic
{
echo "=== share root ==="
ls -la /fsb/home/yutingwang/share/ | head -40
echo
echo "=== WPS_GeoStatic root ==="
ls "$G/"
echo
echo "=== modis_landuse_20class_15s ==="
ls -la "$G/modis_landuse_20class_15s/" | head -25
echo
echo "=== index (first 300 bytes, od -c) ==="
head -c 300 "$G/modis_landuse_20class_15s/index" | od -c | head -25
echo
echo "=== greenfrac_fpar_modis ==="
ls -la "$G/greenfrac_fpar_modis/" | head -20
echo
echo "=== index of greenfrac (first 300 bytes) ==="
head -c 300 "$G/greenfrac_fpar_modis/index" | od -c | head -25
echo
echo "=== topo dirs ==="
ls -d "$G"/*topo* "$G"/*GMTED* "$G"/*gtopo* 2>/dev/null
echo
echo "=== VEGPARM / GEOGRID.TBL / namelist search under share (maxdepth 5) ==="
find /fsb/home/yutingwang/share -maxdepth 5 -iname "VEGPARM*" 2>/dev/null | head
find /fsb/home/yutingwang/share -maxdepth 5 -iname "GEOGRID.TBL*" 2>/dev/null | head
find /fsb/home/yutingwang/share -maxdepth 5 -iname "namelist*" 2>/dev/null | head -20
echo
echo "=== repo copies ==="
find "$HOME/schrodinger-bridge-sr-micrometeorology" -iname "VEGPARM*" -o -iname "GEOGRID.TBL*" 2>/dev/null | head
} > "$OUT" 2>&1
echo "exit=$?" >> "$OUT"
cd ops && git add . && git commit -m "result 52 geog probe" && git push
