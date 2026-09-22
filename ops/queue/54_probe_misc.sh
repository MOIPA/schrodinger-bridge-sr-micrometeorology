#!/bin/bash
# 54 · 附带探查:LES namelist 内容(eta/参考大气常数)、VEGPARM.TBL 全盘搜索
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
OUT=ops/result/54_probe_misc.txt
S=/fsb/home/yutingwang/share
{
echo "=== namelist.input_SZ.6d ==="
cat "$S/Data_WRFout/case01_Shenzhen/les_20200730/namelist.input_SZ.6d"
echo
echo "=== namelist.input_les.1d (head 80) ==="
head -80 "$S/Data_WRFout/case01_Shenzhen/les_20200730/namelist.input_les.1d"
echo
echo "=== VEGPARM search (deeper) ==="
find "$S/software" "$S/MyLibs" "$S/MyModules" -maxdepth 8 -iname "VEGPARM*" 2>/dev/null | head
find "$S" -maxdepth 7 -iname "GEOGRID.TBL*" 2>/dev/null | head
echo
echo "=== any *.TBL in share (depth<=6, first 30) ==="
find "$S" -maxdepth 6 -iname "*.TBL" 2>/dev/null | head -30
echo
echo "=== meso myj dir listing (first 15) ==="
ls "$S/Data_WRFout/case01_Shenzhen/meso_202007_myj/" | head -15
echo
echo "=== Data_WRFout root ==="
ls "$S/Data_WRFout/" | head -20
} > "$OUT" 2>&1
echo "exit=$?" >> "$OUT"
cd ops && git add . && git commit -m "result 54 probe misc" && git push
