#!/bin/bash
# 28_probe_d03.sh — 探测原生 d03 wrfout(真实低精度数据源) + 回传评估任务结果
# 手机执行: bash ops/queue/28_probe_d03.sh (约 1 分钟)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/28_probe_d03.txt
: > "$OUT"

echo "===== 1. 评估任务 sz_eval 状态 =====" >> "$OUT"
bjobs -w 2>/dev/null | grep sz_eval >> "$OUT" || echo "(不在队列,可能已结束)" >> "$OUT"
LATEST=$(ls -t logs/sz_eval_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "评估日志: $LATEST ($(stat -c %y "$LATEST" | cut -d. -f1))" >> "$OUT"
  grep -a "Evaluating\|Test samples\|SKIP\|Saved\|complete\|Error\|Traceback" "$LATEST" 2>/dev/null | tail -15 >> "$OUT"
  echo "--- 日志尾部 ---" >> "$OUT"
  tail -5 "$LATEST" >> "$OUT"
fi
if ls results/sz_eval/*.json >/dev/null 2>&1; then
  echo "--- sz_eval 已产出 ---" >> "$OUT"
  ls -la results/sz_eval/ >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 2. d03 wrfout 文件清单 =====" >> "$OUT"
for d in meso_202007_myj meso_202007_ysu; do
  DIR=/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/$d
  echo "--- $d ---" >> "$OUT"
  ls "$DIR"/wrfout_d03_* 2>/dev/null | head -2 >> "$OUT"
  echo "d03 文件数: $(ls "$DIR"/wrfout_d03_* 2>/dev/null | wc -l)" >> "$OUT"
  echo "各域文件数: $(ls "$DIR"/wrfout_d0*_* 2>/dev/null | sed 's/.*wrfout_d\([0-9]*\)_.*/\1/' | sort | uniq -c | tr '\n' ';')" >> "$OUT"
done

echo "" >> "$OUT"
echo "===== 3. 单个 d03 文件结构详情 =====" >> "$OUT"
/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python - <<'PYEOF' >> "$OUT" 2>&1
import glob
from netCDF4 import Dataset

for d in ["meso_202007_myj", "meso_202007_ysu"]:
    files = sorted(glob.glob("/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/%s/wrfout_d03_*" % d))
    print("### %s: d03 共 %d 个文件" % (d, len(files)))
    if not files:
        continue
    with Dataset(files[0]) as nc:
        print("  文件:", files[0].split('/')[-1])
        n_t = nc.variables['Times'].shape[0]
        ts = []
        for i in range(min(n_t, 4)):
            chars = nc.variables['Times'][i]
            s = b''.join([c if isinstance(c, bytes) else c.encode('utf-8') for c in chars]).decode('utf-8').strip()
            ts.append(s)
        print("  Times 个数: %d, 前4个: %s" % (n_t, ts))
        for v in ['XLAT','XLONG','U','V','W','T2','HGT','LU_INDEX','TSK','HFX','LH','PSFC','PBLH','SWDOWN','GLW','ZNU','ZNW']:
            if v in nc.variables:
                print("  %-8s shape=%s" % (v, nc.variables[v].shape))
            else:
                print("  %-8s (缺失)" % v)

# d04 参考文件的 XLAT/XLONG(插值目标网格)
d04files = sorted(glob.glob("/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_myj/wrfout_d04_*"))
if d04files:
    with Dataset(d04files[0]) as nc:
        print("### d04 参考 XLAT shape:", nc.variables['XLAT'].shape,
              " XLONG shape:", nc.variables['XLONG'].shape)
PYEOF

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传(含 sz_eval 结果,如有)
cd ops && git add . && git add ../results/sz_eval 2>/dev/null; git commit -m "result 28_probe_d03" && git push
