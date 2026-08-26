#!/bin/bash
# 29_d03_prepare.sh — 原生 d03 预处理:冒烟验证 + 启动全量后台任务
# 手机执行: bash ops/queue/29_d03_prepare.sh (约 3-5 分钟,全量在后台跑 ~40 分钟)
# 全量进度用 30 脚本查
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result logs
OUT=ops/result/29_d03_prepare.txt
: > "$OUT"

PY=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python

echo "===== 1. 冒烟测试(处理 1 个 d03 文件 = 24 时次) =====" >> "$OUT"
$PY scripts/prepare_wind_data_3d_sz_d03.py --scheme myj --limit 1 --workers 4 --skip_stats 2>&1 | tail -8 >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 冒烟结果验证 =====" >> "$OUT"
$PY - <<'PYEOF' >> "$OUT" 2>&1
import glob
import numpy as np
files = sorted(glob.glob("prepare_npz_wind_3d_sz_d03/myj_*.npz"))
print("冒烟产出:", len(files), "个 npz")
ok = (len(files) >= 24)
if files:
    with np.load(files[0]) as d:
        keys = sorted(d.keys())
        print("npz keys 数:", len(keys))
        print("hr_u_ml0 shape:", d['hr_u_ml0'].shape, " lr_u_ml0 shape:", d['lr_u_ml0'].shape)
        print("lr_t2 shape:", d['lr_t2'].shape, " swdown:", float(d['swdown'].mean()))
        bad = [k for k in keys if np.any(~np.isfinite(d[k]))]
        print("含非有限值 key:", bad if bad else "无")
        if d['hr_u_ml0'].shape != (96, 112) or d['lr_u_ml0'].shape != (96, 112):
            ok = False
        if bad:
            ok = False
    # 抽查 d03 场与 d04 场的分布差异(分布偏移预览)
    with np.load(files[0]) as d:
        lr_std = float(np.std(d['lr_u_ml0']))
        hr_std = float(np.std(d['hr_u_ml0']))
        print("lr_u_ml0 std: {:.4f}  hr_u_ml0 std: {:.4f} (真实 d03 更平滑则 lr 略小)".format(lr_std, hr_std))
print("验证:", "通过,启动全量" if ok else "失败,不启动全量")
raise SystemExit(0 if ok else 1)
PYEOF
SMOKE_OK=$?

echo "" >> "$OUT"
echo "===== 3. 启动全量后台任务 =====" >> "$OUT"
if [ $SMOKE_OK -eq 0 ]; then
  if [ -f logs/prep_d03_full.pid ] && kill -0 "$(cat logs/prep_d03_full.pid)" 2>/dev/null; then
    echo "全量任务已在跑 (PID $(cat logs/prep_d03_full.pid)),跳过启动" >> "$OUT"
  else
    nohup $PY scripts/prepare_wind_data_3d_sz_d03.py --scheme both --workers 8 \
      > logs/prep_d03_full.log 2>&1 &
    echo $! > logs/prep_d03_full.pid
    echo "已启动全量预处理 (PID $!), 日志 logs/prep_d03_full.log" >> "$OUT"
  fi
else
  echo "冒烟未通过,不启动全量。请把 ops/result/29_d03_prepare.txt 发给我" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 4. 5 秒后冒烟产物清单 =====" >> "$OUT"
sleep 5
ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | head -5 >> "$OUT"
echo "共 $(ls prepare_npz_wind_3d_sz_d03/ 2>/dev/null | wc -l) 个文件" >> "$OUT"

echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 29_d03_prepare" && git push
