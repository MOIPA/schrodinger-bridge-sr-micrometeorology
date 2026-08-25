#!/bin/bash
# 01_check_prep.sh — 检查深圳预处理状态(进程/数量/key/日志)与服务器仓库状态
# 手机在服务器项目根目录执行: bash ops/queue/01_check_prep.sh
# 结果写入 ops/result/01_check_prep.txt,然后 git add/commit/push 回传

cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result || exit 1
OUT=ops/result/01_check_prep.txt
: > "$OUT"

echo "===== 1. 服务器时间 =====" >> "$OUT"
date >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. nohup 预处理进程 (PID 26565) =====" >> "$OUT"
ps -p 26565 -o pid,etime,cmd 2>/dev/null >> "$OUT" || echo "(PID 26565 不存在)" >> "$OUT"
ps aux | grep "prepare_wind_data_3d_sz" | grep -v grep >> "$OUT" || echo "(无运行中的预处理进程)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 输出目录 npz 数量 =====" >> "$OUT"
ls prepare_npz_wind_3d_sz/ 2>/dev/null | wc -l >> "$OUT"
echo "--- 按 scheme 统计 ---" >> "$OUT"
ls prepare_npz_wind_3d_sz/ 2>/dev/null | sed 's/_.*//' | sort | uniq -c >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 日志尾部 (logs/prep_sz_nohup.out) =====" >> "$OUT"
tail -40 logs/prep_sz_nohup.out 2>/dev/null || echo "(日志不存在)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 5. 抽查 npz 的 key (检查是否有 lr_* 条件变量) =====" >> "$OUT"
PYBIN=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
if [ -x "$PYBIN" ]; then
  "$PYBIN" - "$OUT" <<'PYEOF'
import sys, glob
import numpy as np
out = sys.argv[1]
def w(s):
    with open(out, 'a') as f:
        f.write(s + '\n')
files = sorted(glob.glob('prepare_npz_wind_3d_sz/*.npz'))
w('--- npz 总数: %d' % len(files))
if files:
    d = np.load(files[0])
    keys = list(d.keys())
    w('--- 示例文件: ' + files[0])
    w('--- key 数: %d' % len(keys))
    w('--- keys: ' + ', '.join(keys))
    lr_conds = [k for k in keys if k.startswith('lr_')]
    w('--- lr_* 条件变量 key 数: %d' % len(lr_conds))
    for k in lr_conds:
        w('    %s %s' % (k, str(d[k].shape)))
    # 统计前 50 个文件中含 lr_ key 的文件数(新版本脚本才有)
    n_with_lr = 0
    for f in files[:50]:
        try:
            dd = np.load(f)
            if any(k.startswith('lr_') for k in dd.keys()):
                n_with_lr += 1
        except Exception:
            pass
    w('--- 前50个文件中含 lr_ key 的文件数: %d / 50' % n_with_lr)
else:
    w('--- (输出目录为空)')
PYEOF
else
  echo "(pytorch-gpu python 不存在,请检查 env 路径)" >> "$OUT"
fi

echo "" >> "$OUT"
echo "===== 6. 服务器仓库状态 =====" >> "$OUT"
git status --short | head -10 >> "$OUT"
echo "--- 最近提交 ---" >> "$OUT"
git log --oneline -5 >> "$OUT"
echo "--- 分支同步情况 ---" >> "$OUT"
git status -sb | head -2 >> "$OUT"

echo "" >> "$OUT"
echo "===== 完成 =====" >> "$OUT"
cat "$OUT"
