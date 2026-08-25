#!/bin/bash
# 02_investigate_prep.sh — 调查深圳预处理中断情况(真实日志/时间范围/资源)
# 手机在服务器项目根目录执行: bash ops/queue/02_investigate_prep.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/02_investigate_prep.txt
: > "$OUT"

echo "===== 1. logs/ 目录内容 (按时间排序) =====" >> "$OUT"
ls -lt logs/ 2>/dev/null | head -25 >> "$OUT"
echo "--- logs/ 总文件数 ---" >> "$OUT"
ls logs/ 2>/dev/null | wc -l >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 项目根目录的 nohup/out 文件 =====" >> "$OUT"
ls -lt nohup*.out 2>/dev/null | head -10 >> "$OUT"
ls -lt *.out 2>/dev/null | head -10 >> "$OUT"

echo "" >> "$OUT"
echo "===== 3. 输出目录时间分布 =====" >> "$OUT"
echo "--- 最新 5 个文件 ---" >> "$OUT"
ls -lt prepare_npz_wind_3d_sz/ | head -5 >> "$OUT"
echo "--- 最早 3 个文件 ---" >> "$OUT"
ls -lt prepare_npz_wind_3d_sz/ | tail -3 >> "$OUT"

echo "" >> "$OUT"
echo "===== 4. 各 scheme 覆盖时间范围 =====" >> "$OUT"
echo "--- ysu 最早 ---" >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep '^ysu' | sort | head -2 >> "$OUT"
echo "--- ysu 最晚 ---" >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep '^ysu' | sort | tail -2 >> "$OUT"
echo "--- myj 最早 ---" >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep '^myj' | sort | head -2 >> "$OUT"
echo "--- myj 最晚 ---" >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep '^myj' | sort | tail -2 >> "$OUT"
echo "--- 数量: myj / ysu / 其他 ---" >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep -c '^myj' >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep -c '^ysu' >> "$OUT"
ls prepare_npz_wind_3d_sz/ | grep -vc '^\(myj\|ysu\)' >> "$OUT"

echo "" >> "$OUT"
echo "===== 5. 当前 python 进程 =====" >> "$OUT"
ps aux | grep -i python | grep -v grep >> "$OUT" || echo "(无 python 进程)" >> "$OUT"

echo "" >> "$OUT"
echo "===== 6. 内存与磁盘 =====" >> "$OUT"
free -g | head -3 >> "$OUT"
df -h /fsb /fs00 2>/dev/null >> "$OUT"

echo "" >> "$OUT"
echo "===== 7. 预处理脚本 OUTPUT_BASE 与 git 版本 =====" >> "$OUT"
grep -n "OUTPUT_BASE" scripts/prepare_wind_data_3d_sz.py | head -2 >> "$OUT"
git log --oneline -3 -- scripts/prepare_wind_data_3d_sz.py >> "$OUT"

echo "" >> "$OUT"
echo "===== 完成 =====" >> "$OUT"
cat "$OUT"
