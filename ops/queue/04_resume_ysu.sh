#!/bin/bash
# 04_resume_ysu.sh — 登录节点 nohup 续跑 ysu 剩余部分(7-10 18:00 后,~510 文件)
# 启动后立即回传状态;跑完约 30-40 分钟,之后跑 05_check_final.sh 验证
# 手机执行: bash ops/queue/04_resume_ysu.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result

PYBIN=/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python
LOG=logs/prep_sz_ysu_resume.out

# 防重复启动
if pgrep -f "prepare_wind_data_3d_sz.py --scheme ysu" > /dev/null 2>&1; then
  echo "!!! 已有 ysu 预处理在运行,不重复启动 !!!"
else
  nohup "$PYBIN" scripts/prepare_wind_data_3d_sz.py --scheme ysu --workers 4 > "$LOG" 2>&1 &
  echo "已启动 nohup, PID=$!  日志=$LOG"
fi

OUT=ops/result/04_resume_ysu.txt
{
  echo "===== 04 启动状态 ====="
  date
  pgrep -af "prepare_wind_data_3d_sz" || echo "(无进程)"
  echo "--- 当前 ysu 数量 ---"
  ls prepare_npz_wind_3d_sz/ 2>/dev/null | grep -c '^ysu'
} > "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 04_resume_ysu" && git push
