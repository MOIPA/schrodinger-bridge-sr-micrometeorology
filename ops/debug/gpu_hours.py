#!/usr/bin/env python3
"""统计 LSF 作业实际占用的 GPU 小时。

数据来源:LSF 在作业结束时写到 `logs/<jobname>_<jobid>.out` 的邮件正文,里面有
  Started at <time> / Terminated at <time> / CPU time : <sec> / 队列 / 主机
以及提交时的命令行(含 `-gpu num=1:mode=exclusive_process`)。

口径:
- 墙钟小时 = Terminated − Started(该作业真正占着节点的时间);
- GPU 小时 = 墙钟小时 × 每个作业申请的 GPU 数(从 -gpu/-R 里解析,默认 1);
- 只统计"有 Started 和 Terminated 两行"的作业(排除从未运行的);
- 同时给出 CPU 小时(所有核加起来),用来判断是不是被卡在 CPU 上。

用法:
    python ops/debug/gpu_hours.py --logdir logs --out ops/result/gpu_hours.txt
    python ops/debug/gpu_hours.py --logdir logs --since "2026-10-01" --group
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

TIME_RE = re.compile(r"(\w{3})\s+(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})\s+(\d{4})")
JOB_RE = re.compile(r"Subject:\s*Job\s+(\d+):\s*<([^>]*)>")
QUEUE_RE = re.compile(r"in queue <([^>]*)>")
HOST_RE = re.compile(r"was executed on host\(s\) <([^>]*)>")
GPU_RE = re.compile(r"-gpu\s+num=(\d+)")
GPU_R_RE = re.compile(r"rusage\[ngpus_physical=(\d+)\]")


def parse_time(text: str):
    m = TIME_RE.search(text)
    if not m:
        return None
    mon, day, hh, mm, ss, year = m.groups()
    import datetime as _dt
    return _dt.datetime(int(year), MONTHS[mon], int(day), int(hh), int(mm), int(ss))


def parse_one(path: str):
    try:
        txt = open(path, "r", errors="replace").read()
    except OSError:
        return None
    job = JOB_RE.search(txt)
    started = parse_time(txt.split("Started at", 1)[1]) if "Started at" in txt else None
    term = parse_time(txt.split("Terminated at", 1)[1]) if "Terminated at" in txt else None
    if not (started and term):
        return None
    queue = QUEUE_RE.search(txt)
    host = HOST_RE.search(txt)
    # LSF 的 .out 不回写 bsub 的资源选项,所以按"每个作业 1 张卡"计(本项目从未多卡训练);
    # 若以后有多卡作业,提交时可显式写进命令行,这里能解析到就覆盖。
    g = GPU_RE.search(txt) or GPU_R_RE.search(txt)
    ngpu = int(g.group(1)) if g else 1
    cpu = 0.0
    mc = re.search(r"CPU time\s*:\s*([\d.]+)\s*sec", txt)
    if mc:
        cpu = float(mc.group(1))
    return dict(
        file=os.path.basename(path),
        jobid=int(job.group(1)) if job else 0,
        name=(job.group(2) if job else os.path.basename(path)),
        queue=(queue.group(1) if queue else "?"),
        hosts=(host.group(1) if host else "?"),
        started=started, terminated=term,
        wall_h=(term - started).total_seconds() / 3600.0,
        ngpu=ngpu, cpu_h=cpu / 3600.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--since", default=None, help="只统计 Started >= 该日期 (YYYY-MM-DD)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.logdir, "*.out"))):
        r = parse_one(p)
        if r:
            rows.append(r)
    if args.since:
        import datetime as _dt
        cut = _dt.datetime.fromisoformat(args.since)
        rows = [r for r in rows if r["started"] >= cut]
    rows.sort(key=lambda r: r["started"])

    lines = []
    by_group = defaultdict(lambda: [0.0, 0])
    for r in rows:
        g = by_group[prefix_group(r["name"])]
        g[0] += r["wall_h"]
        g[1] += 1
    lines.append("")
    lines.append("按阶段分组(墙钟小时 ≈ GPU 卡时):")
    for tag, (h, n) in sorted(by_group.items(), key=lambda kv: -kv[1][0]):
        lines.append("  %-14s %2d 个作业  %7.2f h" % (tag, n, h))

    lines.append("作业名                          队列          主机          起-止                       墙钟h  GPU  GPU·h    CPU·h")
    for r in rows:
        lines.append("%-30s %-13s %-13s %s → %s %6.2f %4d %6.2f %8.1f" % (
            r["name"][:30], r["queue"][:13], r["hosts"][:13],
            r["started"].strftime("%m-%d %H:%M"), r["terminated"].strftime("%m-%d %H:%M"),
            r["wall_h"], r["ngpu"], r["wall_h"] * r["ngpu"], r["cpu_h"]))

    tot_wall = sum(r["wall_h"] for r in rows)
    tot_gpu = sum(r["wall_h"] * r["ngpu"] for r in rows)
    tot_cpu = sum(r["cpu_h"] for r in rows)

    by_name = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        b = by_name[r["name"]]
        b[0] += r["wall_h"] * r["ngpu"]
        b[1] += r["cpu_h"]
        b[2] += 1

    def prefix_group(name: str) -> str:
        for pre, tag in (("p1", "阶段0/1"), ("p2", "阶段2"), ("p3", "阶段3"),
                         ("sz_", "深圳(8–9月)"), ("abl_", "消融(7–8月)"),
                         ("sz", "深圳(8–9月)")):
            if name.startswith(pre):
                return tag
        return "工具/杂项"

    lines.append("")
    lines.append("按作业名汇总:")
    for name, (g, c, n) in sorted(by_name.items(), key=lambda kv: -kv[1][0]):
        lines.append("  %-30s 作业数 %2d  GPU·h %7.2f  CPU·h %8.1f" % (name[:30], n, g, c))

    lines.append("")
    lines.append("合计:%d 个跑完的作业,墙钟 %.2f h,GPU·h %.2f,CPU·h %.1f" % (
        len(rows), tot_wall, tot_gpu, tot_cpu))
    if rows:
        lines.append("时间范围:%s → %s" % (
            rows[0]["started"].strftime("%F %T"), rows[-1]["terminated"].strftime("%F %T")))

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
