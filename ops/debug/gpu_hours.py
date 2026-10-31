#!/usr/bin/env python3
"""统计 LSF 作业实际占用的 GPU 小时。

数据来源:LSF 在作业结束时写到 `logs/<jobname>_<jobid>.out` 的邮件正文,里面有
  Started at <time> / Terminated at <time> / CPU time : <sec> / 队列 / 主机

口径(2026-10-31 定):
- 墙钟小时 = Terminated − Started,即该作业真正占着节点的时间;
- 本项目所有训练/评估作业都是单卡(-gpu num=1:mode=exclusive_process),所以
  **GPU 卡时 ≈ 墙钟小时 × 1**;LSF 的 .out 不回写 bsub 的资源选项,若命令行里能
  解析到 `-gpu num=N` 或 `rusage[ngpus_physical=N]`,按解析值覆盖;
- 只统计同时有 Started 和 Terminated 的作业(排除从未真正跑起来的);
- 同时给出 CPU 小时(所有核加起来),用来看是否被 CPU 侧拖住。

用法:
    python ops/debug/gpu_hours.py --logdir logs --out ops/result/gpu_hours.txt
    python ops/debug/gpu_hours.py --logdir logs --since 2026-10-01
"""
from __future__ import annotations

import argparse
import datetime as dt
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
CPU_RE = re.compile(r"CPU time\s*:\s*([\d.]+)\s*sec")

GROUPS = (("p1", "阶段0/1"), ("p2", "阶段2"), ("p3", "阶段3"),
          ("sz", "深圳(8-9月)"), ("abl", "消融(7-8月)"))


def parse_time(text: str):
    m = TIME_RE.search(text)
    if not m:
        return None
    mon, day, hh, mm, ss, year = m.groups()
    return dt.datetime(int(year), MONTHS[mon], int(day), int(hh), int(mm), int(ss))


def parse_one(path: str):
    try:
        txt = open(path, "r", errors="replace").read()
    except OSError:
        return None
    started = parse_time(txt.split("Started at", 1)[1]) if "Started at" in txt else None
    term = parse_time(txt.split("Terminated at", 1)[1]) if "Terminated at" in txt else None
    if not (started and term):
        return None
    job = JOB_RE.search(txt)
    queue = QUEUE_RE.search(txt)
    host = HOST_RE.search(txt)
    g = GPU_RE.search(txt) or GPU_R_RE.search(txt)
    ngpu = int(g.group(1)) if g else 1
    cpu = CPU_RE.search(txt)
    return dict(
        name=(job.group(2) if job else os.path.basename(path)),
        jobid=int(job.group(1)) if job else 0,
        queue=(queue.group(1) if queue else "?"),
        hosts=(host.group(1) if host else "?"),
        started=started, terminated=term,
        wall_h=(term - started).total_seconds() / 3600.0,
        ngpu=ngpu,
        cpu_h=float(cpu.group(1)) / 3600.0 if cpu else 0.0,
    )


def group_of(name: str) -> str:
    for pre, tag in GROUPS:
        if name.startswith(pre):
            return tag
    return "工具/杂项"


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
        cut = dt.datetime.fromisoformat(args.since)
        rows = [r for r in rows if r["started"] >= cut]
    rows.sort(key=lambda r: r["started"])

    tot_wall = sum(r["wall_h"] for r in rows)
    tot_gpu = sum(r["wall_h"] * r["ngpu"] for r in rows)
    tot_cpu = sum(r["cpu_h"] for r in rows)

    by_group = defaultdict(lambda: [0.0, 0])
    by_name = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        g = by_group[group_of(r["name"])]
        g[0] += r["wall_h"] * r["ngpu"]
        g[1] += 1
        b = by_name[r["name"]]
        b[0] += r["wall_h"] * r["ngpu"]
        b[1] += r["cpu_h"]
        b[2] += 1

    out = []
    out.append("按阶段分组(墙钟 x 卡数 = GPU 卡时):")
    for tag, (h, n) in sorted(by_group.items(), key=lambda kv: -kv[1][0]):
        out.append("  %-14s %3d 个作业  %8.2f h" % (tag, n, h))
    out.append("")
    out.append("合计:%d 个跑完的作业,墙钟 %.2f h,GPU 卡时 %.2f,CPU 核时 %.1f" % (
        len(rows), tot_wall, tot_gpu, tot_cpu))
    if rows:
        out.append("时间范围:%s -> %s" % (
            rows[0]["started"].strftime("%F %T"), rows[-1]["terminated"].strftime("%F %T")))

    out.append("")
    out.append("明细(按开始时间):")
    out.append("%-30s %-12s %-12s %-13s %7s %4s %7s %8s" % (
        "作业名", "队列", "主机", "起-止", "墙钟h", "卡", "GPU·h", "CPU·h"))
    for r in rows:
        out.append("%-30s %-12s %-12s %s->%s %7.2f %4d %7.2f %8.1f" % (
            r["name"][:30], r["queue"][:12], r["hosts"][:12],
            r["started"].strftime("%m-%d %H:%M"), r["terminated"].strftime("%m-%d %H:%M"),
            r["wall_h"], r["ngpu"], r["wall_h"] * r["ngpu"], r["cpu_h"]))

    out.append("")
    out.append("按作业名汇总(前 20):")
    for name, (g, c, n) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:20]:
        out.append("  %-30s 作业数 %2d  GPU·h %7.2f  CPU·h %8.1f" % (name[:30], n, g, c))

    text = "\n".join(out)
    print(text)
    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
