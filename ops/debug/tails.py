#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tails.py — 一眼看四个训练日志的真实进度(处理 tqdm 的 \\r 刷新)。

为什么不用 tail:训练日志里 tqdm 用回车原地刷新,`tail -n 5` 只会拿到一行很长的字符串,
并且 screen 的 stuff 通道对含转义字符的命令极不友好。这个脚本干三件事:
  1) 打印当前时间(时钟对齐,便于判断"是不是卡了");
  2) 每个日志取最后一条含 Epoch 的进度 + 最后一条 Train end/Best epoch;
  3) 给出"距上次日志更新多少分钟",超过 THRESH 分钟标 ⚠️(可能被抢占或在等 IO)。

用法: python3 ops/debug/tails.py [--dir logs] [--tags a,b,c] [--thresh 15]
"""
import argparse
import os
import re
import time

DEFAULT_TAGS = ["sq27to9_reg", "sq9to3_reg", "sq3to1_reg", "p4_27to1_reg"]


def last_lines(path, limit=4000):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - limit))
        blob = f.read().decode("utf-8", "replace")
    return [s for s in blob.replace("\r", "\n").split("\n") if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs")
    ap.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    ap.add_argument("--thresh", type=int, default=15, help="多少分钟没更新就标警")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    now = time.time()
    print("现在 {}  (epoch 时间 = 本机 {})".format(
        time.strftime("%Y-%m-%d %H:%M:%S"), time.strftime("%Z")))
    for tag in tags:
        path = os.path.join(args.dir, "{}.log".format(tag))
        if not os.path.exists(path):
            print("  {:<16s} 无日志 {}".format(tag, path))
            continue
        lines = last_lines(path)
        prog = [s for s in lines if "Epoch" in s and ("it]" in s or "s/it" in s)]
        done = [s for s in lines if ("Train end" in s or "Best epoch" in s)]
        age = (now - os.path.getmtime(path)) / 60.0
        flag = " ⚠️ 日志 {} 分钟没更新".format(int(age)) if age > args.thresh else ""
        print("  {:<16s} 更新于 {:.0f} 分钟前{}".format(tag, age, flag))
        if prog:
            print("      进度: {}".format(re.sub(r"\s+", " ", prog[-1])[-110:]))
        if done:
            for d in done[-2:]:
                print("      {}".format(d.strip()[:110]))


if __name__ == "__main__":
    main()
