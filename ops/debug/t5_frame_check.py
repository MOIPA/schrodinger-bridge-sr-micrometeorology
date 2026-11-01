#!/usr/bin/env python3
"""阶段5 前置核查:细端 10 分钟帧是否连续、形状是否一致、每小时是否正好 6 帧。

只读 prepare_npz_outline_fine,不占 GPU。用法:
  python ops/debug/t5_frame_check.py --dir prepare_npz_outline_fine --scheme myj --days 2
"""
import argparse
import datetime as dt
import os
import re
import sys

import numpy as np

PAT = re.compile(r'f_(\w+?)_(\d{8})T(\d{6})\.npz$')


def parse(name):
    m = PAT.match(name)
    if not m:
        return None
    scheme, d, t = m.groups()
    return scheme, dt.datetime.strptime(d + t, '%Y%m%d%H%M%S')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='prepare_npz_outline_fine')
    ap.add_argument('--scheme', default='myj')
    ap.add_argument('--days', type=int, default=0, help='只检查前 N 天;0=全部')
    ap.add_argument('--shapes', type=int, default=3, help='抽查几个文件的形状')
    args = ap.parse_args()

    names = sorted(os.listdir(args.dir))
    frames = []
    for n in names:
        p = parse(n)
        if p and p[0] == args.scheme:
            frames.append((p[1], os.path.join(args.dir, n)))
    frames.sort()
    print('[t5] %s / %s: %d 帧' % (args.dir, args.scheme, len(frames)))
    if not frames:
        return 1

    # 1) 首尾与总跨度
    t0, t1 = frames[0][0], frames[-1][0]
    span_h = (t1 - t0).total_seconds() / 3600.0
    print('[t5] 时间范围 %s → %s(%.1f h)' % (t0, t1, span_h))
    print('[t5] 期望帧数 = 跨度/10min + 1 = %d,实际 %d %s'
          % (int(span_h * 6) + 1, len(frames),
             '✓' if abs(span_h * 6 + 1 - len(frames)) <= 1 else '✗ 有缺口'))

    # 2) 逐日:每小时是否正好 6 帧、间隔是否正好 600 s
    by_day = {}
    for t, path in frames:
        by_day.setdefault(t.date(), []).append((t, path))
    bad_gap = bad_count = 0
    worst = []
    days = sorted(by_day)[:args.days] if args.days > 0 else sorted(by_day)
    for d in days:
        items = by_day[d]
        hours = {}
        for t, path in items:
            hours.setdefault(t.hour, []).append(t)
        for h, ts in sorted(hours.items()):
            if len(ts) != 6:
                bad_count += 1
                worst.append('%s %02d 时 %d 帧' % (d, h, len(ts)))
        ts_all = [t for t, _ in items]
        for a, b in zip(ts_all, ts_all[1:]):
            if (b - a).total_seconds() != 600:
                bad_gap += 1
                if len(worst) < 8:
                    worst.append('%s→%s 间隔 %ds' % (a.strftime('%d %H:%M'),
                                                    b.strftime('%d %H:%M'),
                                                    (b - a).total_seconds()))
    print('[t5] 检查 %d 天:每小时非 6 帧的 %d 个,间隔异常 %d 处' % (len(days), bad_count, bad_gap))
    for w in worst[:8]:
        print('        %s' % w)

    # 3) 形状抽查 + 相邻帧差异(判断 10 分钟变率是否有量)
    step = max(1, len(frames) // max(args.shapes, 1))
    for t, path in frames[::step][:args.shapes]:
        z = np.load(path)
        keys = sorted(z.files)
        shapes = {k: z[k].shape for k in keys}
        print('[t5] %s %s %s' % (t, os.path.basename(path), shapes))
        # 和下一帧比,看 u10 的 10 分钟变化量级
        i = frames.index((t, path))
        if i + 1 < len(frames):
            z2 = np.load(frames[i + 1][1])
            d = float(np.sqrt(np.mean((z['f_u10'] - z2['f_u10']) ** 2)))
            print('        相邻帧 u10 差异 RMS = %.4f m/s' % d)
    return 0


if __name__ == '__main__':
    sys.exit(main())
