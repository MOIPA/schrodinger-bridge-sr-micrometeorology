# -*- coding: utf-8 -*-
"""阶段 5 · 时间降尺度:构造 (t, t+1h, alpha) 三元组索引。

为什么单独一个脚本:数据集只负责"读帧 + 拼接通道",而"哪两帧配成一对、
中间要插哪几个 10 分钟时刻"是纯文件名/时间逻辑,不需要 torch、不需要
数据内容,可以在登录节点甚至笔记本上跑通并验证。

配对规则(与 docs/阶段5_时间降尺度_设计方案.md 一致):
- 只用整点帧(stamp 的分钟位 = 00),因为两端点必须是同一小时的两个整点;
- 一对 = (T, T+1h),中间 5 个待预测时刻 alpha = k/6, k = 1..5;
- 一对贡献 5 个样本(不是 1 个);
- 划分沿用阶段 3/4 的 split.json:一对属于哪个子集,由**起点 T 所在小时**决定
  (若 T 在 test 而 T+1 在 valid,按 T 计;跨子集的边界对单独计数并打印)。

用法:
    python scripts/outline/70_time_pair_index.py \
        --fine_dir prepare_npz_outline_fine \
        --split_json prepare_npz_outline_static/split.json \
        --out_json results/phase5/time_pairs.json

输出 json:{"alphas":[1/6..5/6], "splits":{"train":[{"t0":..,"t1":..,"alpha":..}], ...}, "stats":{...}}
"""
import argparse
import json
import os
import re

STAMP_RE = re.compile(r"_(\d{8}T\d{6})\.npz$")


def parse_stamp(path):
    m = STAMP_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def hour_key(stamp):
    """'YYYYMMDDTHHMMSS' -> 'YYYYMMDDTHH'"""
    return stamp[:11]


def collect_hours(fine_dir, scheme="myj"):
    """返回 {(yyyy,mm,dd,HH): path},只保留整点帧。"""
    tag = "_" + scheme + "_"
    hours = {}
    dup = 0
    for name in sorted(os.listdir(fine_dir)):
        if not name.endswith(".npz") or tag not in name:
            continue
        st = parse_stamp(name)
        if st is None or st[11:13] != "00":
            continue
        k = "{}T{}".format(st[:8], st[9:11])   # 与 split.json 同格式 YYYYMMDDTHH
        if k in hours:
            dup += 1
            continue
        hours[k] = os.path.join(fine_dir, name)
    return hours, dup


def load_split_hours(split_json):
    """把 split.json 归一化成 {'train': set('YYYYMMDDTHH'), ...}。

    真实格式见 scripts/outline/30_blocks_and_split.py:生成的 json 里 `hours`
    是 {'train': ['YYYY-MM-DDTHH:MM:SS', ...], ...};老/手写版本可能只有
    `blocks`(每天一条,带 'split' 字段),此时按天展开成逐小时兜底。
    """
    with open(split_json) as f:
        d = json.load(f)

    def norm(stamp):
        s = str(stamp).replace("-", "").replace(":", "")
        if len(s) == 10:                 # YYYYMMDDHH
            return s[:8] + "T" + s[8:10]
        return s[:8] + "T" + s[9:11]     # YYYYMMDDTHHMMSS

    out = {}
    if isinstance(d.get("hours"), dict):
        for k, v in d["hours"].items():
            s = out.setdefault(k, set())
            if isinstance(v, (list, tuple)):
                s.update(norm(x) for x in v)
            else:
                s.add(norm(v))
    if not out and isinstance(d.get("blocks"), list):
        for b in d["blocks"]:
            s = out.setdefault(b.get("split", "train"), set())
            s.update(enumerate_hours(b["start"], b["end"]))
    if not out:
        raise SystemExit("split.json 结构无法识别:{}".format(list(d.keys())[:5]))
    return out


def enumerate_hours(start_stamp, end_stamp):
    """从 start 到 end(含),先按 split.json 里出现的两种时间格式各试一次。"""
    import datetime as dt

    def parse(s):
        s = s.replace("-", "").replace(":", "")
        if len(s) == 8:             # YYYYMMDD(块按天给)
            return dt.datetime.strptime(s, "%Y%m%d")
        if len(s) == 10:            # YYYYMMDDHH
            return dt.datetime.strptime(s, "%Y%m%d%H")
        return dt.datetime.strptime(s[:13], "%Y%m%dT%H")

    a, b = parse(start_stamp), parse(end_stamp)
    if len(str(end_stamp).replace("-", "")) == 8:   # 端点只给到天 -> 含当天 23 时
        b = b + dt.timedelta(hours=23)
    out = []
    while a <= b:
        out.append(a.strftime("%Y%m%dT%H"))
        a += dt.timedelta(hours=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine_dir", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--scheme", default="myj")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n_alpha", type=int, default=5)
    args = ap.parse_args()

    hours, dup = collect_hours(args.fine_dir, args.scheme)
    splits = load_split_hours(args.split_json)
    alphas = [k / (args.n_alpha + 1) for k in range(1, args.n_alpha + 1)]

    # 小时 -> 子集(一个小时只应属于一个子集;重叠/未覆盖都要报出来)
    hour2split = {}
    overlap = 0
    for name, hs in splits.items():
        for h in hs:
            if h in hour2split and hour2split[h] != name:
                overlap += 1
            hour2split[h] = name

    out = {"alphas": alphas,
           "n_alpha": args.n_alpha,
           "splits": {k: [] for k in splits},
           "stats": {}}
    missing_endpoint = 0
    cross_split = 0
    uncovered = 0
    import datetime as dt
    for h in sorted(hours):
        h2 = (dt.datetime.strptime(h, "%Y%m%dT%H") + dt.timedelta(hours=1)).strftime("%Y%m%dT%H")
        if h2 not in hours:
            missing_endpoint += 1
            continue
        sp = hour2split.get(h)
        if sp is None:
            uncovered += 1
            continue
        if hour2split.get(h2) not in (None, sp):
            cross_split += 1
        for a in alphas:
            out["splits"].setdefault(sp, []).append(
                {"t0": hours[h], "t1": hours[h2], "alpha": round(a, 6)})

    out["stats"] = {
        "n_hour_frames": len(hours),
        "duplicate_frames": dup,
        "missing_next_hour": missing_endpoint,
        "hours_not_in_split": uncovered,
        "cross_split_pairs": cross_split,
        "split_hour_overlap": overlap,
        "n_samples": {k: len(v) for k, v in out["splits"].items()},
        "n_pairs": {k: len(v) // max(len(alphas), 1) for k, v in out["splits"].items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    s = out["stats"]
    print("[ok] 整点帧 {} 个 -> 样本 {} (对 {}),写入 {}".format(
        s["n_hour_frames"], s["n_samples"], s["n_pairs"], args.out_json))
    for k in ("duplicate_frames", "missing_next_hour", "hours_not_in_split",
              "cross_split_pairs", "split_hour_overlap"):
        if s[k]:
            print("[注意] {} = {}".format(k, s[k]))


if __name__ == "__main__":
    main()
