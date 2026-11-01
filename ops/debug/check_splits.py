# -*- coding: utf-8 -*-
"""比较若干 static 目录的 split.json 是否一致(时次划分必须同一套)。

split.json 结构(30_blocks_and_split.py 产出)里有 'hours' 与 'blocks' 两套描述,
本脚本把两者都归一化成"每个 split 的时次集合"再比。
"""
import argparse
import json
import os


def norm_hours(j):
    """从 split.json 里抽出 {split: set(时次字符串)}。"""
    out = {}
    h = j.get("hours")
    if isinstance(h, dict):
        for k, v in h.items():
            if isinstance(v, list) and v and isinstance(v[0], str):
                out[k] = set(v)
            elif isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                out[k] = set(x[0] for x in v)
    if not out and isinstance(j.get("blocks"), list):
        for b in j["blocks"]:
            k = b.get("split") or b.get("kind") or b.get("name")
            hs = b.get("hours") or []
            if k and hs:
                out.setdefault(k, set()).update(
                    x if isinstance(x, str) else x[0] for x in hs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    sp = {}
    for d in a.dirs:
        with open(os.path.join(d, "split.json")) as f:
            sp[d] = norm_hours(json.load(f))
    for d in a.dirs:
        s = sp[d]
        txt = " ".join("{}={}".format(k, len(v)) for k, v in sorted(s.items()))
        rng = ""
        if s.get("test"):
            rng = " test:[{}..{}]".format(min(s["test"])[:13], max(s["test"])[:13])
        print("[{}] {}{}".format(os.path.basename(d.rstrip("/")), txt, rng))
    ref = a.dirs[0]
    ok = True
    for d in a.dirs[1:]:
        for k in ("train", "valid", "test"):
            if sp[d].get(k) != sp[ref].get(k):
                ok = False
                inter = len(sp[d].get(k, set()) & sp[ref].get(k, set()))
                dif = sorted(sp[d].get(k, set()) ^ sp[ref].get(k, set()))[:3]
                print("[差异] {} {}: 交集 {} / {} vs {};例 {}".format(
                    os.path.basename(d.rstrip("/")), k, inter,
                    len(sp[d].get(k, ())), len(sp[ref].get(k, ())), dif))
    print("[结论]", "split 完全一致" if ok else "split 不一致(评估不可直接对比)")


if __name__ == "__main__":
    main()
