# -*- coding: utf-8 -*-
"""比较若干 static 目录的 split.json 是否一致(时次划分必须同一套)。"""
import argparse
import json
import os


def load(d):
    p = os.path.join(d, "split.json")
    with open(p) as f:
        j = json.load(f)
    out = {}
    for k, v in j.items():
        if isinstance(v, list) and v and isinstance(v[0], str):
            out[k] = sorted(os.path.basename(x) for x in v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    splits = {d: load(d) for d in a.dirs}
    ref = a.dirs[0]
    for d in a.dirs:
        s = splits[d]
        print("[{}] ".format(d) + " ".join(
            "{}={}".format(k, len(v)) for k, v in sorted(s.items())))
    ok = True
    for d in a.dirs[1:]:
        for k in ("train", "valid", "test"):
            if splits[d].get(k) != splits[ref].get(k):
                ok = False
                same = len(set(splits[d].get(k, [])) & set(splits[ref].get(k, [])))
                print("[差异] {} {}: 交集 {} / {} vs {}".format(
                    d, k, same, len(splits[d].get(k, [])), len(splits[ref].get(k, []))))
    print("[结论]", "三套 split 完全一致" if ok else "split 不一致(评估不可直接对比)")


if __name__ == "__main__":
    main()
