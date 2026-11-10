#!/usr/bin/env python3
"""把配置里的旧集群绝对路径改写成本机的新根目录。

背景:2026-11 从 NJU HPC(LSF)搬到组内服务器 multiscale(Slurm)后,configs/ 下所有
yml 里的 data.* 路径都还是旧集群的 `/fsb/home/yutingwang/ytw_tangzq/...`,直接跑会在
`split_paths_by_manifest` 里报 FileNotFoundError。这里只做**前缀替换**,不动其它键。

用法:
    python ops/slurm/fix_config_paths.py --root /data/users/tangziqing/sbr [--write]

默认只打印将改动的行(--dry),加 --write 才落盘。
"""
import argparse
import pathlib
import re

OLD = "/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="新机器上仓库的绝对路径")
    ap.add_argument("--glob", default="**/*.yml")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    hits = 0
    files = 0
    old_b = OLD.encode()
    new_b = str(root).encode()
    for p in sorted(pathlib.Path("configs").rglob("*.yml")):
        # 按字节处理:部分老配置不是 UTF-8(含 GBK 注释),read_text() 会 UnicodeDecodeError
        raw = p.read_bytes()
        if old_b not in raw:
            continue
        files += 1
        n = raw.count(old_b)
        hits += n
        if not args.write:
            print("{}: {} 处旧路径".format(p, n))
        else:
            p.write_bytes(raw.replace(old_b, new_b))
    print("[fix-config-paths] {} 个文件、{} 行,{}".format(
        files, hits, "已改写" if args.write else "未改写(加 --write 落盘)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
