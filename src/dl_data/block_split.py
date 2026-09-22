# -*- coding: utf-8 -*-
"""阶段0(导师大纲)T0.6:按 split.json 的时间戳清单做块级划分(替代 shuffle=False 尾部切分)。"""
import json
import os
import re

_STAMP = re.compile(r'_(\d{8}T\d{2})(\d{2})(\d{2})\.npz$')


def _hour_key(basename):
    m = _STAMP.search(os.path.basename(basename))
    if not m:
        return None
    return '{}-{}-{}T{}'.format(m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:8], m.group(2))


def split_paths_by_manifest(paths, manifest_path):
    """split.json 的 hours.{train,valid,test} -> {kind: [paths]}。"""
    with open(manifest_path) as f:
        manifest = json.load(f)
    hour_sets = {k: set(h[:13] for h in v) for k, v in manifest['hours'].items()}
    out = {'train': [], 'valid': [], 'test': []}
    unmatched = []
    for p in paths:
        hk = _hour_key(p)
        placed = False
        for kind, hs in hour_sets.items():
            if hk in hs:
                out[kind].append(p)
                placed = True
                break
        if not placed:
            unmatched.append(p)
    return out, unmatched
