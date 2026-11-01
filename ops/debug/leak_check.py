#!/usr/bin/env python3
"""训练期泄漏的廉价体检:checkpoint 是不是"旧划分"时代的产物?

背景(2026-11-01 真实事故):外传三档统一划分后重训时,训练脚本发现
`data/DL_result/<Exp>/config_wind_canvas_<tag>/checkpoint.pth` 已存在,就自动 full resume。
旧权重是在旧划分上训的,而旧训练集包含新测试期的时次 —— 等于把测试期喂进训练。
事故靠"作业才跑 5 分钟、日志已经到 Epoch 46"发现,属于运气。

本脚本提供一个便宜的事前/事后检查,两条判据:

  A. 时间戳:checkpoint 的 mtime 早于该配置引用的 `split.json` 的 mtime
     → 说明划分改过而权重没重训,高度可疑(要人工确认,不是断言)。
  B. 训练期与测试期重叠:从 `logs/<tag>.log` 里读训练开始时间(文件 mtime 或 Train end 行),
     与 split.json 的 test 时间范围比 —— 只能做粗判,日志里没有时次列表,故不在这里做。

用法:
  python ops/debug/leak_check.py --tags p3_arch_reg,sq27to9_reg --cfg_root configs/深圳
  python ops/debug/leak_check.py --tags sq27to9_reg --cfg_root sidequest_scale3x/configs

输出:每个 tag 一行,标 [OK] / [可疑] / [缺],并给出两个 mtime。
退出码:有 [可疑] 时返回 1,便于在流水线里当闸门(默认 --soft 时总返回 0)。
"""
import argparse
import glob
import os
import sys
import time

EXP = 'ExperimentSchrodingerBridgeWindCanvas'


def find_cfg(cfg_root, tag):
    hits = glob.glob(os.path.join(cfg_root, '**', 'config_wind_canvas_%s.yml' % tag), recursive=True)
    return sorted(hits)[0] if hits else None


def yaml_get(path, key):
    """极简:只取 `key: value` 形式的单行值,不引 yaml 依赖。"""
    try:
        for ln in open(path, encoding='utf-8'):
            s = ln.strip()
            if s.startswith(key + ':'):
                return s.split(':', 1)[1].strip().strip('"\'')
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', required=True)
    ap.add_argument('--cfg_root', default='configs/深圳')
    ap.add_argument('--data_root', default='data/DL_result')
    ap.add_argument('--slack', type=float, default=600.0, help='容差秒数,默认 10 分钟')
    ap.add_argument('--soft', action='store_true', help='永远返回 0(只提示)')
    args = ap.parse_args()

    bad = 0
    for tag in [t for t in args.tags.split(',') if t]:
        cfg = find_cfg(args.cfg_root, tag)
        if cfg is None:
            print('[缺] %-24s 找不到配置(config_wind_canvas_%s.yml)' % (tag, tag))
            continue
        split = yaml_get(cfg, 'split_manifest')
        if not split:
            print('[缺] %-24s 配置里没有 split_manifest' % tag)
            continue
        ck = os.path.join(args.data_root, EXP, 'config_wind_canvas_%s' % tag, 'checkpoint.pth')
        if not os.path.isfile(split):
            print('[缺] %-24s split.json 不存在: %s' % (tag, split))
            continue
        if not os.path.isfile(ck):
            print('[缺] %-24s checkpoint 不存在(还没训过)' % tag)
            continue
        t_ck, t_sp = os.path.getmtime(ck), os.path.getmtime(split)
        ck_s, sp_s = time.strftime('%m-%d %H:%M', time.localtime(t_ck)), time.strftime('%m-%d %H:%M', time.localtime(t_sp))
        if t_ck + args.slack < t_sp:
            print('[可疑] %-22s ckpt %s 早于 split %s → 可能是在旧划分上训的旧权重' % (tag, ck_s, sp_s))
            bad += 1
        else:
            print('[OK]   %-22s ckpt %s ≥ split %s' % (tag, ck_s, sp_s))

    print('')
    print('说明:判据只是"时间先后",不是泄漏的证明;ckpt 时间早于划分时,'
          '要么重训,要么人工确认该 ckpt 的训练集与新划分不冲突。')
    if bad and not args.soft:
        print('结论:%d 个 tag 可疑。' % bad)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
