# -*- coding: utf-8 -*-
"""阶段 3 配置生成:AGL 空间监督三变体,基座 = p2_l1r2_lr2e4(保证与基线的唯一差异)。

变体(tag 供 ops/63 使用,必须与文件名一致):
  p3_agl    AGL 项替换数据项:si.agl_weight=1.0、agl_replace_data=true、agl_w_weight=0.5;
            data.return_agl_tables=true
  p3_joint  双空间联合监督:si.agl_weight=lambda_joint(88 探针:data/agl 量级比)、
            agl_replace_data=false、agl_w_weight=0.5;data.return_agl_tables=true
  p3_agllw  低层等效通道加权(消融,无 AGL 字段):si.channel_weights=channel_weights_72
            (88 探针,72 floats);data.return_agl_tables=false

基座逐键照抄(yaml.safe_load),只动上面列出的键:seed/lr/save_interval/输入组等原样。
补充键(AGL 监督运行必需,写盘时在改动清单里逐条明示):
  p3_agl / p3_joint 的 si.phys_scale / phys_dz / phys_div_tau(基座没有该三键):
  - phys_scale:_canvas_agl_raw 反标准化必需,训练就会用到;
  - phys_dz / phys_div_tau:训练(return_parts=False 且物理权重 0)不触发,但 63 设置波
    的冒烟与任何 return_parts=True 的调用会要求全套(与 96/88 同款口径),故一并写入;
  - 来源依次为 88 探针 meta(phys_scale)、基座、config_wind_canvas_p2_div_mid.yml
    (与 60d 的物理参数来源一致,三者同值,已入库)。

运行(仓库根目录,只需 pyyaml;不读服务器数据):
  python scripts/outline/87_gen_phase3_configs.py --dry_run        # 只打印改动键清单
  python scripts/outline/87_gen_phase3_configs.py                  # 写 configs/深圳/phase3
  python scripts/outline/87_gen_phase3_configs.py --probe_json results/phase3/probe.json
"""
import argparse
import copy
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "configs", "深圳", "phase3")
BASE_CONFIG = os.path.join(ROOT, "configs", "深圳", "phase2",
                           "config_wind_canvas_p2_l1r2_lr2e4.yml")
PHYS_FALLBACK = os.path.join(ROOT, "configs", "深圳", "phase2",
                             "config_wind_canvas_p2_div_mid.yml")
PROBE_DEFAULT = os.path.join(ROOT, "results", "phase3", "probe.json")


def disp(path):
    """仓库内路径显示为相对路径,仓库外(如 /tmp 测试)显示绝对路径。"""
    rel = os.path.relpath(path, ROOT)
    return rel if not rel.startswith(os.pardir) else path


def brief(v):
    """改动清单里的紧凑值表示(避免把 72 个浮点整段打印)。"""
    if isinstance(v, list):
        head = ",".join("{:g}".format(x) if isinstance(x, (int, float)) else str(x)
                        for x in v[:2])
        if not v:
            return "list(0)"
        return "list({})[{}{}]".format(len(v), head, ",..." if len(v) > 2 else "")
    return repr(v)


def flat(d, prefix=""):
    """嵌套 dict -> {'si.agl_weight': ...} 形式;列表值视为叶子。"""
    out = {}
    for k, v in d.items():
        key = "{}.{}".format(prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


def diff_keys(base, new):
    """逐键对比,返回 '键 旧 -> 新' 清单(含新增/删除)。"""
    fb, fn = flat(base), flat(new)
    lines = []
    for k in sorted(set(fb) | set(fn)):
        if k not in fb:
            lines.append("{:<24} (新增) {}".format(k, brief(fn[k])))
        elif k not in fn:
            lines.append("{:<24} (删除) {}".format(k, brief(fb[k])))
        elif fb[k] != fn[k]:
            lines.append("{:<24} {} -> {}".format(k, brief(fb[k]), brief(fn[k])))
    return lines


def load_phys_params(probe, base_cfg):
    """AGL 监督所需的物理参数:探针 meta(phys_scale)-> 基座 -> p2_div_mid.yml。

    phys_scale 长度 = out_channel;phys_dz / phys_div_tau 长度 = len(target_levels)。
    返回 (dict, 来源说明);找不到时直接报错(不静默用默认值)。
    """
    n_out = base_cfg.get("model", {}).get("out_channel")
    n_lev = len((base_cfg.get("data") or {}).get("target_levels") or [])
    meta = probe.get("meta") or {}
    si_base = base_cfg.get("si") or {}
    fb_si = {}
    if os.path.isfile(PHYS_FALLBACK):
        with open(PHYS_FALLBACK) as f:
            fb_si = yaml.safe_load(f).get("si") or {}
    want = {"phys_scale": n_out, "phys_dz": n_lev, "phys_div_tau": n_lev}
    got, srcs = {}, []
    for k, n in want.items():
        for src, v in (("probe.meta.{}".format(k), meta.get(k)),
                       ("基座 si.{}".format(k), si_base.get(k)),
                       ("{} si.{}".format(os.path.basename(PHYS_FALLBACK), k),
                        fb_si.get(k))):
            if isinstance(v, list) and len(v) == n:
                got[k] = [float(x) for x in v]
                srcs.append("{}<-{}".format(k, src))
                break
    missing = [k for k in want if k not in got]
    if missing:
        sys.exit("错误: 找不到长度合规的 {}。\n"
                 "      先跑 88_phase3_probe.py 写出 probe.json(meta.phys_scale),\n"
                 "      或确认 {} 存在。".format("、".join(missing), PHYS_FALLBACK))
    return got, ", ".join(srcs)


def check_probe(probe, n_out):
    """探针 json 必需键校验(缺失/量纲不对时直接报错,不静默用默认值)。

    注意:channel_weights_72 允许为 0——AGL 算子只覆盖到 1000 m,更高的模式层
    (含 1 km 以上背景层)诱导权重本来就是 0(纯 AGL 臂同样不监督这些层)。
    """
    need = []
    for k in ("lambda_joint", "channel_weights_72"):
        if probe.get(k) is None:
            need.append(k)
    if need:
        sys.exit("错误: 探针 json 缺 {}".format("、".join(need)))
    lam = float(probe["lambda_joint"])
    if not (lam > 0):
        sys.exit("错误: lambda_joint={} 非正".format(lam))
    cw = [float(x) for x in probe["channel_weights_72"]]
    if len(cw) != n_out:
        sys.exit("错误: channel_weights_72 长度 {} != out_channel {}".format(len(cw), n_out))
    if min(cw) < 0:
        sys.exit("错误: channel_weights_72 存在负值")
    if sum(cw) <= 0:
        sys.exit("错误: channel_weights_72 全零")
    return lam, cw


def variants(probe, phys):
    """(tag, si/data 覆盖)——与 docs 阶段 3 T3.5 计划一致;lambda/权重全部来自探针。

    phys: dict(phys_scale/phys_dz/phys_div_tau),AGL 监督运行必需(见模块 docstring)。
    """
    lam, cw = probe["lambda_joint"], probe["channel_weights_72"]
    agl_common = {"agl_w_weight": 0.5, "phys_scale": list(phys["phys_scale"]),
                  "phys_dz": list(phys["phys_dz"]),
                  "phys_div_tau": list(phys["phys_div_tau"])}
    return [
        ("p3_agl", {
            "si": dict(agl_common, agl_weight=1.0, agl_replace_data=True),
            "data": {"return_agl_tables": True}}),
        ("p3_joint", {
            "si": dict(agl_common, agl_weight=float(lam), agl_replace_data=False),
            "data": {"return_agl_tables": True}}),
        ("p3_agllw", {   # 无 AGL 字段:只做低层等效通道加权
            "si": {"channel_weights": [float(x) for x in cw]},
            "data": {"return_agl_tables": False}}),
    ]


def yml_header(tag, over, base_rel, probe_rel, phys_src):
    lines = [
        "# 阶段 3 配置 {}:基座 = {}".format(tag, base_rel),
        "# 生成:scripts/outline/87_gen_phase3_configs.py(探针值 {})".format(probe_rel),
        "# 与基线的差异(逐键,基座逐键照抄):",
    ]
    fb = flat(over)
    for k in sorted(fb):
        lines.append("#   {:<24} -> {}".format(k, brief(fb[k])))
    if any(k.startswith("si.phys_") for k in fb):
        lines.append("# si.phys_* 来源: {}(AGL 损失反标准化 + return_parts 诊断必需,基座无该三键)"
                     .format(phys_src))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="阶段 3 配置生成(AGL 监督三变体)")
    ap.add_argument("--probe_json", default=PROBE_DEFAULT,
                    help="88 探针输出 json(默认 results/phase3/probe.json;缺失报错)")
    ap.add_argument("--out_dir", default=OUT_DIR, help="输出目录(默认 configs/深圳/phase3)")
    ap.add_argument("--base_config", default=BASE_CONFIG,
                    help="基座 yml(默认 p2_l1r2_lr2e4,保证与基线唯一差异)")
    ap.add_argument("--dry_run", action="store_true", help="只打印改动键,不写文件")
    args = ap.parse_args()

    probe_path = args.probe_json if os.path.isabs(args.probe_json) \
        else os.path.join(ROOT, args.probe_json)
    if not os.path.isfile(probe_path):
        sys.exit("错误: 找不到探针 json {}\n"
                 "      先跑 88_phase3_probe.py 生成(服务器 torch 环境),或用 --probe_json 指定"
                 .format(probe_path))
    with open(probe_path) as f:
        probe = json.load(f)
    base_path = args.base_config if os.path.isabs(args.base_config) \
        else os.path.join(ROOT, args.base_config)
    with open(base_path) as f:
        base = yaml.safe_load(f)
    for sect in ("data", "si", "model", "train", "loader"):
        if sect not in base:
            sys.exit("错误: 基座 {} 缺 {} 段".format(base_path, sect))

    lam, cw = check_probe(probe, base["model"]["out_channel"])
    phys, phys_src = load_phys_params(probe, base)
    print("基座: {}".format(disp(base_path)))
    print("探针: {}  lambda_joint={:.6g}  channel_weights_72 均值={:.4g} 零权重通道={}".format(
        disp(probe_path), lam, sum(cw) / len(cw), sum(1 for x in cw if x == 0.0)))
    print("物理参数来源: {} (phys_scale {} / phys_dz {} / phys_div_tau {})".format(
        phys_src, len(phys["phys_scale"]), len(phys["phys_dz"]), len(phys["phys_div_tau"])))

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    if not args.dry_run and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    var_list = variants({"lambda_joint": lam, "channel_weights_72": cw}, phys)
    for tag, over in var_list:
        cfg = copy.deepcopy(base)
        for sect, kv in over.items():
            cfg.setdefault(sect, {}).update(kv)
        name = "config_wind_canvas_{}.yml".format(tag)
        path = os.path.join(out_dir, name)
        header = yml_header(tag, over, disp(base_path), disp(probe_path), phys_src)
        if not args.dry_run:
            with open(path, "w") as f:
                f.write(header)
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print("\n==== {} -> {}{} ====".format(tag, path, "(dry_run, 未写盘)" if args.dry_run else ""))
        for line in diff_keys(base, cfg):
            print("  " + line)
    print("\n{} {} 个配置 -> {}".format(
        "dry_run:" if args.dry_run else "生成", len(var_list), out_dir))


if __name__ == "__main__":
    main()
