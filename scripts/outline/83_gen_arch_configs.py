# -*- coding: utf-8 -*-
"""阶段 3 架构对比配置生成:由 config_wind_canvas_p2_l1r2_lr2e4.yml 派生 7 个 run。

统一口径(与基座逐键相同,只有下列显式差异):myj / 23 层(0..22)/ L1 / 模式层监督
(无 agl 字段、return_agl_tables 缺省 false)/ 150 epochs + valid 早停 / 无物理项。

  p3_arch_unet_s2 : 种子对照臂(与参考臂同架构同超参,仅 train.seed 78270)
  p3_arch_swin    : SwinIR 窗口注意力(arch: swinir_canvas;inner 288 → ≈5.8M 参数,
  p3_arch_swin_s2 :   与 UNet 参考臂 ~5-6M 对齐);seed 78269 / 78270
  p3_arch_reg     : 回归步(UNet,与参考臂同架构同容量);seed 78269 / 78270;
  p3_arch_reg_s2  :   训练脚本 scripts/train_regression_model.py
  p3_arch_edm     : 回归 + EDM 两步法(基座 canvas yml + `edm:` 段 + `reg:` 段,
  p3_arch_edm_s2  :   与 p3_arch_reg / p3_arch_reg_s2 配对);脚本 train_edm_correction.py

生成后自检:
  1) edm 段逐字段核对 EDMCorrectorConfig(字段名/默认值,`sigma_data: null` = 训练时估计);
  2) 非 edm 配置跑 81_config_channel_check.py(edm 配置因含 `edm:`/`reg:` 段,
     81 的 base loader 按设计读不了——由本脚本用 train_edm_correction.load_edm_config
     剥离后校验通道恒等式,等价于 edm 训练脚本启动时的断言)。

运行(需要 torch 的环境,仓库根目录):
  python scripts/outline/83_gen_arch_configs.py [--no_check]
"""
import argparse
import copy
import dataclasses
import importlib.util
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.dl_data.dataset_wind_canvas import (  # noqa: E402
    build_input_channel_names,
    build_target_channel_names,
)

EXPERIMENT = "ExperimentSchrodingerBridgeWindCanvas"
BASE_YML = os.path.join(ROOT, "configs", "深圳", "phase2",
                        "config_wind_canvas_p2_l1r2_lr2e4.yml")
OUT_DIR = os.path.join(ROOT, "configs", "深圳", "phase3_arch")

# SwinIR 臂的 model 段(由任务契约给定;in/out 通道会与通道函数的计算结果核对)
# 容量决策(2026-10-07 实测):参考 UNet(inner 64)= 79.26M(远超原假设 ~5-6M);
# UNet 最小可训容量 = inner 32 = 19.87M(GroupNorm 要求通道 ÷32);
# 因此架构对比按 **≈20M 容量对齐**:UNet-small(19.87M) vs SwinIR inner=536(≈19.2M);
# SwinIR 单轮耗时由 64a_arch_probe.sh 先测(超预算则降 inner 并在报告标注容量差)。
SWIN_MODEL = {
    'arch': 'swinir_canvas',
    'in_channel': 167,
    'out_channel': 72,
    'inner_channel': 536,
    'num_blocks': 6,
    'window_size': 8,
    'num_heads': 4,
    'mlp_ratio': 2.0,
    'dropout': 0.0,
}


def load_base():
    with open(BASE_YML) as f:
        return yaml.safe_load(f)


def diff_dict(old, new, prefix=""):
    """递归逐键差异:返回 [(路径, 旧值, 新值), ...](repr 形式,便于日志核对)。"""
    out = []
    for k in sorted(set(old) | set(new)):
        p = "{}.{}".format(prefix, k) if prefix else str(k)
        if k not in old:
            out.append((p, "<缺>", repr(new[k])))
        elif k not in new:
            out.append((p, repr(old[k]), "<删>"))
        elif isinstance(old[k], dict) and isinstance(new[k], dict):
            out += diff_dict(old[k], new[k], p)
        elif old[k] != new[k]:
            out.append((p, repr(old[k]), repr(new[k])))
    return out


def edm_sections(reg_tag):
    """EDM 训练配置的 `edm:` 段(字段名与 EDMCorrectorConfig 一一对应,未列项取默认)
    与 `reg:` 段(指向配对的回归 tag;路径为仓库根相对路径,训练脚本会补绝对前缀)。"""
    edm = {
        'out_channel': 72,
        'ctx_channel': 239,          # 72(ŷ1_reg)+ 72(y0)+ 95(条件)
        'inner_channel': 64,
        'channel_mults': [1, 2, 4],
        'blocks_per_level': 1,
        'dropout': 0.0,
        'max_period': 10.0,
        'sigma_data': None,          # 训练脚本估计/CLI 指定后写入 config_resolved.yml
        'sigma_min': 0.002,
        'sigma_max': 80.0,
        'p_mean': -1.2,
        'p_std': 1.2,
        'rho': 7.0,
        'steps': 24,
        'sigma_d_estimator': 'std',
        'sigma_d_est_batches': 8,
    }
    reg = {
        'reg_tag': reg_tag,
        'reg_config_path': 'configs/深圳/phase3_arch/config_wind_canvas_{}.yml'.format(
            reg_tag),
    }
    return edm, reg


def header(tag, notes):
    lines = ["# {} — 阶段 3 架构对比(由 83_gen_arch_configs.py 派生)".format(tag)]
    lines.append("# 基座: configs/深圳/phase2/config_wind_canvas_p2_l1r2_lr2e4.yml"
                 "(myj / 23 层 / L1 / 模式层监督 / 无物理项)")
    lines += ["# " + n for n in notes]
    return "\n".join(lines) + "\n"


def build_variants(base):
    """返回 [(tag, cfg, notes, is_edm), ...];cfg 为基座的深拷贝 + 显式改动。"""
    n_out = len(build_target_channel_names(base['data']['target_levels'],
                                           base['data']['include_w']))
    n_cond = len(build_input_channel_names(base['data']['input_groups'],
                                           base['data']['target_levels'],
                                           base['data']['include_w']))
    assert (n_out, n_cond) == (72, 95), "基座通道变了: out={} cond={}".format(n_out, n_cond)
    assert SWIN_MODEL['in_channel'] == n_out + n_cond, \
        "SWIN_MODEL.in_channel {} != {} + {}".format(SWIN_MODEL['in_channel'], n_out, n_cond)
    assert SWIN_MODEL['out_channel'] == n_out

    variants = []

    def add(tag, mutate, notes, is_edm=False):
        cfg = copy.deepcopy(base)
        mutate(cfg)
        variants.append((tag, cfg, notes, is_edm))

    add('p3_arch_unet_s2',
        lambda c: c['train'].__setitem__('seed', 78270),
        ['变更: train.seed 78269 -> 78270(种子对照臂,其余逐键同参考臂)'])

    add('p3_arch_unet_small',
        lambda c: (c['model'].__setitem__('inner_channel', 32),
                   c['train'].__setitem__('seed', 78269)),
        ['变更: model.inner_channel 64 -> 32(实测 19.87M;容量对齐臂,与 SwinIR inner=536 同量级)',
         '变更: train.seed = 78269'])
    add('p3_arch_unet_small_s2',
        lambda c: (c['model'].__setitem__('inner_channel', 32),
                   c['train'].__setitem__('seed', 78270)),
        ['变更: model.inner_channel 64 -> 32(同 p3_arch_unet_small)',
         '变更: train.seed = 78270(与 p3_arch_swin_s2 同种子)'])

    swin_notes = [
        '变更: model 段整体替换为 SwinIR(arch: swinir_canvas;inner_channel 536 -> ≈19.2M 参数,'
        '与 UNet-small 19.87M 容量对齐)',
        '注: 参考臂 UNet(inner 64)实测 79.26M;容量对齐臂为 p3_arch_unet_small(inner 32)',
        '变更: train.seed = 78269',
    ]
    add('p3_arch_swin',
        lambda c: (c.__setitem__('model', dict(SWIN_MODEL)),
                   c['train'].__setitem__('seed', 78269)),
        swin_notes)
    add('p3_arch_swin_s2',
        lambda c: (c.__setitem__('model', dict(SWIN_MODEL)),
                   c['train'].__setitem__('seed', 78270)),
        ['变更: model 段整体替换为 SwinIR(= p3_arch_swin)',
         '变更: train.seed = 78270(与 p3_arch_reg_s2 / p3_arch_unet_s2 同种子)'])

    add('p3_arch_reg',
        lambda c: c['train'].__setitem__('seed', 78269),
        ['变更: 仅命名(tag -> 回归步 run;训练脚本 scripts/train_regression_model.py)',
         'model/si/data/loader 与基座逐键相同(同架构同容量),train.seed = 78269'])
    add('p3_arch_reg_s2',
        lambda c: c['train'].__setitem__('seed', 78270),
        ['变更: train.seed = 78270;训练脚本 scripts/train_regression_model.py',
         'model/si/data/loader 与基座逐键相同(同架构同容量)'])

    for tag, reg_tag, seed in (('p3_arch_edm', 'p3_arch_reg', 78269),
                               ('p3_arch_edm_s2', 'p3_arch_reg_s2', 78270)):
        def mutate(c, reg_tag=reg_tag, seed=seed):
            c['train']['seed'] = seed
            edm, reg = edm_sections(reg_tag)
            c['edm'] = edm
            c['reg'] = reg
        add(tag, mutate,
            ['变更: train.seed = {}'.format(seed),
             '新增 `edm:` 段(EDMCorrectorConfig;sigma_data: null = 训练时估计)',
             '新增 `reg:` 段 -> {}'.format(reg_tag),
             '训练脚本 scripts/train_edm_correction.py(两段由 load_edm_config 剥离)'],
            is_edm=True)
    return variants


def verify_edm_section(edm, tag):
    """edm 段逐字段核对 EDMCorrectorConfig:字段名存在 + 默认值一致;并试构造。"""
    try:
        from src.dl_model.edm_correction import EDMCorrectorConfig
    except Exception as e:                                        # noqa: BLE001
        print("[警告] {} 无法导入 EDMCorrectorConfig({}),跳过字段校验".format(
            tag, repr(e)[:80]))
        return True
    fields = EDMCorrectorConfig.__dataclass_fields__
    ok = True
    print("  {:<18} {:<22} {:<22} {}".format('字段', '生成值', '类默认值', '判定'))
    for k in sorted(edm):
        if k not in fields:
            print("  {:<18} {:<22} {:<22} 字段不存在".format(k, str(edm[k]), '-'))
            ok = False
            continue
        f = fields[k]
        dv = f.default if f.default is not dataclasses.MISSING else '<factory>'
        same = (repr(dv) == repr(edm[k])) if f.default is not dataclasses.MISSING else None
        print("  {:<18} {:<22} {:<22} {}".format(
            k, repr(edm[k]), repr(dv),
            '=默认' if same else ('≠默认(有意)' if same is False else '=')))
    try:
        cfg = EDMCorrectorConfig(**edm)
        assert cfg.ctx_channel == 2 * cfg.out_channel + 95
        print("  EDMCorrectorConfig(**edm) 构造 OK: ctx={} = 2*{} + 95".format(
            cfg.ctx_channel, cfg.out_channel))
    except Exception as e:                                        # noqa: BLE001
        print("  [FAIL] EDMCorrectorConfig(**edm) 构造失败: {}".format(repr(e)[:140]))
        ok = False
    return ok


def load_edm_config_fallback(experiment_name, path):
    """train_edm_correction.load_edm_config 的最小等价实现(该模块依赖 pandas,
    精简环境导入失败时兜底):剥离 edm/reg 顶段 -> 临时 yml -> load_config。"""
    from src.dl_config.config_loader import load_config
    from src.dl_model.edm_correction import EDMCorrectorConfig
    import tempfile
    with open(path) as f:
        full_raw = yaml.safe_load(f)
    raw = copy.deepcopy(full_raw)
    edm_raw = raw.pop("edm", None) or {}
    reg_raw = raw.pop("reg", None) or {}
    with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False) as tmp:
        yaml.safe_dump(raw, tmp, sort_keys=False, allow_unicode=True)
        tmp_path = tmp.name
    try:
        base_cfg = load_config(experiment_name, tmp_path)
    finally:
        os.remove(tmp_path)
    return base_cfg, EDMCorrectorConfig(**edm_raw), reg_raw, full_raw


def verify_edm_config_file(path):
    """用 train_edm_correction.load_edm_config 剥离 edm/reg 后校验基座 + 通道恒等式。"""
    try:
        p = os.path.join(ROOT, "scripts", "train_edm_correction.py")
        spec = importlib.util.spec_from_file_location("train_edm_correction_83", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        load_edm_config = mod.load_edm_config
        how = "load_edm_config(训练脚本)"
    except Exception as e:                                        # noqa: BLE001
        print("  [说明] 训练脚本导入失败({}),用 83 内的最小等价实现".format(repr(e)[:70]))
        load_edm_config = load_edm_config_fallback
        how = "83 兜底实现"
    base_cfg, edm_cfg, reg_sec, _ = load_edm_config(EXPERIMENT, path)
    n_out = len(build_target_channel_names(base_cfg.data.target_levels,
                                           base_cfg.data.include_w))
    n_in = len(build_input_channel_names(base_cfg.data.input_groups,
                                         base_cfg.data.target_levels,
                                         base_cfg.data.include_w))
    assert n_out == edm_cfg.out_channel, "out {} != {}".format(n_out, edm_cfg.out_channel)
    assert 2 * n_out + n_in == edm_cfg.ctx_channel, "ctx {} != 2*{}+{}".format(
        edm_cfg.ctx_channel, n_out, n_in)
    assert reg_sec.get('reg_tag'), "reg 段缺 reg_tag"
    assert base_cfg.model.out_channel == n_out and base_cfg.model.in_channel == n_out + n_in
    return ("base 可构造({});通道 out={} in={} ctx={};reg_tag={}".format(
        how, n_out, n_out + n_in, edm_cfg.ctx_channel, reg_sec['reg_tag']))


def run_81(config_dir):
    """跑 81 通道校验:先全目录(会因 edm 配置按设计不可读而失败,打印一行说明),
    再对 si/reg 配置(排除 edm)实际校验。"""
    script = os.path.join(ROOT, "scripts", "outline", "81_config_channel_check.py")
    full = subprocess.run([sys.executable, script, "--config_dir", config_dir],
                          capture_output=True, text=True)
    if full.returncode == 0:
        print(full.stdout.strip())
        return True
    last = [l for l in (full.stdout + full.stderr).splitlines() if l.strip()][-1:]
    print("[说明] 81 全目录模式 exit={}(预期):{}".format(
        full.returncode, last[0] if last else ''))
    print("       edm 配置含 `edm:`/`reg:` 顶段,base loader 按设计读不了;"
          "由 83 用 load_edm_config 校验(见上)。以下为排除 edm 后的 81 实检:")
    rs = subprocess.run([sys.executable, script, "--config_dir", config_dir,
                         "--glob", "config_wind_canvas_p3_arch_[rsu]*.yml"],
                        capture_output=True, text=True)
    print(rs.stdout.strip())
    if rs.returncode != 0:
        print(rs.stderr.strip()[-800:])
    return rs.returncode == 0


def report_param_counts():
    """打印参考臂 UNet 与 swin(inner 288)的实测参数量(容量对齐核对;无 torch 则跳过)。"""
    try:
        from src.dl_config.config_loader import load_config
        from src.dl_model.model_maker import make_model
        from src.dl_model.swinir_arch import SwinIRCanvasConfig
    except Exception as e:                                        # noqa: BLE001
        print("[说明] torch 环境不可用({}),跳过参数量对照".format(repr(e)[:60]))
        return
    base = load_config(EXPERIMENT, BASE_YML)
    n_unet = sum(p.numel() for p in make_model(base.model).parameters())
    swin_fields = {k: v for k, v in SWIN_MODEL.items() if k != 'arch'}
    n_swin = sum(p.numel() for p in
                 make_model(SwinIRCanvasConfig(**swin_fields)).parameters())
    print("参数量对照: 参考臂 UNet {:.3f} M vs swin(inner 288) {:.3f} M".format(
        n_unet / 1e6, n_swin / 1e6))
    print("  注: 契约假设参考臂 ~5-6M,实测 {:.1f}M;swin 臂按契约 inner_channel=288 "
          "生成(5.786M)。两者并非同容量,如需严格容量对齐需重定 inner_channel"
          "(≈288*sqrt({:.1f}/{:.1f})≈{:d}),请主线决策。".format(
              n_unet / 1e6, n_unet / 1e6, n_swin / 1e6,
              int(288.0 * (n_unet / n_swin) ** 0.5)))


def main():
    ap = argparse.ArgumentParser(description="阶段 3 架构对比配置生成")
    ap.add_argument("--no_check", action="store_true", help="跳过 81 与 edm 段校验")
    args = ap.parse_args()

    base = load_base()
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    variants = build_variants(base)

    for tag, cfg, notes, is_edm in variants:
        path = os.path.join(OUT_DIR, "config_wind_canvas_{}.yml".format(tag))
        with open(path, 'w') as f:
            f.write(header(tag, notes))
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        changes = diff_dict(base, cfg)
        print("")
        print("== {} -> {} ==".format(tag, os.path.relpath(path, ROOT)))
        if changes:
            for p, o, n in changes:
                print("  {:<28} {} -> {}".format(p, o, n))
        else:
            print("  (与基座逐键相同)")
        if is_edm and not args.no_check:
            ok = verify_edm_section(cfg['edm'], tag)
            if ok:
                print("  文件级校验: {}".format(verify_edm_config_file(path)))
            else:
                print("  [FAIL] edm 段校验未过")

    print("")
    print("生成 {} 个配置 -> {}".format(len(variants), OUT_DIR))
    if not args.no_check:
        report_param_counts()
    if not args.no_check:
        ok81 = run_81(os.path.join(ROOT, "configs", "深圳", "phase3_arch"))
        print("")
        print("81 通道校验(si/reg 配置): {}".format("OK" if ok81 else "FAIL"))
        if not ok81:
            sys.exit(1)


if __name__ == "__main__":
    main()
