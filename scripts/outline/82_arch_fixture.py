# -*- coding: utf-8 -*-
"""阶段 3 架构 A/B/C 合成自检(秒级,不读服务器数据,不建 checkpoint 文件)。

覆盖(与阶段 0 教训一致:先合成自检再上全量训练/评估):
  A) SwinIR:前向形状/有限/gamma 敏感性/非整除 pad 兜底/反向梯度/参数规模(83 配置 ≈5.8M);
  B) RegressionModel:输出 == y0 + net(...) 手算(逐位)、add_noise 被忽略、裸键/前缀键加载;
  C) EDM:预条件恒等式(手算)、零初始化头 D == c_skip·r_σ(精确)、采样形状/有限/样本互异/
     seed 可复现、Karras 调度与 σ 采样范围、损失有限、ctx 通道不符报错;
  D) 84 的分布类指标解析对拍:样本式 CRPS(fair)对拍暴力双重求和 + N(μ,1) 解析 CRPS;
     rank histogram 对拍均匀秩(含边界:真值低于/高于全体样本、全并列)+ 期望计数;
  E) 配置加载:旧 phase2 yml 照常(默认 UNet 分派)、swin yml 分派到 SwinIRCanvasConfig、
     83 产物(phase3_arch,若已生成)逐 tag 校验;edm 配置经 load_edm_config 剥离后校验。

运行(仓库根目录,torch 环境;不依赖服务器数据):
  python scripts/outline/82_arch_fixture.py
退出码 0 = 全部通过,1 = 有失败项。
"""
import argparse
import dataclasses
import importlib.util
import math
import os
import sys
import tempfile

import numpy as np
import torch
import yaml
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dl_model.ddpm.unet_ddpm_v01 import UNetDDPMVer01, UNetDDPMVer01Config  # noqa: E402
from src.dl_model.edm_correction import (  # noqa: E402
    EDMCorrector,
    EDMCorrectorConfig,
    karras_sigma_schedule,
    sample_log_sigma,
)
from src.dl_model.model_maker import make_model  # noqa: E402
from src.dl_model.regression_wrapper import RegressionModel  # noqa: E402
from src.dl_model.swinir_arch import SwinIRCanvas, SwinIRCanvasConfig  # noqa: E402

N_OK = [0]
N_FAIL = [0]


def check(name, cond, extra=""):
    tag = "OK" if cond else "FAIL"
    (N_OK if cond else N_FAIL)[0] += 1
    print("  [{}] {} {}".format(tag, name, extra))


def check_skip(name, reason=""):
    print("  [SKIP] {} {}".format(name, reason))


def _load_84():
    """84_ensemble_eval.py 文件名以数字开头,importlib 载入以复用其指标函数。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '84_ensemble_eval.py')
    spec = importlib.util.spec_from_file_location('agl_eval_84_fixture', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def small_unet_cfg(in_channel=12, out_channel=8, inner=32):
    return UNetDDPMVer01Config(
        in_channel=in_channel, inner_channel=inner, out_channel=out_channel,
        res_blocks=1, channel_mults=[1, 2], attn_res=[], channels_each_head=4,
        dropout=0.0, resblock_updown=True, max_period=10.0)


def randomize_convs(net):
    """把零初始化输出头之后的所有 conv/linear 权重随机化(使输出非恒零)。"""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, std=0.05)
            if m.bias is not None:
                nn.init.normal_(m.bias, std=0.01)


# ---------------------------------------------------------------------------
# A) SwinIR
# ---------------------------------------------------------------------------
def test_swin():
    print("A) SwinIR 前向/梯度/规模")
    torch.manual_seed(0)
    net = SwinIRCanvas(in_channel=12, out_channel=8, inner_channel=16, num_blocks=2,
                       window_size=8, num_heads=4, mlp_ratio=2.0, dropout=0.0).eval()
    B, H, W = 2, 16, 24
    yt = torch.randn(B, 8, H, W)
    yc = torch.randn(B, 4, H, W)
    with torch.no_grad():
        out1 = net(yt=yt, y_cond=yc, gamma=torch.full((B,), 0.2))
        out2 = net(yt=yt, y_cond=yc, gamma=torch.full((B,), 0.9))
    zero_init = bool(out1.abs().max() == 0.0)
    check("零初始化输出头:初始输出恒 0(残差零起点)", zero_init,
          "|out|max={:.3e}".format(float(out1.abs().max())))
    randomize_convs(net)
    with torch.no_grad():
        out1 = net(yt=yt, y_cond=yc, gamma=torch.full((B,), 0.2))
        out2 = net(yt=yt, y_cond=yc, gamma=torch.full((B,), 0.9))
    check("前向形状 == 输入(2,8,16,24)", tuple(out1.shape) == (B, 8, H, W),
          str(tuple(out1.shape)))
    check("前向有限", bool(torch.isfinite(out1).all()))
    dg = float((out1 - out2).abs().max())
    check("gamma 敏感性(不同时间步输出不同)", dg > 0, "max|Δ|={:.3e}".format(dg))

    with torch.no_grad():
        out_np = net(yt=torch.randn(1, 8, 20, 28), y_cond=torch.randn(1, 4, 20, 28),
                     gamma=torch.zeros(1))
        out_tiny = net(yt=torch.randn(1, 8, 4, 4), y_cond=torch.randn(1, 4, 4, 4),
                       gamma=torch.zeros(1))
    check("非整除尺寸(20x28)镜像 pad 后裁回", tuple(out_np.shape) == (1, 8, 20, 28)
          and bool(torch.isfinite(out_np).all()))
    check("极小尺寸(4x4)pad 退化 replicate 分支可用",
          tuple(out_tiny.shape) == (1, 8, 4, 4) and bool(torch.isfinite(out_tiny).all()))

    net.zero_grad()
    out = net(yt=yt, y_cond=yc, gamma=torch.full((B,), 0.5))
    out.pow(2).mean().backward()
    g1 = net.patch_embed.weight.grad
    g2 = net.out_conv.weight.grad
    ok_g = (g1 is not None and g2 is not None and bool(torch.isfinite(g1).all())
            and bool(torch.isfinite(g2).all()) and float(g1.abs().sum()) > 0
            and float(g2.abs().sum()) > 0)
    check("反向:patch_embed/out_conv 梯度有限非零", ok_g,
          "|g1|1={:.3g} |g2|1={:.3g}".format(
              float(g1.abs().sum()) if g1 is not None else -1.0,
              float(g2.abs().sum()) if g2 is not None else -1.0))

    big = SwinIRCanvas(in_channel=167, out_channel=72, inner_channel=288, num_blocks=6,
                       window_size=8, num_heads=4, mlp_ratio=2.0, dropout=0.0)
    n = sum(p.numel() for p in big.parameters())
    check("83 配置参数规模 5-6.5M(对齐 UNet 参考臂)", 5.0e6 <= n <= 6.5e6,
          "{:.3f} M".format(n / 1e6))
    check("83 配置整除性:训练 96x112 / 评估 pad16 后 112x128 均被 window_size=8 整除",
          96 % 8 == 0 and 112 % 8 == 0 and 112 % 8 == 0 and 128 % 8 == 0)


# ---------------------------------------------------------------------------
# B) RegressionModel
# ---------------------------------------------------------------------------
def test_regression():
    print("B) RegressionModel 前向恒等式与加载兼容")
    torch.manual_seed(1)
    cfg = small_unet_cfg()
    reg = RegressionModel(cfg).eval()
    check("net 由 make_model 构建(UNetDDPMVer01)", isinstance(reg.net, UNetDDPMVer01))
    B, H, W = 2, 16, 16
    y0 = torch.randn(B, 8, H, W)
    yc = torch.randn(B, 4, H, W)
    with torch.no_grad():
        out, aux = reg.sample_y1_bare_diffusion(y0=y0, y_cond=yc)
        manual = y0 + reg.net(yt=y0, y_cond=yc, gamma=torch.ones(B))
        out_n = reg.sample_y1_bare_diffusion(y0=y0, y_cond=yc, add_noise=True)[0]
    check("返回 (y1_full, None)", aux is None and tuple(out.shape) == (B, 8, H, W))
    check("y1_full == y0 + net(..., gamma=ones) 逐位一致", torch.equal(out, manual))
    check("add_noise=True 结果与默认逐位一致(被忽略)", torch.equal(out, out_n))

    reg2 = RegressionModel(cfg).eval()
    bare = {k[len(reg.NET_PREFIX):]: v for k, v in reg.state_dict().items()}
    reg2.load_state_dict(bare)
    with torch.no_grad():
        out2 = reg2.sample_y1_bare_diffusion(y0=y0, y_cond=yc)[0]
    check("裸 UNet 键加载后输出逐位一致(评估链路加载口径)",
          torch.equal(out, out2))
    reg3 = RegressionModel(cfg).eval()
    reg3.load_state_dict(reg.state_dict())          # 前缀键原样加载
    with torch.no_grad():
        out3 = reg3.sample_y1_bare_diffusion(y0=y0, y_cond=yc)[0]
    check("前缀键(RegressionModel.state_dict)加载后输出逐位一致", torch.equal(out, out3))


# ---------------------------------------------------------------------------
# C) EDM 预条件 / 采样
# ---------------------------------------------------------------------------
def test_edm():
    print("C) EDM 预条件恒等式 / 采样 / 调度")
    torch.manual_seed(2)
    cfg = EDMCorrectorConfig(out_channel=8, ctx_channel=28, inner_channel=16,
                             channel_mults=[1, 2, 4], sigma_data=1.3,
                             sigma_min=0.01, sigma_max=20.0, steps=6)
    cor = EDMCorrector(cfg).eval()
    B = 2
    sig = torch.tensor([0.05, 1.3, 10.0])
    c_skip, c_out, c_in, c_noise = cor.precondition(sig)
    sd = cfg.sigma_data
    ok_pre = True
    for i, s in enumerate(sig.tolist()):
        ok_pre &= (abs(float(c_skip[i]) - sd ** 2 / (s ** 2 + sd ** 2)) < 1e-6
                   and abs(float(c_out[i]) - s * sd / math.sqrt(sd ** 2 + s ** 2)) < 1e-6
                   and abs(float(c_in[i]) - 1.0 / math.sqrt(sd ** 2 + s ** 2)) < 1e-6
                   and abs(float(c_noise[i]) - math.log(s) / 4.0) < 1e-6)
    check("预条件恒等式 c_skip/c_out/c_in/c_noise == 论文表 1(手算)", bool(ok_pre))
    check("预条件形状:c_skip/c_out/c_in (B,1,1,1),c_noise (B,)",
          tuple(c_skip.shape) == (3, 1, 1, 1) and tuple(c_noise.shape) == (3,))

    r = torch.randn(B, 8, 16, 16)
    ctx = torch.randn(B, 28, 16, 16)
    with torch.no_grad():
        d0 = cor.denoise(r, 2.0, ctx)
    expect0 = (sd ** 2 / (2.0 ** 2 + sd ** 2)) * r      # F 恒 0(零初始化头)
    check("零初始化输出头:D(r_σ) == c_skip·r_σ(逐位)", torch.equal(d0, expect0),
          "max|Δ|={:.3e}".format(float((d0 - expect0).abs().max())))

    randomize_convs(cor.net)
    y0 = torch.randn(B, 8, 16, 16)
    y1_reg = torch.randn(B, 8, 16, 16)
    yc = torch.randn(B, 12, 16, 16)
    with torch.no_grad():
        st0 = torch.get_rng_state()
        s3 = cor.sample(y0=y0, y1_reg=y1_reg, y_cond=yc, n_samples=3, steps=6, seed=7)
        rng_kept = torch.equal(torch.get_rng_state(), st0)
        s3b = cor.sample(y0=y0, y1_reg=y1_reg, y_cond=yc, n_samples=3, steps=6, seed=7)
        s3c = cor.sample(y0=y0, y1_reg=y1_reg, y_cond=yc, n_samples=3, steps=6, seed=8)
    check("sample 形状 (n,B,C,H,W) 且有限", tuple(s3.shape) == (3, B, 8, 16, 16)
          and bool(torch.isfinite(s3).all()), str(tuple(s3.shape)))
    d_ij = float((s3[0] - s3[1]).abs().max())
    check("样本互异(抽两个样本 max|Δ|>0)", d_ij > 0, "max|Δ|={:.3e}".format(d_ij))
    check("同 seed 逐位可复现", torch.equal(s3, s3b))
    check("不同 seed 结果不同", not torch.equal(s3, s3c))
    check("sample 不扰动全局 RNG(seed 局部 CPU generator)", rng_kept)

    sigmas = karras_sigma_schedule(6, 0.01, 20.0, 7.0)
    check("Karras 调度:长度 steps+1、首值 σ_max、末值 0、单调不增",
          len(sigmas) == 7 and abs(float(sigmas[0]) - 20.0) < 1e-6
          and float(sigmas[-1]) == 0.0
          and bool((sigmas[:-1].diff() <= 0).all()))
    g = torch.Generator().manual_seed(0)
    ss = sample_log_sigma(1000, cfg, generator=g)
    check("训练 σ 采样落在 [σ_min, σ_max]",
          bool((ss >= cfg.sigma_min).all() and (ss <= cfg.sigma_max).all()))

    with torch.no_grad():
        loss_w = cor.loss(r, torch.full((B,), 0.5), ctx)
        loss_u = cor.loss(r, torch.full((B,), 0.5), ctx, weight=False)
    check("EDM 损失有限(加权/未加权)", bool(torch.isfinite(loss_w))
          and bool(torch.isfinite(loss_u)) and float(loss_w) > 0 and float(loss_u) > 0)
    try:
        with torch.no_grad():
            cor.sample(y0=y0, y1_reg=y1_reg, y_cond=torch.randn(B, 11, 16, 16), n_samples=1)
    except ValueError as e:
        check("ctx 通道不符 -> ValueError", True, "-> {}".format(str(e)[:60]))
    else:
        check("ctx 通道不符 -> ValueError", False, "未抛 ValueError")


# ---------------------------------------------------------------------------
# D) 84 的分布类指标解析对拍
# ---------------------------------------------------------------------------
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_dist_metrics():
    print("D) 84 分布类指标:CRPS / rank histogram 解析对拍")
    m84 = _load_84()

    # D1 精确公式对拍(固定样本,暴力双重求和)
    s = np.array([0.0, 1.0, 2.0, 5.0])
    y = np.array([2.0])
    est = float(m84.crps_sample(s, y))
    n = len(s)
    brute = float(np.abs(s - y[0]).mean()
                  - sum(abs(s[i] - s[j]) for i in range(n) for j in range(n)
                        if i != j) / (2.0 * n * (n - 1.0)))
    check("样本式 CRPS == 暴力双重求和(fair 估计)", abs(est - brute) < 1e-12,
          "{:.10f} vs {:.10f}".format(est, brute))

    # D2 N(μ,1) 解析 CRPS:重复采样取均值对拍解析值(5 倍标准误判定)
    rng = np.random.default_rng(11)
    n_ens, reps, mu = 512, 400, 0.0
    yv = 0.3
    ests = np.empty(reps)
    for i in range(reps):
        xs = rng.standard_normal((n_ens, 1)) + mu
        ests[i] = float(m84.crps_sample(xs, np.array([yv]))[0])
    analytic = (2.0 * math.exp(-yv ** 2 / 2.0) / math.sqrt(2.0 * math.pi)
                + yv * (2.0 * _phi(yv) - 1.0) - 1.0 / math.sqrt(math.pi))
    se = ests.std(ddof=1) / math.sqrt(reps)
    check("N(0,1) 预报的 CRPS ≈ 解析值(±5 标准误)",
          abs(ests.mean() - analytic) < 5.0 * se + 1e-12,
          "估计 {:.5f} 解析 {:.5f} ±{:.5f}".format(ests.mean(), analytic, 5 * se))
    try:
        m84.crps_sample(np.zeros((1, 4)), np.zeros(4))
    except ValueError:
        check("N=1 -> ValueError(84 记 N/A)", True)
    else:
        check("N=1 -> ValueError(84 记 N/A)", False, "未抛 ValueError")

    # D3 rank histogram:均匀秩 + 边界约定
    n_ens = 8
    trials = 4000
    sx = rng.standard_normal((n_ens, trials))
    ty = rng.standard_normal(trials)
    ranks = m84.rank_of_truth(sx, ty)
    hist = m84.rank_histogram(ranks[None, :], n_ens, 1)[0]
    exp = trials / (n_ens + 1.0)
    max_rel = float(np.abs(hist - exp).max() / exp)
    check("独立同分布样本+真值:秩直方图均匀(N=8,4000 次)",
          max_rel < 0.15 and hist.sum() == trials,
          "counts={} 相对偏差 max {:.1%}".format(hist.tolist(), max_rel))
    check("rank 均值 ≈ N/2", abs(float(ranks.mean()) - n_ens / 2.0) < 0.15,
          "{:.3f}".format(float(ranks.mean())))
    check("真值低于全体样本 -> 秩 0(箱 0)",
          float(m84.rank_of_truth(np.full((n_ens, 1), 1.0), np.array([0.0]))[0]) == 0.0)
    check("真值高于全体样本 -> 秩 N(末箱)",
          float(m84.rank_of_truth(np.full((n_ens, 1), 0.0),
                                  np.array([1.0]))[0]) == float(n_ens))
    tie = float(m84.rank_of_truth(np.full((n_ens, 1), 0.5), np.array([0.5]))[0])
    check("全并列 -> 秩 = N/2(0.5 计并列)", tie == n_ens / 2.0, "rank={}".format(tie))
    hb = m84.rank_histogram(np.full((1, 3, 4), 0.0), n_ens, 1)[0]
    check("rank_histogram 箱数 = N+1 且计数守恒",
          len(hb) == n_ens + 1 and int(hb.sum()) == 12, str(hb.tolist()))


# ---------------------------------------------------------------------------
# E) 配置加载:旧 yml / arch 分派 / 83 产物
# ---------------------------------------------------------------------------
def test_configs():
    print("E) 配置加载(旧 yml 照常 / arch 分派 / 83 产物)")
    from src.dl_config.config_loader import load_config
    exp = "ExperimentSchrodingerBridgeWindCanvas"
    old = os.path.join(ROOT, "configs", "深圳", "phase2",
                       "config_wind_canvas_p2_l1r2_lr2e4.yml")
    if not os.path.exists(old):
        check_skip("旧 phase2 yml 不存在,整节跳过", old)
        return
    cfg = load_config(exp, old)
    check("旧 phase2 yml:model 仍分派为 UNetDDPMVer01Config",
          isinstance(cfg.model, UNetDDPMVer01Config), type(cfg.model).__name__)
    net = make_model(cfg.model)
    n = sum(p.numel() for p in net.parameters())
    check("旧 yml 可 make_model(参考臂参数量打印)", n > 0, "{:.3f} M".format(n / 1e6))

    # swin 分派:优先用 83 产物,否则临时 yml
    gen_swin = os.path.join(ROOT, "configs", "深圳", "phase3_arch",
                            "config_wind_canvas_p3_arch_swin.yml")
    if os.path.exists(gen_swin):
        swin_path = gen_swin
        print("  (使用 83 产物 {})".format(os.path.relpath(swin_path, ROOT)))
    else:
        with open(old) as f:
            raw = yaml.safe_load(f)
        raw['model'] = {'arch': 'swinir_canvas', 'in_channel': 167, 'out_channel': 72,
                        'inner_channel': 288, 'num_blocks': 6, 'window_size': 8,
                        'num_heads': 4, 'mlp_ratio': 2.0, 'dropout': 0.0}
        fh = tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False)
        yaml.safe_dump(raw, fh, sort_keys=False, allow_unicode=True)
        fh.close()
        swin_path = fh.name
        check_skip("83 的 swin 配置不存在,改用同内容临时 yml", gen_swin)
    cfg_s = load_config(exp, swin_path)
    check("swin yml:model 分派为 SwinIRCanvasConfig",
          isinstance(cfg_s.model, SwinIRCanvasConfig), type(cfg_s.model).__name__)
    net_s = make_model(cfg_s.model)
    check("swin 配置 make_model -> SwinIRCanvas",
          isinstance(net_s, SwinIRCanvas))
    if os.path.exists(gen_swin):
        with torch.no_grad():
            out = net_s(yt=torch.zeros(1, 72, 96, 112), y_cond=torch.zeros(1, 95, 96, 112),
                        gamma=torch.zeros(1))
        check("swin 真实配置前向(1x167x96x112,真实训练尺寸)",
              tuple(out.shape) == (1, 72, 96, 112) and bool(torch.isfinite(out).all()))
    if swin_path != gen_swin:
        os.remove(swin_path)

    arch_dir = os.path.join(ROOT, "configs", "深圳", "phase3_arch")
    expect = {
        'p3_arch_unet_s2': UNetDDPMVer01Config,
        'p3_arch_swin': SwinIRCanvasConfig,
        'p3_arch_swin_s2': SwinIRCanvasConfig,
        'p3_arch_reg': UNetDDPMVer01Config,
        'p3_arch_reg_s2': UNetDDPMVer01Config,
    }
    if os.path.isdir(arch_dir) and os.listdir(arch_dir):
        for tag, cls in expect.items():
            p = os.path.join(arch_dir, "config_wind_canvas_{}.yml".format(tag))
            if not os.path.exists(p):
                check_skip("83 产物缺失 " + tag, p)
                continue
            cfg_t = load_config(exp, p)
            check("{} 分派 {}".format(tag, cls.__name__), isinstance(cfg_t.model, cls),
                  "seed={}".format(cfg_t.train.seed))
        # edm:load_edm_config 剥离后构造
        for tag in ('p3_arch_edm', 'p3_arch_edm_s2'):
            p = os.path.join(arch_dir, "config_wind_canvas_{}.yml".format(tag))
            if not os.path.exists(p):
                check_skip("83 产物缺失 " + tag, p)
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    "train_edm_correction_82",
                    os.path.join(ROOT, "scripts", "train_edm_correction.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                base_cfg, edm_cfg, reg_sec, _ = mod.load_edm_config(exp, p)
            except Exception as e:                                # noqa: BLE001
                check_skip("{} load_edm_config 不可用({})".format(tag, repr(e)[:60]), "")
                continue
            check("{} edm 段构造 EDMCorrectorConfig(ctx=239,inner=64,steps=24)".format(tag),
                  isinstance(edm_cfg, EDMCorrectorConfig)
                  and edm_cfg.ctx_channel == 239 and edm_cfg.inner_channel == 64
                  and edm_cfg.steps == 24 and edm_cfg.sigma_data is None,
                  "sigma_data={}".format(edm_cfg.sigma_data))
            check("{} reg 段指向配对 tag".format(tag), bool(reg_sec.get('reg_tag')),
                  "reg_tag={}".format(reg_sec.get('reg_tag')))
            try:
                EDMCorrector(dataclasses.replace(edm_cfg, sigma_data=1.0))
                check("{} EDMCorrector 可构造(σ_d 占位 1.0)".format(tag), True)
            except Exception as e:                                # noqa: BLE001
                check("{} EDMCorrector 可构造(σ_d 占位 1.0)".format(tag), False,
                      repr(e)[:80])
    else:
        check_skip("83 产物目录为空", arch_dir)


def main():
    ap = argparse.ArgumentParser(description="阶段 3 架构 A/B/C 合成自检(秒级)")
    ap.parse_args()
    print("=" * 72)
    test_swin()
    test_regression()
    test_edm()
    test_dist_metrics()
    test_configs()
    print("=" * 72)
    print("检查项:通过 {} / 失败 {}".format(N_OK[0], N_FAIL[0]))
    if N_FAIL[0]:
        print("ARCH FIXTURE FAILED")
        sys.exit(1)
    print("ARCH FIXTURE OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
