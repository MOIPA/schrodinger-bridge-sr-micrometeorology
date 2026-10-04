# -*- coding: utf-8 -*-
"""阶段 2 损失链合成自检(秒级,不读服务器数据):canvas 四物理项 + 旧行为回归。

覆盖(与阶段 0 的教训一致:先合成自检再上全量训练):
  ① physics_canvas 纯函数自检(解析场/真值侧对拍/梯度)直接跑一遍;
  ② 旧行为逐位回归:权重全 0 时 5 个基准数值与改造前完全一致(tol=0,
     基准值由对拍脚本在改造前捕获,此处内嵌,不依赖 /tmp);
  ③ 新四项:parts 有限、total == data + Σ w_i·part_i、t=1 完美估计时
     vort/spectral/extreme≈0 且 div hinge≥0、mask 全 False 时四项=0 且
     total==data、分项原始值与权重无关、各报错路径;
  ④ 梯度:四项对 net 输出各有有限非零梯度;
  ⑤ 旧 yml(phase1 / phase1r V*)经 load_config 可构造(环境不支持时跳过);
  ⑥ 阶段 3 AGL 空间监督:
     a) 可微算子 agl_interp_batched 与 60 号 numpy 算子逐点等价(field10 有无两种);
     b) ln z 线性合成场经表插值精确还原;10 m(-2)/低于首层(-1)分支 torch↔numpy;
     c) **crop 对齐端到端**:训练路(place→dataset crop→split/destagger→可微插值)
        与评估路(整画布评估 destagger + numpy 算子→切)在 (0,0)/(3,8) 两个 crop 下
        逐点一致(表同 crop 也逐元素相等);
     d) 集成:parts["agl"] 与公共算子复算一致、replace/add 加权恒等式、梯度有限、
        agl_weight=0 时传表/不传表 total 逐位相同。

运行(仓库根目录,torch 环境;不依赖服务器数据):
  python scripts/outline/95_phase2_loss_fixture.py
退出码 0 = 全部通过,1 = 有失败项。
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dl_data.wind_canvas_statics import (  # noqa: E402
    TARGET_AGL, CanvasStatics, build_agl_tables_native)
from src.dl_model.si_follmer import physics_canvas  # noqa: E402
from src.dl_model.si_follmer.agl_canvas import agl_interp_batched  # noqa: E402
from src.dl_model.si_follmer.physics_canvas import (  # noqa: E402
    destagger_canvas, split_canvas_state)
from src.dl_model.si_follmer.si_follmer_framework import (  # noqa: E402
    SIFollmerConfig, StochasticInterpolantFollmer)

# 画布窗口与层数(canvas_resid 等基准的原始形状)
B, C, L = 2, 72, 23
H, W = 96, 112
DX = 1000.0

# 任务 A 改造前由 /tmp/si_baseline_capture.py 捕获的"权重全 0 旧行为"数值。
# 同一机器/同一 torch 上逐位复现;跨机器/跨版本因 float32 归约顺序不同会有
# ~1e-7 相对偏差(2026-10-02 服务器 wind3d 实测 7e-8),故用 rel 1e-6 判定:
# 若旧路径被改动,偏差会远大于此。
BASELINE = {
    "canvas_resid_L2": 2.9703762531280518,
    "canvas_direct_L1_cw": 5.213955879211426,
    "canvas_direct_L2_cw": 14.20551872253418,
    "interleaved_olddiv": 4.563587188720703,
    "interleaved_resid_olddiv": 3.9180166721343994,
}

N_OK = [0]
N_FAIL = [0]


def check(name, cond, extra=""):
    tag = "OK" if cond else "FAIL"
    (N_OK if cond else N_FAIL)[0] += 1
    print("  [{}] {} {}".format(tag, name, extra))


def check_skip(name, reason=""):
    """环境缺依赖/缺文件时的跳过(不计入失败,但明确打印原因)。"""
    print("  [SKIP] {} {}".format(name, reason))


# ---------------------------------------------------------------------------
# 通用 fake net / 配置 / 数据
# ---------------------------------------------------------------------------
class FakeNet(nn.Module):
    """固定权重的线性 fake net(阶段 2 集成测试用)。

    mode="normal"  : 1x1 conv 小随机权重(④ 梯度项用,与 /tmp 集成测试同构);
    mode="baseline": linspace 权重 + 条件项(② 旧行为对拍口径,逐位复现基准值)。
    """

    def __init__(self, c, mode="normal"):
        super().__init__()
        self.mode = mode
        self.lin = nn.Conv2d(c, c, 1)
        with torch.no_grad():
            if mode == "linspace":
                self.lin.weight.copy_(torch.linspace(-0.1, 0.1, c * c).view(c, c, 1, 1))
            else:
                self.lin.weight.normal_(0, 0.02)
            self.lin.bias.zero_()

    def forward(self, yt, y_cond, gamma):
        if self.mode == "linspace":
            return self.lin(yt) + 0.05 * yt + 0.01 * y_cond[:, : yt.shape[1]]
        return self.lin(yt) + 0.05 * yt


def phys_scale():
    return [3.0] * L + [3.0] * L + [1.0] * (L + 1) + [4.0, 4.0]


def make_si(**over):
    """构造 canvas SI;over 里的键经 dataclass 字段过滤(兼容未知键)。"""
    cfg = dict(n_timestep=10, eps=0.2, formula="quadratic", loss_type="L2",
               residual_output=True, state_layout="canvas",
               phys_scale=phys_scale(), phys_dz=[30.0 + 8.0 * k for k in range(L)],
               phys_dx=DX, phys_min_t=0.5, phys_div_tau=[2e-3] * L,
               divergence_weight=0.7, vorticity_weight=0.3,
               spectral_weight=0.2, extreme_weight=0.1)
    cfg.update(over)
    return StochasticInterpolantFollmer(
        SIFollmerConfig(**{k: v for k, v in cfg.items()
                           if k in SIFollmerConfig.__dataclass_fields__}),
        FakeNet(C), device="cpu")


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    y0 = torch.randn(B, C, H, W, generator=g)
    y1 = torch.randn(B, C, H, W, generator=g)
    y_cond = torch.randn(B, C, H, W, generator=g)
    rho = 1.1 + 0.1 * torch.rand(B, L, H, W, generator=g)
    return y0, y1, y_cond, rho


def force_timestep(si, ts):
    """把 _sample_timestep 钉死在指定时间步索引(去掉随机性,便于断言)。"""
    def _f(batch_size):
        return (torch.full((batch_size,), int(ts), dtype=torch.int64),
                torch.full((batch_size,), float(ts)))

    si._sample_timestep = _f


# ---------------------------------------------------------------------------
# ① physics_canvas 纯函数自检
# ---------------------------------------------------------------------------
def test_physics_canvas():
    print("① physics_canvas 纯函数自检(解析场/真值对拍/梯度)")
    ok = physics_canvas._self_test()
    check("physics_canvas._self_test() 全绿", ok)


# ---------------------------------------------------------------------------
# ② 旧行为逐位回归(改造前捕获的 5 个基准值)
# ---------------------------------------------------------------------------
def make_cfg(**over):
    base = dict(n_timestep=10, eps=0.2, formula="quadratic", loss_type="L2",
                channel_weights=None, residual_output=False, state_layout="canvas")
    base.update(over)
    return SIFollmerConfig(**base)


def run_one(cfg, shape, seed, cw=None):
    torch.manual_seed(seed)
    net = FakeNet(shape[1], mode="linspace")
    si = StochasticInterpolantFollmer(cfg, net, device="cpu")
    if cw is not None:
        si.channel_weights = torch.tensor(cw, dtype=torch.float32).view(1, -1, 1, 1)
    torch.manual_seed(seed)
    y0 = torch.randn(*shape)
    y1 = torch.randn(*shape)
    y_cond = torch.randn(*shape)
    torch.manual_seed(seed + 1)
    loss = si(y0=y0, y1=y1, y_cond=y_cond)
    return float(loss)


def test_baseline_regression():
    print("② 旧行为回归(权重全 0,rel 1e-6)")
    s = (2, 72, 96, 112)   # canvas 窗口
    o = (2, 18, 64, 64)    # 旧 interleaved(Z=6 层)
    cw = [1.0] * 72
    cw[46:70] = [10.0] * 24   # W 通道加权
    got = {
        "canvas_resid_L2": run_one(make_cfg(residual_output=True), s, 1234),
        "canvas_direct_L1_cw": run_one(
            make_cfg(residual_output=False, loss_type="L1"), s, 4321, cw=cw),
        "canvas_direct_L2_cw": run_one(
            make_cfg(residual_output=False, channel_weights=cw), s, 999, cw=cw),
        "interleaved_olddiv": run_one(
            make_cfg(residual_output=False, state_layout="interleaved_uvw",
                     divergence_weight=0.5, vorticity_weight=0.3), o, 7),
        "interleaved_resid_olddiv": run_one(
            make_cfg(residual_output=True, state_layout="interleaved_uvw",
                     divergence_weight=0.2, vorticity_weight=0.1), o, 11),
    }
    for k in BASELINE:
        ref = BASELINE[k]
        ok = abs(got[k] - ref) <= 1e-6 * max(abs(ref), 1.0)
        check("{} 数值一致(rel 1e-6)".format(k), ok,
              "{} vs {}".format(repr(got[k]), repr(ref)))


# ---------------------------------------------------------------------------
# ③ 新四项:一致性 / 退化情形 / 报错路径
# ---------------------------------------------------------------------------
def test_new_parts():
    print("③ 新四项:有限性/加权一致性/退化情形")
    si = make_si()
    y0, y1, y_cond, rho = make_batch(0)
    torch.manual_seed(0)
    out = si(y0=y0, y1=y1, y_cond=y_cond, rho=rho, return_parts=True)
    print("   parts: {}".format(
        {k: (round(float(v), 6) if torch.is_tensor(v) else v) for k, v in out.items()}))
    keys = ("div", "vort", "spectral", "extreme")
    check("四项原始值有限", all(torch.is_tensor(out[k]) and torch.isfinite(out[k])
                                 for k in keys))
    check("四项原始值>0", all(float(out[k]) > 0 for k in keys))
    check("total 有限", bool(torch.isfinite(out["total"])))
    w = dict(div=0.7, vort=0.3, spectral=0.2, extreme=0.1)
    lhs = float(out["total"])
    rhs = float(out["data"]) + sum(w[k] * float(out[k]) for k in w)
    check("total == data + Σ w_i·part_i", abs(lhs - rhs) <= 1e-6 * abs(lhs),
          "{} vs {}".format(repr(lhs), repr(rhs)))

    # 分项原始值与权重无关(权重 0 vs 非 0,同 seed/同网络)
    torch.manual_seed(7)
    si_a = make_si(divergence_weight=0.0, vorticity_weight=0.0,
                   spectral_weight=0.0, extreme_weight=0.0)
    torch.manual_seed(7)
    si_b = make_si(divergence_weight=0.7, vorticity_weight=0.3,
                   spectral_weight=0.2, extreme_weight=0.1)
    torch.manual_seed(0)
    out_a = si_a(y0=y0, y1=y1, y_cond=y_cond, rho=rho, return_parts=True)
    torch.manual_seed(0)
    out_b = si_b(y0=y0, y1=y1, y_cond=y_cond, rho=rho, return_parts=True)
    diffs = {k: abs(float(out_a[k]) - float(out_b[k])) for k in keys}
    check("权重 0/非 0 分项一致", all(v == 0.0 for v in diffs.values()), str(diffs))
    check("权重 0 时 total==data", float(out_a["total"]) == float(out_a["data"]))
    # rho=None 且散度权重 0 的探针场景:div=None,其余正常
    si_probe = make_si(divergence_weight=0.0)
    torch.manual_seed(0)
    o = si_probe(y0=y0, y1=y1, y_cond=y_cond, return_parts=True)
    check("rho=None+div 权重=0:div=None,其余正常",
          o["div"] is None and float(o["vort"]) > 0 and float(o["extreme"]) > 0)

    # t=1 完美估计 (b̂ = 2·(y1−y0),quadratic dot_beta(1)=2):物理项应退化
    si_p = make_si()
    force_timestep(si_p, 10)
    y0p, y1p, y_condp, rhop = make_batch(1)
    b_perfect = 2.0 * (y1p - y0p)
    raw = si_p._canvas_physics_raw(y0p, y1p, b_perfect,
                                   torch.full((B,), 10, dtype=torch.int64), rhop)
    vals = {k: float(v) for k, v in raw.items()}
    print("   perfect: {}".format({k: round(v, 8) for k, v in vals.items()}))
    check("t=1 完美估计 vort=0", abs(vals["vort"]) < 1e-6)
    check("t=1 完美估计 spectral=0", abs(vals["spectral"]) < 1e-6)
    check("t=1 完美估计 extreme=0", abs(vals["extreme"]) < 1e-6)
    check("t=1 完美估计 div≥0(hinge)", vals["div"] >= 0.0)

    # mask 全 False(t=0 档,diff 下 dot_beta=0 恒被排除)
    si_z = make_si()
    force_timestep(si_z, 0)
    torch.manual_seed(0)
    out_z = si_z(y0=y0, y1=y1, y_cond=y_cond, rho=rho, return_parts=True)
    zs = {k: float(out_z[k]) for k in keys}
    check("mask 全 False 时四项=0", all(v == 0.0 for v in zs.values()), str(zs))
    check("mask 全 False 时 total==data", float(out_z["total"]) == float(out_z["data"]))
    check("mask 全 False 时无 NaN", bool(torch.isfinite(out_z["total"])))

    print("③b 报错路径")

    def expect_raise(name, fn):
        try:
            fn()
        except ValueError as e:
            check(name, True, "-> {}".format(str(e)[:70]))
        else:
            check(name, False, "未抛 ValueError")

    expect_raise("canvas+div 权重>0 缺 phys_scale", lambda: make_si(phys_scale=None))
    expect_raise("canvas+div 权重>0 缺 phys_dz", lambda: make_si(phys_dz=None))
    expect_raise("canvas+div 权重>0 缺 phys_div_tau", lambda: make_si(phys_div_tau=None))
    expect_raise("canvas+spectral>0 缺 phys_scale",
                 lambda: make_si(phys_scale=None, divergence_weight=0.0))
    expect_raise("canvas 散度 rho=None", lambda: make_si()(
        y0=y0, y1=y1, y_cond=y_cond))
    expect_raise("interleaved+spectral>0", lambda: make_si(
        state_layout="interleaved_uvw", spectral_weight=0.1))
    expect_raise("phys_scale 长度不符", lambda: make_si(phys_scale=[3.0] * 71)(
        y0=y0, y1=y1, y_cond=y_cond, rho=rho))
    expect_raise("rho 形状不符", lambda: make_si()(
        y0=y0, y1=y1, y_cond=y_cond, rho=rho[:, :L - 1]))


# ---------------------------------------------------------------------------
# ④ 梯度:四项对 net 输出各有有限非零梯度
# ---------------------------------------------------------------------------
def test_gradients():
    print("④ 四项对 net 输出的梯度(有限非零)")
    si = make_si()
    y0, y1, y_cond, rho = make_batch(0)
    torch.manual_seed(0)
    out = si(y0=y0, y1=y1, y_cond=y_cond, rho=rho, return_parts=True)
    for key in ("div", "vort", "spectral", "extreme"):
        si.zero_grad()
        out[key].backward(retain_graph=True)
        g = si.net.lin.weight.grad
        if g is None:
            check("{} 梯度存在".format(key), False, "grad 为 None")
            continue
        check("{} 梯度有限非零".format(key),
              bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0,
              "|g|1={:.4g}".format(float(g.abs().sum())))
    # total 的梯度(与 F1 相同路径,确认加权后仍回传)
    si.zero_grad()
    torch.manual_seed(0)
    out = si(y0=y0, y1=y1, y_cond=y_cond, rho=rho, return_parts=True)
    out["total"].backward()
    g = si.net.lin.weight.grad
    check("total 梯度有限非零", bool(torch.isfinite(g).all())
          and float(g.abs().sum()) > 0, "|g|1={:.4g}".format(float(g.abs().sum())))


# ---------------------------------------------------------------------------
# ⑤ 旧 yml 经 load_config 构造(新字段必须有默认值)
# ---------------------------------------------------------------------------
def test_old_ymls():
    print("⑤ 旧 yml 经 load_config 构造")
    try:
        from src.dl_config.config_loader import load_config
    except Exception as e:                                   # noqa: BLE001
        check_skip("load_config 导入失败(本环境 python<3.10 不支持 src 的"
                   " PEP 604 注解,改用 py3.10 环境可跑此项)", repr(e)[:100])
        return
    experiment = "ExperimentSchrodingerBridgeWindCanvas"
    cases = [
        ("configs/深圳/phase1/config_wind_canvas_p1_t16_residual.yml",
         dict(residual_output=True)),
        ("configs/深圳/phase1r/config_wind_canvas_p1r_t14_noenc_cos.yml",
         dict(residual_output=True)),          # V* 模板
    ]
    for rel, exp in cases:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            check_skip(rel + " 不存在", "")
            continue
        cfg = load_config(experiment, path)
        si = cfg.si
        ok = (isinstance(si, SIFollmerConfig)
              and si.state_layout == "canvas"
              and si.spectral_weight == 0.0 and si.extreme_weight == 0.0
              and si.phys_scale is None and si.phys_dz is None
              and si.phys_div_tau is None
              and si.phys_dx == 1000.0 and si.phys_min_t == 0.5)
        ok = ok and all(getattr(si, k) == v for k, v in exp.items())
        check(os.path.basename(rel) + " 可构造且新字段为默认值", ok,
              "out={} in={}".format(cfg.model.out_channel, cfg.model.in_channel))
        # 旧 config 还要能直接构造 SI 模块(权重 0 时不做物理校验)
        try:
            StochasticInterpolantFollmer(si, FakeNet(cfg.model.out_channel), device="cpu")
            built = True
        except Exception as e:                               # noqa: BLE001
            built = False
            print("     构造 SI 失败: {}".format(repr(e)[:120]))
        check(os.path.basename(rel) + " 可构造 SI 模块", built)


# ---------------------------------------------------------------------------
# ⑥ 阶段 3 AGL 空间监督(agl_canvas 可微插值层 ↔ 60 号 numpy 算子/评估侧接线)
# ---------------------------------------------------------------------------
AGL_NT = 11                       # TARGET_AGL 目标层数
AGL_NLEV = 23                     # 质量层数(canvas 3L+3 的 L)


def _load_numpy_agl_interp():
    """importlib 载入 60_agl_operator.py 的 numpy agl_interp(评估侧唯一实现)。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "60_agl_operator.py")
    spec = importlib.util.spec_from_file_location("agl_operator_60_fixture", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agl_interp


def _eval_destagger_canvas():
    """评估侧 destagger(agl_eval_common,与 89/90 同一实现)。"""
    import agl_eval_common
    return agl_eval_common.destagger_canvas


def _synth_zagl(ny=99, nx=120):
    """合成 z_agl:23 质量层等间隔 20..1000 m + 阶梯地形(右半 +200 m);界面 24 层。

    右半 +200 m 使 30..200 m 目标落到首层以下(-1 分支),左半 20 m 首层使 30 m
    走 idx>=0、10 m 走 -2 分支,三类分支在表里同时出现。
    """
    hgt = np.zeros((ny, nx), dtype=np.float64)
    hgt[:, nx // 2:] = 200.0
    z_m = np.linspace(20.0, 1000.0, AGL_NLEV)[:, None, None] + hgt[None]
    z_i = np.linspace(0.0, 1000.0, AGL_NLEV + 1)[:, None, None] + hgt[None]
    return z_m, z_i


def _synth_agl_batch(b=2, h=96, w=112, ny=99, nx=120):
    """合成 AGL 表 batch(dict, (B,11,h,w));与 dataset 同口径:先 place 成画布再 crop。"""
    z_m, z_i = _synth_zagl(ny, nx)
    tabs = build_agl_tables_native(z_m, z_i, list(range(AGL_NLEV)))
    out = {}
    for k, v in tabs.items():
        placed = CanvasStatics.place(v)[None][:, :, :h, :w]     # (1,11,h,w)
        t = torch.from_numpy(np.repeat(placed, b, axis=0))
        out[k] = t.long() if "idx" in k else t
    return out


def _window_destagger(u, v, w, u10, v10):
    """训练窗口口径 destagger(与 physics_canvas.destagger_canvas 一致):(…,95,111)。

    注意与评估侧 agl_eval_common.destagger_canvas 的区别:后者按整画布 (100,121)
    的 99/120 行/列切片,对窗口输入尺寸不匹配;两者在 crop 全落原生区内时逐点相同
    (由 ⑥c 端到端验证)。
    """
    um = 0.5 * (u[..., :, :-1] + u[..., :, 1:])[..., :-1, :]
    vm = 0.5 * (v[..., :-1, :] + v[..., 1:, :])[..., :, :-1]
    return um, vm, w[..., :-1, :-1], u10[..., :-1, :-1], v10[..., :-1, :-1]


def _agl_raw_numpy(err_canvas, agl):
    """公共算子复算 AGL 项:60 号 numpy 插值 + 训练窗口 destagger,逐样本后取批均值。"""
    np_interp = _load_numpy_agl_interp()
    total = 0.0
    for i in range(err_canvas.shape[0]):
        u, v, w, u10, v10 = split_canvas_state(
            torch.from_numpy(err_canvas[i])[None], AGL_NLEV)
        u, v, w, u10, v10 = (x[0].numpy() for x in (u, v, w, u10, v10))
        um, vm, wm, u10m, v10m = _window_destagger(u, v, w, u10, v10)
        hh, ww = um.shape[-2], um.shape[-1]
        au = np_interp(um, agl["idx_m"][i][:, :hh, :ww].numpy().astype(np.int64),
                       agl["w_m"][i][:, :hh, :ww].numpy(), field10=u10m)
        av = np_interp(vm, agl["idx_m"][i][:, :hh, :ww].numpy().astype(np.int64),
                       agl["w_m"][i][:, :hh, :ww].numpy(), field10=v10m)
        aw = np_interp(wm, agl["idx_i"][i][:, :hh, :ww].numpy().astype(np.int64),
                       agl["w_i"][i][:, :hh, :ww].numpy(), field10=None)
        total += (np.abs(au).mean() + np.abs(av).mean()
                  + 0.5 * np.abs(aw).mean()) / 2.5       # agl_w_weight=0.5 口径
    return total / err_canvas.shape[0]


def _agl_replay(si, y0, y1, y_cond, b_est_seed):
    """按 forward 的随机数消耗顺序复放同一次 (timestep, noise),取 b_est/b_true。

    前提:timestep 已被 force_timestep 钉死(_sample_timestep 不消耗 RNG),
    于是 seed 后的第一个 randn_like 与 forward 里那次一致。
    """
    torch.manual_seed(b_est_seed)
    y0r, y1r = y0, y1
    if si.c.residual_output:
        y1r, y0r = y1 - y0, torch.zeros_like(y0)
    timestep, t = si._sample_timestep(y0.shape[0])
    noise = torch.randn_like(y0r)
    yt = si._sample_yt(y0=y0r, y1=y1r, noise=noise, timestep=timestep)
    b_true = si._calc_b_true(y0=y0r, y1=y1r, noise=noise, timestep=timestep)
    b_est = si.net(yt=yt, y_cond=y_cond, gamma=t)
    return b_est, b_true


def test_agl_operator_equivalence():
    print("⑥a 可微算子 ↔ numpy 算子等价(idx 覆盖 -2/-1/0..21,field10 有无)")
    np_interp = _load_numpy_agl_interp()
    rng = np.random.default_rng(11)
    b, ny, nx = 2, 8, 9
    field = rng.standard_normal((b, AGL_NLEV, ny, nx)).astype(np.float32)
    field10 = rng.standard_normal((b, ny, nx)).astype(np.float32)
    vals = np.array([-2, -1] + list(range(AGL_NLEV - 1)))          # -2,-1,0..21
    flat = np.tile(vals, int(np.ceil(b * AGL_NT * ny * nx / float(vals.size))))
    flat = flat[:b * AGL_NT * ny * nx].copy()
    rng.shuffle(flat)
    idx = flat.reshape(b, AGL_NT, ny, nx).astype(np.int64)
    w = (0.05 + 0.9 * rng.random((b, AGL_NT, ny, nx))).astype(np.float32)
    check("合成 idx 全覆盖 -2/-1/0..21", sorted(np.unique(idx).tolist())
          == sorted(vals.tolist()))
    field_t = torch.from_numpy(field)
    idx_t = torch.from_numpy(idx)
    w_t = torch.from_numpy(w)
    for name, f10, f10_t in (("field10 有", field10, torch.from_numpy(field10)),
                             ("field10 无", None, None)):
        got = agl_interp_batched(field_t, idx_t, w_t, f10_t).numpy()
        err = 0.0
        for i in range(b):
            ref = np_interp(field[i], idx[i], w[i],
                            field10[i] if f10 is not None else None)
            err = max(err, float(np.abs(got[i] - ref).max()))
        check("算子等价({})".format(name), err < 1e-5, "max|Δ|={:.3e}".format(err))


def test_agl_analytic():
    print("⑥b 解析自检(ln z 线性场精确还原 / -2 与 -1 分支 torch↔numpy)")
    np_interp = _load_numpy_agl_interp()
    a, c = 3.0, 2.0
    z_m, z_i = _synth_zagl()
    u = (a + c * np.log(np.maximum(z_m, 1e-3))).astype(np.float32)       # (23,99,120)
    wf = (a + c * np.log(np.maximum(z_i, 1e-3))).astype(np.float32)      # (24,99,120)
    u10 = np.full((99, 120), a + c * np.log(10.0), dtype=np.float32)
    tabs = build_agl_tables_native(z_m, z_i, list(range(AGL_NLEV)))
    analytic = np.broadcast_to((a + c * np.log(TARGET_AGL))[:, None, None], (AGL_NT, 99, 120))
    m_ge = tabs["idx_m"] >= 0
    m_1 = tabs["idx_m"] == -1
    m_2 = tabs["idx_m"] == -2
    check("质量表覆盖 -2/-1/≥0 三类分支",
          bool(m_2.any()) and bool(m_1.any()) and bool(m_ge.any()),
          "-2={} -1={} ≥0={}".format(int(m_2.sum()), int(m_1.sum()), int(m_ge.sum())))

    au_t = agl_interp_batched(
        torch.from_numpy(u)[None], torch.from_numpy(tabs["idx_m"])[None].long(),
        torch.from_numpy(tabs["w_m"])[None], torch.from_numpy(u10)[None])[0].numpy()
    au_n = np_interp(u, tabs["idx_m"], tabs["w_m"], field10=u10)
    e_t = float(np.abs(au_t[m_ge] - analytic[m_ge]).max())
    e_n = float(np.abs(au_n[m_ge] - analytic[m_ge]).max())
    check("ln z 线性场 idx≥0 精确还原(torch/numpy)", e_t < 1e-5 and e_n < 1e-5,
          "torch {:.2e} numpy {:.2e}".format(e_t, e_n))
    e_2 = max(float(np.abs(au_t[m_2] - np.broadcast_to(u10, au_t.shape)[m_2]).max()),
              float(np.abs(au_n[m_2] - np.broadcast_to(u10, au_t.shape)[m_2]).max()))
    check("idx==-2 直接取 10 m 通道(torch/numpy 一致)", e_2 < 1e-5,
          "max|Δ|={:.3e}".format(e_2))
    e_1 = float(np.abs(au_t[m_1] - au_n[m_1]).max())
    check("idx==-1 分支 torch↔numpy 逐点一致", e_1 < 1e-5,
          "max|Δ|={:.3e}".format(e_1))
    print("   (记录)-1 分支与解析值偏差 {:.3e}(评估侧权重口径,仅 -1 列)".format(
        float(np.abs(au_n[m_1] - analytic[m_1]).max())))
    # 界面层(W)表:无 -1/-2;目标落在界面范围内(不触发首/末层钳制)时精确还原
    aw_t = agl_interp_batched(
        torch.from_numpy(wf)[None], torch.from_numpy(tabs["idx_i"])[None].long(),
        torch.from_numpy(tabs["w_i"])[None], None)[0].numpy()
    aw_n = np_interp(wf, tabs["idx_i"], tabs["w_i"], field10=None)
    inside = ((z_i[0][None] <= TARGET_AGL[:, None, None])
              & (TARGET_AGL[:, None, None] <= z_i[-1][None]))
    e_w = max(float(np.abs(aw_t[inside] - analytic[inside]).max()),
              float(np.abs(aw_n[inside] - analytic[inside]).max()),
              float(np.abs(aw_t - aw_n).max()))
    check("界面层表(区间内)精确还原 + torch↔numpy", e_w < 1e-5,
          "max|Δ|={:.3e}(区间外为首/末层钳制,按设计不外推)".format(e_w))


def test_agl_crop_alignment():
    print("⑥c crop 对齐端到端(训练路 place→crop→split/destagger→可微插值 vs 评估路)")
    rng = np.random.default_rng(2026)
    u_nat = rng.standard_normal((AGL_NLEV, 99, 121)).astype(np.float32)
    v_nat = rng.standard_normal((AGL_NLEV, 100, 120)).astype(np.float32)
    w_nat = rng.standard_normal((AGL_NLEV + 1, 99, 120)).astype(np.float32)
    u10 = rng.standard_normal((99, 120)).astype(np.float32)
    v10 = rng.standard_normal((99, 120)).astype(np.float32)
    z_m, z_i = _synth_zagl()
    tabs = build_agl_tables_native(z_m, z_i, list(range(AGL_NLEV)))
    stack = np.concatenate([CanvasStatics.place(u_nat), CanvasStatics.place(v_nat),
                            CanvasStatics.place(w_nat), CanvasStatics.place(u10),
                            CanvasStatics.place(v10)], axis=0)        # (72,100,121)
    tabs_c = {k: CanvasStatics.place(v) for k, v in tabs.items()}     # (11,100,121)
    eval_destagger = _eval_destagger_canvas()
    um_e, vm_e, wm_e, u10_e, v10_e = eval_destagger(u_nat, v_nat, w_nat, u10, v10)
    np_interp = _load_numpy_agl_interp()
    au_e = np_interp(um_e, tabs["idx_m"], tabs["w_m"], field10=u10_e)
    av_e = np_interp(vm_e, tabs["idx_m"], tabs["w_m"], field10=v10_e)
    aw_e = np_interp(wm_e, tabs["idx_i"], tabs["w_i"], field10=None)
    for h, x0 in ((0, 0), (3, 8)):       # dataset 随机 crop 的边界两档(全在原生区内)
        crop = torch.from_numpy(np.ascontiguousarray(stack[:, h:h + 96, x0:x0 + 112]))[None]
        u_c, v_c, w_c, u10_c, v10_c = split_canvas_state(crop, AGL_NLEV)
        um, vm, wm = destagger_canvas(u_c, v_c, w_c)                  # (1,23,95,111)

        def got(k):
            arr = np.ascontiguousarray(
                tabs_c[k][:, h:h + 96, x0:x0 + 112][:, :95, :111])
            t = torch.from_numpy(arr)[None]
            return t.long() if "idx" in k else t

        au_t = agl_interp_batched(um, got("idx_m"), got("w_m"), u10_c[..., :-1, :-1])
        av_t = agl_interp_batched(vm, got("idx_m"), got("w_m"), v10_c[..., :-1, :-1])
        aw_t = agl_interp_batched(wm, got("idx_i"), got("w_i"), None)
        d = max(
            float((au_t[0] - torch.from_numpy(
                au_e[:, h:h + 95, x0:x0 + 111].copy())).abs().max()),
            float((av_t[0] - torch.from_numpy(
                av_e[:, h:h + 95, x0:x0 + 111].copy())).abs().max()),
            float((aw_t[0] - torch.from_numpy(
                aw_e[:, h:h + 95, x0:x0 + 111].copy())).abs().max()))
        check("crop({},{}) 两路 AGL 场一致".format(h, x0), d < 1e-5,
              "max|Δ|={:.3e}".format(d))
        ok_tab = all(np.array_equal(
            tabs_c[k][:, h:h + 96, x0:x0 + 112][:, :95, :111],
            tabs[k][:, h:h + 95, x0:x0 + 111]) for k in tabs)
        check("crop({},{}) 表同 crop 逐元素相等".format(h, x0), ok_tab)


def test_agl_integration():
    print("⑥d AGL 集成(parts 复算 / replace-add 恒等式 / 梯度 / weight=0 逐位)")
    y0, y1, y_cond, rho = make_batch(0)
    agl = _synth_agl_batch()
    ps = torch.tensor(phys_scale(), dtype=torch.float32).view(1, -1, 1, 1)
    w = dict(div=0.7, vort=0.3, spectral=0.2, extreme=0.1)

    # replace 模式:total = w·agl + Σ 物理项(数据项被替换掉)
    si_r = make_si(agl_weight=1.0, agl_replace_data=True, agl_w_weight=0.5)
    force_timestep(si_r, 5)
    torch.manual_seed(101)
    out_r = si_r(y0=y0, y1=y1, y_cond=y_cond, rho=rho, agl=agl, return_parts=True)
    check("AGL 项有限且>0", torch.isfinite(out_r["agl"]) and float(out_r["agl"]) > 0,
          "agl={:.6f}".format(float(out_r["agl"])))
    b_est, b_true = _agl_replay(si_r, y0, y1, y_cond, 101)
    ref = _agl_raw_numpy(((b_est - b_true) * ps).detach().numpy(), agl)
    check("parts['agl'] ≈ 公共算子复算(rel 1e-4)",
          abs(float(out_r["agl"]) - ref) <= 1e-4 * max(abs(ref), 1e-6),
          "{:.6f} vs {:.6f}".format(float(out_r["agl"]), ref))
    lhs = float(out_r["total"])
    rhs_r = float(out_r["agl"]) + sum(w[k] * float(out_r[k]) for k in w)
    check("replace: total == w·agl + Σ w_i·part_i", abs(lhs - rhs_r) <= 1e-6 * lhs,
          "{} vs {}".format(repr(lhs), repr(rhs_r)))
    check("replace 模式 data 项仍照常返回",
          torch.isfinite(out_r["data"]) and float(out_r["data"]) > 0)

    # add 模式:total = data + w·agl + Σ 物理项
    si_a = make_si(agl_weight=1.0, agl_replace_data=False, agl_w_weight=0.5)
    force_timestep(si_a, 5)
    torch.manual_seed(101)
    out_a = si_a(y0=y0, y1=y1, y_cond=y_cond, rho=rho, agl=agl, return_parts=True)
    lhs_a = float(out_a["total"])
    rhs_a = float(out_a["data"]) + float(out_a["agl"]) \
        + sum(w[k] * float(out_a[k]) for k in w)
    check("add: total == data + w·agl + Σ w_i·part_i", abs(lhs_a - rhs_a) <= 1e-6 * lhs_a,
          "{} vs {}".format(repr(lhs_a), repr(rhs_a)))

    # 梯度:total 对 net 输出有限非零(replace 模式)
    si_g = make_si(agl_weight=1.0, agl_replace_data=True, agl_w_weight=0.5)
    force_timestep(si_g, 5)
    torch.manual_seed(102)
    out_g = si_g(y0=y0, y1=y1, y_cond=y_cond, rho=rho, agl=agl)
    out_g.backward()
    g = si_g.net.lin.weight.grad
    check("total 梯度有限非零", g is not None and bool(torch.isfinite(g).all())
          and float(g.abs().sum()) > 0,
          "|g|1={:.4g}".format(float(g.abs().sum()) if g is not None else -1.0))

    # agl_weight=0:传表与不传表 total 逐位相同(旧行为 bit-identical)
    si_z = make_si(agl_weight=0.0)
    force_timestep(si_z, 5)
    torch.manual_seed(103)
    t_no = float(si_z(y0=y0, y1=y1, y_cond=y_cond, rho=rho))
    torch.manual_seed(103)
    t_yes = float(si_z(y0=y0, y1=y1, y_cond=y_cond, rho=rho, agl=agl))
    check("agl_weight=0 传表/不传表 total 逐位相同", t_no == t_yes,
          "{} vs {}".format(repr(t_no), repr(t_yes)))

    # 报错路径:agl_weight>0 未传表
    try:
        torch.manual_seed(104)
        si_z2 = make_si(agl_weight=1.0)
        force_timestep(si_z2, 5)
        si_z2(y0=y0, y1=y1, y_cond=y_cond, rho=rho)
    except ValueError as e:
        check("agl_weight>0 缺 agl 表 -> ValueError", True, "-> {}".format(str(e)[:60]))
    else:
        check("agl_weight>0 缺 agl 表 -> ValueError", False, "未抛 ValueError")


def main():
    ap = argparse.ArgumentParser(
        description="阶段 2 损失链合成自检(canvas 四物理项 + 旧行为回归,秒级)")
    ap.parse_args()
    print("=" * 72)
    test_physics_canvas()
    test_baseline_regression()
    test_new_parts()
    test_gradients()
    test_old_ymls()
    test_agl_operator_equivalence()
    test_agl_analytic()
    test_agl_crop_alignment()
    test_agl_integration()
    print("=" * 72)
    print("检查项:通过 {} / 失败 {}".format(N_OK[0], N_FAIL[0]))
    if N_FAIL[0]:
        print("PHASE2 LOSS FIXTURE FAILED")
        sys.exit(1)
    print("PHASE2 LOSS FIXTURE OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
