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
  ⑤ 旧 yml(phase1 / phase1r V*)经 load_config 可构造(环境不支持时跳过)。

运行(仓库根目录,torch 环境;不依赖服务器数据):
  python scripts/outline/95_phase2_loss_fixture.py
退出码 0 = 全部通过,1 = 有失败项。
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.dl_model.si_follmer import physics_canvas  # noqa: E402
from src.dl_model.si_follmer.si_follmer_framework import (  # noqa: E402
    SIFollmerConfig, StochasticInterpolantFollmer)

# 画布窗口与层数(canvas_resid 等基准的原始形状)
B, C, L = 2, 72, 23
H, W = 96, 112
DX = 1000.0

# 任务 A 改造前由 /tmp/si_baseline_capture.py 捕获的"权重全 0 旧行为"数值
# (同一机器/同一 torch 上应逐位复现,tol=0;若不一致说明旧路径被改动了)
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
    print("② 旧行为逐位回归(权重全 0,tol=0)")
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
        check("{} 逐位一致".format(k), got[k] == BASELINE[k],
              "{} vs {}".format(repr(got[k]), repr(BASELINE[k])))


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
    print("=" * 72)
    print("检查项:通过 {} / 失败 {}".format(N_OK[0], N_FAIL[0]))
    if N_FAIL[0]:
        print("PHASE2 LOSS FIXTURE FAILED")
        sys.exit(1)
    print("PHASE2 LOSS FIXTURE OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
