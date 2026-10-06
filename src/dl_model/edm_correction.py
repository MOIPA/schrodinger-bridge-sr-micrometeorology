"""EDM 订正步(Karras et al. 2022 口径,自实现,不依赖 PhysicsNeMo)。

阶段 3 架构对比「回归 + 扩散两步法」的第二步:
- 步 1(冻结的回归网,checkpoint 与 SI 训练脚本同构):
      ŷ1_reg = y0 + net_reg(yt=y0, y_cond=x, gamma=ones)
- 步 2(本模块):在残差空间学去噪器
      r = y1 − ŷ1_reg,  r_σ = r + σ·ε,  D(r_σ; σ, ctx) ≈ r
      ctx = concat([ŷ1_reg, y0, y_cond])(当前配置 72 + 72 + 95 = 239 通道)
- 采样(Karras Heun,ρ=7,默认 24 步)从 N(0, σ_max²) 出发积分到 0,
  最终样本 = ŷ1_reg + r̂0。

预条件(论文表 1):
    c_skip  = σ_d² / (σ² + σ_d²)
    c_out   = σ·σ_d / √(σ_d² + σ²)
    c_in    = 1 / √(σ_d² + σ²)
    c_noise = ln(σ) / 4
    D = c_skip·r_σ + c_out·F(c_in·r_σ, c_noise, ctx)
训练损失 λ(σ)·‖D − r‖²,λ = 1/c_out²(等价于对网络输出做 (r−c_skip·r_σ)/c_out 的回归);
训练 σ 分布:ln σ ~ N(p_mean, p_std²),σ ∈ [σ_min, σ_max]。

σ_d(σ_data)由训练脚本在冻结回归网上按前 N 个 batch 的残差尺度估计(std 或
mean-abs),训练/评估必须复用同一数值:训练脚本把它写进 checkpoint 的
'sigma_data'/'edm_config' 键与结果目录 config_resolved.yml。

与评估链路(84 集合评估)的接口(cfg 来自训练产物 config_resolved.yml 的 edm 段,
或 checkpoint 的 'edm_config' 键;两者都带训练时用的 σ_d):
    cfg = EDMCorrectorConfig(**resolved["edm"])
    corrector = EDMCorrector(cfg).eval()
    corrector.net.load_state_dict(ckpt['model_state_dict'])
    out = corrector.sample(y0, y1_reg, y_cond, n_samples=16, steps=cfg.steps, seed=0)
    # out: (n_samples, B, out_channel, H, W),内部按样本循环控显存
"""

import dataclasses
import math
import typing
from logging import getLogger

import torch
import torch.nn.functional as F
from src.dl_config.base_config import YamlConfig
from src.dl_model.ddpm.blocks import gamma_embedding
from src.dl_model.util import initialize_to_zero
from torch import nn

logger = getLogger()


@dataclasses.dataclass()
class EDMCorrectorConfig(YamlConfig):
    """EDM 订正步配置(dataclass 口径与 SIFollmerConfig 相同,可写进 yml `edm:` 段)。

    通道口径:网络输入 = out_channel(带噪残差)+ ctx_channel(上下文),
    ctx_channel = out_channel(ŷ1_reg)+ out_channel(y0)+ 条件通道数(当前 72+72+95=239)。
    """

    out_channel: int = 72
    ctx_channel: int = 239
    inner_channel: int = 64
    channel_mults: typing.List[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 4])
    blocks_per_level: int = 1
    dropout: float = 0.0
    max_period: float = 10.0

    # --- σ_d:残差尺度,None 表示未定(训练脚本估计后填入/或用 --sigma_d 指定)---
    sigma_data: typing.Optional[float] = None

    # --- 训练 σ 分布与 Karras 常量 ---
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2

    # --- Heun 采样 ---
    rho: float = 7.0
    steps: int = 24

    # --- σ_d 估计(训练脚本用)---
    sigma_d_estimator: str = "std"  # "std" | "mean_abs"
    sigma_d_est_batches: int = 8

    def __post_init__(self):
        # yml 里 sigma_data 常写成 null(YAML None),统一转 float 或 None
        if self.sigma_data is not None:
            self.sigma_data = float(self.sigma_data)
        assert self.sigma_min > 0 and self.sigma_max > self.sigma_min
        assert self.rho > 0 and self.steps >= 2
        assert self.p_std > 0


def sample_log_sigma(
    batch_size: int,
    config: EDMCorrectorConfig,
    device: typing.Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: typing.Optional[torch.Generator] = None,
) -> torch.Tensor:
    """训练用 σ 采样:ln σ ~ N(p_mean, p_std²),截断到 [σ_min, σ_max]。

    在 CPU 上抽 (B,) 的标量再搬到 device(便于用固定 generator 做可复现的 valid)。
    """
    z = torch.randn(batch_size, generator=generator, dtype=dtype)
    sigma = torch.exp(z * config.p_std + config.p_mean)
    sigma = sigma.clamp(min=config.sigma_min, max=config.sigma_max)
    return sigma if device is None else sigma.to(device)


def karras_sigma_schedule(
    steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: typing.Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Karras 噪声调度,返回 steps+1 个递减值:[σ_max=σ0, ..., σ_min, 0]。

    前 steps 个按 ρ 幂律插值;末尾 0 是 EDM 采样器最后一步的 Euler 落点
    (参考实现 t_steps = cat([schedule, 0]) 的口径)。
    """
    if steps < 2:
        raise ValueError(f"steps 至少为 2(当前 {steps})")
    idx = torch.arange(steps, device=device, dtype=torch.float64)
    inv_max = sigma_max ** (1.0 / rho)
    inv_min = sigma_min ** (1.0 / rho)
    t = (inv_max + idx / (steps - 1) * (inv_min - inv_max)) ** rho
    t = torch.cat([t, torch.zeros(1, device=device, dtype=torch.float64)])
    return t.to(dtype)


def _group_count(channels: int) -> int:
    """GroupNorm 组数 = min(8, channels) 的可整除口径(小配置 inner=16 也能用)。"""
    return math.gcd(int(channels), 8)


class _ResBlock(nn.Module):
    """紧凑 ResBlock:adaGN 尺度/平移(σ 嵌入)+ 残差,second conv 零初始化。"""

    def __init__(self, in_channel: int, out_channel: int, emb_dim: int,
                 dropout: float = 0.0, down: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channel), in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3,
                               stride=2 if down else 1, padding=1)
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_dim, 2 * out_channel))
        self.norm2 = nn.GroupNorm(_group_count(out_channel), out_channel)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = initialize_to_zero(
            nn.Conv2d(out_channel, out_channel, 3, padding=1))
        if down or in_channel != out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, 1,
                                  stride=2 if down else 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # σ 嵌入在第一次卷积后调制(与 ddpm.blocks.ResBlock2DEmb 同序,兼容 in≠out)
        scale, shift = self.emb_layers(emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.skip(x) + h


class EDMDenoiserNet(nn.Module):
    """紧凑 UNet 风格去噪器(纯 torch,conv 编解码 + stride-2 下采样)。

    forward(x, c_noise) -> F:
      x       = concat([c_in·r_σ, ctx]),(B, out_channel + ctx_channel, H, W)
      c_noise = ln(σ)/4,(B,),经 gamma_embedding(同 UNet 的正弦嵌入)+ MLP 成条件向量
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        inner_channel: int = 64,
        channel_mults: typing.Sequence[int] = (1, 2, 4),
        blocks_per_level: int = 1,
        dropout: float = 0.0,
        max_period: float = 10.0,
    ):
        super().__init__()
        assert len(channel_mults) >= 2, "至少 2 个层级(需要一次下采样)"
        self.inner_channel = int(inner_channel)
        self.max_period = float(max_period)
        chs = [int(inner_channel * m) for m in channel_mults]
        emb_dim = 2 * self.inner_channel

        self.emb_mlp = nn.Sequential(
            nn.Linear(self.inner_channel, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.stem = nn.Conv2d(in_channel, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList([
            nn.ModuleList([
                _ResBlock(chs[i], chs[i], emb_dim, dropout)
                for _ in range(blocks_per_level)
            ]) for i in range(len(chs))
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1)
            for i in range(len(chs) - 1)
        ])
        self.mid = _ResBlock(chs[-1], chs[-1], emb_dim, dropout)
        self.up_convs = nn.ModuleList([
            nn.Conv2d(chs[i + 1], chs[i], 3, padding=1)
            for i in range(len(chs) - 1)
        ])
        self.dec_blocks = nn.ModuleList([
            _ResBlock(2 * chs[i], chs[i], emb_dim, dropout)
            for i in range(len(chs) - 1)
        ])
        self.head = nn.Sequential(
            nn.GroupNorm(_group_count(chs[0]), chs[0]),
            nn.SiLU(),
            initialize_to_zero(nn.Conv2d(chs[0], out_channel, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        c_noise = c_noise.reshape(-1)
        emb = self.emb_mlp(
            gamma_embedding(c_noise.float(), self.inner_channel, self.max_period))

        h = self.stem(x)
        skips = []
        for level, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                h = block(h, emb)
            if level < len(self.downs):
                skips.append(h)
                h = self.downs[level](h)
        h = self.mid(h, emb)
        for level in reversed(range(len(self.downs))):
            skip = skips[level]
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = self.up_convs[level](h)
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[level](h, emb)
        return self.head(h).to(torch.float32)


class EDMCorrector(nn.Module):
    """EDM 去噪器 + Karras 预条件 + Heun 采样 + 训练损失(见模块 docstring)。

    可训练参数全部在 `self.net`(EDMDenoiserNet)里;checkpoint 按项目口径存
    `net.state_dict()`(裸键,与 SI 脚本的 'model_state_dict' 一致)。
    """

    def __init__(self, config: EDMCorrectorConfig,
                 net: typing.Optional[nn.Module] = None):
        super().__init__()
        self.c = config
        if config.sigma_data is None or float(config.sigma_data) <= 0:
            raise ValueError(
                "EDMCorrectorConfig.sigma_data 未设置(须先用训练脚本估计或显式指定)")
        self.sigma_d = float(config.sigma_data)
        if net is None:
            net = EDMDenoiserNet(
                in_channel=config.out_channel + config.ctx_channel,
                out_channel=config.out_channel,
                inner_channel=config.inner_channel,
                channel_mults=config.channel_mults,
                blocks_per_level=config.blocks_per_level,
                dropout=config.dropout,
                max_period=config.max_period,
            )
        self.net = net

    # ------------------------------------------------------------------ 预条件
    def precondition(
        self,
        sigma: typing.Union[float, torch.Tensor],
        dtype: torch.dtype = torch.float32,
        device: typing.Optional[torch.device] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (c_skip, c_out, c_in, c_noise);前三个形状 (B,1,1,1),c_noise (B,)。"""
        sigma = torch.as_tensor(sigma, dtype=dtype, device=device).reshape(-1)
        sd2 = self.sigma_d ** 2
        sig2 = sigma * sigma
        denom = sig2 + sd2
        c_skip = sd2 / denom
        c_out = sigma * self.sigma_d / torch.sqrt(denom)
        c_in = 1.0 / torch.sqrt(denom)
        c_noise = torch.log(sigma) / 4.0
        view = (-1, 1, 1, 1)
        return (c_skip.reshape(view), c_out.reshape(view), c_in.reshape(view), c_noise)

    def denoise(self, r_sigma: torch.Tensor,
                sigma: typing.Union[float, torch.Tensor],
                ctx: torch.Tensor) -> torch.Tensor:
        """D(r_σ; σ, ctx) = c_skip·r_σ + c_out·F(c_in·r_σ, c_noise, ctx)。"""
        c_skip, c_out, c_in, c_noise = self.precondition(
            sigma, dtype=r_sigma.dtype, device=r_sigma.device)
        net_in = torch.cat([c_in * r_sigma, ctx], dim=1)
        return c_skip * r_sigma + c_out * self.net(net_in, c_noise)

    # ------------------------------------------------------------------ 损失
    def loss(self, r: torch.Tensor, sigma: torch.Tensor, ctx: torch.Tensor,
             noise: typing.Optional[torch.Tensor] = None,
             weight: bool = True) -> torch.Tensor:
        """EDM 训练损失:λ(σ)·‖D(r_σ) − r‖²,λ = 1/c_out²。

        r: 干净残差 y1−ŷ1_reg (B,C,H,W);sigma: (B,) 或 (B,1,1,1);
        noise: 可传入固定噪声(valid 可复现);weight=False 时退化为未加权 MSE。
        """
        if noise is None:
            noise = torch.randn_like(r)
        _, c_out, _, _ = self.precondition(
            sigma, dtype=r.dtype, device=r.device)
        r_sigma = r + sigma.reshape(-1, 1, 1, 1) * noise
        d = self.denoise(r_sigma, sigma, ctx)
        err = (d - r).pow(2).flatten(1).mean(dim=1)
        if weight:
            err = err / c_out.reshape(-1).pow(2)
        return err.mean()

    # ------------------------------------------------------------------ 采样
    def _device_dtype(self) -> typing.Tuple[torch.device, torch.dtype]:
        p = next(self.parameters(), None)
        if p is None:  # 解析构造(自检/玩具)可能没有参数
            return torch.device("cpu"), torch.float32
        return p.device, p.dtype

    def _ode_derivative(self, x: torch.Tensor, sigma: torch.Tensor,
                        ctx: torch.Tensor) -> torch.Tensor:
        d = self.denoise(x, sigma, ctx)
        return (x - d) / sigma

    def _heun_sample(self, r_init: torch.Tensor, ctx: torch.Tensor,
                     sigmas: torch.Tensor) -> torch.Tensor:
        """Karras Heun:除最后一步(σ_next=0)外都做二阶校正。"""
        x = r_init
        for i in range(sigmas.shape[0] - 1):
            s_cur, s_next = sigmas[i], sigmas[i + 1]
            d_cur = self._ode_derivative(x, s_cur, ctx)
            x_next = x + d_cur * (s_next - s_cur)
            if s_next > 0:
                d_next = self._ode_derivative(x_next, s_next, ctx)
                x_next = x + 0.5 * (d_cur + d_next) * (s_next - s_cur)
            x = x_next
        return x

    @torch.no_grad()
    def sample(self, y0: torch.Tensor, y1_reg: torch.Tensor, y_cond: torch.Tensor,
               n_samples: int = 1, steps: int = 24,
               seed: typing.Optional[int] = None) -> torch.Tensor:
        """返回 (n_samples, B, out_channel, H, W);内部按样本循环以控显存。

        y1_reg = y0 + net_reg(yt=y0, y_cond=x, gamma=1)(回归步输出,冻结);
        seed 给定时用局部 CPU generator(不扰动全局 RNG),同 seed 逐位可复现。
        """
        device, dtype = self._device_dtype()
        y0 = y0.to(device=device, dtype=dtype)
        y1_reg = y1_reg.to(device=device, dtype=dtype)
        y_cond = y_cond.to(device=device, dtype=dtype)
        assert y0.shape == y1_reg.shape, "y0 与 y1_reg 形状必须一致"
        assert y_cond.dim() == 4 and y_cond.shape[0] == y0.shape[0]
        ctx = torch.cat([y1_reg, y0, y_cond], dim=1)
        if ctx.shape[1] != self.c.ctx_channel:
            raise ValueError(
                f"ctx 通道 {ctx.shape[1]} != config.ctx_channel {self.c.ctx_channel}")

        sigmas = karras_sigma_schedule(
            steps, self.c.sigma_min, self.c.sigma_max, self.c.rho,
            device=device, dtype=dtype)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        outs = []
        for _ in range(int(n_samples)):
            r = torch.randn(y0.shape, generator=generator, dtype=torch.float32)
            r = r.to(device=device, dtype=dtype) * self.c.sigma_max
            r = self._heun_sample(r, ctx, sigmas)
            outs.append(y1_reg + r)
        return torch.stack(outs, dim=0)
