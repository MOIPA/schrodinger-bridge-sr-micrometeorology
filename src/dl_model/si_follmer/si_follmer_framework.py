import copy
import dataclasses
import sys
import typing
import warnings
from functools import partial
from logging import getLogger
from typing import Literal, Optional, Union

import numpy as np
import torch
from src.dl_config.base_config import YamlConfig
from src.dl_model.si_follmer.physics_canvas import (
    destagger_canvas,
    divergence_rho_u,
    estimate_residual,
    infer_n_levels,
    radial_log_spectra,
    split_canvas_state,
    vorticity,
    windspeed_levels,
)
from torch import nn

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def _make_time_alpha_beta_sigma_gF_A_for_linear(n_timestep: int, eps: float):
    #
    t = np.linspace(0.0, 1.0, n_timestep, endpoint=True, dtype=np.float128)
    #
    alpha = (1.0 - t).astype(np.float64)
    beta = (t).astype(np.float64)
    sigma = (eps * (1.0 - t)).astype(np.float64)
    gF = (eps * np.sqrt((1.0 - t) * (1.0 + t))).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (1.0 / (eps**2 * t * (1.0 - t))).astype(np.float64)
        # A becomes inf when t == 0 or 1, but this value is NOT used in the calculation.
        # To notice errors if A at t == 0 or 1 is used, we remain this inf value.
    t_sqrt = (np.sqrt(t)).astype(np.float64)
    #
    dot_alpha = -np.ones_like(alpha)
    dot_beta = np.ones_like(alpha)
    dot_sigma = -eps * np.ones_like(alpha)

    return (
        t.astype(np.float64),
        alpha,
        beta,
        sigma,
        gF,
        A,
        t_sqrt,
        dot_alpha,
        dot_beta,
        dot_sigma,
    )


def _make_time_alpha_beta_sigma_gF_A_for_quadratic(n_timestep: int, eps: float):
    #
    t = np.linspace(0.0, 1.0, n_timestep, endpoint=True, dtype=np.float64)
    #
    alpha = (1.0 - t).astype(np.float64)
    beta = (t**2).astype(np.float64)
    sigma = (eps * (1.0 - t)).astype(np.float64)
    gF = (eps * np.sqrt((3.0 - t) * (1.0 - t))).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (1.0 / (eps**2 * t**2 * (1.0 - t) * (2.0 - t))).astype(np.float64)
        # A becomes inf when t == 0 or 1, but this value is NOT used in the calculation.
        # To notice errors if A at t == 0 or 1 is used, we remain this inf value.
    t_sqrt = (np.sqrt(t)).astype(np.float64)
    #
    dot_alpha = -np.ones_like(alpha)
    dot_beta = (2.0 * t).astype(np.float64)
    dot_sigma = -eps * np.ones_like(alpha)

    return (
        t.astype(np.float64),
        alpha,
        beta,
        sigma,
        gF,
        A,
        t_sqrt,
        dot_alpha,
        dot_beta,
        dot_sigma,
    )


@dataclasses.dataclass()
class SIFollmerConfig(YamlConfig):
    n_timestep: int
    eps: float
    formula: Literal["linear", "quadratic"]
    loss_type: Literal["L2", "L1"] = "L2"
    channel_weights: typing.Optional[list] = None  # 通道加权，如 [1,1,10, 1,1,10, ...]
    divergence_weight: float = 0.0  # 散度约束权重（interleaved: ∂U/∂x+∂V/∂y+∂W/∂z=0；canvas: ∇·(ρu) hinge），0 表示不启用
    vorticity_weight: float = 0.0  # 涡度约束权重（interleaved: 压向 0；canvas: 与真值 ζ 的 L1 结构差），0 表示不启用
    residual_output: bool = False  # True: 桥接对象为残差 y−y0(从 0 出发),采样时再加回 y0
    state_layout: Literal["interleaved_uvw", "canvas"] = "interleaved_uvw"  # canvas 布局用阶段 2 原生 C 网格口径
    # ---- 阶段 2:canvas 布局物理损失(权重全为 0 时完全不参与计算) ----
    spectral_weight: float = 0.0  # 径向 log 谱 L2 差权重（仅 canvas）
    extreme_weight: float = 0.0  # 风速 max/min 极值结构差权重（仅 canvas）
    extreme_levels: typing.Optional[list] = None  # 极值层索引；None -> 最低 10 层 [0..9]
    phys_scale: typing.Optional[list] = None  # 长度 out_channel 的每通道物理 σ（标准化场乘回物理单位）
    phys_dz: typing.Optional[list] = None  # 长度 n_levels 的层厚 m（散度垂直差分）
    phys_dx: float = 1000.0  # 水平网格距 m（d04 1 km；真值侧 x/y 同用 dx）
    phys_min_t: float = 0.5  # 单步估计掩码阈值：仅用 dot_beta>=2*min_t（quadratic 下 t>=min_t）
    phys_div_tau: typing.Optional[list] = None  # 长度 n_levels 的散度 hinge 阈值 kg m^-3 s^-1（T0.5 定标）


class StochasticInterpolantFollmer(nn.Module):
    def __init__(
        self,
        config: SIFollmerConfig,
        neural_net: nn.Module,
        device: Union[None, str] = None,
    ):
        super().__init__()

        if device is None:
            d = next(neural_net.parameters()).device
            self.device = str(d)
        else:
            self.device = device

        self.c = copy.deepcopy(config)
        self.net = neural_net
        self.dtype = torch.float32
        self._set_alpha_beta_gamma()

        # 通道加权 (用于 W 分量加权等)
        if self.c.channel_weights is not None:
            w = torch.tensor(self.c.channel_weights, dtype=self.dtype, device=self.device)
            self.register_buffer("channel_weights", w.view(1, -1, 1, 1))
            logger.info(f"Channel weights enabled: {self.c.channel_weights}")
        else:
            self.channel_weights = None

        # 物理约束项配置校验:
        #   interleaved_uvw —— 旧三元组口径(压向 0),行为原样保留;
        #   canvas          —— 阶段 2 原生 C 网格口径(散度 hinge / 涡度结构差 / 谱 / 极值)。
        canvas_phys_weights = (
            self.c.divergence_weight,
            self.c.vorticity_weight,
            self.c.spectral_weight,
            self.c.extreme_weight,
        )
        if self.c.state_layout == "canvas":
            if any(w > 0 for w in canvas_phys_weights):
                # canvas 物理损失都在物理单位下定义,必须能反标准化
                if self.c.phys_scale is None:
                    raise ValueError(
                        "canvas 物理损失需要 si.phys_scale(长度=out_channel 的每通道物理 σ)"
                    )
                if self.c.divergence_weight > 0:
                    if self.c.phys_dz is None:
                        raise ValueError(
                            "canvas 散度损失需要 si.phys_dz(长度=n_levels 的层厚 m)")
                    if self.c.phys_div_tau is None:
                        raise ValueError(
                            "canvas 散度损失需要 si.phys_div_tau(长度=n_levels 的 hinge 阈值)")
                    if not self.c.phys_dx > 0:
                        raise ValueError(
                            f"canvas 散度损失需要 si.phys_dx>0,当前 {self.c.phys_dx}")
        else:
            # 旧交错布局没有谱/极值的定义,拒绝误用
            if self.c.spectral_weight > 0 or self.c.extreme_weight > 0:
                raise ValueError("谱/极值损失仅支持 state_layout=canvas")

        if self.c.state_layout == "canvas" and any(w > 0 for w in canvas_phys_weights):
            logger.info(
                "Canvas physics losses enabled: div=%g, vort=%g, spectral=%g, extreme=%g, "
                "dx=%g, min_t=%g, levels=%s",
                self.c.divergence_weight, self.c.vorticity_weight,
                self.c.spectral_weight, self.c.extreme_weight, self.c.phys_dx,
                self.c.phys_min_t,
                self.c.extreme_levels if self.c.extreme_levels is not None
                else list(range(10)),
            )
        elif self.c.state_layout != "canvas":
            # 三维散度物理约束 (∂U/∂x + ∂V/∂y + ∂W/∂z = 0)
            if self.c.divergence_weight > 0:
                logger.info(f"3D Divergence constraint enabled: weight={self.c.divergence_weight}")

            # 涡度约束 (大尺度涡度守恒)
            if self.c.vorticity_weight > 0:
                logger.info(f"Vorticity constraint enabled: weight={self.c.vorticity_weight}")

    def _set_alpha_beta_gamma(self):
        # Time index is from 0 to T (t is an N+1 size array)
        if self.c.formula == "linear":
            t, alpha, beta, sigma, gF, A, t_sqrt, dot_alpha, dot_beta, dot_sigma = (
                _make_time_alpha_beta_sigma_gF_A_for_linear(
                    n_timestep=self.c.n_timestep + 1, eps=self.c.eps
                )
            )
        elif self.c.formula == "quadratic":
            t, alpha, beta, sigma, gF, A, t_sqrt, dot_alpha, dot_beta, dot_sigma = (
                _make_time_alpha_beta_sigma_gF_A_for_quadratic(
                    n_timestep=self.c.n_timestep + 1, eps=self.c.eps
                )
            )
        else:
            raise ValueError(f"Unknown formula: {self.c.formula}")

        to_torch = partial(torch.tensor, dtype=self.dtype, device=self.device)

        self.register_buffer("time", to_torch(t))
        self.register_buffer("alpha", to_torch(alpha))
        self.register_buffer("beta", to_torch(beta))
        self.register_buffer("sigma", to_torch(sigma))
        self.register_buffer("gF", to_torch(gF))
        self.register_buffer("A", to_torch(A))
        self.register_buffer("time_sqrt", to_torch(t_sqrt))
        self.register_buffer("dot_alpha", to_torch(dot_alpha))
        self.register_buffer("dot_beta", to_torch(dot_beta))
        self.register_buffer("dot_sigma", to_torch(dot_sigma))

        self.dt = (1.0 / torch.tensor(self.c.n_timestep, dtype=torch.float64)).to(
            self.dtype
        )

        if self.c.formula == "linear":
            coeff1_bF = 1.0 + t
        elif self.c.formula == "quadratic":
            coeff1_bF = 1.0 + 1.0 / (2.0 - t)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeff2_bF = 1.0 / (t * (2.0 - t))
                # coeff2 becomes inf when t == 0, but this value is not used in the calculation
                # To notice errors when it is used, we remain inf here.
            coeff3_bF = 2.0 - t
            self.register_buffer("coeff2_bF", to_torch(coeff2_bF))
            self.register_buffer("coeff3_bF", to_torch(coeff3_bF))

        self.register_buffer("coeff1_bF", to_torch(coeff1_bF))

    def _sample_timestep(self, batch_size: int):
        # Time index here is from 0 to T
        timestep = torch.randint(
            0, self.c.n_timestep + 1, (batch_size,), device=self.device
        )
        timestep = timestep.to(torch.int64)  # array index needs to be int64 in PyTorch
        t = torch.gather(self.time, dim=-1, index=timestep)

        return timestep, t

    def _sample_yt(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ):
        # y0: LR data, dim = batch, channel, y, and x
        # y1: HR data, dim = batch, channel, y, and x
        # noise has the same shape as y0 and y1
        # timestep: indices, dim = batch

        a = torch.gather(self.alpha, dim=-1, index=timestep)[:, None, None, None]
        b = torch.gather(self.beta, dim=-1, index=timestep)[:, None, None, None]
        s = torch.gather(self.sigma, dim=-1, index=timestep)[:, None, None, None]
        t_sq = torch.gather(self.time_sqrt, dim=-1, index=timestep)[:, None, None, None]
        # Add channel, y, and x dims

        return a * y0 + b * y1 + s * t_sq * noise

    def _calc_b_true(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ):
        d_a = torch.gather(self.dot_alpha, dim=-1, index=timestep)[:, None, None, None]
        d_b = torch.gather(self.dot_beta, dim=-1, index=timestep)[:, None, None, None]
        d_s = torch.gather(self.dot_sigma, dim=-1, index=timestep)[:, None, None, None]
        t_sq = torch.gather(self.time_sqrt, dim=-1, index=timestep)[:, None, None, None]

        return d_a * y0 + d_b * y1 + d_s * t_sq * noise

    def _calc_bF(
        self,
        b: torch.Tensor,
        y0: torch.Tensor,
        yt: torch.Tensor,
        timestep: torch.Tensor,
    ):
        c1 = torch.gather(self.coeff1_bF, dim=-1, index=timestep)[:, None, None, None]

        if self.c.formula == "linear":
            bF = c1 * b - yt + y0
        elif self.c.formula == "quadratic":
            c2 = torch.gather(self.coeff2_bF, dim=-1, index=timestep)
            c2 = c2[:, None, None, None]
            c3 = torch.gather(self.coeff3_bF, dim=-1, index=timestep)
            c3 = c3[:, None, None, None]
            bF = c1 * b - c2 * (2.0 * yt - c3 * y0)
        else:
            raise ValueError(f"Unknown formula: {self.c.formula}")

        return bF

    def _calc_3d_divergence_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        三维散度约束：∂U/∂x + ∂V/∂y + ∂W/∂z = 0
        pred: [B, 18, H, W]，通道顺序为 [U0,V0,W0, U1,V1,W1, ..., U5,V5,W5]
        水平导数用中心差分，垂直导数用层间中心差分。
        """
        div_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        n_levels = pred.shape[1] // 3  # 6 层

        for i in range(n_levels):
            u = pred[:, i * 3, :, :]      # [B, H, W]
            v = pred[:, i * 3 + 1, :, :]  # [B, H, W]
            w = pred[:, i * 3 + 2, :, :]  # [B, H, W]

            # 水平中心差分
            du_dx = (u[:, :, 2:] - u[:, :, :-2]) / 2.0  # [B, H, W-2]
            dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / 2.0  # [B, H-2, W]

            # 垂直中心差分 ∂W/∂z
            if 0 < i < n_levels - 1:
                w_above = pred[:, (i + 1) * 3 + 2, :, :]
                w_below = pred[:, (i - 1) * 3 + 2, :, :]
                dw_dz = (w_above - w_below) / 2.0
            elif i == 0:
                w_above = pred[:, 1 * 3 + 2, :, :]
                dw_dz = w_above - w  # 底层前向差分
            else:
                w_below = pred[:, (i - 1) * 3 + 2, :, :]
                dw_dz = w - w_below  # 顶层后向差分

            # 对齐到公共区域 [B, H-2, W-2]
            du_dx_crop = du_dx[:, 1:-1, :]   # [B, H-2, W-2]
            dv_dy_crop = dv_dy[:, :, 1:-1]   # [B, H-2, W-2]
            dw_dz_crop = dw_dz[:, 1:-1, 1:-1]  # [B, H-2, W-2]

            divergence = du_dx_crop + dv_dy_crop + dw_dz_crop
            div_loss = div_loss + torch.mean(divergence ** 2)

        return div_loss / n_levels

    def _calc_vorticity_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        涡度约束：模型修正场不应引入虚假的大尺度涡度。
        pred: [B, 18, H, W]，通道顺序为 [U0,V0,W0, ...]
        对每层计算 ζ = ∂V/∂x - ∂U/∂y，经 avg_pool 提取大尺度分量后惩罚。
        物理依据：NS 方程的运动学部分——大尺度涡度在无粘条件下守恒。
        """
        vort_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        n_levels = pred.shape[1] // 3

        for i in range(n_levels):
            u = pred[:, i * 3, :, :]
            v = pred[:, i * 3 + 1, :, :]

            dv_dx = (v[:, :, 2:] - v[:, :, :-2]) / 2.0  # [B, H, W-2]
            du_dy = (u[:, 2:, :] - u[:, :-2, :]) / 2.0  # [B, H-2, W]

            dv_dx_crop = dv_dx[:, 1:-1, :]  # [B, H-2, W-2]
            du_dy_crop = du_dy[:, :, 1:-1]  # [B, H-2, W-2]

            zeta = dv_dx_crop - du_dy_crop  # [B, H-2, W-2]

            # avg_pool 提取大尺度涡度（kernel=4，抑制小尺度噪声）
            zeta_coarse = torch.nn.functional.avg_pool2d(
                zeta.unsqueeze(1), kernel_size=4
            ).squeeze(1)

            vort_loss = vort_loss + torch.mean(zeta_coarse ** 2)

        return vort_loss / n_levels

    def _canvas_phys_active(self) -> bool:
        """canvas 四个物理项是否有任一权重>0(全 0 时不做任何多余计算)。"""
        return (self.c.divergence_weight > 0 or self.c.vorticity_weight > 0
                or self.c.spectral_weight > 0 or self.c.extreme_weight > 0)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """逐样本值 (B,) 按 mask (B,) bool 平均;mask 全 False 时返回 0 张量(无 NaN)。"""
        mf = mask.to(dtype=values.dtype)
        return (values * mf).sum() / mf.sum().clamp(min=1.0)

    def _phys_vec(self, name: str, n_expected: int, device, dtype) -> torch.Tensor:
        """把 si.<name> 的列表配置转成 (n_expected,) 张量;缺失/长度不符时报错。"""
        seq = getattr(self.c, name)
        if seq is None:
            raise ValueError(f"canvas 物理损失需要 si.{name}")
        t = torch.as_tensor(seq, dtype=dtype, device=device)
        if t.numel() != n_expected:
            raise ValueError(f"si.{name} 元素数 {t.numel()} != 期望 {n_expected}")
        return t

    def _extreme_levels(self, n_levels: int) -> list:
        """极值项层索引;extreme_levels=None 时默认最低 10 层 [0..9](受层数截断)。"""
        levels = self.c.extreme_levels
        if levels is None:
            levels = list(range(min(10, n_levels)))
        levels = [int(l) for l in levels]
        if len(levels) == 0 or min(levels) < 0 or max(levels) >= n_levels:
            raise ValueError(
                f"si.extreme_levels={levels} 非法(需非空且落在 [0,{n_levels}))")
        return levels

    def _canvas_physics_raw(
        self,
        y0_orig: torch.Tensor,
        y1_orig: torch.Tensor,
        b_est: torch.Tensor,
        timestep: torch.Tensor,
        rho: Optional[torch.Tensor],
    ) -> dict:
        """canvas 原生口径四个物理项的批次标量(未加权、已按 min_t 掩码平均)。

        单步可微估计 ŷ1 = y0_orig + r̂(b̂/dot_beta),目标为 y1_orig;反标准化用
        phys_scale 乘回物理单位后交给 physics_canvas 的算子。返回 dict:
        {'div','vort','spectral','extreme'};div 需要 rho,rho=None 且散度权重为 0
        的探针场景该项为 None,权重>0 时直接报错(不会静默丢项)。
        """
        d_b = torch.gather(self.dot_beta, dim=-1, index=timestep)[:, None, None, None]
        r_hat, mask = estimate_residual(b_est, d_b, min_t=self.c.phys_min_t)
        y_hat = y0_orig + r_hat

        device, dtype = y_hat.device, y_hat.dtype
        n_levels = infer_n_levels(y_hat.shape[1])
        if self.c.phys_scale is None:
            raise ValueError(
                "canvas 物理损失需要 si.phys_scale(长度=out_channel 的每通道物理 σ)")
        ps = torch.as_tensor(self.c.phys_scale, dtype=dtype, device=device)
        if ps.numel() != y_hat.shape[1]:
            raise ValueError(
                f"si.phys_scale 元素数 {ps.numel()} != out_channel {y_hat.shape[1]}")
        ps = ps.view(1, -1, 1, 1)
        y_hat_phys = y_hat * ps
        y1_phys = y1_orig * ps

        u_p, v_p, w_p, u10_p, v10_p = split_canvas_state(y_hat_phys, n_levels)
        u_t, v_t, w_t, u10_t, v10_t = split_canvas_state(y1_phys, n_levels)
        out = {"div": None, "vort": None, "spectral": None, "extreme": None}

        # ① 可压缩散度 ∇·(ρu) 的 hinge 约束:max(0, |D|/τ_k − 1)²
        if rho is None:
            if self.c.divergence_weight > 0:
                raise ValueError(
                    "canvas 散度损失需要在 forward(..., rho=...) 传入干空气密度 (B,L,H,W)")
        else:
            rho_t = rho.to(device=device, dtype=dtype)
            exp_shape = (y_hat.shape[0], n_levels) + tuple(y_hat.shape[-2:])
            if tuple(rho_t.shape) != exp_shape:
                raise ValueError(f"rho 形状 {tuple(rho_t.shape)} != 期望 {exp_shape}")
            dz = self._phys_vec("phys_dz", n_levels, device, dtype)
            tau = self._phys_vec("phys_div_tau", n_levels, device, dtype)
            tau = tau.view(1, n_levels, 1, 1)
            div = divergence_rho_u(u_p, v_p, w_p, rho_t, self.c.phys_dx, dz)
            hinge = torch.clamp(div.abs() / tau - 1.0, min=0.0) ** 2
            out["div"] = self._masked_mean(hinge.mean(dim=(1, 2, 3)), mask)

        # ② 涡度结构差:逐层 L1 差取均值
        zeta_p = vorticity(u_p, v_p, self.c.phys_dx, self.c.phys_dx)
        zeta_t = vorticity(u_t, v_t, self.c.phys_dx, self.c.phys_dx)
        out["vort"] = self._masked_mean(
            (zeta_p - zeta_t).abs().mean(dim=(1, 2, 3)), mask)

        # ③ 径向 log 谱 L2 差(bins×层×u,v 取均值)
        um_p, vm_p, _ = destagger_canvas(u_p, v_p, w_p)
        um_t, vm_t, _ = destagger_canvas(u_t, v_t, w_t)
        spec_p = radial_log_spectra(um_p, vm_p)
        spec_t = radial_log_spectra(um_t, vm_t)
        out["spectral"] = self._masked_mean(
            ((spec_p - spec_t) ** 2).mean(dim=(1, 2, 3)), mask)

        # ④ 极值结构差:每层 0.5(|Δmax_s|+|Δmin_s|),再对层取均值
        levels = self._extreme_levels(n_levels)
        spd_p = windspeed_levels(u_p, v_p, u10_p, v10_p, levels)
        spd_t = windspeed_levels(u_t, v_t, u10_t, v10_t, levels)
        ext = 0.5 * ((spd_p.amax(dim=(2, 3)) - spd_t.amax(dim=(2, 3))).abs()
                     + (spd_p.amin(dim=(2, 3)) - spd_t.amin(dim=(2, 3))).abs())
        out["extreme"] = self._masked_mean(ext.mean(dim=1), mask)
        return out

    def forward(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        y_cond: torch.Tensor,
        rho: Optional[torch.Tensor] = None,
        return_parts: bool = False,
        **kwargs,
    ):
        # y0: LR data, dim = batch, channel, y, and x
        # y1: HR data, dim = batch, channel, y, and x
        # y0 and y1 have the same shape.
        # y_cond: condition for y0 and y1, such as building data, dim = batch, channel, y, and x
        # rho: 可空,canvas 布局的干空气密度 (B,L,H,W) kg/m³(散度损失用)
        # return_parts: True 返回各项损失 dict(canvas 探针定标用),默认返回标量 total

        # 物理约束项的目标是原始 y0/y1,必须在残差变换前保存
        y0_orig, y1_orig = y0, y1

        if self.c.residual_output:
            # 残差参数化:桥接 0 -> (y1 − y0);采样端从 0 出发、末尾再加回 y0
            y1 = y1 - y0
            y0 = torch.zeros_like(y0)

        timestep, t = self._sample_timestep(batch_size=y0.shape[0])
        noise = torch.randn_like(y0)

        yt = self._sample_yt(y0=y0, y1=y1, noise=noise, timestep=timestep)
        b_true = self._calc_b_true(y0=y0, y1=y1, noise=noise, timestep=timestep)

        b_est = self.net(yt=yt, y_cond=y_cond, gamma=t)

        if self.c.loss_type == "L2":
            diff = (b_true - b_est) ** 2
        elif self.c.loss_type == "L1":
            diff = (b_true - b_est).abs()  # channel_weights 在 L1 下为线性加权
        else:
            raise NotImplementedError(
                f"{self.c.loss_type} loss type is not implemented."
            )

        if self.channel_weights is not None:
            diff = diff * self.channel_weights
        data_loss = torch.mean(diff)
        total_loss = data_loss

        # ---- 物理约束项 ----
        if self.c.state_layout == "canvas":
            # 阶段 2:原生 C 网格口径(散度/涡度/谱/极值),逐样本掩码平均后再加权。
            # 权重全为 0 且 return_parts=False 时整块跳过,与旧行为完全一致。
            if return_parts or self._canvas_phys_active():
                raw = self._canvas_physics_raw(
                    y0_orig=y0_orig, y1_orig=y1_orig, b_est=b_est,
                    timestep=timestep, rho=rho)
                for key, weight in (("div", self.c.divergence_weight),
                                    ("vort", self.c.vorticity_weight),
                                    ("spectral", self.c.spectral_weight),
                                    ("extreme", self.c.extreme_weight)):
                    if weight > 0 and raw.get(key) is not None:
                        total_loss = total_loss + weight * raw[key]
                if return_parts:
                    return {
                        "data": data_loss,
                        "div": raw.get("div"),
                        "vort": raw.get("vort"),
                        "spectral": raw.get("spectral"),
                        "extreme": raw.get("extreme"),
                        "total": total_loss,
                    }
        else:
            # 旧交错布局(interleaved_uvw):旧三元组口径与行为原样保留
            if self.c.divergence_weight > 0:
                div_loss = self._calc_3d_divergence_loss(b_est)
                total_loss = total_loss + self.c.divergence_weight * div_loss

            if self.c.vorticity_weight > 0:
                vort_loss = self._calc_vorticity_loss(b_est)
                total_loss = total_loss + self.c.vorticity_weight * vort_loss

            if return_parts:
                return {
                    "data": data_loss,
                    "div": (self._calc_3d_divergence_loss(b_est)
                            if self.c.divergence_weight > 0 else None),
                    "vort": (self._calc_vorticity_loss(b_est)
                             if self.c.vorticity_weight > 0 else None),
                    "spectral": None,
                    "extreme": None,
                    "total": total_loss,
                }

        return total_loss

    @torch.no_grad()
    def sample_y1_bare_diffusion(
        self,
        y0: torch.Tensor,
        y_cond: torch.Tensor,
        n_return_step: Optional[int] = None,
        hide_progress_bar: bool = True,
        add_noise: bool = True,
        **kwargs,
    ):
        #
        assert not self.net.training
        #
        if n_return_step is not None:
            inter = self.c.n_timestep // n_return_step
            intermidiates = {}
        else:
            inter = None
            intermidiates = None

        b = y0.shape[0]  # batch size
        yt = (torch.zeros_like(y0) if self.c.residual_output else y0).detach().clone()

        # Time index here is from 0 to T
        for step in tqdm(range(0, self.c.n_timestep + 1), disable=hide_progress_bar):
            if inter is not None and step % inter == 0:
                if step > 0:
                    intermidiates[step] = yt

            t = torch.broadcast_to(self.time[step][None, None], size=(b, 1))
            b_est = self.net(yt=yt, y_cond=y_cond, gamma=t)
            yt = yt + self.dt * b_est

            if add_noise and step < self.c.n_timestep:
                s = self.sigma[step]
                yt = yt + s * torch.sqrt(self.dt) * torch.randn_like(yt)
            # Theoretically, noise is zero when step == N (i.e., self.sigma[N] == 0).
            # But, just in case, we skip adding noise when step == N.

        if self.c.residual_output:
            yt = yt + y0
        return yt, intermidiates

    @torch.no_grad()
    def sample_y1_follmer_diffusion(
        self,
        y0: torch.Tensor,
        y_cond: torch.Tensor,
        n_return_step: Optional[int] = None,
        hide_progress_bar: bool = True,
        add_noise: bool = True,
        **kwargs,
    ):
        #
        assert not self.net.training
        #
        if n_return_step is not None:
            inter = self.c.n_timestep // n_return_step
            intermidiates = {}
        else:
            inter = None
            intermidiates = None

        b = y0.shape[0]  # batch size
        y0_eff = torch.zeros_like(y0) if self.c.residual_output else y0
        yt = y0_eff.detach().clone()

        # Time index here is from 0 to T
        for step in tqdm(range(0, self.c.n_timestep + 1), disable=hide_progress_bar):
            if inter is not None and step % inter == 0:
                if step > 0:
                    intermidiates[step] = yt

            t = torch.broadcast_to(self.time[step][None, None], size=(b, 1))
            b_est = self.net(yt=yt, y_cond=y_cond, gamma=t)

            if step == 0:
                yt = yt + self.dt * b_est
                if add_noise:
                    s = self.sigma[step]
                    yt = yt + s * torch.sqrt(self.dt) * torch.randn_like(yt)
            else:
                _step = torch.broadcast_to(torch.tensor(step), size=(b,))
                _step = _step.to(self.device)
                bF_est = self._calc_bF(b=b_est, y0=y0_eff, yt=yt, timestep=_step)
                yt = yt + self.dt * bF_est

                if add_noise and step < self.c.n_timestep:
                    s = self.gF[step]
                    yt = yt + s * torch.sqrt(self.dt) * torch.randn_like(yt)
                # Theoretically, noise is zero when step == N (i.e., self.gF[N] == 0).
                # But, just in case, we skip adding noise when step == N.

        if self.c.residual_output:
            yt = yt + y0
        return yt, intermidiates
