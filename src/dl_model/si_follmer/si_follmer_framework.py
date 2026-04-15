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
    loss_type: Literal["L2"] = "L2"
    channel_weights: typing.Optional[list] = None  # 通道加权，如 [1,1,10, 1,1,10, ...]


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
            w = torch.tensor(self.c.channel_weights, dtype=self.dtype)
            self.register_buffer("channel_weights", w.view(1, -1, 1, 1))
            logger.info(f"Channel weights enabled: {self.c.channel_weights}")
        else:
            self.channel_weights = None

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

    def forward(
        self, y0: torch.Tensor, y1: torch.Tensor, y_cond: torch.Tensor, **kwargs
    ):
        # y0: LR data, dim = batch, channel, y, and x
        # y1: HR data, dim = batch, channel, y, and x
        # y0 and y1 have the same shape.
        # y_cond: condition for y0 and y1, such as building data, dim = batch, channel, y, and x

        timestep, t = self._sample_timestep(batch_size=y0.shape[0])
        noise = torch.randn_like(y0)

        yt = self._sample_yt(y0=y0, y1=y1, noise=noise, timestep=timestep)
        b_true = self._calc_b_true(y0=y0, y1=y1, noise=noise, timestep=timestep)

        b_est = self.net(yt=yt, y_cond=y_cond, gamma=t)

        if self.c.loss_type == "L2":
            diff_sq = (b_true - b_est) ** 2
            if self.channel_weights is not None:
                diff_sq = diff_sq * self.channel_weights
            return torch.mean(diff_sq)
        else:
            raise NotImplementedError(
                f"{self.c.loss_type} loss type is not implemented."
            )

    @torch.no_grad()
    def sample_y1_bare_diffusion(
        self,
        y0: torch.Tensor,
        y_cond: torch.Tensor,
        n_return_step: Optional[int] = None,
        hide_progress_bar: bool = True,
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
        yt = y0.detach().clone()

        # Time index here is from 0 to T
        for step in tqdm(range(0, self.c.n_timestep + 1), disable=hide_progress_bar):
            if inter is not None and step % inter == 0:
                if step > 0:
                    intermidiates[step] = yt

            t = torch.broadcast_to(self.time[step][None, None], size=(b, 1))
            b_est = self.net(yt=yt, y_cond=y_cond, gamma=t)
            yt = yt + self.dt * b_est

            if step < self.c.n_timestep:
                s = self.sigma[step]
                yt = yt + s * torch.sqrt(self.dt) * torch.randn_like(yt)
            # Theoretically, noise is zero when step == N (i.e., self.sigma[N] == 0).
            # But, just in case, we skip adding noise when step == N.

        return yt, intermidiates

    @torch.no_grad()
    def sample_y1_follmer_diffusion(
        self,
        y0: torch.Tensor,
        y_cond: torch.Tensor,
        n_return_step: Optional[int] = None,
        hide_progress_bar: bool = True,
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
        yt = y0.detach().clone()

        # Time index here is from 0 to T
        for step in tqdm(range(0, self.c.n_timestep + 1), disable=hide_progress_bar):
            if inter is not None and step % inter == 0:
                if step > 0:
                    intermidiates[step] = yt

            t = torch.broadcast_to(self.time[step][None, None], size=(b, 1))
            b_est = self.net(yt=yt, y_cond=y_cond, gamma=t)

            if step == 0:
                yt = yt + self.dt * b_est
                s = self.sigma[step]
                yt = yt + s * torch.sqrt(self.dt) * torch.randn_like(yt)
            else:
                _step = torch.broadcast_to(torch.tensor(step), size=(b,))
                _step = _step.to(self.device)
                bF_est = self._calc_bF(b=b_est, y0=y0, yt=yt, timestep=_step)
                yt = yt + self.dt * bF_est

                if step < self.c.n_timestep:
                    s = self.gF[step]
                    yt = yt + s * torch.sqrt(self.dt) * torch.randn_like(yt)
                # Theoretically, noise is zero when step == N (i.e., self.gF[N] == 0).
                # But, just in case, we skip adding noise when step == N.

        return yt, intermidiates
