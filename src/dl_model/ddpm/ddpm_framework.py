import copy
import dataclasses
import math
import sys
from functools import partial
from logging import getLogger
from typing import Literal, Union

import numpy as np
import torch
from torch import nn

from src.dl_config.base_config import YamlConfig

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


@dataclasses.dataclass()
class BetaConfig(YamlConfig):
    schedule: str
    start: float
    end: float
    n_timestep: int


@dataclasses.dataclass()
class DDPMConfig(YamlConfig):
    beta_schedules: dict[str, BetaConfig]
    dtype: str


class DDPM(nn.Module):
    def __init__(
        self, config: DDPMConfig, neural_net: nn.Module, device: Union[None, str] = None
    ):
        super().__init__()

        self.device = (
            device if device is not None else str(next(neural_net.parameters()).device)
        )

        self.c = copy.deepcopy(config)
        self.net = neural_net
        logger.info("This is DDPM from ddpm_framework2 (using the SDE formulas)")

        if self.c.dtype == "float32":
            self.dtype = torch.float32
        elif self.c.dtype == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Not supported dtype = {self.c.dtype}")

    def set_noise_schedule(self, phase: Literal["train", "test"]):
        assert phase in ["train", "test"]

        to_torch = partial(torch.tensor, dtype=self.dtype, device=self.device)
        #
        betas = make_beta_schedule(
            schedule=self.c.beta_schedules[phase].schedule,
            start=self.c.beta_schedules[phase].start,
            end=self.c.beta_schedules[phase].end,
            n_timestep=self.c.beta_schedules[phase].n_timestep,
        )
        times = np.linspace(
            0.0, 1.0, num=len(betas) + 1, endpoint=True, dtype=np.float64
        )
        times = times[1:]  # skip the initial value

        self.n_timestep = len(betas)
        assert self.n_timestep == self.c.beta_schedules[phase].n_timestep

        self.dt = 1.0 / self.n_timestep
        self.sqrt_dt = math.sqrt(self.dt)

        # variance-preserving SDE
        frictions = 0.5 * betas
        sigmas = np.sqrt(betas)

        decays, vars = precompute_ou(mu=frictions, sigma=sigmas, dt=self.dt)
        stds = np.sqrt(vars)
        # the OU solution is expressed as x_t = decay * x_0 + std * epsilon (epsilon ~ N(0,1))

        # the number of elements in each param is equal to self.n_timestep
        self.register_buffer("frictions", to_torch(frictions))
        self.register_buffer("sigmas", to_torch(sigmas))
        self.register_buffer("times", to_torch(times))

        # register params except for the initial values because std is zero
        self.register_buffer("decays", to_torch(decays[1:]))
        self.register_buffer("stds", to_torch(stds[1:]))

        assert (
            self.frictions.shape
            == self.sigmas.shape
            == self.times.shape
            == self.decays.shape
            == self.stds.shape
            == (self.n_timestep,)
        )
        assert torch.all(self.sigmas > 0.0) and torch.all(self.stds > 0.0)

    def _extract_params(
        self, params: torch.Tensor, t_indices: torch.Tensor, for_broadcast: bool = True
    ) -> torch.Tensor:

        def select(array):
            return torch.index_select(array, dim=0, index=t_indices)
            # Select diffusion times along batch dim

        (n_batches,) = t_indices.shape

        selected = select(params)
        assert selected.shape == (n_batches,)

        # add channel, y, and x dims
        if for_broadcast:
            return selected.requires_grad_(False)[:, None, None, None]
        else:
            return selected.requires_grad_(False)

    def _forward_sample_y(
        self, y0: torch.Tensor, t_index: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        #
        a = self._extract_params(self.decays, t_index)
        b = self._extract_params(self.stds, t_index)
        return a * y0 + b * noise

    @torch.no_grad()
    def _backward_sample_y(
        self, yt: torch.Tensor, y_cond: torch.Tensor, t_index: torch.Tensor
    ) -> torch.Tensor:

        friction = self._extract_params(self.frictions, t_index)
        sigma = self._extract_params(self.sigmas, t_index)
        std = self._extract_params(self.stds, t_index)
        t = self._extract_params(self.times, t_index, for_broadcast=False)
        t = t[:, None]  # add channel dim

        est_noise = self.net(yt=yt, y_cond=y_cond, gamma=t, t_index=t_index)
        score = -est_noise / std

        mean = yt + self.dt * (friction * yt + (sigma**2) * score)
        dW = self.sqrt_dt * torch.randn_like(yt)

        n_batches = yt.shape[0]
        mask = (1 - (t_index == 0).float()).reshape(n_batches, *((1,) * (yt.ndim - 1)))
        mask = mask.to(dtype=self.dtype, device=self.device)
        # no noise at t_index == 0

        return mean + mask * sigma * dW

    @torch.no_grad()
    def backward_sample_y(
        self,
        y_cond: torch.Tensor,
        n_return_step: int = None,
        tqdm_disable: bool = False,
    ):
        assert not self.net.training

        n_batches, _, *space = y_cond.shape
        yt = torch.randn(
            size=(n_batches, self.net.out_channel, *space), device=y_cond.device
        )
        yt = self.stds[-1] * yt

        if n_return_step is not None:
            inter = self.n_timestep // n_return_step
            intermediates = {}
        else:
            inter = None
            intermediates = None

        for i in tqdm(
            reversed(range(0, self.n_timestep)),
            total=self.n_timestep,
            disable=tqdm_disable,
        ):
            if inter is not None and (i + 1) % inter == 0:
                intermediates[i + 1] = yt.cpu().detach().clone()

            ind = torch.full((n_batches,), i, device=self.device, dtype=torch.long)
            yt = self._backward_sample_y(yt=yt, y_cond=y_cond, t_index=ind)

        if intermediates is not None:
            assert 0 not in intermediates
            intermediates[0] = yt.cpu().detach().clone()

        return yt, intermediates

    def forward(self, y0: torch.Tensor, y_cond: torch.Tensor, **kwargs) -> torch.Tensor:

        b = y0.shape[0]
        t_index = torch.randint(0, self.n_timestep, (b,), device=self.device).long()

        noise = torch.randn_like(y0)

        yt = self._forward_sample_y(y0=y0, t_index=t_index, noise=noise)
        t = self._extract_params(self.times, t_index, for_broadcast=False)
        t = t[:, None]  # add channel dim
        noise_hat = self.net(yt=yt, y_cond=y_cond, gamma=t, t_index=t_index)

        return noise, noise_hat


def make_beta_schedule(
    schedule: str,
    start: float,
    end: float,
    n_timestep: int,
) -> np.ndarray:
    logger.info(
        f"\nMaking betas for DDPM2:\n{schedule=}\n{start=}\n{end=}\n{n_timestep}\n"
    )

    if schedule == "linear":
        betas = np.linspace(start, end, n_timestep, dtype=np.float64, endpoint=True)
    elif schedule == "warmup10":
        betas = _warmup_beta(start, end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(start, end, n_timestep, 0.5)
    elif schedule == "const":
        betas = end * np.ones(n_timestep, dtype=np.float64)
    else:
        raise NotImplementedError(f"Not supported: {schedule=}")
    return betas


def _warmup_beta(
    start: float, end: float, n_timestep: int, warmup_frac: float, **kwargs
) -> np.ndarray:
    #
    betas = end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        start, end, warmup_time, dtype=np.float64, endpoint=True
    )
    return betas


def precompute_ou(
    mu: np.ndarray,
    sigma: np.ndarray,
    dt: float | np.ndarray,
    init_variance: float = 0.0,
):
    mu = np.array(mu, dtype=np.float64)
    assert np.all(mu >= 0.0)

    sigma = np.array(sigma, dtype=np.float64)
    assert np.all(sigma >= 0.0)

    if isinstance(dt, float):
        dt = np.full_like(mu, dt, dtype=np.float64)
    else:
        dt = np.array(dt, dtype=np.float64)
    assert mu.shape == sigma.shape == dt.shape
    assert init_variance >= 0.0

    N = mu.size
    m = np.empty(N + 1, dtype=np.float64)  # mean
    v = np.empty(N + 1, dtype=np.float64)  # variance
    m[0] = 1.0
    v[0] = init_variance

    for n in range(N):
        decay = np.exp(-mu[n] * dt[n])
        m[n + 1] = decay * m[n]
        if mu[n] == 0.0:
            q = sigma[n] ** 2 * dt[n]
        else:
            q = sigma[n] ** 2 * (1.0 - decay**2) / (2.0 * mu[n])
        v[n + 1] = decay**2 * v[n] + q

    return np.array(m), np.array(v)
