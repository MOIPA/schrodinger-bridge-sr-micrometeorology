import math
from abc import abstractmethod
from logging import getLogger
from typing import Union

import torch
from torch import nn

from src.dl_model.block.downsample import Downsample2D
from src.dl_model.block.norm import GroupNorm32
from src.dl_model.block.upsample import Upsample2DNearest
from src.dl_model.util import initialize_to_zero

logger = getLogger()


class EmbedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock2DEmb(EmbedBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channel: Union[int, None] = None,
        use_conv_in_down: bool = False,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        assert not (
            (up == True) and (down == True)
        ), "Unexpected input: both up and down are True."

        assert (
            channels % 32 == 0
        ), f"For GroupNorm32, channels must be divisible by 32. {channels=}"

        if out_channel is not None:
            assert (
                out_channel % 32 == 0
            ), f"For GroupNorm32, out_channel must be divisible by 32. {out_channel=}"

        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channel, 3, padding=1),
        )

        self.updown = up or down

        if up:
            logger.debug("Up is selected.")
            self.h_upd = Upsample2DNearest(channels, use_conv=False)
            self.x_upd = Upsample2DNearest(channels, use_conv=False)
        elif down:
            logger.debug(f"Down is selected. {use_conv_in_down=}")
            self.h_upd = Downsample2D(channels, use_conv=use_conv_in_down)
            self.x_upd = Downsample2D(channels, use_conv=use_conv_in_down)
        else:
            logger.debug("No Up or Down.")
            self.h_upd = self.x_upd = nn.Identity()

        logger.debug(f"{use_scale_shift_norm=}")

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32, self.out_channel),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            initialize_to_zero(
                nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
            ),
        )

        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x dim: batch, channel, y, and x
        # emb dim: batch, emb_channel

        if not self.updown:
            h = self.in_layers(x)
        else:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            h = in_conv(h)

            x = self.x_upd(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = emb_out[..., None, None]  # add y and x dims

        if not self.use_scale_shift_norm:
            h = h + emb_out
            h = self.out_layers(h)
        else:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)  # split along channel dim.
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        return self.skip_connection(x) + h


def gamma_embedding(gammas: torch.Tensor, dim: int, max_period: float):
    # gammas only have batch dim (gammas.ndim == 1)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=gammas.device)

    args = gammas[:, None].float() * freqs[None, ...]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2 == 1:
        # ensures embedding == batch x "dim"
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding
