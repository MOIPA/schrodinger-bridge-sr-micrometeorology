from logging import getLogger
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

logger = getLogger()


class Upsample2DNearest(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool, out_channel: Union[int, None] = None
    ):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            logger.debug(f"{use_conv=}, {self.out_channel=}")
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest-exact")
        if self.use_conv:
            x = self.conv(x)
        return x
