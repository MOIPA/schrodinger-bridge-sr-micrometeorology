from logging import getLogger
from typing import Union

from torch import nn

logger = getLogger()


class Downsample2D(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool, out_channel: Union[int, None] = None
    ):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            logger.debug(f"{use_conv=}, {self.out_channel=}")
            self.op = nn.Conv2d(
                self.channels, self.out_channel, 3, stride=stride, padding=1
            )
        else:
            logger.debug(f"Use AvgPool2d, {self.out_channel=}")
            assert self.channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)
