from logging import getLogger

from torch import nn

logger = getLogger()


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
