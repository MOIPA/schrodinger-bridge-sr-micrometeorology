from logging import getLogger

from torch import nn

logger = getLogger()


def initialize_to_zero(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module
