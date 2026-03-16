from logging import getLogger

import torch
from torch import nn

from src.dl_config.base_config import BaseLossConfig

logger = getLogger()


def make_loss(config: BaseLossConfig) -> nn.Module:
    if config.loss_name == "L2":
        logger.info("L2 loss is created.")
        return L2Loss()
    elif config.loss_name == "L1":
        logger.info("L1 loss is created.")
        return L1Loss()
    else:
        raise ValueError(f"{config.loss_name} is not supported.")


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        return self.loss(predicts, targets)


class L1Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        return self.loss(predicts, targets)
