from logging import getLogger

import torch
from torch import nn

from src.dl_config.base_config import BaseTrainConfig

logger = getLogger()


def make_optimizer(config: BaseTrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optim_name == "AdamW":
        logger.info("Optimizer is AdamW.")
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    elif config.optim_name == "Adam":
        logger.info("Optimizer is Adam.")
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Optimizer {config.optim_name} is not supported")
