import random
import typing
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer
from src.dl_train.exp_moving_ave import EMA
from src.utils.average_meter import AverageMeter

logger = getLogger()


def optimize_si(
    dataloader: DataLoader,
    si: StochasticInterpolantFollmer,
    optimizer: Optimizer,
    epoch: int,
    mode: typing.Union[str, typing.Literal["train", "valid", "test"]],
    scaler: GradScaler,
    use_amp: bool,
    ema: EMA,
    ema_net: nn.Module,
) -> float:
    #
    loss_meter = AverageMeter()

    d = next(si.net.parameters()).device
    device = str(d)

    if mode == "train":
        si.net.train()
    elif mode in ["valid", "test"]:
        si.net.eval()
    else:
        raise ValueError(f"{mode} is not supported.")

    random.seed(epoch)
    np.random.seed(epoch)

    device_type = "cuda" if "cuda" in device else "cpu"
    for batch in dataloader:
        y0 = batch["y0"].to(device, non_blocking=True)
        y1 = batch["y"].to(device, non_blocking=True)
        y_cond = batch["x"].to(device, non_blocking=True)

        if mode == "train":
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    loss = si.forward(y0=batch["y0"], y1=batch["y"], y_cond=batch["x"])
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard CPU training forward and backward pass
                loss = si.forward(y0=batch["y0"], y1=batch["y"], y_cond=batch["x"])
                loss.backward()
                optimizer.step()
    
            if ema.decay is not None and 0.0 < ema.decay < 1.0:
                ema.update_model_average(current_model=si.net, ma_model=ema_net)

            else:
                with torch.no_grad(), torch.autocast(
                    device_type=device_type, dtype=torch.float16, enabled=use_amp
                ):
                    loss = si(y0=y0, y1=y1, y_cond=y_cond)
            
        loss_meter.update(loss.item(), n=batch["x"].shape[0])
    logger.info(f"{mode} error: avg loss = {loss_meter.avg:.8f}")

    return loss_meter.avg
