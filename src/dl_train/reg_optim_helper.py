"""回归步(regression step)训练循环:单次前向 L1 残差回归。

阶段 3 架构对比「回归 + 扩散两步法」的回归步。前向/损失口径与 SI 残差输出对齐:
- pred = net(yt=y0, y_cond=x, gamma=ones(B)) —— 固定 t=1 的常数嵌入。
  SI 框架中 t∈[0,1]、t=1 为桥接终点(采样 `sample_y1_bare_diffusion` 的最后
  一步即 gamma=1、残差输出加回 y0);回归步只在该端点做一次前向;
- loss = L1(pred, y1 - y0) —— 残差输出口径(阶段 1/3 结论:残差输出 + L1)。

工程行为(AMP / NaN/Inf 跳过计数 / 梯度裁剪 / EMA / 日志行)逐条复刻
`optimize_si`,使训练日志与 SI 脚本同族,valid 分支同 `optimize_si`。
"""

import random
import typing
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.dl_train.exp_moving_ave import EMA
from src.utils.average_meter import AverageMeter

logger = getLogger()


def optimize_reg(
    dataloader: DataLoader,
    net: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    mode: typing.Union[str, typing.Literal["train", "valid", "test"]],
    scaler: GradScaler,
    use_amp: bool,
    ema: EMA,
    ema_net: nn.Module,
) -> float:
    """跑一个 epoch(或一次 valid/test),返回平均损失。

    参数结构对齐 `optimize_si`:原 `si` 参数由裸 `net`(UNet)取代,
    其余同名同义(epoch 用于 `random.seed`/`np.random.seed` 复刻)。
    """
    loss_meter = AverageMeter()

    d = next(net.parameters()).device
    device = str(d)

    if mode == "train":
        net.train()
    elif mode in ["valid", "test"]:
        net.eval()
    else:
        raise ValueError(f"{mode} is not supported.")

    random.seed(epoch)
    np.random.seed(epoch)

    device_type = "cuda" if "cuda" in device else "cpu"
    nan_skipped = 0
    for batch in dataloader:
        y0 = batch["y0"].to(device, non_blocking=True)
        y1 = batch["y"].to(device, non_blocking=True)
        y_cond = batch["x"].to(device, non_blocking=True)
        # 固定 t=1 常数嵌入:gamma 形状 (B,)(UNet 内部 gamma.view(-1) 消费);
        # 与 RegressionModel.sample_y1_bare_diffusion 的推理前向完全一致
        gamma_fixed = torch.ones(y0.shape[0], device=device, dtype=torch.float32)

        if mode == "train":
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    pred = net(yt=y0, y_cond=y_cond, gamma=gamma_fixed)
                    loss = F.l1_loss(pred, y1 - y0)

                # 检查 loss 是否 NaN/Inf，是则跳过这个 batch
                if not torch.isfinite(loss):
                    nan_skipped += 1
                    continue

                scaler.scale(loss).backward()
                # 梯度裁剪（在 unscale 之后再裁剪）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard CPU training forward and backward pass
                pred = net(yt=y0, y_cond=y_cond, gamma=gamma_fixed)
                loss = F.l1_loss(pred, y1 - y0)

                # 检查 loss 是否 NaN/Inf，是则跳过这个 batch
                if not torch.isfinite(loss):
                    nan_skipped += 1
                    continue

                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            if ema.decay is not None and 0.0 < ema.decay < 1.0:
                ema.update_model_average(current_model=net, ma_model=ema_net)

        else:
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=torch.float16, enabled=use_amp
            ):
                pred = net(yt=y0, y_cond=y_cond, gamma=gamma_fixed)
                loss = F.l1_loss(pred, y1 - y0)

        loss_meter.update(loss.item(), n=batch["x"].shape[0])
    if nan_skipped > 0:
        logger.warning(f"{mode}: skipped {nan_skipped} batches with NaN/Inf loss")
    logger.info(f"{mode} error: avg loss = {loss_meter.avg:.8f}")

    return loss_meter.avg
