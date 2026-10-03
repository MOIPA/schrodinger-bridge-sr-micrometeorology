import dataclasses
from logging import getLogger

from src.dl_config.base_config import BaseTrainConfig
from torch import nn

logger = getLogger()


@dataclasses.dataclass()
class TrainEMAConfig(BaseTrainConfig):
    ema_decay: float
    save_interval: int
    lr_schedule: str = "none"  # "none"=恒定;"cosine"=按 epoch 余弦衰减到 0(稳定性诊断,2026-10-03)


class EMA:
    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay
        logger.info(f"EMA {self.decay=}")

    def update_model_average(self, current_model: nn.Module, ma_model: nn.Module):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            new_weight, old_weight = (
                current_params.data,
                ma_params.data,
            )
            ma_params.data = self._update_average(new_weight, old_weight)

    def _update_average(self, new, old):
        return old * self.decay + (1 - self.decay) * new
