import dataclasses

from src.dl_config.base_config import BaseDataloaderConfig, YamlConfig
from src.dl_data.dataset_2d_residual_tm2m import Dataset2dResidualTemperature2mConfig
from src.dl_model.ddpm.ddpm_framework import DDPMConfig
from src.dl_model.ddpm.unet_ddpm_v01 import UNetDDPMVer01Config
from src.dl_train.exp_moving_ave import TrainEMAConfig


@dataclasses.dataclass
class ExperimentDiffusionModelConfig(YamlConfig):
    data: Dataset2dResidualTemperature2mConfig
    loader: BaseDataloaderConfig
    train: TrainEMAConfig
    model: UNetDDPMVer01Config
    ddpm: DDPMConfig
