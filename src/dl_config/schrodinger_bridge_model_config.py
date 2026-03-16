import dataclasses

from src.dl_config.base_config import BaseDataloaderConfig, YamlConfig
from src.dl_data.dataset_2d_tm2m import Dataset2dTemperature2mConfig
from src.dl_model.ddpm.unet_ddpm_v01 import UNetDDPMVer01Config
from src.dl_model.si_follmer.si_follmer_framework import SIFollmerConfig
from src.dl_train.exp_moving_ave import TrainEMAConfig


@dataclasses.dataclass
class ExperimentSchrodingerBridgeModelConfig(YamlConfig):
    data: Dataset2dTemperature2mConfig
    loader: BaseDataloaderConfig
    train: TrainEMAConfig
    model: UNetDDPMVer01Config
    si: SIFollmerConfig
