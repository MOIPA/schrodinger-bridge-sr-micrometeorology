import dataclasses

from src.dl_config.base_config import BaseDataloaderConfig, YamlConfig
from src.dl_data.dataset_2d_tm2m import Dataset2dTemperature2mConfig
from src.dl_data.dataset_3d_wind import Dataset3dWindConfig
from src.dl_data.dataset_wind_canvas import DatasetWindCanvasConfig
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


@dataclasses.dataclass
class ExperimentSchrodingerBridge3dWindConfig(YamlConfig):
    """3D风场超分实验配置，使用 Dataset3dWindConfig"""
    data: Dataset3dWindConfig
    loader: BaseDataloaderConfig
    train: TrainEMAConfig
    model: UNetDDPMVer01Config
    si: SIFollmerConfig


@dataclasses.dataclass
class ExperimentSchrodingerBridgeWindCanvasConfig(YamlConfig):
    """阶段0(导师大纲)canvas 管线:原生 C 网格 + AGL + 块级划分。"""
    data: DatasetWindCanvasConfig
    loader: BaseDataloaderConfig
    train: TrainEMAConfig
    model: UNetDDPMVer01Config
    si: SIFollmerConfig
