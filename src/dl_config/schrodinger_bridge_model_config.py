import dataclasses
import typing

from src.dl_config.base_config import BaseDataloaderConfig, YamlConfig
from src.dl_data.dataset_2d_tm2m import Dataset2dTemperature2mConfig
from src.dl_data.dataset_3d_wind import Dataset3dWindConfig
from src.dl_data.dataset_wind_canvas import DatasetWindCanvasConfig
from src.dl_model.ddpm.unet_ddpm_v01 import UNetDDPMVer01Config
from src.dl_model.si_follmer.si_follmer_framework import SIFollmerConfig
from src.dl_model.swinir_arch import SwinIRCanvasConfig
from src.dl_train.exp_moving_ave import TrainEMAConfig

# model 段为多架构联合注解(Union 不是类,YamlConfig.load 不会自动构造,
# 由 config_loader.load_config 按 yml `model.arch` 键分派构造):
#   arch: unet_ddpm_v01(缺省,旧 yml 不写也原样工作)-> UNetDDPMVer01Config
#   arch: swinir_canvas                                   -> SwinIRCanvasConfig
#
# swinir_canvas 的 model 段示例(未列字段取 SwinIRCanvasConfig 默认值):
#   model:
#     arch: swinir_canvas
#     in_channel: 167        # 72 状态 + 95 条件
#     out_channel: 72
#     inner_channel: 96
#     num_blocks: 6
#     window_size: 8
#     num_heads: 4
#     mlp_ratio: 2.0
#     dropout: 0.0


ModelConfig = typing.Union[UNetDDPMVer01Config, SwinIRCanvasConfig]


@dataclasses.dataclass
class ExperimentSchrodingerBridgeModelConfig(YamlConfig):
    data: Dataset2dTemperature2mConfig
    loader: BaseDataloaderConfig
    train: TrainEMAConfig
    model: ModelConfig
    si: SIFollmerConfig


@dataclasses.dataclass
class ExperimentSchrodingerBridge3dWindConfig(YamlConfig):
    """3D风场超分实验配置，使用 Dataset3dWindConfig"""
    data: Dataset3dWindConfig
    loader: BaseDataloaderConfig
    train: TrainEMAConfig
    model: ModelConfig
    si: SIFollmerConfig


@dataclasses.dataclass
class ExperimentSchrodingerBridgeWindCanvasConfig(YamlConfig):
    """阶段0(导师大纲)canvas 管线:原生 C 网格 + AGL + 块级划分。"""
    data: DatasetWindCanvasConfig
    loader: BaseDataloaderConfig
    train: TrainEMAConfig
    model: ModelConfig
    si: SIFollmerConfig
