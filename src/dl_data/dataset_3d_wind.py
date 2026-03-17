# -*- coding: utf-8 -*-
"""
3D风场数据集
继承自 Dataset2dTemperature2m，仅覆盖 _get_var_name 使用完整变量名。
这样旧的温度/2D风场配置不受影响，只有3D配置走这个子类。
"""

import dataclasses
import typing

from src.dl_data.dataset_2d_tm2m import Dataset2dTemperature2m, Dataset2dTemperature2mConfig


@dataclasses.dataclass()
class Dataset3dWindConfig(Dataset2dTemperature2mConfig):
    """3D风场配置，覆盖 dataset_name"""
    dataset_name: typing.ClassVar[str] = "Dataset3dWind"


class Dataset3dWind(Dataset2dTemperature2m):
    """3D风场数据集，使用完整变量名作为bias/scale的key"""

    def _get_var_name(self, name: str):
        return name  # 不截断，使用完整变量名如 hr_u_ml0, lr_v_ml10 等
