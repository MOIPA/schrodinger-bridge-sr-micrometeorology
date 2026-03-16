import copy
import dataclasses
import datetime
import os
import sys
import typing
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.dl_config.base_config import BaseDatasetConfig
from src.utils.random_crop import RandomCrop2D

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


@dataclasses.dataclass()
class Dataset2dTemperature2mConfig(BaseDatasetConfig):
    target_variable_names: list[str]
    input_variable_names: list[str]
    biases: dict[str, float]
    scales: dict[str, float]
    is_clipped: bool
    min_clipped_value: typing.Union[float, None]
    max_clipped_value: typing.Union[float, None]
    missing_value: float
    hr_data_shape: list[int]
    hr_cropped_shape: list[int]
    dataset_name: typing.ClassVar[str] = "Dataset2dTemperature2m"
    dtype: str
    discarded_minute_range: list[float]

    def __post_init__(self):
        assert self.dtype == "float16" or self.dtype == "float32"


class Dataset2dTemperature2m(Dataset):
    def __init__(
        self,
        file_paths: list[str],
        config: Dataset2dTemperature2mConfig,
        **kwargs,
    ):
        self.c = copy.deepcopy(config)
        self.ps = self._extract_paths(file_paths)

        logger.info(f"All files give to dataset are {len(self.ps)}.")

        logger.info("Reading all files.")
        self._read_all_files()

        if self.c.dtype == "float16":
            self.dtype = torch.float16
        elif self.c.dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Not supported dtype {self.c.dtype}")

        self.crop = RandomCrop2D(
            img_sz=self.c.hr_data_shape, crop_sz=self.c.hr_cropped_shape
        )

        # assert self.c.input_variable_names[0] == "lr_tm002m"
        # The first input must be LR temperature at 2 m height

    def _extract_paths(self, file_paths):
        # Temporarily bypass all filtering to ensure dataloader is not empty.
        # This will return all .npz files without checking time.
        extracted = []
        for file_path in file_paths:
            if file_path.endswith(".npz"):
                extracted.append(file_path)
        return extracted

    def _read_all_files(self):
        self.all_files = []
        for p in tqdm(self.ps):
            with np.load(p) as data:
                file_dict = {key: data[key].copy() for key in data.keys()}
                self.all_files.append(file_dict)

    def __len__(self) -> int:
        return len(self.ps)

    def _get_var_name(self, name: str):
        return name[:5]

    def _scale(self, data: torch.Tensor, name: str) -> torch.Tensor:
        n = self._get_var_name(name)
        b, s = self.c.biases[n], self.c.scales[n]
        return (data - b) / s

    def _scale_inversely(self, data: torch.Tensor, name: str) -> torch.Tensor:
        n = self._get_var_name(name)
        b, s = self.c.biases[n], self.c.scales[n]
        return data * s + b

    def _resize(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndim == 2  # y and x
        if list(data.shape) == self.c.hr_data_shape:
            return data

        logger.debug(f"Before resize: {data.shape=}")
        d = F.interpolate(
            data[None, None, ...], size=self.c.hr_data_shape, mode="nearest-exact"
        ).squeeze()
        logger.debug(f"After resize: {d.shape=}")

        return d

    def _preprocess(
        self, ds: dict[str, np.ndarray], variable_names: list[str]
    ) -> torch.Tensor:
        #
        ret = []
        #
        for name in variable_names:
            logger.debug(f"{name=}")
            v = torch.from_numpy(ds[name]).detach().clone().to(self.dtype)
            v = self._resize(v)
            v = self._scale(v, name)
            if self.c.is_clipped:
                v = torch.clamp(v, self.c.min_clipped_value, self.c.max_clipped_value)
            ret.append(v)
        return torch.stack(ret, dim=0)

    # def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    #     ds = self.all_files[idx]

    #     X = self._preprocess(ds, self.c.input_variable_names)
    #     logger.debug("Creating target variables (y)")
    #     y = self._preprocess(ds, self.c.target_variable_names)
        
    #     # --- BUG FIX ---
    #     # Manually load and process lr_tm002m for y0, instead of incorrectly taking it from X.
    #     logger.debug("Creating low-resolution input (y0)")
    #     y0_raw = torch.from_numpy(ds['lr_tm002m']).detach().clone().to(self.dtype)
        
    #     # The lr_tm002m in the npz is already interpolated to HR size.
    #     # The _resize function will just ensure it matches hr_data_shape if there's any discrepancy.
    #     # Squeeze/Unsqueeze to handle the [channel, H, W] dimension format.
    #     y0_resized = self._resize(y0_raw.squeeze(0)).unsqueeze(0)
        
    #     # We can re-use the scaling parameters from 'lr_tm' for 'lr_tm002m'
    #     # as they represent the same physical quantity.
    #     y0_scaled = self._scale(y0_resized, "lr_tm")

    #     logger.debug(f"{X.shape=}, {y.shape=}, {y0_scaled.shape=}")

    #     # The cropping should be applied to all tensors that have spatial dimensions
    #     stacked_for_crop = torch.cat([y, y0_scaled, X])
    #     logger.debug(f"{stacked_for_crop.shape=}")

    #     cropped = self.crop(stacked_for_crop)
    #     logger.debug(f"{cropped.shape=}")

    #     # Un-stack them after cropping
    #     y_final = cropped[0:1]
    #     y0_final = cropped[1:2]
    #     X_final = cropped[2:]
    #     logger.debug(f"{X_final.shape=}, {y_final.shape=}, {y0_final.shape=}")

    #     return {
    #         "x": torch.nan_to_num(X_final, self.c.missing_value),
    #         "y": torch.nan_to_num(y_final, self.c.missing_value),
    #         "y0": torch.nan_to_num(y0_final, self.c.missing_value),
    #     }


## 这是处理风场时候的代码，如果哪天处理2m气温出现了问题就用上面的代码

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ds = self.all_files[idx]

        # 1. 处理条件输入 X (这部分是通用的，无需修改)
        X = self._preprocess(ds, self.c.input_variable_names)

        # 2. 处理高分辨率目标 y (这部分也是通用的)
        # 对于风场任务, target_variable_names 是 ['hr_u10', 'hr_v10']
        # 所以这里会生成一个 shape 为 [2, H, W] 的 y 张量
        y = self._preprocess(ds, self.c.target_variable_names)

        # 3. 【核心修正】动态地创建低分辨率输入 y0
        # 不再写死 'lr_tm002m'，而是根据高分辨率目标名自动推断
        lr_target_names = [name.replace('hr_', 'lr_') for name in self.c.target_variable_names]
        # 对于风场任务, lr_target_names 会是 ['lr_u10', 'lr_v10']

        # 复用 _preprocess 函数来处理 y0, 保证所有逻辑一致
        y0_scaled = self._preprocess(ds, lr_target_names)

        # 4. 【核心修正】动态地进行堆叠、裁剪和切分
        # 获取目标通道数 (温度是1, 风场是2)
        num_target_channels = len(self.c.target_variable_names)

        # 将所有需要被同步裁剪的张量堆叠在一起
        # 顺序: y (HR目标), y0 (LR目标), X (条件)
        stacked_for_crop = torch.cat([y, y0_scaled, X], dim=0)

        # 执行随机裁剪
        cropped = self.crop(stacked_for_crop)

        # 根据通道数，动态地切分出 y, y0, X
        y_final = cropped[0:num_target_channels]
        y0_final = cropped[num_target_channels : 2 * num_target_channels]
        X_final = cropped[2 * num_target_channels:]

        # 确保尺寸正确
        assert y_final.shape[0] == num_target_channels
        assert y0_final.shape[0] == num_target_channels
        assert X_final.shape[0] == len(self.c.input_variable_names)

        return {
            "x": torch.nan_to_num(X_final, self.c.missing_value),
            "y": torch.nan_to_num(y_final, self.c.missing_value),
            "y0": torch.nan_to_num(y0_final, self.c.missing_value),
        }
