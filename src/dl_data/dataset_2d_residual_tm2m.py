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
class Dataset2dResidualTemperature2mConfig(BaseDatasetConfig):
    input_variable_names: list[str]
    biases: dict[str, float]
    scales: dict[str, float]
    is_clipped: bool
    min_clipped_value: typing.Union[float, None]
    max_clipped_value: typing.Union[float, None]
    missing_value: float
    hr_data_shape: list[int]
    hr_cropped_shape: list[int]
    dataset_name: typing.ClassVar[str] = "Dataset2dResidualTemperature2m"
    dtype: str
    discarded_minute_range: list[float]

    def __post_init__(self):
        assert self.dtype == "float16" or self.dtype == "float32"


class Dataset2dResidualTemperature2m(Dataset):
    def __init__(
        self,
        file_paths: list[str],
        config: Dataset2dResidualTemperature2mConfig,
        **kwargs,
    ):
        self.c = copy.deepcopy(config)
        self.ps = self._extract_paths(file_paths)

        logger.info("This is Dataset2dResidualTemperature2m")
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

        assert self.c.input_variable_names[0] == "lr_tm002m"
        # The first input must be LR temperature at 2 m height

    def _extract_paths(self, file_paths):
        extracted = []

        for file_path in file_paths:
            if not file_path.endswith(".npz"):
                continue
            dt = datetime.datetime.strptime(
                os.path.basename(file_path).replace(".npz", ""), "%Y%m%dT%H%M%S"
            )

            if (
                self.c.discarded_minute_range[0]
                <= dt.minute
                <= self.c.discarded_minute_range[1]
            ):
                continue

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

    def _preprocess_gt(self, ds: dict[str, np.ndarray]) -> torch.Tensor:
        #
        name = "re_tm"  # residual temperature
        #
        y = torch.from_numpy(ds["hr_tm002m"]).detach().clone().to(self.dtype)
        x = torch.from_numpy(ds["lr_tm002m"]).detach().clone().to(self.dtype)
        logger.debug(f"{name=}, {y.shape=}, {x.shape=}")
        x = self._resize(x)
        assert x.shape == y.shape == tuple(self.c.hr_data_shape)
        v = y - x  # residual
        v = self._scale(v, name)

        return v[None, ...]  # Add a new dimension for the channel

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ds = self.all_files[idx]

        logger.debug("Creating input variables")
        X = self._preprocess(ds, self.c.input_variable_names)
        logger.debug("Creating target variables")
        y = self._preprocess_gt(ds)
        logger.debug(f"{X.shape=},{y.shape=}")

        stacked = torch.cat([y, X])
        logger.debug(f"{stacked.shape=}")

        cropped = self.crop(stacked)
        logger.debug(f"{cropped.shape=}")

        n = 1  # == y.shape[0], one channel
        logger.debug(f"{n=}")
        y = stacked[:n]
        X = stacked[n:]
        logger.debug(f"{X.shape=},{y.shape=}")

        v = torch.from_numpy(ds["lr_tm002m"]).detach().clone().to(self.dtype)
        x_raw = self._resize(v)

        y_raw = torch.from_numpy(ds["hr_tm002m"]).detach().clone().to(self.dtype)

        return {
            "x": torch.nan_to_num(X, self.c.missing_value),
            "y": torch.nan_to_num(y, self.c.missing_value),
            "lr_tm002m": x_raw,
            "hr_tm002m": y_raw,
        }
