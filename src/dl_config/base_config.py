import dataclasses
import inspect
import json
import os
import typing

import yaml


@dataclasses.dataclass
class YamlConfig:
    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(dataclasses.asdict(self), indent=indent)

    def save(self, config_path: str):
        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path, "w") as f:
            yaml.safe_dump(convert_dict(dataclasses.asdict(self)), f)

    @classmethod
    def load(cls, config_path: str):
        assert os.path.exists(config_path), f"YAML config {config_path} does not exist"

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if inspect.isclass(child_class) and issubclass(child_class, YamlConfig):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            # recursively convert config item to Config
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)


@dataclasses.dataclass()
class BaseDatasetConfig(YamlConfig):
    dataset_name: typing.ClassVar[str]


@dataclasses.dataclass()
class BaseDataloaderConfig(YamlConfig):
    batch_size: int
    dl_data_ver: str
    train_valid_test_ratios: list[float]
    num_workers: int
    seed: int

    def __post_init__(self):
        assert (
            len(self.train_valid_test_ratios) == 3
        )  # train, valid, test, i.e., three ratios
        assert sum(self.train_valid_test_ratios) == 1.0
        assert all([r > 0 for r in self.train_valid_test_ratios])


@dataclasses.dataclass()
class BaseLossConfig(YamlConfig):
    loss_name: str


@dataclasses.dataclass()
class BaseTrainConfig(YamlConfig):
    epochs: int
    early_stopping_patience: int
    seed: int
    learning_rate: float
    optim_name: str
    loss: BaseLossConfig
    use_amp: bool


@dataclasses.dataclass()
class BaseModelConfig(YamlConfig):
    model_name: typing.ClassVar[str]
