import glob
import typing
from logging import getLogger

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.dl_config.base_config import BaseDataloaderConfig, BaseDatasetConfig
from src.dl_data.dataset_2d_residual_tm2m import (
    Dataset2dResidualTemperature2m,
    Dataset2dResidualTemperature2mConfig,
)
from src.dl_data.dataset_2d_tm2m import (
    Dataset2dTemperature2m,
    Dataset2dTemperature2mConfig,
)
from src.dl_data.dataset_3d_wind import Dataset3dWind
from src.utils.random_seed_helper import get_torch_generator, seed_worker

logger = getLogger()


def make_dataloaders_and_samplers(
    *,
    root_dir: str,
    loader_config: BaseDataloaderConfig,
    dataset_config: BaseDatasetConfig,
    world_size: typing.Union[int, None],
    rank: typing.Union[int, None],
    train_valid_test_kinds: list[str] = ["train", "valid", "test"],
) -> tuple[dict[str, DataLoader], dict[str, DistributedSampler]]:
    #

    if dataset_config.dataset_name == "Dataset2dTemperature2m":
        logger.info(f"{dataset_config.dataset_name=}")
        dataset_initializer = Dataset2dTemperature2m
        extension = "npz"
        assert isinstance(dataset_config, Dataset2dTemperature2mConfig)
    elif dataset_config.dataset_name == "Dataset2dResidualTemperature2m":
        logger.info(f"{dataset_config.dataset_name=}")
        dataset_initializer = Dataset2dResidualTemperature2m
        extension = "npz"
        assert isinstance(dataset_config, Dataset2dResidualTemperature2mConfig)
    elif dataset_config.dataset_name == "Dataset3dWind":
        logger.info(f"{dataset_config.dataset_name=}")
        dataset_initializer = Dataset3dWind
        extension = "npz"
    else:
        raise NotImplementedError(
            f"Dataset {dataset_config.dataset_name} is not supported."
        )

    data_dir_path = f"{root_dir}/data/DL_data/{loader_config.dl_data_ver}"
    all_file_paths = _get_file_paths(data_dir_path, extension)
    logger.info(
        f"The total files = {len(all_file_paths)}. A part of them is used for training."
    )

    dict_file_paths = _split_paths_into_train_valid_test(
        all_file_paths, loader_config.train_valid_test_ratios
    )

    return _make_dataloaders_and_samplers(
        dataset_initializer=dataset_initializer,
        dict_file_paths=dict_file_paths,
        dataset_config=dataset_config,
        loader_config=loader_config,
        train_valid_test_kinds=train_valid_test_kinds,
        world_size=world_size,
        rank=rank,
    )


def _split_paths_into_train_valid_test(
    paths: list[str], train_valid_test_ratios: list[float]
) -> dict[str, list[str]]:
    #
    logger.debug(f"train, valid, test ratios = {train_valid_test_ratios}")

    assert len(train_valid_test_ratios) == 3  # train, valid, test, three ratios
    assert sum(train_valid_test_ratios) == 1.0
    assert all([r > 0 for r in train_valid_test_ratios])

    test_size = train_valid_test_ratios[-1]
    _paths, test_paths = train_test_split(paths, test_size=test_size, shuffle=False)

    valid_size = train_valid_test_ratios[1] / (
        train_valid_test_ratios[0] + train_valid_test_ratios[1]
    )
    train_paths, valid_paths = train_test_split(
        _paths, test_size=valid_size, shuffle=False
    )

    assert set(train_paths).isdisjoint(set(valid_paths))
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(valid_paths).isdisjoint(set(test_paths))

    logger.debug(
        f"train: {len(train_paths)}, valid: {len(valid_paths)}, test: {len(test_paths)}"
    )

    return {"train": train_paths, "valid": valid_paths, "test": test_paths}


def _get_file_paths(data_dir_path: str, extension: str) -> list[str]:
    #
    if extension is not None:
        return sorted([p for p in glob.glob(f"{data_dir_path}/*.{extension}")])
    else:
        return sorted([p for p in glob.glob(f"{data_dir_path}/*")])


def _make_dataloaders_and_samplers(
    *,
    dict_file_paths: dict[str, list[str]],
    dataset_initializer,
    dataset_config: BaseDatasetConfig,
    loader_config: BaseDataloaderConfig,
    train_valid_test_kinds: list[str],
    world_size: typing.Union[int, None],
    rank: typing.Union[int, None],
    **kwargs,
):
    logger.debug(f"world_size = {world_size}, rank = {rank}")

    if world_size is not None and rank is not None:
        assert isinstance(rank, int)
        assert (
            loader_config.batch_size % world_size == 0
        ), "batch_size % world_size /= 0."

    dict_dataloaders: dict[str, DataLoader] = {}
    dict_samplers: dict[str, DistributedSampler] = {}

    for kind in train_valid_test_kinds:
        logger.debug(f"\n{kind} dataloader and sampler is being made:")
        dataset = dataset_initializer(
            file_paths=dict_file_paths[kind],
            config=dataset_config,
        )

        if world_size is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=loader_config.batch_size,
                drop_last=(True if kind == "train" else False),
                shuffle=(True if kind == "train" else False),
                pin_memory=True,
                num_workers=loader_config.num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=loader_config.seed,
                shuffle=(True if kind == "train" else False),
                drop_last=(True if kind == "train" else False),
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=loader_config.batch_size // world_size,
                pin_memory=True,
                num_workers=loader_config.num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(),
                drop_last=(True if kind == "train" else False),
            )

            if rank == 0:
                logger.debug(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}"
                )
    return dict_dataloaders, dict_samplers
