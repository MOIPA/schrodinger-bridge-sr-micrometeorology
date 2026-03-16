import argparse
import copy
import datetime
import gc
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.ddpm.ddpm_framework import DDPM
from src.dl_model.model_maker import make_model
from src.dl_train.ddpm_optim_helper import optimize_ddpm
from src.dl_train.exp_moving_ave import EMA
from src.dl_train.loss_maker import make_loss
from src.dl_train.optim_helper import make_optimizer
from src.utils.io_pickle import write_pickle
from src.utils.random_seed_helper import set_seeds

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42)

ROOT_DIR = str(pathlib.Path(os.environ["PYTHONPATH"]).resolve())
EXPERIMENT_NAME = "ExperimentDiffusionModel"

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")


def get_model_ddpm_loader(
    org_config,
    model_weight_path,
    data_kind,
    device,
    ddpm_config=None,
    root_dir=None,
):
    config = copy.deepcopy(org_config)

    assert config.data.is_clipped == False
    config.data.hr_cropped_shape = config.data.hr_data_shape
    assert config.data.hr_cropped_shape == config.data.hr_data_shape

    if ddpm_config is None:
        ddpm_config = config.ddpm

    model = make_model(config.model).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    _ = model.eval()

    ddpm = DDPM(config=ddpm_config, neural_net=model)
    assert id(ddpm.net) == id(model)
    ddpm.set_noise_schedule(phase="test")

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=(ROOT_DIR if root_dir is None else root_dir),
        loader_config=config.loader,
        dataset_config=config.data,
        world_size=None,
        rank=None,
        train_valid_test_kinds=[data_kind],
    )

    return model, ddpm, dict_loaders[data_kind]


def make_data_for_inference(n_data, dataset):
    y0, y_cond, x = [], [], []
    np.random.seed(42)
    indices = np.random.choice(len(dataset), n_data, replace=False)

    if n_data >= len(dataset):
        indices = np.arange(len(dataset))
        logger.info("All data will be used for inference.")

    for i in indices:
        data = dataset[i]
        y0.append(data["hr_tm002m"][None, ...])  # add channel dim.
        y_cond.append(data["x"])
        x.append(data["lr_tm002m"][None, ...])  # add channel dim.
    y0 = torch.stack(y0)
    y_cond = torch.stack(y_cond)
    x = torch.stack(x)
    del data
    assert y0.shape == x.shape == (n_data, 1, 320, 320)
    assert y_cond.shape == (n_data, 31, 320, 320)
    return y0, y_cond, x


if __name__ == "__main__":
    try:
        config_path: str = parser.parse_args().config_path
        device: str = parser.parse_args().device

        config_name = os.path.basename(config_path).split(".")[0]

        config = load_config(EXPERIMENT_NAME, config_path)

        result_dir_path = f"{ROOT_DIR}/data/DL_result/{EXPERIMENT_NAME}/{config_name}"
        os.makedirs(result_dir_path, exist_ok=True)
        logger.addHandler(FileHandler(f"{result_dir_path}/log.txt"))

        logger.info("\n" + "*" * 50)
        logger.info("Show configuration")
        logger.info("*" * 50 + "\n")
        logger.info(f"{EXPERIMENT_NAME=}")
        logger.info(f"{config_name=}")
        logger.info(f"{config_path=}")
        logger.info(f"{result_dir_path=}")
        logger.info(f"\nInput config = {config.to_json_str()}\n")

        dict_loaders, _ = make_dataloaders_and_samplers(
            root_dir=ROOT_DIR,
            loader_config=config.loader,
            dataset_config=config.data,
            world_size=None,
            rank=None,
            train_valid_test_kinds=["train", "valid"],
        )

        set_seeds(config.train.seed)
        net = make_model(config.model).to(device)
        ema_net = copy.deepcopy(net).to(device)
        ddpm = DDPM(config=config.ddpm, neural_net=net)
        assert id(ddpm.net) == id(net)
        assert id(ema_net) != id(net)

        ema = EMA(config.train.ema_decay)
        loss_fn = make_loss(config.train.loss)
        optimizer = make_optimizer(config.train, ddpm.net)
        scaler = torch.cuda.amp.GradScaler()

        model_weight_path = f"{result_dir_path}/model_weight.pth"
        ema_model_weight_path = f"{result_dir_path}/ema_model_weight.pth"
        loss_history_path = f"{result_dir_path}/model_loss_history.csv"

        all_scores = []
        best_epoch = 0
        best_loss = np.inf
        es_cnt = 0

        ddpm.set_noise_schedule(phase="train")

        logger.info("\n" + "*" * 50)
        logger.info("Train model")
        logger.info("*" * 50 + "\n")
        logger.info(f"Train start: {datetime.datetime.now(datetime.timezone.utc)} UTC")
        logger.info(f"Saving interval = {config.train.save_interval}")
        logger.info(f"EMA decay rate = {ema.decay}")

        set_seeds(config.train.seed)
        start_time = time.time()

        for epoch in tqdm(range(config.train.epochs + 1)):
            _time = time.time()
            logger.info(f"Epoch {epoch+1} / {config.train.epochs}")

            losses = {}
            for mode in ["train"]:
                loss = optimize_ddpm(
                    dataloader=dict_loaders[mode],
                    ddpm=ddpm,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    mode=mode,
                    epoch=epoch,
                    scaler=scaler,
                    use_amp=config.train.use_amp,
                    ema=ema,
                    ema_net=ema_net,
                )
                losses[mode] = loss
            all_scores.append(losses)

            if losses["train"] > best_loss:
                es_cnt += 1
                logger.info(f"ES count = {es_cnt}")
                if es_cnt >= config.train.early_stopping_patience:
                    break
            else:
                es_cnt = 0
                best_epoch = epoch + 1
                best_loss = losses["train"]
                logger.info("Best loss is updated and ES count is reset.")
                torch.save(ddpm.net.state_dict(), model_weight_path)
                if ema.decay is not None and 0.0 < ema.decay < 1.0:
                    torch.save(ema_net.state_dict(), ema_model_weight_path)

            if (epoch + 1) % config.train.save_interval == 0:
                logger.info(
                    f"Epoch = {(epoch + 1)}. Save the model and ema model results"
                )
                p = f"{result_dir_path}/model_weight_{(epoch + 1):04}.pth"
                torch.save(ddpm.net.state_dict(), p)
                if ema.decay is not None and 0.0 < ema.decay < 1.0:
                    p = f"{result_dir_path}/ema_model_weight_{(epoch + 1):04}.pth"
                    torch.save(ema_net.state_dict(), p)

            if epoch % 10 == 0:
                pd.DataFrame(all_scores).to_csv(loss_history_path, index=False)

            logger.info(f"Elapsed time = {time.time() - _time} sec")
            logger.info("-" * 10)

        pd.DataFrame(all_scores).to_csv(loss_history_path, index=False)
        end_time = time.time()

        logger.info(
            f"Train end: {datetime.datetime.now(datetime.timezone.utc).isoformat()} UTC"
        )
        logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")
        logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")

        del (
            dict_loaders,
            net,
            ema_net,
            ddpm,
            ema,
            loss_fn,
            optimizer,
            scaler,
            all_scores,
        )
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("\n" + "*" * 50)
        logger.info("Make inference")
        logger.info("*" * 50 + "\n")

        for epoch in range(0, config.train.epochs + 1, config.train.save_interval):
            p = f"{result_dir_path}/model_weight_{epoch:04}.pth"
            if not os.path.exists(p):
                continue
            pickle_file_path = f"{result_dir_path}/inference_{epoch:04}.pickle"
            logger.info(f"\nInference at epoch {epoch}.")
            start_time = time.time()

            model, ddpm, loader = get_model_ddpm_loader(
                org_config=config, model_weight_path=p, data_kind="valid", device=device
            )
            n_data = 60
            y0, y_cond, x = make_data_for_inference(
                n_data=n_data, dataset=loader.dataset
            )

            set_seeds(config.train.seed)
            pred, _ = ddpm.backward_sample_y(
                y_cond=y_cond.to(device), n_return_step=None
            )
            assert pred.shape == x.shape == y0.shape == (n_data, 1, 320, 320)

            pred = (
                loader.dataset._scale_inversely(pred, "re_tm")
                .detach()
                .cpu()
                .to(torch.float32)
            )
            pred = pred + x  # add residual to lr data

            y0 = y0.detach().cpu().to(torch.float32)

            write_pickle({"y0": y0, "pred": pred}, pickle_file_path)
            end_time = time.time()

            logger.info(f"Inference out = {pickle_file_path}")
            logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min\n")

            del model, ddpm, loader, y0, y_cond, pred
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("End all inference processes.")

    except Exception as e:
        logger.info("\n" + "*" * 50)
        logger.info("Error")
        logger.info("*" * 50 + "\n")
        logger.error(e)
        logger.error(traceback.format_exc())
