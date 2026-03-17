import sys
import pathlib

# Add project root to the Python path so 'src' module can be found when running locally
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import argparse
import copy
import datetime
import gc
import os
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dl_config.config_loader import load_config
from src.dl_config.schrodinger_bridge_model_config import (
    ExperimentSchrodingerBridgeModelConfig,
)
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.model_maker import make_model
from src.dl_model.si_follmer.si_follmer_framework import (
    SIFollmerConfig,
    StochasticInterpolantFollmer,
)
from src.dl_train.exp_moving_ave import EMA
from src.dl_train.optim_helper import make_optimizer
from src.dl_train.si_optim_helper import optimize_si
from src.utils.io_pickle import write_pickle
from src.utils.random_seed_helper import set_seeds

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42)

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
EXPERIMENT_NAME = "ExperimentSchrodingerBridgeModel"

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--experiment_name", type=str, default=EXPERIMENT_NAME,
                    help="Experiment name, e.g. ExperimentSchrodingerBridgeModel or ExperimentSchrodingerBridge3dWind")


def get_model_si_loader(
    org_config: ExperimentSchrodingerBridgeModelConfig,
    model_weight_path: str,
    data_kind: str,
    device: str,
    si_config: Optional[SIFollmerConfig] = None,
    root_dir: Optional[str] = None,
):
    config = copy.deepcopy(org_config)

    assert config.data.is_clipped == False
    config.data.hr_cropped_shape = config.data.hr_data_shape
    assert config.data.hr_cropped_shape == config.data.hr_data_shape

    model = make_model(config.model).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    _ = model.eval()

    si = StochasticInterpolantFollmer(
        config=(config.si if si_config is None else si_config), neural_net=model
    )
    assert id(si.net) == id(model)

    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=(ROOT_DIR if root_dir is None else root_dir),
        loader_config=config.loader,
        dataset_config=config.data,
        world_size=None,
        rank=None,
        train_valid_test_kinds=[data_kind],
    )

    return model, si, dict_loaders[data_kind]


def make_data_for_inference(n_data, dataset):
    np.random.seed(42)
    
    num_available = len(dataset)
    if n_data >= num_available:
        logger.info(f"Requested {n_data} samples, but only {num_available} are available. Using all available data.")
        n_data = num_available
        indices = np.arange(num_available)
    else:
        indices = np.random.choice(num_available, n_data, replace=False)

    y0, y1, y_cond = [], [], []
    for i in indices:
        data = dataset[i]
        y0.append(data["y0"])
        y1.append(data["y"])
        y_cond.append(data["x"])

    y0 = torch.stack(y0)
    y1 = torch.stack(y1)
    y_cond = torch.stack(y_cond)
    del data

    # NOTE: The shape assertion for y_cond might need adjustment based on final data shape after cropping
    # The number of channels should now be 30. The spatial dimensions depend on hr_cropped_shape.
    # assert y0.shape[1] == 1 and y1.shape[1] == 1 and y_cond.shape[1] == 30
    
    return y0, y1, y_cond


if __name__ == "__main__":
    try:
        args = parser.parse_args()
        config_path: str = args.config_path
        device: str = args.device
        experiment_name: str = args.experiment_name

        config_name = os.path.basename(config_path).split(".")[0]

        config = load_config(
            experiment_name, config_path
        )

        result_dir_path = f"{ROOT_DIR}/data/DL_result/{experiment_name}/{config_name}"
        os.makedirs(result_dir_path, exist_ok=True)
        logger.addHandler(FileHandler(f"{result_dir_path}/log.txt"))

        logger.info("\n" + "*" * 50)
        logger.info("Show configuration")
        logger.info("*" * 50 + "\n")
        logger.info(f"{experiment_name=}")
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
        logger.info(f"DEBUG-DICT_LOADERS:{dict_loaders['train'].__len__()}")

        set_seeds(config.train.seed)
        net = make_model(config.model).to(device)
        ema_net = copy.deepcopy(net).to(device)
        si = StochasticInterpolantFollmer(config=config.si, neural_net=net)
        assert id(si.net) == id(net)
        assert id(ema_net) != id(net)

        ema = EMA(config.train.ema_decay)
        optimizer = make_optimizer(config.train, si.net)
        scaler = torch.cuda.amp.GradScaler()

        loss_history_path = f"{result_dir_path}/model_loss_history.csv"
        checkpoint_path = f"{result_dir_path}/checkpoint.pth"

        all_scores = []
        best_epoch = 0
        best_loss = np.inf
        es_cnt = 0
        start_epoch = 0

        # --- Check for existing checkpoint ---
        if os.path.exists(checkpoint_path):
            logger.info(f"Checkpoint found at '{checkpoint_path}'. Loading...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check for new dictionary format vs old raw state_dict format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                logger.info("New format checkpoint detected. Performing full resume.")
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                es_cnt = checkpoint.get('es_cnt', 0)
                
                if 'ema_model_state_dict' in checkpoint and checkpoint['ema_model_state_dict'] is not None:
                    ema_net.load_state_dict(checkpoint['ema_model_state_dict'])
                
                logger.info(f"Resuming from epoch {start_epoch}. Best loss so far: {best_loss:.8f}")
            else:
                # Handle old format where only the model's state_dict was saved
                logger.warning("Old format checkpoint detected. Loading model weights only.")
                logger.warning("Optimizer state and epoch number will not be restored. Training will start with a fresh optimizer.")
                net.load_state_dict(checkpoint)
                # start_epoch, best_loss, etc., will remain at their default initial values
            
        else:
            logger.info(f"No checkpoint found at '{checkpoint_path}'. Starting training from scratch.")

        logger.info("\n" + "*" * 50)
        logger.info("Train model")
        logger.info("*" * 50 + "\n")
        logger.info(f"Train start: {datetime.datetime.now(datetime.timezone.utc)} UTC")
        logger.info(f"Saving interval = {config.train.save_interval}")
        logger.info(f"EMA decay rate = {ema.decay}")

        set_seeds(config.train.seed + start_epoch) # Offset seed by start_epoch
        start_time = time.time()

        for epoch in tqdm(range(start_epoch, config.train.epochs + 1)):
            _time = time.time()
            logger.info(f"Epoch {epoch+1} / {config.train.epochs}")

            losses = {}
            for mode in ["train"]:
                loss = optimize_si(
                    dataloader=dict_loaders[mode],
                    si=si,
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

            if losses["train"] < best_loss:
                es_cnt = 0
                best_epoch = epoch + 1
                best_loss = losses["train"]
                logger.info("Best loss is updated and ES count is reset.")
                
                # --- Save comprehensive checkpoint ---
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': si.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'es_cnt': es_cnt,
                    'ema_model_state_dict': ema_net.state_dict() if ema.decay is not None and 0.0 < ema.decay < 1.0 else None,
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved to '{checkpoint_path}'")

            else:
                es_cnt += 1
                logger.info(f"ES count = {es_cnt}")
                if es_cnt >= config.train.early_stopping_patience:
                    break

            if (epoch + 1) % config.train.save_interval == 0:
                logger.info(
                    f"Epoch = {(epoch + 1)}. Save a periodic model snapshot."
                )
                # Still save periodic snapshots separately if needed
                p = f"{result_dir_path}/model_weight_{(epoch + 1):04}.pth"
                torch.save(si.net.state_dict(), p)
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
            si,
            ema,
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

            model, si, loader = get_model_si_loader(
                org_config=config, model_weight_path=p, data_kind="valid", device=device
            )
            y0, y1, y_cond = make_data_for_inference(n_data=60, dataset=loader.dataset)

            set_seeds(config.train.seed)
            pred, _ = si.sample_y1_bare_diffusion(
                y0=y0.to(device), y_cond=y_cond.to(device), n_return_step=None
            )

            pred = (
                loader.dataset._scale_inversely(pred, "hr_tm")
                .detach()
                .cpu()
                .to(torch.float32)
            )
            y1 = (
                loader.dataset._scale_inversely(y1, "hr_tm")
                .detach()
                .cpu()
                .to(torch.float32)
            )
            write_pickle({"y1": y1, "pred": pred}, pickle_file_path)
            end_time = time.time()

            logger.info(f"Inference out = {pickle_file_path}")
            logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min\n")

            del model, si, loader, y0, y1, y_cond, pred
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("End all inference processes.")

    except Exception as e:
        logger.info("\n" + "*" * 50)
        logger.info("Error")
        logger.info("*" * 50 + "\n")
        logger.error(e)
        logger.error(traceback.format_exc())
