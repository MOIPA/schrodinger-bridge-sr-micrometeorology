"""回归步训练脚本(阶段 3 架构对比「回归 + 扩散两步法」的回归步)。

外壳逐段复用 scripts/train_schrodinger_bridge_model.py,产物与该脚本完全同构:
- 实验目录 data/DL_result/{experiment_name}/{config_name}/;
- checkpoint.pth 为同一 dict 结构('epoch'/'model_state_dict'/
  'optimizer_state_dict'/'best_loss'/'es_cnt'/'ema_model_state_dict'),
  同一批 eval/ops(如 scripts/outline/90_agl_eval_phase1.py 的 build_si 式加载、
  src/dl_model/regression_wrapper.py 的 RegressionModel)可按同路径同键加载;
- 同一 `Train end: ` 结束标记、model_loss_history.csv、model_weight_XXXX.pth /
  ema_model_weight_XXXX.pth 周期快照、seed 与 early-stop 逻辑。

与 SI 脚本的差异(去掉 SI 专属逻辑):
- 训练循环为 optimize_reg(单次前向 L1 残差回归,gamma 固定 t=1 常数嵌入),
  无 SI 对象、无物理项/物理 warmup;
- 不做训练后推理(原 SI 推理块依赖 si.sample_y1_bare_diffusion),评估统一走
  共享评估链路消费 checkpoint.pth。

用法(仓库根目录):
  python scripts/train_regression_model.py \
      --config_path configs/深圳/phase3/config_wind_canvas_reg_xxx.yml \
      --experiment_name ExperimentSchrodingerBridgeWindCanvas --device cuda:0
"""

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

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.model_maker import make_model
from src.dl_train.exp_moving_ave import EMA
from src.dl_train.optim_helper import make_optimizer
from src.dl_train.reg_optim_helper import optimize_reg
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
                    help="Experiment name, e.g. ExperimentSchrodingerBridgeModel or ExperimentSchrodingerBridgeWindCanvas")


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

        # canvas 数据集:通道数由输入组/layer 集合决定,配置里的 in/out_channel 必须一致
        ds_train = dict_loaders["train"].dataset
        if hasattr(ds_train, "input_channel_names"):
            n_in = len(ds_train.input_channel_names())
            n_out = len(ds_train.target_channel_names())
            assert n_out == config.model.out_channel, (
                f"target 通道 {n_out} != model.out_channel {config.model.out_channel}")
            assert n_in + n_out == config.model.in_channel, (
                f"状态 {n_out} + 条件 {n_in} != model.in_channel {config.model.in_channel}")
            logger.info(f"channel check: {n_out} (state) + {n_in} (cond) = {n_in + n_out}")

        set_seeds(config.train.seed)
        net = make_model(config.model).to(device)
        ema_net = copy.deepcopy(net).to(device)
        assert id(ema_net) != id(net)

        ema = EMA(config.train.ema_decay)
        optimizer = make_optimizer(config.train, net)
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

        set_seeds(config.train.seed + start_epoch)  # Offset seed by start_epoch
        start_time = time.time()

        for epoch in tqdm(range(start_epoch, config.train.epochs + 1)):
            _time = time.time()
            logger.info(f"Epoch {epoch+1} / {config.train.epochs}")

            # 学习率调度(稳定性诊断用):cosine = 按 epoch 余弦衰减到 0
            _sched = getattr(config.train, "lr_schedule", "none") or "none"
            if _sched == "cosine":
                _lr = float(config.train.learning_rate) * 0.5 * (
                    1.0 + np.cos(np.pi * float(epoch) / max(1, config.train.epochs)))
                for _g in optimizer.param_groups:
                    _g["lr"] = _lr
                logger.info(f"lr_schedule=cosine: lr={_lr:.3e}")

            losses = {}
            for mode in (["train", "valid"] if "valid" in dict_loaders else ["train"]):
                loss = optimize_reg(
                    dataloader=dict_loaders[mode],
                    net=net,
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

            # 模型选择优先用验证损失;无 valid 加载器时回退训练损失
            select_loss = losses.get("valid", losses["train"])
            if select_loss < best_loss:
                es_cnt = 0
                best_epoch = epoch + 1
                best_loss = select_loss
                logger.info(
                    f"Best loss is updated and ES count is reset "
                    f"(train={losses['train']:.8f}, valid={losses.get('valid', float('nan')):.8f})"
                )

                # --- Save comprehensive checkpoint (结构与 SI 脚本逐键一致) ---
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
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
                torch.save(net.state_dict(), p)
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
            ema,
            optimizer,
            scaler,
            all_scores,
        )
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("\n" + "*" * 50)
        logger.info("Skip post-train inference (regression step)")
        logger.info("*" * 50 + "\n")
        logger.info(
            "回归步训练脚本不做训练后推理;评估由共享评估链路消费 "
            f"'{checkpoint_path}' 与周期快照统一完成"
        )

    except Exception as e:
        logger.info("\n" + "*" * 50)
        logger.info("Error")
        logger.info("*" * 50 + "\n")
        logger.error(e)
        logger.error(traceback.format_exc())
