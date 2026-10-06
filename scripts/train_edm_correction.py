"""EDM 订正步训练脚本(阶段 3 架构对比「回归 + 扩散两步法」的扩散/订正步)。

外壳逐段复用 scripts/train_schrodinger_bridge_model.py(与 train_regression_model.py
同族),产物与该脚本完全同构:
- 实验目录 data/DL_result/{experiment_name}/{config_name}/;
- checkpoint.pth 为同一 dict 结构('epoch'/'model_state_dict'/'optimizer_state_dict'/
  'best_loss'/'es_cnt'/'ema_model_state_dict'),model_state_dict 是 EDMCorrector.net
  的裸键,可用 `EDMCorrector(cfg).net.load_state_dict(ckpt['model_state_dict'])` 加载;
  另附 EDM 专属键('sigma_data'/'edm_config'/'reg'),不与 SI 键冲突;
- 同一 `Train end: ` 结束标记、model_loss_history.csv、model_weight_XXXX.pth /
  ema_model_weight_XXXX.pth 周期快照、seed 与 early-stop 逻辑。

流程:
1. 配置:base(config_path,= canvas yml)+ `edm:` 段(EDMCorrectorConfig)+
   可选 `reg:` 段(reg_tag/reg_config_path);CLI 同名参数优先。
   因 base 配置类不接受 edm/reg 键,脚本把这两段剥离后经临时 yml 走 load_config。
2. 冻结回归网:按 reg_config_path 建 net 并加载
   data/DL_result/{exp}/config_wind_canvas_{reg_tag}/checkpoint.pth 的
   'model_state_dict',requires_grad_(False)+eval();
   每 batch 前向 ŷ1_reg = y0 + net(yt=y0, y_cond=x, gamma=ones)(与
   src/dl_model/regression_wrapper.py / reg_optim_helper.optimize_reg 同口径)。
3. σ_d:优先 --sigma_d,其次 resume 的 checkpoint 里存的值,否则用前 N 个 train
   batch 的残差估计(std 或 mean-abs),打印并写进 config_resolved.yml 与 checkpoint。
4. 训练:逐 batch 取样 σ(截断对数正态)→ EDM 加权损失 → 反传(AMP/NaN 跳过/梯度
   裁剪/EMA 与 SI 同款);valid 固定 per-epoch generator(σ 与噪声可复现)。

配置里的 `edm:`/`reg:` 段(83 配置生成按此写;未列字段取 EDMCorrectorConfig 默认值):
  edm:
    inner_channel: 64
    channel_mults: [1, 2, 4]
    sigma_data: null          # 脚本估计/CLI 指定后写入 config_resolved.yml 与 checkpoint
    sigma_d_estimator: std    # std | mean_abs
    sigma_d_est_batches: 8
  reg:
    reg_tag: p3_reg
    reg_config_path: configs/深圳/arch/config_wind_canvas_p3_reg.yml

用法(仓库根目录):
  python scripts/train_edm_correction.py \
      --config_path configs/深圳/arch/config_wind_canvas_edm_p3_reg.yml \
      --reg_tag p3_reg \
      --reg_config_path configs/深圳/arch/config_wind_canvas_p3_reg.yml \
      --experiment_name ExperimentSchrodingerBridgeWindCanvas --device cuda:0
"""

import sys
import pathlib

# Add project root to the Python path so 'src' module can be found when running locally
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import argparse
import copy
import dataclasses
import datetime
import gc
import os
import random
import tempfile
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.edm_correction import (
    EDMCorrector,
    EDMCorrectorConfig,
    sample_log_sigma,
)
from src.dl_model.model_maker import make_model
from src.dl_train.exp_moving_ave import EMA
from src.dl_train.optim_helper import make_optimizer
from src.utils.average_meter import AverageMeter
from src.utils.random_seed_helper import set_seeds

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42)

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
EXPERIMENT_NAME = "ExperimentSchrodingerBridgeWindCanvas"

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True,
                    help="EDM 配置(canvas yml + `edm:` 段,可选 `reg:` 段)")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--experiment_name", type=str, default=EXPERIMENT_NAME,
                    help="实验名(决定 data/DL_result 下的目录),默认 WindCanvas")
parser.add_argument("--reg_tag", type=str, default=None,
                    help="回归产物 tag;回归 checkpoint 路径为 "
                         "data/DL_result/<reg_exp>/config_wind_canvas_<reg_tag>/checkpoint.pth")
parser.add_argument("--reg_config_path", type=str, default=None,
                    help="回归配置 yml(建回归网用,通道/arch 须与训练时一致)")
parser.add_argument("--reg_experiment_name", type=str, default=None,
                    help="回归产物的实验名,默认与 --experiment_name 相同")
parser.add_argument("--reg_checkpoint_path", type=str, default=None,
                    help="直接指定回归 checkpoint 路径(优先于 --reg_tag 拼路径)")
parser.add_argument("--reg_weights", type=str, default="model",
                    choices=["model", "ema"],
                    help="回归网取 checkpoint 里的 model_state_dict(默认,与评估"
                         "链路 --weights 默认一致)或 ema_model_state_dict")
parser.add_argument("--sigma_d", type=float, default=None,
                    help="直接指定 σ_d(优先于估计与 checkpoint 里存的值)")
parser.add_argument("--sigma_d_estimator", type=str, default=None,
                    choices=[None, "std", "mean_abs"],
                    help="σ_d 估计口径,默认取 edm.sigma_d_estimator(std)")
parser.add_argument("--sigma_d_est_batches", type=int, default=None,
                    help="σ_d 估计用前 N 个 train batch,默认取 edm.sigma_d_est_batches")


def load_edm_config(experiment_name: str, config_path: str):
    """读 yml,剥离 edm/reg 段后走 load_config 建 base 配置。

    返回 (base_config, edm_config, reg_section: dict, full_raw: dict)。
    """
    with open(config_path) as f:
        full_raw = yaml.safe_load(f)
    raw = copy.deepcopy(full_raw)
    edm_raw = raw.pop("edm", None) or {}
    reg_raw = raw.pop("reg", None) or {}
    if not edm_raw:
        logger.warning("配置无 `edm:` 段,全部取 EDMCorrectorConfig 默认值")

    # base 配置类不接受 edm/reg 键:写到临时 yml 再交给统一 loader(路径已在日志中给出)
    tmp_dir = tempfile.mkdtemp(prefix="edm_correction_cfg_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(config_path))
    with open(tmp_path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
    try:
        base_config = load_config(experiment_name, tmp_path)
    finally:
        os.remove(tmp_path)
        os.rmdir(tmp_dir)
    return base_config, EDMCorrectorConfig(**edm_raw), reg_raw, full_raw


def reg_forward(reg_net, y0, y_cond) -> torch.Tensor:
    """ŷ1_reg = y0 + net_reg(yt=y0, y_cond=x, gamma=ones)(gamma 形状 (B,))。"""
    gamma = torch.ones(y0.shape[0], device=y0.device, dtype=torch.float32)
    return y0 + reg_net(yt=y0, y_cond=y_cond, gamma=gamma)


def estimate_sigma_d(dataloader, reg_net, n_batches, estimator, device):
    """用前 N 个 batch 的残差 r = y1 − ŷ1_reg 估计 σ_d(全局标量,日志打印双口径)。"""
    chunks = []
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        y0 = batch["y0"].to(device, non_blocking=True)
        y1 = batch["y"].to(device, non_blocking=True)
        y_cond = batch["x"].to(device, non_blocking=True)
        with torch.no_grad():
            y1_reg = reg_forward(reg_net, y0, y_cond)
        chunks.append((y1 - y1_reg).flatten())
    if not chunks:
        raise RuntimeError("σ_d 估计失败:train loader 为空")
    r = torch.cat(chunks).to(torch.float32)
    mean_abs = r.abs().mean().item()
    std = r.std(unbiased=True).item()
    logger.info(
        f"sigma_d estimation over {len(chunks)} train batches "
        f"({r.numel()} residual values): mean_abs={mean_abs:.6g}, std={std:.6g}, "
        f"estimator={estimator}")
    if estimator == "mean_abs":
        return mean_abs
    if estimator == "std":
        return std
    raise ValueError(f"未知 sigma_d_estimator: {estimator}")


def run_epoch(
    dataloader,
    corrector: EDMCorrector,
    reg_net,
    optimizer,
    epoch: int,
    mode: str,
    scaler,
    use_amp: bool,
    ema: EMA,
    ema_net,
):
    """跑一个 epoch(或一次 valid),返回平均损失;工程行为对齐 optimize_si/optimize_reg。"""
    loss_meter = AverageMeter()

    device = str(next(corrector.net.parameters()).device)

    if mode == "train":
        corrector.net.train()
    elif mode in ["valid", "test"]:
        corrector.net.eval()
    else:
        raise ValueError(f"{mode} is not supported.")

    random.seed(epoch)
    np.random.seed(epoch)

    device_type = "cuda" if "cuda" in device else "cpu"
    nan_skipped = 0
    for step, batch in enumerate(dataloader):
        y0 = batch["y0"].to(device, non_blocking=True)
        y1 = batch["y"].to(device, non_blocking=True)
        y_cond = batch["x"].to(device, non_blocking=True)

        # 冻结回归网前向(fp32、no_grad;与评估端回归包装同口径)
        with torch.no_grad():
            y1_reg = reg_forward(reg_net, y0, y_cond)
        r = y1 - y1_reg
        ctx = torch.cat([y1_reg, y0, y_cond], dim=1)

        # valid 用 per-(epoch,step) 固定 generator:σ 与噪声都可复现
        gen = None
        if mode != "train":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(epoch) * 1000003 + step)
        sigma = sample_log_sigma(y0.shape[0], corrector.c, device=device,
                                 generator=gen)
        noise = None
        if gen is not None:
            noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
            noise = noise.to(device=device, dtype=r.dtype)

        if mode == "train":
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    loss = corrector.loss(r, sigma, ctx)
                if not torch.isfinite(loss):
                    nan_skipped += 1
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(corrector.net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = corrector.loss(r, sigma, ctx)
                if not torch.isfinite(loss):
                    nan_skipped += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(corrector.net.parameters(), max_norm=1.0)
                optimizer.step()

            if ema.decay is not None and 0.0 < ema.decay < 1.0:
                ema.update_model_average(current_model=corrector.net, ma_model=ema_net)
        else:
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=torch.float16, enabled=use_amp
            ):
                loss = corrector.loss(r, sigma, ctx, noise=noise)

        loss_meter.update(loss.item(), n=y_cond.shape[0])

    if nan_skipped > 0:
        logger.warning(f"{mode}: skipped {nan_skipped} batches with NaN/Inf loss")
    logger.info(f"{mode} error: avg loss = {loss_meter.avg:.8f}")
    return loss_meter.avg


if __name__ == "__main__":
    try:
        args = parser.parse_args()
        config_path: str = args.config_path
        device: str = args.device
        experiment_name: str = args.experiment_name
        reg_experiment_name: str = args.reg_experiment_name or experiment_name

        config_name = os.path.basename(config_path).split(".")[0]

        config, edm_config, reg_section, full_raw = load_edm_config(
            experiment_name, config_path)

        # --- 回归产物定位 ---
        reg_tag = args.reg_tag or reg_section.get("reg_tag")
        reg_config_path = args.reg_config_path or reg_section.get("reg_config_path")
        if reg_config_path and not os.path.isabs(reg_config_path):
            reg_config_path = os.path.join(ROOT_DIR, reg_config_path)
        if reg_tag is None and args.reg_checkpoint_path is None:
            raise SystemExit("缺少回归产物标识:给 --reg_tag(或配置 reg.reg_tag)或 --reg_checkpoint_path")
        reg_ckpt_path = args.reg_checkpoint_path or (
            f"{ROOT_DIR}/data/DL_result/{reg_experiment_name}/"
            f"config_wind_canvas_{reg_tag}/checkpoint.pth")
        if reg_config_path is None:
            logger.warning(
                "--reg_config_path 未给:回退用 EDM 配置的 model 段建回归网"
                "(仅当两者逐键一致时才正确)")
            reg_config = config
        else:
            reg_config = load_config(experiment_name, reg_config_path)
        if not os.path.exists(reg_ckpt_path):
            raise SystemExit(f"回归 checkpoint 不存在: {reg_ckpt_path}")

        # --- σ_d 估计/指定的口径(CLI > 配置 > 默认) ---
        if args.sigma_d_estimator is not None:
            edm_config.sigma_d_estimator = args.sigma_d_estimator
        if args.sigma_d_est_batches is not None:
            edm_config.sigma_d_est_batches = args.sigma_d_est_batches

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
        logger.info(f"{reg_experiment_name=}, {reg_tag=}")
        logger.info(f"{reg_config_path=}")
        logger.info(f"{reg_ckpt_path=}")
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

        # canvas 数据集:通道数由输入组/layer 集合决定,配置里的 in/out/ctx 必须一致
        ds_train = dict_loaders["train"].dataset
        if hasattr(ds_train, "input_channel_names"):
            n_in = len(ds_train.input_channel_names())
            n_out = len(ds_train.target_channel_names())
            assert n_out == edm_config.out_channel, (
                f"target 通道 {n_out} != edm.out_channel {edm_config.out_channel}")
            assert 2 * n_out + n_in == edm_config.ctx_channel, (
                f"ctx 通道 {2 * n_out + n_in} != edm.ctx_channel {edm_config.ctx_channel}")
            assert n_out == getattr(reg_config.model, "out_channel", n_out), (
                "回归配置 out_channel 与 EDM 配置不一致")
            logger.info(
                f"channel check: state {n_out} + ctx({n_out}+{n_out}+{n_in}) "
                f"= ctx_channel {edm_config.ctx_channel}")

        # --- 回归网:冻结 + eval ---
        set_seeds(config.train.seed)
        reg_net = make_model(reg_config.model).to(device)
        reg_checkpoint = torch.load(reg_ckpt_path, map_location=device,
                                    weights_only=False)
        if isinstance(reg_checkpoint, dict) and "model_state_dict" in reg_checkpoint:
            if args.reg_weights == "ema":
                assert reg_checkpoint.get("ema_model_state_dict") is not None, (
                    "回归 checkpoint 无 ema_model_state_dict,不能用 --reg_weights ema")
                reg_net.load_state_dict(reg_checkpoint["ema_model_state_dict"])
            else:
                reg_net.load_state_dict(reg_checkpoint["model_state_dict"])
            logger.info(
                f"回归 checkpoint 加载({args.reg_weights}): "
                f"epoch={reg_checkpoint.get('epoch')}, "
                f"best_loss={reg_checkpoint.get('best_loss')}")
        else:
            logger.warning("回归 checkpoint 为裸 state_dict 格式,仅加载权重")
            reg_net.load_state_dict(reg_checkpoint)
        reg_net.requires_grad_(False)
        reg_net.eval()

        # --- σ_d 解析(CLI > resume checkpoint > 估计) ---
        loss_history_path = f"{result_dir_path}/model_loss_history.csv"
        checkpoint_path = f"{result_dir_path}/checkpoint.pth"
        resume_checkpoint = None
        if os.path.exists(checkpoint_path):
            resume_checkpoint = torch.load(checkpoint_path, map_location=device,
                                           weights_only=False)
            if not (isinstance(resume_checkpoint, dict) and "model_state_dict" in resume_checkpoint):
                resume_checkpoint = None

        if args.sigma_d is not None:
            edm_config.sigma_data = float(args.sigma_d)
            logger.info(f"sigma_d 来源 CLI: {edm_config.sigma_data:.6g}")
        elif resume_checkpoint is not None and resume_checkpoint.get("sigma_data") is not None:
            edm_config.sigma_data = float(resume_checkpoint["sigma_data"])
            logger.info(
                f"sigma_d 来源 resume checkpoint: {edm_config.sigma_data:.6g} "
                "(与已训权重预条件口径一致,跳过估计)")
        else:
            edm_config.sigma_data = estimate_sigma_d(
                dataloader=dict_loaders["train"], reg_net=reg_net,
                n_batches=edm_config.sigma_d_est_batches,
                estimator=edm_config.sigma_d_estimator, device=device)
            logger.info(
                f"sigma_d 来源在线估计({edm_config.sigma_d_estimator}, "
                f"前 {edm_config.sigma_d_est_batches} 个 train batch): "
                f"{edm_config.sigma_data:.6g}")

        # 把生效的 edm 段(含 σ_d)与 reg 定位落盘:84 集合评估据此重建 corrector,
        # 保证采样复用与训练完全相同的 σ_d 与网络超参
        full_raw["edm"] = dataclasses.asdict(edm_config)
        full_raw["reg"] = {
            "reg_tag": reg_tag,
            "reg_config_path": reg_config_path,
            "reg_checkpoint_path": reg_ckpt_path,
            "reg_experiment_name": reg_experiment_name,
        }
        resolved_path = f"{result_dir_path}/config_resolved.yml"
        with open(resolved_path, "w") as f:
            yaml.safe_dump(full_raw, f, sort_keys=False, allow_unicode=True)
        logger.info(f"resolved config(edm 段含 sigma_data) -> {resolved_path}")

        # --- 订正网/优化器/EMA ---
        corrector = EDMCorrector(edm_config).to(device)
        ema_net = copy.deepcopy(corrector.net).to(device)
        assert id(ema_net) != id(corrector.net)
        n_params = sum(p.numel() for p in corrector.net.parameters())
        logger.info(f"EDMCorrector: {n_params / 1e6:.3f} M 参数, "
                    f"sigma_d={corrector.sigma_d:.6g}, "
                    f"steps={edm_config.steps}, rho={edm_config.rho}, "
                    f"sigma∈[{edm_config.sigma_min}, {edm_config.sigma_max}]")

        ema = EMA(config.train.ema_decay)
        optimizer = make_optimizer(config.train, corrector.net)
        scaler = torch.cuda.amp.GradScaler()

        all_scores = []
        best_epoch = 0
        best_loss = np.inf
        es_cnt = 0
        start_epoch = 0

        # --- Check for existing checkpoint(键结构与 SI 脚本一致)---
        if resume_checkpoint is not None:
            logger.info(f"Checkpoint found at '{checkpoint_path}'. Loading...")
            logger.info("New format checkpoint detected. Performing full resume.")
            corrector.net.load_state_dict(resume_checkpoint['model_state_dict'])
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            start_epoch = resume_checkpoint['epoch']
            best_loss = resume_checkpoint['best_loss']
            es_cnt = resume_checkpoint.get('es_cnt', 0)
            if resume_checkpoint.get('ema_model_state_dict') is not None:
                ema_net.load_state_dict(resume_checkpoint['ema_model_state_dict'])
            logger.info(f"Resuming from epoch {start_epoch}. "
                        f"Best loss so far: {best_loss:.8f}")
        else:
            logger.info(f"No checkpoint found at '{checkpoint_path}'. "
                        "Starting training from scratch.")

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

            # 学习率调度(与 SI/回归脚本同款):cosine = 按 epoch 余弦衰减到 0
            _sched = getattr(config.train, "lr_schedule", "none") or "none"
            if _sched == "cosine":
                _lr = float(config.train.learning_rate) * 0.5 * (
                    1.0 + np.cos(np.pi * float(epoch) / max(1, config.train.epochs)))
                for _g in optimizer.param_groups:
                    _g["lr"] = _lr
                logger.info(f"lr_schedule=cosine: lr={_lr:.3e}")

            losses = {}
            for mode in (["train", "valid"] if "valid" in dict_loaders else ["train"]):
                loss = run_epoch(
                    dataloader=dict_loaders[mode],
                    corrector=corrector,
                    reg_net=reg_net,
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

                # --- Save comprehensive checkpoint(+EDM 专属键)---
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': corrector.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'es_cnt': es_cnt,
                    'ema_model_state_dict': ema_net.state_dict() if ema.decay is not None and 0.0 < ema.decay < 1.0 else None,
                    'sigma_data': corrector.sigma_d,
                    'edm_config': dataclasses.asdict(edm_config),
                    'reg': {
                        'reg_tag': reg_tag,
                        'reg_config_path': reg_config_path,
                        'reg_checkpoint_path': reg_ckpt_path,
                        'reg_experiment_name': reg_experiment_name,
                    },
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved to '{checkpoint_path}'")

            else:
                es_cnt += 1
                logger.info(f"ES count = {es_cnt}")
                if es_cnt >= config.train.early_stopping_patience:
                    break

            if (epoch + 1) % config.train.save_interval == 0:
                logger.info(f"Epoch = {(epoch + 1)}. Save a periodic model snapshot.")
                p = f"{result_dir_path}/model_weight_{(epoch + 1):04}.pth"
                torch.save(corrector.net.state_dict(), p)
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

        del (dict_loaders, reg_net, corrector, ema_net, ema, optimizer, scaler, all_scores)
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("\n" + "*" * 50)
        logger.info("Skip post-train inference (EDM correction step)")
        logger.info("*" * 50 + "\n")
        logger.info(
            f"订正步训练脚本不做训练后推理;评估由共享评估链路消费 '{checkpoint_path}'"
            f"(sigma_data={edm_config.sigma_data:.6g},见 config_resolved.yml)统一完成"
        )

    except Exception as e:
        logger.info("\n" + "*" * 50)
        logger.info("Error")
        logger.info("*" * 50 + "\n")
        logger.error(e)
        logger.error(traceback.format_exc())
