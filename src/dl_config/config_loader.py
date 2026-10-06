from logging import getLogger

from src.dl_config.diffusion_model_config import ExperimentDiffusionModelConfig
from src.dl_config.schrodinger_bridge_model_config import (
    ExperimentSchrodingerBridgeModelConfig,
    ExperimentSchrodingerBridge3dWindConfig,
    ExperimentSchrodingerBridgeWindCanvasConfig,
)
from src.dl_model.ddpm.ddpm_framework import BetaConfig
from src.dl_model.ddpm.unet_ddpm_v01 import UNetDDPMVer01Config
from src.dl_model.swinir_arch import SwinIRCanvasConfig

logger = getLogger()

# model 段 arch -> dataclass 分派表(仿照本文件 ExperimentDiffusionModel 分支的
# 手动转换先例)。旧 yml 不写 arch,默认 unet_ddpm_v01,与改动前逐位一致。
MODEL_ARCH_TO_CONFIG = {
    "unet_ddpm_v01": UNetDDPMVer01Config,
    "swinir_canvas": SwinIRCanvasConfig,
}


def resolve_model_config(model_config):
    """yml model 段(Union 注解下 YamlConfig.load 保留的原始 dict)按 arch 键构造。

    arch 缺省视为 unet_ddpm_v01(旧 yml 零影响);已构造的 dataclass 实例原样返回。
    """
    if not isinstance(model_config, dict):
        return model_config
    model_config = dict(model_config)
    arch = model_config.pop("arch", "unet_ddpm_v01")
    if arch not in MODEL_ARCH_TO_CONFIG:
        raise ValueError(
            f"Model arch {arch} is not supported. "
            f"Supported: {sorted(MODEL_ARCH_TO_CONFIG)}"
        )
    return MODEL_ARCH_TO_CONFIG[arch](**model_config)


def load_config(experiment_name: str, config_path: str):
    if experiment_name == "ExperimentSchrodingerBridgeModel":
        logger.info("Experiment Schrodinger-Bridge Model is selected.")
        config = ExperimentSchrodingerBridgeModelConfig.load(config_path)

    elif experiment_name == "ExperimentSchrodingerBridge3dWind":
        logger.info("Experiment Schrodinger-Bridge 3D Wind Model is selected.")
        config = ExperimentSchrodingerBridge3dWindConfig.load(config_path)

    elif experiment_name == "ExperimentSchrodingerBridgeWindCanvas":
        logger.info("Experiment Schrodinger-Bridge Wind Canvas is selected.")
        config = ExperimentSchrodingerBridgeWindCanvasConfig.load(config_path)

    elif experiment_name == "ExperimentDiffusionModel":
        logger.info("Experiment Diffusion Model is selected.")
        config = ExperimentDiffusionModelConfig.load(config_path)

        for k in config.ddpm.beta_schedules.keys():
            config.ddpm.beta_schedules[k] = BetaConfig(**config.ddpm.beta_schedules[k])

        return config

    else:
        raise ValueError(f"{experiment_name} is not supported.")

    # model 段按 arch 分派构造(缺省 unet_ddpm_v01 -> 旧行为不变)
    config.model = resolve_model_config(config.model)
    return config
