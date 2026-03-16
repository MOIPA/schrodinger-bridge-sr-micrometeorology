from logging import getLogger

from src.dl_config.diffusion_model_config import ExperimentDiffusionModelConfig
from src.dl_config.schrodinger_bridge_model_config import (
    ExperimentSchrodingerBridgeModelConfig,
)
from src.dl_model.ddpm.ddpm_framework import BetaConfig

logger = getLogger()


def load_config(experiment_name: str, config_path: str):
    if experiment_name == "ExperimentSchrodingerBridgeModel":
        logger.info("Experiment Schrodinger-Bridge Model is selected.")
        return ExperimentSchrodingerBridgeModelConfig.load(config_path)

    elif experiment_name == "ExperimentDiffusionModel":
        logger.info("Experiment Diffusion Model is selected.")
        config = ExperimentDiffusionModelConfig.load(config_path)

        for k in config.ddpm.beta_schedules.keys():
            config.ddpm.beta_schedules[k] = BetaConfig(**config.ddpm.beta_schedules[k])

        return config

    else:
        raise ValueError(f"{experiment_name} is not supported.")
