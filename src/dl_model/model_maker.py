from logging import getLogger

logger = getLogger()

from src.dl_config.base_config import BaseModelConfig
from src.dl_model.ddpm.unet_ddpm_v01 import UNetDDPMVer01, UNetDDPMVer01Config
from src.dl_model.swinir_arch import SwinIRCanvas, SwinIRCanvasConfig


def make_model(config: BaseModelConfig):

    if config.model_name == "unet_ddpm_v01":
        logger.info("UNetDDPMVer01 is created")
        assert isinstance(config, UNetDDPMVer01Config)
        return UNetDDPMVer01(**config.__dict__)
    elif config.model_name == "swinir_canvas":
        logger.info("SwinIRCanvas is created")
        assert isinstance(config, SwinIRCanvasConfig)
        return SwinIRCanvas(**config.__dict__)
    else:
        raise ValueError(f"Model {config.model_name} is not supported.")
