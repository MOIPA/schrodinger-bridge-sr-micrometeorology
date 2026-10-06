"""回归步模型的评估包装:与 SI 采样接口同名同形,供共享评估链路直接消费。

`RegressionModel` 包装由 config 建的 UNet(`make_model`),提供确定性输出:
    y1_full = y0 + net(yt=y0, y_cond=y_cond, gamma=ones(B))
    sample_y1_bare_diffusion(...) 返回 (y1_full, None)
即训练时 `optimize_reg` 的同款前向(gamma 固定 t=1 常数嵌入)。评估脚本把模型
构建/加载包一层本类后,可与 SI 模型走同一批 ops/eval 调用(`.eval()/.to()`/
`state_dict()` 均为标准 nn.Module 行为)。

checkpoint 兼容:`scripts/train_regression_model.py` 与
`scripts/train_schrodinger_bridge_model.py` 保存的 'model_state_dict' 是裸
UNet 键(无 "net." 前缀,与 90_agl_eval_phase1.build_si 的加载口径一致);
本类 `load_state_dict` 自动补前缀,使
    RegressionModel(config.model).load_state_dict(ckpt['model_state_dict'])
可直接加载同一批 checkpoint。
"""

from logging import getLogger

import torch
from torch import nn

from src.dl_config.base_config import BaseModelConfig
from src.dl_model.model_maker import make_model

logger = getLogger()


class RegressionModel(nn.Module):
    """回归步 = UNet 在 t=1 的单次前向 + 残差加回 (y1 = y0 + net(...))。"""

    NET_PREFIX = "net."

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.net = make_model(config)

    def forward(self, yt: torch.Tensor, y_cond: torch.Tensor, gamma: torch.Tensor,
                **kwargs):
        # 与 UNet forward 同签名,透传给 net
        return self.net(yt=yt, y_cond=y_cond, gamma=gamma, **kwargs)

    @torch.no_grad()
    def sample_y1_bare_diffusion(self, y0: torch.Tensor, y_cond: torch.Tensor,
                                 add_noise: bool = False):
        """返回 (y1_full, None):y1_full = y0 + net(yt=y0, y_cond=y_cond, gamma=ones(B))。

        add_noise 仅为对齐 SI 采样接口签名而保留;回归步为确定性输出,忽略该参数。
        """
        b = y0.shape[0]
        gamma = torch.ones(b, device=y0.device, dtype=torch.float32)
        return y0 + self.net(yt=y0, y_cond=y_cond, gamma=gamma), None

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """兼容两种键名:本类前缀形式("net.xxx")与训练脚本保存的裸 UNet 键("xxx")。"""
        if not any(k.startswith(self.NET_PREFIX) for k in state_dict):
            state_dict = {self.NET_PREFIX + k: v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
