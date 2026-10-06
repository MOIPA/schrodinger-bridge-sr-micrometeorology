"""SwinIR 类窗口注意力网络(canvas 口径,阶段 3 架构对比 A 组)。

与 SI 接口的对应(与 ddpm/unet_ddpm_v01.py 完全对齐):
- 训练/采样统一调用 ``net(yt=, y_cond=, gamma=)``,返回漂移场 b_est,
  形状 (B, out_channel, H, W);yt 为状态 (B, out_channel, H, W),
  y_cond 为条件 (B, in_channel - out_channel, H, W),内部先
  ``cat([yt, y_cond], dim=1)`` 再进 patch-embed(3x3 conv)。
- gamma 为扩散时间(训练 (B,),采样 (B,1),内部 reshape(-1) 统一),
  经 gamma_embedding(正弦)+ MLP 得 (B, inner_channel) 向量,每个块内以
  FiLM(scale/shift)调制(与 UNet ResBlock 的 use_scale_shift_norm 同口径:
  norm 后 ``x * (1 + scale) + shift``)。不同 gamma 产生不同输出。
- 输出头 3x3 conv C->out_channel 且 zero-init(initialize_to_zero),与 UNet 的
  b 漂移头一致:训练起步 b_est == 0(残差参数化的零起点)。
- 返回前 ``.to(torch.float32)``(同 UNet 输出头,AMP 下口径一致)。

窗口参数:
- 训练空间 96x112、评估 pad16 后 112x128,均被 window_size=8 整除,正常路径无 pad;
- 非整除时镜像 pad(reflect,极小时退化为 replicate)补到整数倍,SW-MSA 用
  torch.roll 移位不改变尺寸,输出前裁回原尺寸;
- W-MSA 与 SW-MSA 交替(shift = window_size // 2),SW-MSA 用标准 Swin 掩码
  阻断 roll 造成的跨区窗口;省略相对位置偏置(保持实现最小,见任务约定),
  窗口注意力 + 移位窗口 + 残差 + LayerNorm + MLP 完整。

参数量(实测,见文件末尾自检口径):
- 默认配置 inner_channel=96, num_blocks=6 实测 0.786M;
- 参数近似随 inner_channel^2 增长:inner_channel=288, num_blocks=6 实测 5.786M
  (落在 5-10M 目标区间,需要与 UNet 容量对齐时用该量级配置)。
"""

import dataclasses
import typing
from logging import getLogger

import torch
import torch.nn.functional as F
from src.dl_config.base_config import BaseModelConfig
from src.dl_model.ddpm.blocks import gamma_embedding
from src.dl_model.util import initialize_to_zero
from torch import nn

logger = getLogger()


@dataclasses.dataclass()
class SwinIRCanvasConfig(BaseModelConfig):
    model_name: typing.ClassVar[str] = "swinir_canvas"
    in_channel: int  # 状态 + 条件总通道(现用 72 + 95 = 167)
    out_channel: int  # 状态通道 = 漂移场输出通道(72)
    inner_channel: int = 96
    num_blocks: int = 6
    window_size: int = 8
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    max_period: float = 10.0  # 与 UNetDDPMVer01Config.max_period 同口径


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B * nH * nW, ws, ws, C),要求 H/W 被 ws 整除。"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """(B * nH * nW, ws, ws, C) -> (B, H, W, C),window_partition 的逆。"""
    B = int(windows.shape[0] // (H // window_size * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


def build_attn_mask(
    H: int, W: int, window_size: int, shift_size: int, device, dtype
) -> torch.Tensor:
    """SW-MSA 掩码:roll 后同一窗口内来自两个不相邻区域的像素互不可见(标准 Swin 做法)。"""
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )
    return attn_mask.to(dtype)


class WindowAttention(nn.Module):
    """窗口内多头自注意力(省略相对位置偏置;mask 非空时用于 SW-MSA)。"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"{dim=} 必须被 {num_heads=} 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: typing.Optional[torch.Tensor] = None):
        # x: (nW, ws * ws, C)
        nW, N, C = x.shape
        qkv = (
            self.qkv(x)
            .view(nW, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 (nW, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(1)  # (nW, 1, N, N) 广播到 heads
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(nW, N, C)
        return self.proj_drop(self.proj(x))


class SwinCanvasBlock(nn.Module):
    """LN -> (SW-)W-MSA -> 残差,LN -> MLP -> 残差;gamma 以 FiLM 调制两个 LN。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
        emb_dim: int,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        # gamma 时间调制: (B, emb_dim) -> scale/shift 各 (B, dim)
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * dim))

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        H: int,
        W: int,
        mask: typing.Optional[torch.Tensor] = None,
    ):
        # x: (B, H, W, C);emb: (B, emb_dim);H/W 为 pad 后尺寸(被 ws 整除)
        B, _, _, C = x.shape
        scale, shift = self.emb_layers(emb).chunk(2, dim=-1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]

        # ---- W-MSA / SW-MSA ----
        h = self.norm1(x) * (1.0 + scale) + shift
        if self.shift_size > 0:
            h = torch.roll(h, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        h_windows = window_partition(h, self.window_size)
        h_windows = h_windows.view(-1, self.window_size * self.window_size, C)
        h_windows = self.attn(h_windows, mask=mask)
        h = window_reverse(
            h_windows.view(-1, self.window_size, self.window_size, C),
            self.window_size,
            H,
            W,
        )
        if self.shift_size > 0:
            h = torch.roll(h, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x + h

        # ---- MLP(同一 FiLM,adaLN 共享口径)----
        h = self.norm2(x) * (1.0 + scale) + shift
        x = x + self.mlp(h)
        return x


class SwinIRCanvas(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        inner_channel: int = 96,
        num_blocks: int = 6,
        window_size: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_period: float = 10.0,
        **kwargs,
    ):
        super().__init__()
        assert window_size % 2 == 0, f"{window_size=} 需为偶数(shift = ws // 2)"
        assert inner_channel % num_heads == 0, (
            f"{inner_channel=} 必须被 {num_heads=} 整除"
        )

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.inner_channel = inner_channel
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.max_period = max_period
        self.num_blocks = num_blocks

        # [yt, y_cond] 拼接后 3x3 conv 提特征
        self.patch_embed = nn.Conv2d(in_channel, inner_channel, 3, padding=1)
        # gamma(标量/B,) -> (B, inner_channel) 时间向量
        self.gamma_mlp = nn.Sequential(
            nn.Linear(inner_channel, inner_channel),
            nn.SiLU(),
            nn.Linear(inner_channel, inner_channel),
        )
        self.blocks = nn.ModuleList(
            [
                SwinCanvasBlock(
                    dim=inner_channel,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    emb_dim=inner_channel,
                )
                for i in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(inner_channel)
        # 与 UNet 的 b 漂移头一致:zero-init,训练起步 b_est == 0
        self.out_conv = initialize_to_zero(
            nn.Conv2d(inner_channel, out_channel, 3, padding=1)
        )

        self._mask_cache: dict = {}

    def _get_attn_mask(self, H: int, W: int, device, dtype) -> torch.Tensor:
        key = (H, W, str(device), str(dtype))
        if key not in self._mask_cache:
            self._mask_cache[key] = build_attn_mask(
                H, W, self.window_size, self.shift_size, device, dtype
            )
        return self._mask_cache[key]

    def forward(self, yt: torch.Tensor, y_cond: torch.Tensor, gamma: torch.Tensor, **kwargs):
        # yt/y_cond: batch, channel, y and x;gamma: (B,) 或 (B, 1)
        x = torch.cat([yt, y_cond], dim=1)
        B, _, H, W = x.shape

        # 非整除兜底:镜像 pad 到 window_size 整数倍(shift/roll 不改变尺寸)
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            mode = "reflect" if min(H, W) > max(pad_h, pad_w) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        Hp, Wp = x.shape[-2:]

        x = self.patch_embed(x)  # (B, C, Hp, Wp)

        emb = self.gamma_mlp(
            gamma_embedding(gamma.reshape(-1), self.inner_channel, self.max_period)
        ).to(x.dtype)

        x = x.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, C)

        mask = None
        if any(blk.shift_size > 0 for blk in self.blocks):
            mask = self._get_attn_mask(Hp, Wp, x.device, x.dtype)
            mask = mask.repeat(B, 1, 1)  # 窗口序为 batch-major,按 batch 复制

        for blk in self.blocks:
            x = blk(x, emb, Hp, Wp, mask=mask if blk.shift_size > 0 else None)

        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        out = self.out_conv(x.to(torch.float32))
        return out[..., :H, :W]
