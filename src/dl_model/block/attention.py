import math
from logging import getLogger

import torch
from src.dl_model.block.norm import GroupNorm32
from src.dl_model.util import initialize_to_zero
from torch import nn

logger = getLogger()


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        channels_each_head: int = -1,
        use_new_attention_order: bool = False,
    ):
        assert channels_each_head == -1 or num_heads == -1

        assert (
            channels % 32 == 0
        ), f"For GroupNorm32, channels must be divisible by 32. {channels=}"

        super().__init__()
        self.channels = channels

        if channels_each_head == -1:
            assert channels % num_heads == 0
            self.num_heads = num_heads
        else:
            assert channels % channels_each_head == 0
            self.num_heads = channels // channels_each_head
        logger.debug(f"{self.num_heads=}, {self.channels=}")

        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)

        if use_new_attention_order:
            # reshape and then split
            self.attention = QKVAttention(self.num_heads)
        else:
            # split and then reshape
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = initialize_to_zero(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)

        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)

        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )

        return a.reshape(bs, -1, length)
