import dataclasses
import typing
from logging import getLogger

import torch
from src.dl_config.base_config import BaseModelConfig
from src.dl_model.block.attention import AttentionBlock
from src.dl_model.block.downsample import Downsample2D
from src.dl_model.block.norm import GroupNorm32
from src.dl_model.block.upsample import Upsample2DNearest
from src.dl_model.ddpm.blocks import EmbedSequential, ResBlock2DEmb, gamma_embedding
from src.dl_model.util import initialize_to_zero
from torch import nn

logger = getLogger()


@dataclasses.dataclass()
class UNetDDPMVer01Config(BaseModelConfig):
    model_name: typing.ClassVar[str] = "unet_ddpm_v01"
    in_channel: int
    inner_channel: int
    out_channel: int
    res_blocks: int
    channel_mults: list[int]
    attn_res: list[int]
    channels_each_head: int
    dropout: float
    resblock_updown: bool
    max_period: float


class UNetDDPMVer01(nn.Module):
    def __init__(
        self,
        in_channel: int,
        inner_channel: int,
        out_channel: int,
        res_blocks: int,
        channel_mults: list[int],
        attn_res: list[int],
        channels_each_head: int,
        dropout: float,
        resblock_updown: bool,
        max_period: int,
        #
        conv_resample: bool = True,
        num_heads: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        use_new_attention_order: bool = False,
        **kwargs,
    ):

        super().__init__()

        self.inner_channel = inner_channel
        self.max_period = max_period
        self.out_channel = out_channel

        cond_embed_dim = inner_channel * 4
        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        ch = input_ch = int(channel_mults[0] * inner_channel)
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        logger.debug(f"input {ch=}")

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1  # scale factor

        for level, mult in enumerate(channel_mults):
            for idx_res in range(res_blocks):
                logger.debug(
                    f"\ndown (False) {ch=}, {level=}, {mult=}, {ds=}, {idx_res=}"
                )
                layers = [
                    ResBlock2DEmb(
                        channels=ch,
                        emb_channels=cond_embed_dim,
                        dropout=dropout,
                        out_channel=int(mult * inner_channel),
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=False,
                    )
                ]
                ch = int(mult * inner_channel)

                logger.debug(f"\n{ds=}, {attn_res=}")
                if ds in attn_res:
                    logger.debug("Attention is added!")
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                            channels_each_head=channels_each_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mults) - 1:
                out_ch = ch
                logger.debug(
                    f"\ndown (True) {out_ch=}, {ch=}, {level=}, {mult=}, {ds=}"
                )
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock2DEmb(
                            channels=ch,
                            emb_channels=cond_embed_dim,
                            dropout=dropout,
                            out_channel=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_conv_in_down=False,
                        )
                        if resblock_updown
                        else Downsample2D(
                            channels=ch, use_conv=conv_resample, out_channel=out_ch
                        )
                    )
                )

                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock2DEmb(
                channels=ch,
                emb_channels=cond_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                channels=ch,
                num_heads=num_heads,
                channels_each_head=channels_each_head,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock2DEmb(
                channels=ch,
                emb_channels=cond_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mults))[::-1]:
            for idx_res in range(res_blocks + 1):
                ich = input_block_chans.pop()
                logger.debug(
                    f"\nup (False) {ch=}, {ich=}, {level=}, {mult=}, {ds=}, {idx_res=}"
                )
                layers = [
                    ResBlock2DEmb(
                        channels=ch + ich,
                        emb_channels=cond_embed_dim,
                        dropout=dropout,
                        out_channel=int(inner_channel * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(inner_channel * mult)

                logger.debug(f"\n{ds=}, {attn_res=}")
                if ds in attn_res:
                    logger.debug("Attention is added!")
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads_upsample,
                            channels_each_head=channels_each_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and idx_res == res_blocks:
                    out_ch = ch
                    logger.debug(
                        f"\nup (True) {ch=}, {ich=}, {out_ch=}, {level=}, {mult=}, {ds=}, {idx_res=}"
                    )
                    layers.append(
                        ResBlock2DEmb(
                            channels=ch,
                            emb_channels=cond_embed_dim,
                            dropout=dropout,
                            out_channel=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample2DNearest(
                            channels=ch, use_conv=conv_resample, out_channel=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            initialize_to_zero(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
        )

    def forward(
        self, yt: torch.Tensor, y_cond: torch.Tensor, gamma: torch.Tensor, **kwargs
    ):
        # yt and y_cond dims: batch, channel, y and x
        # gamma dims: batch, channel (channel = 1)

        emb = self.cond_embed(
            gamma_embedding(gamma.view(-1), self.inner_channel, self.max_period)
        )

        h = torch.cat([yt, y_cond], dim=1)
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            hpop = hs.pop()
            h = torch.cat([h, hpop], dim=1)
            h = module(h, emb)

        return self.out(h.to(torch.float32))
