from dataclasses import dataclass

import torch.nn as nn

from .spatial_transformer import SpatialTransformerConfig
from .unet_blocks.res import ResBlock

"""
References: https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html
"""


@dataclass
class UNetConfig:
    corpus_length: int
    in_channels: int
    out_channels: int
    model_channels: int
    attention_levels: list[int]
    num_res_blocks: int
    channel_multipliers: list[int]
    transformer: SpatialTransformerConfig


class UNet(nn.Module):
    def __init__(self, config: UNetConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
