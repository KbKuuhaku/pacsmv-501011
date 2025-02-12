from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from .unet_blocks.attention import AttentionBlock


@dataclass
class SpatialTransformerConfig:
    context_dim: int
    num_heads: int
    num_layers: int


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer takes in a feature map (spatial) and a context embedding as the input,
    and output a processed feature map

    NOTE: No context embedding included, the context embedding will be generated at
    the start of U-Net
    """

    def __init__(
        self,
        channels: int,
        config: SpatialTransformerConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    query_dim=channels,
                    value_dim=config.context_dim,
                    hidden_dim=channels,  # same as channels to make sure the dim is not changing
                    num_heads=config.num_heads,
                )
                for _ in range(config.num_layers)
            ]
        )

        # 1x1 projection layer
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, feature_map: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature_map.shape

        # Rearrange (permute + reshape)
        x = rearrange(feature_map, pattern="b c h w -> b (h w) c")

        # Attention Blocks
        for block in self.blocks:
            x = block(x, context)

        # Rearrange back (permute + reshape)
        x = rearrange(x, pattern="b (h w) c -> b c h w", h=H, w=W)

        # Residual connection
        ret = feature_map + self.proj_out(x)
        return ret
