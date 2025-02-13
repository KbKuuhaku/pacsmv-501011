import torch
import torch.nn as nn

from ..spatial_transformer import SpatialTransformer
from .residual import ResidualBlock


class EncoderBlockError(Exception): ...


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        spatial_transformer: SpatialTransformer | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Residual Blocks
        res_blocks = [
            ResidualBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                time_dim=time_dim,
            )
            for i in range(num_res_blocks)
        ]
        self.res_blocks = nn.ModuleList(res_blocks)

        # Spatial Transformer
        self.spatial_transformer = spatial_transformer

        # Downsampling: ((w + 1 x 2) - 3 + 1) / 2 = w / 2
        self.downsampling = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Resisual Blocks
        res_out = x
        for res_block in self.res_blocks:
            res_out = res_block(res_out, t)

        # Spatial Transformer
        if self.spatial_transformer:
            if context is None:
                raise EncoderBlockError("`SpatialTransformer` is added but `context` is None")

            res_out = self.spatial_transformer(res_out, context=context)

        # Downsampling
        ret = self.downsampling(res_out)

        return ret, res_out
