import warnings

import torch
import torch.nn as nn

from .residual import ResidualBlock


class CenterCropError(Exception): ...


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Upsampling: (w - 1) x 2 - 2 x 1 + (3 - 1) + 1 + 1 = 2w
        self.upsampling = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            bias=False,
            padding=1,
            output_padding=1,
        )

        concat_in_channels = out_channels + enc_channels  # skip connection

        # Residual Blocks
        res_blocks = [
            ResidualBlock(
                in_channels=concat_in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                time_dim=time_dim,
            )
            for i in range(num_res_blocks)
        ]
        self.res_blocks = nn.ModuleList(res_blocks)

    def forward(self, x: torch.Tensor, enc_h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Upsampling
        up_x = self.upsampling(x)

        # cropped_enc_h = self._center_crop(src=enc_h, tar=up_x)
        cropped_enc_h = enc_h

        res_in = torch.cat([cropped_enc_h, up_x], dim=1)  # skip connection on dim1 of (B, C, H, W)

        # Residual Blocks
        res_out = res_in
        for res_block in self.res_blocks:
            res_out = res_block(res_out, t)

        return res_out

    def _center_crop(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        """
        Center-crop the feature map `src` to be the same size as `tar`
        """
        src_w = src.shape[-1]
        tar_w = tar.shape[-1]

        if src_w == tar_w:
            warnings.warn(
                f"Performing ceter cropping, but src_w and tar_w are the same ({tar_w}), skipped"
            )
            return src

        if src_w < tar_w:
            raise CenterCropError(
                f"Performing ceter cropping, but src_w ({src_w}) < tar_w ({tar_w})"
            )

        start = (src_w - tar_w) // 2
        end = start + tar_w

        return src[:, :, start:end, start:end]
