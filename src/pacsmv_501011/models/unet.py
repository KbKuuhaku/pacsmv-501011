from dataclasses import dataclass

import torch
import torch.nn as nn

from .spatial_transformer import SpatialTransformer, SpatialTransformerConfig
from .unet_blocks.decoder import DecoderBlock
from .unet_blocks.encoder import EncoderBlock
from .unet_blocks.residual import ResidualBlock

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
    """
    U-Net
    """

    def __init__(self, config: UNetConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.levels = len(config.channel_multipliers)
        self.base_channels = config.model_channels

        # Embeddings
        self.context_embed = nn.Embedding(
            num_embeddings=config.corpus_length,
            embedding_dim=config.transformer.context_dim,
        )
        self.time_embed = TimeEmbedding(time_dim=config.model_channels)

        self.encoder = self._build_encoder(config)

        self.mid_out_channels = self.encoder_channels[-1] * 2
        self.mid_block = MidBlock(
            in_channels=self.encoder_channels[-1],
            out_channels=self.mid_out_channels,
            config=config.transformer,
            time_dim=config.model_channels,
        )

        self.decoder = self._build_decoder(config)

        self.proj = nn.Sequential(
            nn.GroupNorm(32, self.base_channels),
            nn.SiLU(),
            nn.Conv2d(  # (w + 1 x 2) - 3 + 1 = w
                in_channels=self.base_channels,
                out_channels=config.out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def _build_encoder(self, config: UNetConfig) -> nn.ModuleList:
        blocks = []

        self.encoder_channels = [self.base_channels]
        for level, channel_multiplier in zip(range(self.levels), config.channel_multipliers):
            out_channels = self.base_channels * channel_multiplier

            # Only add Spatial Transformer on certain levels
            spatial_transformer = None
            if level in config.attention_levels:
                spatial_transformer = SpatialTransformer(out_channels, config.transformer)

            blocks.append(
                EncoderBlock(
                    in_channels=config.in_channels if level == 0 else self.encoder_channels[-1],
                    out_channels=out_channels,
                    time_dim=config.model_channels,
                    num_res_blocks=config.num_res_blocks,
                    spatial_transformer=spatial_transformer,
                )
            )
            self.encoder_channels.append(out_channels)

        return nn.ModuleList(blocks)

    def _build_decoder(self, config: UNetConfig) -> nn.ModuleList:
        blocks = []

        in_channels = self.mid_out_channels
        for channel_multiplier in reversed(config.channel_multipliers):
            blocks.append(
                DecoderBlock(
                    in_channels=in_channels,
                    enc_channels=self.encoder_channels.pop(),
                    out_channels=self.base_channels * channel_multiplier,
                    time_dim=config.model_channels,
                    num_res_blocks=config.num_res_blocks,
                )
            )

            in_channels = self.base_channels * channel_multiplier

        return nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        context_embed = self.context_embed(context)

        encoder_out = self.encode(x, t_embed, context_embed)

        mid_out = self.mid_block(encoder_out, t_embed, context_embed)

        decoder_out = self.decode(mid_out, t_embed)

        out = self.proj(decoder_out)

        return out

    def encode(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self.encoder_hiddens: list[torch.Tensor] = []

        encoder_out = x
        for encoder_block in self.encoder:
            encoder_out, encoder_hidden = encoder_block(encoder_out, t, context)
            self.encoder_hiddens.append(encoder_hidden)

        return encoder_out

    def decode(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        decoder_out = x
        for decoder_block in self.decoder:
            # Skip connection on (B, C, H, W)
            decoder_out = decoder_block(decoder_out, self.encoder_hiddens.pop(), t)

        return decoder_out


class TimeEmbedding(nn.Module):
    """
    Reference: `GaussianFourierProjection` from
    https://colab.research.google.com/drive/1Y5wr91g5jmpCDiX-RLfWL1eSBWoSuLqO?usp=sharing
    """

    def __init__(self, time_dim: int, scale=30.0):
        super().__init__()
        # Randomly sample weights (frequencies) during initialization.
        # These weights (frequencies) are fixed during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(time_dim // 2) * scale, requires_grad=False)
        self.proj = nn.Linear(in_features=time_dim, out_features=time_dim)

    def forward(self, x):
        # Cosine(2 pi freq x), Sine(2 pi freq x)
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        return self.proj(out)


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        config: SpatialTransformerConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.res1 = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_dim=time_dim,
        )
        self.res2 = ResidualBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            time_dim=time_dim,
        )
        self.st = SpatialTransformer(out_channels, config)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> None:
        res1_out = self.res1(x, t)
        res2_out = self.res2(res1_out, t)
        st_out = self.st(res2_out, context)

        return st_out
