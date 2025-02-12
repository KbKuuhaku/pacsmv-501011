from dataclasses import dataclass

import torch
import torch.nn as nn

from .unet import UNet, UNetConfig
from .vae import VAE, VAEConfig

"""
References: https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml
"""


@dataclass
class LatentDiffusionConfig:
    vae: VAEConfig
    unet: UNetConfig


class LatentDiffusion(nn.Module):
    def __init__(self, config: LatentDiffusionConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vae = VAE(config.vae)
        self.unet = UNet(config.unet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.vae(x)  # latent z
        score = self.unet(z)

        return score
