from dataclasses import dataclass

import torch.nn as nn


@dataclass
class VAEConfig:
    in_channels: int
    out_channels: int
    model_channels: int
    attention_resolutions: list[int]
    num_res_blcoks: int
    channel_mult: list[int]


class VAE(nn.Module):
    """
    Variational Auto Encoder with ResBlock as hidden layers
    """

    def __init__(self, config: VAEConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
