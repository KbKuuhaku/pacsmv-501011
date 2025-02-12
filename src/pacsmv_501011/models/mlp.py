from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    name: str
    hidden1: int
    hidden2: int


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: MLPConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=config.hidden1),
            nn.ReLU(),
            nn.Linear(in_features=config.hidden1, out_features=config.hidden2),
            nn.ReLU(),
            nn.Linear(in_features=config.hidden2, out_features=out_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = image.shape
        # flatten image, then feed into MLP
        x = image.reshape(batch_size, -1)
        logits = self.mlp(x)

        return logits
