from dataclasses import dataclass

from ..core.utils import get_device
from ..models.latent_diffusion import LatentDiffusion, LatentDiffusionConfig


@dataclass
class TrainConfig:
    model: LatentDiffusionConfig

    device: str = get_device()


def train() -> None: ...


def train_latent_diffusion(config: TrainConfig) -> None:
    # model
    model = LatentDiffusion(config.model).to(config.device)
