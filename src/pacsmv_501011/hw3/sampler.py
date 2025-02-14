from dataclasses import dataclass

import torch

from ..core.utils import get_device
from ..models.unet import UNet
from .utils import compute_diffusion_coeff, compute_marginal_prob_std


@dataclass
class SamplerConfig:
    image_size: int
    batch_size: int
    num_steps: int
    eps: float

    device: str = get_device()


class EulerSampler:
    def __init__(
        self,
        config: SamplerConfig,
        sigma: float,
        unet: UNet,
    ) -> None:
        # Time step decreases from 1 to `eps`
        self.time_steps = torch.linspace(1, config.eps, config.num_steps, device=config.device)
        self.step_size = self.time_steps[0] - self.time_steps[1]
        self.unet = unet
        self.sigma = sigma
        self.config = config

    @torch.no_grad
    def sample(self, digit: int) -> torch.Tensor:
        t0 = torch.ones(self.config.batch_size, device=self.config.device)
        std = compute_marginal_prob_std(t0, self.sigma)

        # Images and digit
        images = std * torch.randn(
            self.config.batch_size,
            1,
            self.config.image_size,
            self.config.image_size,
            device=self.config.device,
        )  # starts from a gaussian noise
        images_mean = images
        labels = (digit * torch.ones(self.config.batch_size, 1)).long().to(self.config.device)

        for time_step in self.time_steps:
            t = t0 * time_step
            images, images_mean = self._denoise(t, images, labels)

        return images_mean.clamp(0, 1)  # constrain the result to 0 to 1

    @torch.no_grad
    def _denoise(
        self,
        t: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diffusion_coeff = compute_diffusion_coeff(t, self.sigma)
        std = compute_marginal_prob_std(t, self.sigma)

        score = self.unet(images, t, labels) / std
        images_mean = images + (diffusion_coeff**2) * score * self.step_size

        noise = torch.sqrt(self.step_size) * diffusion_coeff * torch.randn_like(images)
        images = images_mean + noise

        return images, images_mean
