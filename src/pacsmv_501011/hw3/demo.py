import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import rich
import torch
from einops import rearrange
from torchvision.utils import make_grid

from ..core.utils import get_device
from ..models.unet import UNet, UNetConfig
from .sampler import EulerSampler, SamplerConfig


@dataclass
class DemoConfig:
    ckpt_dir: str
    sigma: float
    digit_dir: str

    sampler: SamplerConfig

    device: str = get_device()


def demo(config: DemoConfig) -> None:
    model_config = UNetConfig.from_file(config.ckpt_dir)
    rich.print(model_config)

    model = UNet(model_config)

    # Load checkpoint
    ckpt_file = Path(config.ckpt_dir) / model_config.CKPT_NAME
    model.load_state_dict(torch.load(ckpt_file, weights_only=True))

    model = model.to(get_device())
    print(model)

    digit_dir = Path(config.digit_dir)
    digit_dir.mkdir(parents=True, exist_ok=True)

    while True:
        while digit := int(input("Input your favorite number (0-9): ")):
            if 0 <= digit <= 9:
                break
            print("Invalid number, please try it again")

        # Load sampler
        sampler = EulerSampler(config.sampler, sigma=config.sigma, unet=model)
        samples = sampler.sample(digit)

        grid = make_grid(samples, nrow=int(math.sqrt(config.sampler.batch_size)))

        display_grid = rearrange(grid.cpu(), pattern="c h w -> h w c")

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(display_grid, vmin=0.0, vmax=1.0)
        plt.show()

        save_to = digit_dir / f"hw3-digit-{digit}.png"
        fig.savefig(save_to)

        print(f"Digit saved to {save_to}!")
