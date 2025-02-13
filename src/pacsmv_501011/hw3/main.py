import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rich
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from ..core.config import parse_config
from ..core.utils import get_device
from ..dataset.mnist import MNIST
from ..models.unet import UNet, UNetConfig


@dataclass
class DataConfig:
    root: str
    train_images: str
    train_labels: str
    test_images: str
    test_labels: str


@dataclass
class HparamConfig:
    lr: float
    beta1: float
    beta2: float


@dataclass
class HW3Config:
    seed: int
    batch_size: int
    n_epoch: int
    sigma: float  # for diffusion

    data: DataConfig
    model: UNetConfig
    hparam: HparamConfig
    device: str = get_device()


"""
Reference: https://colab.research.google.com/drive/1CUj3dS42BVQ93eztyEeUKLQSFnsQ_rQc#scrollTo=3FwiV_bpSD7Q

- margin std

- diffusion coeff

- loss
"""


def compute_marginal_prob_std(t: torch.Tensor, sigma: float) -> torch.Tensor:
    numerator = (sigma ** (2 * t)) - 1
    denominator = 2 * np.log(sigma)
    return torch.sqrt(numerator / denominator)


def compute_diffusion_coeff(t: torch.Tensor, sigma: float) -> torch.Tensor:
    return t**sigma


def compute_loss(
    hidden: torch.Tensor,
    z: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    # Normalization
    score = hidden / std
    loss = (((score * std) + z) ** 2).sum(dim=(1, 2, 3)).mean()
    return loss


def eval() -> None: ...


def train(
    config: HW3Config,
    model: nn.Module,
    optimizer: Adam,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
) -> None:
    def add_noise(
        eps: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = images.shape[0]
        t = torch.rand(B, device=config.device) * (1.0 - eps) + eps
        std = compute_marginal_prob_std(t, config.sigma).reshape(-1, 1, 1, 1)

        z = torch.randn_like(images)
        noise = z * std
        perturbed_images = images + noise

        return perturbed_images, t, z, std

    model.train()

    for epoch in range(config.n_epoch):
        pbar = tqdm(train_dataloader)
        running_loss = 0.0

        for step, (images, labels) in enumerate(pbar):
            B, H, W = images.shape
            images = images.reshape(B, 1, H, W).to(config.device)
            labels = labels.reshape(B, 1).long().to(config.device)

            perturbed_images, t, z, std = add_noise()
            # import pdb
            # pdb.set_trace()

            # print(images.shape, images.dtype)
            # print(labels.shape, labels.dtype)
            # print(perturbed_images.shape, perturbed_images.dtype)
            # print(t.shape, t.dtype)
            # print(z.shape, z.dtype)
            # print(std.shape, std.dtype)

            hidden = model(perturbed_images, t, labels)

            loss = compute_loss(hidden, z, std)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f"Epoch: {epoch:>06} | Step: {step:>06} | Loss: {running_loss / (step + 1):.4f}"
            )

    model.eval()


def train_diffusion(
    config: HW3Config,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
) -> None:
    # Model
    model = UNet(config.model).to(config.device).to(config.device)
    optimizer = Adam(
        model.parameters(),
        config.hparam.lr,
        betas=(config.hparam.beta1, config.hparam.beta2),
    )

    train(config, model, optimizer, train_dataloader, eval_dataloader)


def main() -> None:
    config_path = Path("configs", "hw3.toml")
    config = parse_config(path=str(config_path), dataclass_name=HW3Config)
    # random.seed(config.seed)

    rich.print(config)

    # Load MNIST
    train_dataset = MNIST(
        config.data.root,
        config.data.train_images,
        config.data.train_labels,
    )
    eval_dataset = MNIST(
        config.data.root,
        config.data.test_images,
        config.data.test_labels,
    )

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
    )

    train_diffusion(
        config,
        train_dataloader,
        eval_dataloader,
    )
