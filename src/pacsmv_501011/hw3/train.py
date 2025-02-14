import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ..core.utils import get_device
from ..dataset.mnist import MNIST
from ..models.unet import UNet, UNetConfig
from .utils import compute_marginal_prob_std


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
class TrainConfig:
    seed: int
    batch_size: int
    num_epochs: int
    sigma: float  # for diffusion

    data: DataConfig
    model: UNetConfig
    hparam: HparamConfig

    device: str = get_device()
    tb_log_dir: Path = Path("runs") / "hw3"
    ckpt_dir: Path = Path("ckpts") / "diffusion"

    def __post_init__(self) -> None:
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.parent.mkdir(parents=True, exist_ok=True)

    @property
    def ckpt_file(self) -> Path:
        return self.ckpt_dir / self.model.CKPT_NAME

    @property
    def model_config_file(self) -> Path:
        return self.ckpt_dir / self.model.CONFIG_NAME


"""
Reference: https://colab.research.google.com/drive/1CUj3dS42BVQ93eztyEeUKLQSFnsQ_rQc#scrollTo=3FwiV_bpSD7Q

- loss
"""


def compute_loss(
    hidden: torch.Tensor,
    z: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    # Normalization
    score = hidden / std
    loss = (((score * std) + z) ** 2).sum(dim=(1, 2, 3)).mean()
    return loss


def add_noise(
    config: TrainConfig,
    images: torch.Tensor,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = images.shape[0]
    t = torch.rand(B, device=config.device) * (1.0 - eps) + eps
    std = compute_marginal_prob_std(t, config.sigma)

    z = torch.randn_like(images)
    noise = z * std
    perturbed_images = images + noise

    return perturbed_images, t, z, std


def update_pbar_description(
    pbar: tqdm,
    mode: str,
    epoch: int,
    step: int,
    running_loss: float,
) -> None:
    pbar.set_description(
        f"{mode:>15} | Epoch: {epoch:>03} | Step: {step:>06} | Loss: {running_loss / (step + 1):.4f}"
    )


def save_checkpoint(
    config: TrainConfig,
    model: nn.Module,
) -> None:
    import json

    # Save config
    with open(config.model_config_file, "w") as f:
        json.dump(asdict(config.model), f, indent=4)

    # Save Model checkpoint
    torch.save(model.state_dict(), config.ckpt_file)


@torch.no_grad
def evaluate(
    config: TrainConfig,
    model: nn.Module,
    epoch: int,
    eval_dataloader: DataLoader,
) -> float:
    model.eval()

    running_loss = 0.0
    pbar = tqdm(eval_dataloader)

    for step, (images, labels) in enumerate(pbar):
        B, H, W = images.shape
        images = images.reshape(B, 1, H, W).to(config.device)
        labels = labels.reshape(B, 1).long().to(config.device)

        perturbed_images, t, z, std = add_noise(config, images)

        hidden = model(perturbed_images, t, labels)

        loss = compute_loss(hidden, z, std)
        running_loss += loss.item()

        perturbed_images, t, z, std = add_noise(config, images)

        update_pbar_description(pbar, "Evaluating", epoch, step, running_loss)

    model.train()

    return running_loss


def train(
    config: TrainConfig,
    model: nn.Module,
    optimizer: Adam,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
) -> None:
    tb_writer = SummaryWriter(log_dir=config.tb_log_dir)

    model.train()
    global_step = 0

    for epoch in range(config.num_epochs):
        pbar = tqdm(train_dataloader)
        train_running_loss = 0.0

        # Train
        for step, (images, labels) in enumerate(pbar):
            B, H, W = images.shape
            images = images.reshape(B, 1, H, W).to(config.device)
            labels = labels.reshape(B, 1).long().to(config.device)

            perturbed_images, t, z, std = add_noise(config, images)

            hidden = model(perturbed_images, t, labels)

            loss = compute_loss(hidden, z, std)
            train_running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_pbar_description(pbar, "Training", epoch, step, train_running_loss)

            global_step += 1

        # Eval
        eval_running_loss = evaluate(config, model, epoch, eval_dataloader)

        # Tensorboard
        train_avg_loss = train_running_loss / len(train_dataloader)
        eval_avg_loss = eval_running_loss / len(eval_dataloader)

        tb_writer.add_scalar("Loss/Train", train_avg_loss, global_step)
        tb_writer.add_scalar("Loss/Eval", eval_avg_loss, global_step)

    model.eval()

    # Save model
    save_checkpoint(config, model)
    print(f"Checkpoint saved to {config.ckpt_dir}!")


def train_diffusion_model(config: TrainConfig) -> None:
    random.seed(config.seed)

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

    # Model
    model = UNet(config.model).to(config.device).to(config.device)
    optimizer = Adam(
        model.parameters(),
        config.hparam.lr,
        betas=(config.hparam.beta1, config.hparam.beta2),
    )

    train(config, model, optimizer, train_dataloader, eval_dataloader)
