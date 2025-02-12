import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ..core.config import parse_config
from ..core.record import AccuracyRecord, LossRecord
from ..core.train_state import TrainState
from ..core.utils import get_device
from ..dataset.mnist import MNIST, show_samples
from ..models.mlp import MLP, MLPConfig


class OptimizerNotDefined(Exception): ...


@dataclass
class DataConfig:
    root: str
    train_images: str
    train_labels: str
    test_images: str
    test_labels: str


@dataclass
class OptimConfig:
    name: str
    lr: float
    beta1: float
    beta2: float

    def create_optimizer(self, params: Iterable[torch.Tensor]) -> optim.Optimizer:  # type: ignore
        match self.name:
            case "sgd":
                return optim.SGD(params, lr=self.lr)  # type: ignore
            case "adam":
                return optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2))  # type: ignore
            case "adamw":
                return optim.AdamW(params, lr=self.lr, betas=(self.beta1, self.beta2))  # type: ignore
            case _:
                raise OptimizerNotDefined(f"{self.name} not defined")


@dataclass
class HW1Config:
    seed: int
    batch_size: int
    n_epoch: int
    data: DataConfig
    optim: OptimConfig
    model: MLPConfig

    @property
    def name(self) -> str:
        dataset_name = Path(self.data.root).stem
        return f"{self.model.name}_{dataset_name}_{self.optim.name}"

    @property
    def sample_dir(self) -> Path:
        result = Path("pics", self.name)
        result.mkdir(parents=True, exist_ok=True)

        return result

    @property
    def tensorboard_dir(self) -> Path:
        result = Path("runs", self.name)
        result.mkdir(parents=True, exist_ok=True)

        return result


@torch.no_grad
def compute_correct_nums(logits: torch.Tensor, labels: torch.Tensor) -> int:
    probs = F.softmax(logits, dim=-1)
    preds_np = probs.argmax(dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    return (preds_np == labels_np).sum()


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,  # type: ignore
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: HW1Config,
    device: str,
    writer: SummaryWriter | None = None,
) -> None:
    model.train()
    train_state = TrainState(config.optim.lr)

    with Progress(transient=True) as progress:
        n_batch = math.ceil(len(train_dataloader.dataset) / config.batch_size)  # type: ignore
        task_id = progress.add_task(
            "training",
            total=n_batch * config.n_epoch,
        )

        model.train()
        for epoch in range(config.n_epoch):
            train_state.set_epoch(epoch)
            train_loss = LossRecord()
            train_acc = AccuracyRecord()

            for images, labels in train_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)

                loss = F.cross_entropy(logits, labels)

                train_state.update()
                train_loss.update(loss.item() * len(images), len(images))
                train_acc.update(compute_correct_nums(logits, labels), len(images))

                progress.update(task_id, description=f"{train_state}")
                progress.advance(task_id)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Training")
            print(train_loss)
            print(train_acc)
            if writer is not None:
                train_loss.write_avg_to_tensorboard(
                    writer,
                    global_step=train_state.global_step,
                    title="Loss/train",
                )
                train_acc.write_avg_to_tensorboard(
                    writer,
                    global_step=train_state.global_step,
                    title="Accuracy/train",
                )

            evaluate(model, train_state.global_step, eval_dataloader, device, writer)

            print("=" * 100)
        model.eval()


@torch.no_grad
def evaluate(
    model: nn.Module,
    global_step: int,
    eval_dataloader: DataLoader,
    device: str,
    writer: SummaryWriter | None = None,
) -> None:
    model.eval()
    eval_loss = LossRecord()
    eval_acc = AccuracyRecord()

    for step, (images, labels) in enumerate(eval_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        eval_loss.update(loss.item() * len(images), len(images))
        eval_acc.update(compute_correct_nums(logits, labels), len(images))

    if writer is not None:
        eval_loss.write_avg_to_tensorboard(writer, global_step=global_step, title="Loss/eval")
        eval_acc.write_avg_to_tensorboard(writer, global_step=global_step, title="Accuracy/eval")

    print("Evaluating")
    print(eval_loss)
    print(eval_acc)

    model.train()


@torch.no_grad
def predict_and_show(
    model: nn.Module,
    eval_dataloader: DataLoader,
    config: HW1Config,
    image_height: int,
    image_width: int,
    device: str,
) -> None:
    model.eval()

    all_preds = []
    all_images = []

    for images, _ in eval_dataloader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=-1)

        # NOTE: denormalize
        all_images.append(255 * images.cpu().numpy())
        all_preds.append(probs.argmax(dim=-1, keepdim=True).cpu().numpy())

    # NOTE: list of batches -> concatenate because it already has 3 dims
    all_images = np.concatenate(all_images, axis=0).astype(np.uint8)
    all_preds = np.concatenate(all_preds, axis=0).flatten()

    show_samples(
        all_images,
        all_preds,
        image_height,
        image_width,
        n_sample_per_class=10,
        save_to=Path(config.sample_dir, "prediction-samples.png"),
    )


def main() -> None:
    import sys

    config_name = sys.argv[1]
    config_path = Path("configs") / f"{config_name}.toml"

    config = parse_config(path=str(config_path), dataclass_name=HW1Config)

    # Dataset preparation
    rich.print(config)
    random.seed(config.seed)
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
    show_samples(
        eval_dataset.images,
        eval_dataset.labels,
        eval_dataset.image_height,
        eval_dataset.image_width,
        n_sample_per_class=10,
        save_to=Path(config.sample_dir, "dataset-sample.png"),
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
    in_dim = train_dataset.image_height * train_dataset.image_width
    out_dim = train_dataset.n_class
    model = MLP(in_dim, out_dim, config.model)
    print(model)

    device = get_device()
    model.to(device)

    optimizer = config.optim.create_optimizer(model.parameters())

    # Tensorboard
    tb_writer = SummaryWriter(log_dir=config.tensorboard_dir)

    # Training
    train(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        config,
        device,
        writer=tb_writer,
    )

    # Prediction
    predict_and_show(
        model,
        eval_dataloader,
        config,
        eval_dataset.image_height,
        eval_dataset.image_width,
        device,
    )


if __name__ == "__main__":
    main()
