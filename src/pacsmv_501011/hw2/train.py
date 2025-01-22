from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


@dataclass
class DataConfig:
    data_root: str
    config: str

    @property
    def config_file(self) -> Path:
        return Path(self.data_root) / self.config


@dataclass
class TrainConfig:
    pretrained_weight: str
    name: str
    n_epoch: int
    batch_size: int
    resume: bool


@dataclass
class HyperParams:
    dropout: float
    patience: int


@dataclass
class UNOConfig:
    seed: int
    train: TrainConfig
    data: DataConfig
    hparams: HyperParams


def uno_train(config: UNOConfig):
    model = YOLO(config.train.pretrained_weight)

    model.train(
        seed=config.seed,  # for reproducibility
        # data
        data=config.data.config_file,  # will load data based on the dataset config file
        # train
        name=config.train.name,
        epochs=config.train.n_epoch,
        batch=config.train.batch_size,
        dropout=config.hparams.dropout,
        patience=config.hparams.patience,
    )
