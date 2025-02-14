from pathlib import Path

import click
import rich

from ..core.config import parse_config


@click.group()
def cli() -> None: ...


@cli.command()
def train() -> None:
    from .train import TrainConfig, train_diffusion_model

    config_path = Path("configs", "hw3-train.toml")
    config = parse_config(path=str(config_path), dataclass_name=TrainConfig)

    rich.print(config)

    train_diffusion_model(config)


@cli.command()
def demo() -> None:
    from .demo import DemoConfig, demo

    config_path = Path("configs", "hw3-demo.toml")
    config = parse_config(path=str(config_path), dataclass_name=DemoConfig)

    rich.print(config)

    demo(config)
