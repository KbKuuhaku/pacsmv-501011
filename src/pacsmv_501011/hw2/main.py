from pathlib import Path

import click
import rich

from ..core.config import parse_config


@click.group()
def cli(): ...


@cli.command()
def train():
    from .train import UNOConfig, train

    config_path = Path("configs", "hw2-train.toml")
    config = parse_config(path=str(config_path), dataclass_name=UNOConfig)

    rich.print(config)

    train(config)
