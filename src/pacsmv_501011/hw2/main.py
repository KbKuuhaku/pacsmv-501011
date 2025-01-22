from pathlib import Path

import click
import rich

from ..core.config import parse_config


@click.group()
def cli() -> None: ...


@cli.command()
def train() -> None:
    from .train import UNOConfig, uno_train

    config_path = Path("configs", "hw2-train.toml")
    config = parse_config(path=str(config_path), dataclass_name=UNOConfig)

    rich.print(config)

    uno_train(config)


@cli.command()
def demo() -> None:
    from .demo import DetectConfig, uno_detect

    config_path = Path("configs", "hw2-demo.toml")
    config = parse_config(path=str(config_path), dataclass_name=DetectConfig)

    rich.print(config)

    uno_detect(config)
