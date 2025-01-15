import logging.config as lc

from .factories.io import read_to_dict


def setup_logging(config_file: str) -> None:
    config = read_to_dict(config_file)
    lc.dictConfig(config=config)
