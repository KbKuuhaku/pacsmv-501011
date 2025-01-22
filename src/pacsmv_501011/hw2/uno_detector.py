from dataclasses import dataclass
from ultralytics import YOLO
from ..core.config import parse_config


@dataclass
class DetectConfig:
    model_ckpt: str
    image_sample_links: list[str]


def uno_detect(config: DetectConfig) -> None:
    YOLO(config.model_ckpt)
