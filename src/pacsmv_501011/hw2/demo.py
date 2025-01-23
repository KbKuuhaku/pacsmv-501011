import random
from dataclasses import dataclass

import cv2
import numpy as np
import yaml
from PIL import ImageDraw, ImageFont, ImageGrab
from PIL.Image import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from ..core.object_detection.bbox import BBox2D


@dataclass
class DrawConfig:
    bbox_width: int
    text_width: int
    text_y_offset: int
    font: str
    font_size: int

    def __post_init__(self) -> None:
        self.image_font = ImageFont.truetype(self.font, self.font_size)

    def set_colors(self, k: int) -> None:
        self.colors = [f"#{color_hex:x}" for color_hex in random.sample(range(0xFFFFFF), k=k)]
        print(self.colors)

    def get_color_by_index(self, index: int) -> str:
        return self.colors[index]


@dataclass
class DetectConfig:
    seed: int
    screen_capture_bbox: list[int]
    model_ckpt: str
    threshold: float
    data_config: str
    draw: DrawConfig

    def __post_init__(self) -> None:
        with open(self.data_config) as f:
            self.label_id_to_name = yaml.safe_load(f)["names"]

        self.draw.set_colors(len(self.label_id_to_name))


def get_textbox_anchor(bbox: BBox2D, text_y_offset: int) -> tuple[int, int]:
    x0, y0 = bbox.top_left
    return x0, y0 - text_y_offset


def random_color() -> str:
    return f"#{random.randint(0, 0xFFFFFF):06x}"


def detect_and_draw_bboxes(model: YOLO, image: Image, config: DetectConfig) -> None:
    result = model.predict(image, verbose=False)[0]

    # Filter out empty detections
    if not result.boxes:
        return

    # Filter out unconfident predictions
    boxes = [box for box in result.boxes if float(box.conf.item()) >= config.threshold]
    if not boxes:
        return

    print(f"{len(boxes)} bbox(es) detected!")
    for box in boxes:
        draw_bbox(image, box, config)


def draw_bbox(image: Image, box: Boxes, config: DetectConfig) -> None:
    """
    Draw 2d bounding box on image, and write class and confidence on top of it
    """
    image_draw = ImageDraw.Draw(image, mode="RGB")

    # Bounding box
    # xywh actually means (center x, center y, width, height)
    # https://github.com/ultralytics/ultralytics/issues/6575#issuecomment-1829038496
    bbox = BBox2D.from_xyxy(box.xyxy.flatten().cpu().numpy())  # type: ignore

    # Predicted label
    label = int(box.cls.item())
    conf = box.conf.item()

    display_text = f"{config.label_id_to_name[label]} {conf:.2%}"
    print(bbox, display_text)

    color = config.draw.get_color_by_index(label)

    # Bounding box
    image_draw.rectangle(
        bbox.xyxy,
        outline=color,
        width=config.draw.bbox_width,
    )

    # Text
    text_anchor = get_textbox_anchor(bbox, config.draw.text_y_offset)
    # Get the coordinates of text box
    textbbox = image_draw.textbbox(
        text_anchor,
        text=display_text,
        font=config.draw.image_font,
        stroke_width=config.draw.text_width,
    )
    # Render the text box
    image_draw.rectangle(
        textbbox,
        fill=color,
        width=config.draw.bbox_width,
    )
    # Render the text
    image_draw.text(
        text_anchor,
        display_text,
        font=config.draw.image_font,
        stroke_width=config.draw.text_width,
    )


def uno_detect(config: DetectConfig) -> None:
    random.seed(config.seed)

    model = YOLO(config.model_ckpt)
    print(model)

    while True:
        # Get screenshot
        image = ImageGrab.grab(bbox=config.screen_capture_bbox)  # type: ignore

        # Detect UNO
        detect_and_draw_bboxes(model, image, config)

        # Display the picture
        display_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("Screen Capture", display_image)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
