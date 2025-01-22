from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(slots=True, frozen=True)
class BBox2D:
    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def from_xywh(cls, xywh: np.ndarray) -> Self:
        x_center, y_center, width, height = xywh.astype(int)

        x0 = x_center - width // 2
        x1 = x_center + width // 2
        y0 = y_center - height // 2
        y1 = y_center + height // 2

        return cls(x0, y0, x1, y1)

    @classmethod
    def from_xyxy(cls, xyxy: np.ndarray) -> Self:
        x0, y0, x1, y1 = float_numpy_to_int_list(xyxy)
        return cls(x0, y0, x1, y1)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2

    @property
    def top_left(self) -> tuple[int, int]:
        return self.x0, self.y0

    @property
    def bottom_right(self) -> tuple[int, int]:
        return self.x1, self.y1

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        height = self.y1 - self.y0
        width = self.x1 - self.x0
        x_center, y_center = self.center
        return x_center, y_center, width, height


def float_numpy_to_int_list(src: np.ndarray) -> list[int]:
    return src.astype(int).tolist()  # type: ignore
