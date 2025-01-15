from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class COCOImage:
    id: int
    width: int
    height: int
    file_name: str


@dataclass(frozen=True, slots=True)
class COCOAnnotation:
    id: int
    image_id: int
    category_id: float
    area: float
    bbox: list[float]

    def validate_bounding_box(self, image: COCOImage) -> bool:
        x_min, y_min, width, height = self.bbox
        if not (width > 1 and height > 1):
            return False
        x_max = x_min + width
        y_max = y_min + height
        cols = image.width
        rows = image.height

        if not (0 <= x_min <= image.width) or not (0 <= x_max <= image.width):
            return False
        if not (0 <= y_min <= image.height) or not (0 <= y_max <= image.height):
            return False
        if (x_max <= x_min) or (y_max <= y_min):
            return False

        return True


@dataclass(frozen=True, slots=True)
class COCOCategory:
    id: int
    name: str
    supercategory: str


@dataclass(frozen=True, slots=True)
class COCO:
    images: list[COCOImage]
    annotations: list[COCOAnnotation]
    categories: list[COCOCategory]
