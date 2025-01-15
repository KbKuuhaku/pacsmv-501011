import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data.dataset as dataset
from PIL import Image

LABEL_TO_DESC = {
    0: "T-shift/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


# Parsing MNIST format
# Reference: https://yann.lecun.com/exdb/mnist/
def read_image_from_file(file: Path) -> tuple[np.ndarray, int, int]:
    """
    Byte file format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    """
    print(f"Parsing {file}...")
    images = []
    with open(file, "rb") as f:
        f.read(4)  # skip magic number
        n_image = int.from_bytes(f.read(4))
        height = int.from_bytes(f.read(4))
        width = int.from_bytes(f.read(4))

        print(f"#Images: {n_image} | Rows: {height} | Width: {width}")

        chunk_size = height * width
        while chunk := f.read(chunk_size):
            image_raw = np.array(list(chunk)).reshape(height, width)
            images.append(image_raw)

    # NOTE: list of numpy 1d -> use stack because we need a new dim
    images = np.stack(images, axis=0).astype(np.uint8)

    return images, height, width


def read_label_from_file(file: Path) -> np.ndarray:
    """
    Byte file format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    """
    print(f"Parsing {file}...")
    labels = []
    with open(file, "rb") as f:
        f.read(4)  # skip magic number
        n_label = int.from_bytes(f.read(4))

        print(f"#Labels: {n_label}")

        while byte := f.read(1):
            labels.append(int.from_bytes(byte))

    labels = np.array(labels)
    return labels


class MNIST(dataset.Dataset):
    def __init__(self, root: str, image_file: str, label_file: str) -> None:
        super().__init__()

        self.images, self.image_height, self.image_width = read_image_from_file(
            Path(root) / image_file
        )
        self.labels = read_label_from_file(Path(root) / label_file)
        self.n_class = len(LABEL_TO_DESC)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image_raw = torch.tensor(self.images[index]) / 255
        label = torch.tensor(self.labels[index], dtype=torch.uint8)

        return image_raw, label


def show_samples(
    images: np.ndarray,
    labels: np.ndarray,
    image_height: int,
    image_width: int,
    n_sample_per_class: int,
    save_to: Path,
) -> None:
    print(f"Showing {n_sample_per_class} samples of each class...")

    label_to_indexes = _group_indexes(labels)

    canvas_height = image_height * len(LABEL_TO_DESC)
    canvas_width = image_width * n_sample_per_class
    canvas = Image.new(mode="L", size=(canvas_height, canvas_width))

    for label in sorted(label_to_indexes):
        indexes = label_to_indexes[label]
        sample_indexes = random.choices(indexes, k=n_sample_per_class)
        print(label, sample_indexes)

        _draw_row(images, canvas, image_height, image_width, label, sample_indexes)

    canvas.save(save_to)
    print(f"Sample picture saved to {save_to}")


def _draw_row(
    images: np.ndarray,
    canvas: Image.Image,
    image_height: int,
    image_width: int,
    label: int,
    sample_indexes: list[int],
) -> None:
    y = label * image_height
    x = 0

    for sample_index in sample_indexes:
        image_raw = images[sample_index]
        image = Image.fromarray(image_raw, mode="L")

        canvas.paste(image, (x, y))

        x += image_width


def _group_indexes(labels: np.ndarray) -> dict[int, list[int]]:
    label_to_indexes = {}
    for index, label in enumerate(labels):
        if label not in label_to_indexes:
            label_to_indexes[label] = []

        label_to_indexes[label].append(index)

    return label_to_indexes
