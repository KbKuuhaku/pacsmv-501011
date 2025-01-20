# How to use it

### Prerequisite
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

- Python 3.12. Install it using [uv](https://docs.astral.sh/uv/guides/install-python/) 
or [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation). Changing python default version is not recommended

- Model and data will be transferred to specific device automatically (multi devices are not suppported)

```python
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### HW1

Simple MLP trained on Fashion-MNIST.

[HW1 report](src/pacsmv_501011/hw1/report.md)

First, download MNIST using `scripts/download_mnist.sh`, or Fashion-MNIST using `scripts/download_fmnist.sh`

Then run

Adam
```bash
uv run hw1 "hw1-adam" 
```

SGD
```bash
uv run hw1 "hw1-sgd" 
```


### HW2

UNO card detection.

[HW2 report]()

Thanks [uno-cards (v3, aug416)](https://universe.roboflow.com/joseph-nelson/uno-cards) for providing the data source with YOLOv11 format.

First, download the dataset using `scripts/download_uno_cards.sh`.

#### Train YOLOv11

NOTE: Due to the [weird global setting](https://github.com/ultralytics/ultralytics/issues/1809) 
(tldr: configurations have to be overridden before `ultralytics` gets imported), I have to manually set `YOLO_CONFIG_DIR="./configs/` to make it work.

```bash
YOLO_CONFIG_DIR="./configs" uv run hw2 train
```
