- [Prerequisite](#prerequisite)
- [HW1](#hw1)
- [HW2](#hw2)
    - [Dataset Preparation](#dataset-preparation)
    - [Train YOLOv11](#train-yolov11)
    - [Demo](#demo)

## Prerequisite
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

---

## HW1

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

---

## HW2

UNO card detection.

[HW2 report](/src/pacsmv_501011/hw2/report.md).

- Thanks [uno-cards (v3, aug416)](https://universe.roboflow.com/joseph-nelson/uno-cards) for providing the data source with YOLOv11 format.

**The result can be reproduced but you need to manually set something**


### Dataset Preparation
First, download the dataset using `scripts/download_uno_cards.sh`. 

Then, inside of `data/uno-cards/data.yaml`, replace `names` field with these lines

```yaml
# Modified name mapping referenced from https://www.unorules.com/
names:
  0: "0"
  1: "1"
  2: "Wild Draw 4"
  3: "Draw Two"
  4: "Reverse"
  5: "Skip"
  6: "Wild"
  7: "2"
  8: "3"
  9: "4"
  10: "5"
  11: "6"
  12: "7"
  13: "8"
  14: "9"
```

### Train YOLOv11

NOTE: Due to the [weird global setting](https://github.com/ultralytics/ultralytics/issues/1809) 
(tldr: configurations have to be overridden before `ultralytics` gets imported), I have to manually set `YOLO_CONFIG_DIR="./configs/` to make it work.

```bash
YOLO_CONFIG_DIR="./configs" uv run hw2 train
```

You can check the training results in `yolo-runs/detect/uno-cards{suffix}`. 
The `suffix` starts from 0 and will increase every time you rerun the program.

### Demo

I've stored my checkpoint in `ckpts/best.pt` and uploaded it to github, you can try it if you want :)
(the checkpoint should've been stored in google drive but I'm a bit lazy)

```bash
uv run hw2 demo
```

**The demo will capture your monitor with `(0, 0, 1920, 1080)` by default**

Additionally, you can change these in `configs/hw2-demo.toml`:

```toml
...

screen_capture_bbox = [0, 0, 1920, 1080]  # (x0, y0, x1, y1)
model_ckpt = "ckpts/best.pt"

```

