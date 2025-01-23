# HW2 Report

I trained a YOLOv11 model to detect UNO cards (99% acc on dataset) and provides a demo that detects
UNO cards from the captured screen in real-time.

![Demo Screenshot](../../../pics/hw2-demo.png)
![Training result](../../../pics/hw2-train-results.png)

- Thanks to [ultralytics YOLO](https://docs.ultralytics.com), I can train a YOLO model 
and detect bounding boxes with just a few lines of code

- The implementation of demo is based on `pillow` and `cv2`. Specifically,
    - `PIL.ImageDraw`: drawing the bounding boxes with text on top of it
    - `PIL.ImageGrab`: capturing screen(s) and storing the image in `PIL.Image.Image` format
    - `cv2.imshow` and `cv2.waitKey(1)`: output the processed image to a window

## Dataset
An annotated dataset [uno-cards (v3, aug416)](https://universe.roboflow.com/joseph-nelson/uno-cards)
that provides bounding box annotations of UNO cards, data augmentations, and preprocessed popular data formats.

- Train set: 31475 images with bounding boxes and labels
- Valid set: 1798 images with bounding boxes and labels
- Test set: 899 images with bounding boxes and labels

Detailed augmentations are described below:

```
Augmentations
Outputs per training example: 5
Crop: 0% Minimum Zoom, 25% Maximum Zoom
Rotation: Between -10° and +10°
Shear: ±10° Horizontal, ±10° Vertical
Grayscale: Apply to 5% of images
Hue: Between -5° and +5°
Saturation: Between -10% and +10%
Brightness: Between -15% and +15%
Exposure: Between -15% and +15%
Blur: Up to 1.5px
```

## Regularization
`ultralytics` already sets some regularization techniques as default. 
- Early stop (Patience)
- Dropout

You can pass in your setting in `model.train` to override them:
```python
model.train(
    ...
    dropout=config.hparams.dropout,
    patience=config.hparams.patience,
)
```

This is a part of the `args.yaml` auto generated during the run
```yaml
task: detect
mode: train
model: yolo11n.pt
data: data/uno-cards/data.yaml
epochs: 10

...

patience: 5
batch: 64

...

dropout: 0.2
```

## Model Structure

[YOLOv11](https://yolov11.com/#what-is) is a CNN-based model developed by Ultralytics and aimed at object detection tasks.
The model structure is as follows:

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolo11
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

## Implementation Details

### `dataset.yaml` in `ultralytics`

`ultralytics` supports loading dataset based on a yaml config. 
The recommended format in official documentation is as follows:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8 # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Classes (80 COCO classes)
names:
    0: person
    1: bicycle
    2: car
    # ...
    77: teddy bear
    78: hair drier
    79: toothbrush
```

However, [roboflow](https://roboflow.com/formats/yolov11-pytorch-txt) stores `names` as `list` 
instead of `dict`:

```yaml
names: ['0', '1', '10', '11', '12', '13', '14', '2', '3', '4', '5', '6', '7', '8', '9']
```

After some trail-and-errors (which actually takes a long time), I found the conversion from list to dict is `dict(enumerate(names))`.

Adding the exact labels from [unorules](https://www.unorules.com/), I'm able to list the mapping of labels:

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

### Demo

- The color of bounding boxes is depended on the predicted class. Randomly generate `k` colors 
at the start and index them by `label_id` later

```python
def set_colors(self, k: int) -> None:
    self.colors = [f"#{color_hex:x}" for color_hex in random.sample(range(0xFFFFFF), k=k)]
    print(self.colors)
```

- In order to render text above the detected bounding boxes, there're two steps

    - compute the `text_anchor`: `text_anchor` is the top left corner of the displayed text,
    and can be computed from the bounding box of the detected object
    ```python
    def get_textbox_anchor(bbox: BBox2D, text_y_offset: int) -> tuple[int, int]:
        x0, y0 = bbox.top_left
        return x0, y0 - text_y_offset
    ```

    - render text including a bounding box with background color
    ```python
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
    ```
