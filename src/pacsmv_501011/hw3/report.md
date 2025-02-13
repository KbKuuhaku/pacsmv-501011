# HW3 Report

I reimplemented a Diffusion Model, a simplified version of Stable Diffusion, and trained it on MNIST. 


## What is Diffusion Model

Diffusion Model is a generative model that combines 
**Reverse Diffusion**, 
**Spatial Transformer**
and **U-Net**.


### 1. Diffusion (Core)
The key idea of Diffusion Model, at a higher level, is to first add noises gradually in the forward pass (forward diffusion), then find a way to remove noises in the backward pass (reverse diffusion). 

During the reverse diffusion, the update rule is related to **score function**:

$$
    \frac{d}{dx} \log p(x, T - t)
$$

where $x$ is the input of current step, $t$ is the current time step, and $T$ is the total steps we need to take. 

Unfortunately, we don't know how to compute the score function and have to learn it. One way to learn the score function is **training a U-Net, a CNN-based downsampling+upsamping model**, which will be discussed later.

### 2. Spatial Transformer (Text Encoding)

Spatial Transformer is one type of Transformers. Different from the classic one, Spatial Transformer takes in **feature map** as input, adds **conditional constraints** (i.e. numbers in MNIST) onto the existing features, making image generation more controllable. **Attention Block** is the core of Spatial Transformer.

The basic workflow of Attention Block is as follows: 

- Compute the **attention score matrix** from **query** and **key**. 

- Get normalized using softmax and a scale factor. 

- Combine the attention score matrix with **value** and gets a semantic features on query. 

Depending on the source of inputs, attention is split into two categories:

- Self Attention: query, key and value comes from the same embedding.

- Cross Attention: query comes from `token` embedding, while key and value comes from `context` embedding.

Additionally, to speed up the computation on attention, **Multi-head Attention** is introduced. The basic idea is to 
split the hidden dimension into `num_heads` chunks, and then perform a parallel scaled dot product 
attention.

NOTE: In Stable Diffusion, they actually used [CLIP](https://github.com/openai/CLIP), a pretrained multimodal model as a start.


### 3. U-Net (Backbone)

![U-Net Architecture](../../../pics/u-net-architecture.png)

In general, U-Net receives images as input, then outputs a score that will be used in the reverse diffusion. 
However, the score has to be related to the time step $t$, so we also need to add a `time_embedding`
inside our neural network.

The structure can be split into two parts:

- Encoder: Downsampling images while increasing the number of channels. Moreover,

    - Time embedding is added in each encoder block, letting the model know the information of the time step

    - If Spatial Transformer is added on certain levels, combining with the text constraints, U-Net could learn the relation between text and feature maps, therefore controlling the image generation.

- Decoder: Upsampling features while decreasing the number of channels. Based on current image and timestep, the decoder will output a score that can be used in the reverse diffusion. 


## Implementation Details

### U-Net Model

- Copy&Crop: in decoder block, the upsampled feature map `up_x` has to be concatenated with the hidden output `enc_h` of encoder block from the same level. Unfortunately, the width of two feature maps are not equal, and `enc_h` needs a center-cropping. 

![Center and Crop](../../../pics/center-and-crop.png)

The center-cropping code is implemented as follows:

```python
def _center_crop(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    """
    Center-crop the feature map `src` to be the same size as `tar`
    """
    src_w = src.shape[-1]
    tar_w = tar.shape[-1]

    if src_w == tar_w:
        warnings.warn(
            f"Performing ceter cropping, but src_w and tar_w are the same ({tar_w}), skipped"
        )
        return src

    if src_w < tar_w:
        raise CenterCropError(
            f"Performing ceter cropping, but src_w ({src_w}) < tar_w ({tar_w})"
        )

    start = (src_w - tar_w) // 2

    return src[:, :, start:-start, start:-start]
```

## Result


The model can be further improved by including a **VAE** (variational autoencoder) to **compress images into the low-dimensional latent space**. Due to the time constraints, I didn't implement this part, but I might add one in the future to see the improvement on training :)

---

References:

> Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

> Diffusion Model with Cross Attention. https://colab.research.google.com/drive/1Y5wr91g5jmpCDiX-RLfWL1eSBWoSuLqO?usp=sharing#scrollTo=PUdcXlsLN5J4

