# HW3 Report

I reimplemented a Latent Diffusion Model, similar to Stable Diffusion, and trained it on MNIST. 

## What is Stable Diffusion

Stable Diffusion is a generative model that combines 
**Diffusion Model**, 
**U-Net**, 
**Spatial Transformer** 
and **Auto Encoder** together.

### 1. Diffusion Model (Core)
The key idea of Diffusion Model, at a higher level, is to first add noises gradually in the forward pass (forward diffusion), then find a way to remove noises in the backward pass (reverse diffusion). 

### 2. U-Net (Backbone)
During the reverse diffusion, the update rule is related to **score function**:

$$
    \frac{d}{dx} \log p(x, T - t)
$$

where $x$ is the input of current step, $t$ is the current time step, and $T$ is the total steps we need to take. 

Unfortunately, we don't know how to compute the score function and have to learn it. One way to learn the score function is **training a U-Net, a CNN-based downsampling+upsamping model**

![U-Net Architecture](../../../pics/u-net-architecture.png)

The structure can be split into four parts:

- Downsampling:

- Mid:

- Upsamping:

- Skip connection:

U-Net receive images as input, and output a score that will be used in the reverse diffusion. 
However, the score has to be related to the time step $t$, so we also need to add a `time_embedding`
inside our neural network.

### 3. Spatial Transformer (Text Encoding)

NOTE: In Stable Diffusion, they actually used [CLIP ViT](), a pretrained multimodal model as a start.

### 4. Auto Encoder (Image Compression)

One downside of our diffusing process is the **slow speed**, since the whole process works on high dimensional data. However, we can improve this by putting the input into a latent space. This is where **Auto Encoder** comes in: it can help us **compress images into a lower dimensional latent space**.

I reimplement a `VAE` class, a variational autoencoder (VAE) model that can ensure the model performs well on encoding images in general.

**Putting all together, I have reimplemented a `LatentDiffusion` model in this homework, similar to `StableDiffusion` but much smaller, and trained it on MNIST to see the result.**


## Result

---

References:

> Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

> Diffusion Model with Cross Attention. https://colab.research.google.com/drive/1Y5wr91g5jmpCDiX-RLfWL1eSBWoSuLqO?usp=sharing#scrollTo=PUdcXlsLN5J4

