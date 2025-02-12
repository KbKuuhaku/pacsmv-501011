# HW3 Report

I trained a [Ave Mujica](https://anime.bang-dream.com/avemujica/) (an anime) [LoRA](https://huggingface.co/docs/diffusers/en/training/lora) based on [Illustrious-XL](https://civitai.com/models/795765/illustrious-xl). Since the training and text-to-image generation code is very hard to reproduce, I chose [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as a shortcut, and the whole workflow is GUI-based. But I will touch the concept of LoRA and Stable Diffusion in the report.


## Theory
### What is Stable Diffusion

> NOTE: Illustrious-XL is an advanced Stable Diffusion XL (SD XL)-based model, developed by OnomaAI Research, optimized specifically for illustration and animation tasks. I chose it for better quality of image generation on animation characters

### Why LoRA


## Tuning Details

Since I'm using 4070Ti (12GB) to train the LoRA based on SDXL, there're a few things to tweak: 

- [Lora-Training-in-Comfy](https://github.com/LarryJane491/Lora-Training-in-Comfy) only supports Stable Diffusion. I have to find:

    - How to train SDXL using this repo: this custom node implementation is based on [sd-scripts](https://github.com/kohya-ss/sd-scripts), and it provides both `train_network.py` and `sdxl_train_network.py`, so simply replace the former with the latter will work

    - In `comfy-ui/custom_nodes/Lora-Training-in-Comfy/train.py`, it actually called `sd-scripts/train_network.py`, therefore, this is the exact place to do the replacement

- [Text encoder takes a lot of VRAM](https://github.com/kohya-ss/sd-scripts/issues/661#issuecomment-1643945086), so I have to manually set `train_unet_only` to 1 in `comfy-ui/custom_nodes/Lora-Training-in-Comfy/train.py`

 
## Result

---

References:

1. https://www.youtube.com/watch?v=cCH-1tS5OgA&t=243s (How to train a LoRA using ComfyUI)

2. https://www.bilibili.com/video/BV1ZtkRYhEmR (How to generate anime images with a pretrained Illustrious-XL using ComfyUI)

3. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ (What are Diffusion Models)
