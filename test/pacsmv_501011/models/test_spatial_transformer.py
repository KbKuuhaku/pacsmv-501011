import torch
import torch.nn as nn


def test_rearrange() -> None:
    from einops import rearrange

    images = torch.randn(5, 4, 10, 15)
    B, C, H, W = images.shape

    x = rearrange(images, pattern="b c h w -> b (h w) c")

    assert x.shape == (5, 150, 4)

    out = rearrange(x, pattern="b (h w) c -> b c h w", h=H, w=W)

    assert out.shape == images.shape


def test_spatial_transformer() -> None:
    from pacsmv_501011.models.spatial_transformer import (
        SpatialTransformer,
        SpatialTransformerConfig,
    )

    images = torch.randn(5, 256, 10, 15)
    model = SpatialTransformer(
        channels=images.shape[1],
        config=SpatialTransformerConfig(
            context_dim=768,
            num_heads=8,
            num_layers=1,
        ),
    )
    print(model)
    text_embed = nn.Embedding(10, 768)
    context = 4 * torch.ones(5, 1).long()

    out = model(images, text_embed(context))

    assert out.shape == images.shape
