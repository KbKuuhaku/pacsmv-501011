import torch

from pacsmv_501011.models.spatial_transformer import SpatialTransformer, SpatialTransformerConfig
from pacsmv_501011.models.unet_blocks.encoder import EncoderBlock


def test_encoder_no_context() -> None:
    images = torch.randn(5, 256, 32, 32)
    time_embed = torch.randn(5, 128)

    block = EncoderBlock(in_channels=256, out_channels=512, time_dim=128, num_res_blocks=2)
    out1, hidden1 = block(images, time_embed)
    assert hidden1.shape == (5, 512, 32, 32)
    assert out1.shape == (5, 512, 16, 16)

    # assert hidden1.shape == (5, 512, 28, 28)
    # assert out1.shape == (5, 512, 14, 14)


def test_encoder_with_context() -> None:
    images = torch.randn(5, 256, 32, 32)
    time_embed = torch.randn(5, 128)
    spt = SpatialTransformer(
        channels=512,
        config=SpatialTransformerConfig(
            context_dim=768,
            num_heads=8,
            num_layers=1,
        ),
    )
    context = torch.randn(5, 1, 768)
    block = EncoderBlock(
        in_channels=256,
        out_channels=512,
        time_dim=128,
        num_res_blocks=2,
        spatial_transformer=spt,
    )
    out2, hidden2 = block(images, time_embed, context=context)
    assert hidden2.shape == (5, 512, 32, 32)
    assert out2.shape == (5, 512, 16, 16)

    # assert hidden2.shape == (5, 512, 28, 28)
    # assert out2.shape == (5, 512, 14, 14)
