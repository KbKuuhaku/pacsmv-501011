import torch


def test_decoder() -> None:
    from pacsmv_501011.models.unet_blocks.decoder import DecoderBlock

    concat_images = torch.randn(5, 512, 28, 28)
    # enc_images = torch.randn(5, 256, 64, 64)
    enc_images = torch.randn(5, 256, 56, 56)
    time_embed = torch.randn(5, 128)

    block = DecoderBlock(
        in_channels=512,
        enc_channels=256,
        out_channels=256,
        time_dim=128,
        num_res_blocks=2,
    )
    out1 = block(concat_images, enc_images, time_embed)
    # assert out1.shape == (5, 256, 52, 52)
    assert out1.shape == (5, 256, 56, 56)
