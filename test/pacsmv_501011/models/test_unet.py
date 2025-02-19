import torch


def test_unet() -> None:
    from pacsmv_501011.models.spatial_transformer import SpatialTransformerConfig
    from pacsmv_501011.models.unet import UNet, UNetConfig

    corpus_length = 10  # MNIST

    in_channels = 1
    out_channels = 1
    num_res_blocks = 2

    model_channels = 32
    channel_multipliers = [1, 2, 4]  # multipliers on `model_channels` on each level
    attention_levels = [2, 3]  # where the attention block should be added

    context_dim = 256
    num_heads = 8
    num_layers = 1

    config = UNetConfig(
        corpus_length=corpus_length,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        attention_levels=attention_levels,
        num_res_blocks=num_res_blocks,
        channel_multipliers=channel_multipliers,
        transformer=SpatialTransformerConfig(
            context_dim=context_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        ),
    )

    model = UNet(config).cuda()

    images = torch.randn(5, 1, 128, 128).cuda()
    t = torch.randn(images.shape[0]).cuda()
    context = (torch.ones(5, 1) * 4).long().cuda()

    out = model(images, t, context)

    # assert out.shape != (5, 1, 128, 128)
    assert out.shape == (5, 1, 128, 128)
