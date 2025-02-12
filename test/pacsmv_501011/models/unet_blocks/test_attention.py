import torch
from click.termui import hidden_prompt_func


def test_rearrange() -> None:
    from einops import rearrange

    batch_size = 5
    seq_len = 12
    num_heads = 8
    hidden_dim = 768

    x = torch.randn(batch_size, seq_len, hidden_dim)

    intermediate = rearrange(
        x,
        pattern="b t (n_h d_h) -> b n_h t d_h",
        n_h=num_heads,
        d_h=hidden_dim // num_heads,
    )
    assert intermediate.shape == (batch_size, num_heads, seq_len, hidden_dim // num_heads)

    out = rearrange(
        intermediate,
        pattern="b n_h t d_h -> b t (n_h d_h)",
    )
    assert out.shape == x.shape


def test_multihead_attention() -> None:
    from pacsmv_501011.models.unet_blocks.attention import MultiHeadAttention

    value_dim = 256
    query_dim = 768
    hidden_dim = 768
    num_heads = 8

    self_attn = MultiHeadAttention(
        value_dim=value_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    )
    sa_input = torch.randn(5, 10, value_dim)

    assert self_attn(sa_input).shape == (5, 10, hidden_dim)

    cross_attn = MultiHeadAttention(
        value_dim=value_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        query_dim=hidden_dim,
    )
    token_embed = torch.randn(5, 10, query_dim)
    context_embed = torch.randn(5, 1, value_dim)

    assert cross_attn(token_embed, context=context_embed).shape == (5, 10, hidden_dim)
