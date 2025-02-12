import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttentionError(Exception): ...


class AttentionBlock(nn.Module):
    def __init__(
        self,
        query_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(query_dim, num_heads, hidden_dim)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        self.cross_attn = MultiHeadAttention(value_dim, num_heads, hidden_dim, query_dim=query_dim)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        self.mlp = MLP(hidden_dim=hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # NOTE: Norm first, then attention, finally do residual connnection

        self_attn_out = self.self_attn(self.self_attn_norm(x)) + x
        cross_attn_out = (
            self.cross_attn(
                self.cross_attn_norm(self_attn_out),
                context=context,
            )
            + self_attn_out
        )
        mlp_out = self.mlp(self.mlp_norm(cross_attn_out)) + cross_attn_out

        return mlp_out


class MultiHeadAttention(nn.Module):
    """
    This is a mixed version of self/cross attention.

    Self Attention:
        - `query_dim` is None ==> `query_dim` == `value_dim`
        - `context` is None in `forward()`

    Cross Attention:
        - `query_dim` is not None
        - `context` is not None in `forward()`
    """

    def __init__(
        self,
        value_dim: int,
        num_heads: int,
        hidden_dim: int,
        query_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.is_cross_attn = True

        if query_dim is None:
            self.is_cross_attn = False
            query_dim = value_dim

        # NOTE: don't forget to set bias to false
        self.q_proj = nn.Linear(in_features=query_dim, out_features=hidden_dim, bias=False)
        self.k_proj = nn.Linear(in_features=value_dim, out_features=hidden_dim, bias=False)
        self.v_proj = nn.Linear(in_features=value_dim, out_features=hidden_dim, bias=False)

        self.num_heads = num_heads
        self.hidden_dim_per_head = hidden_dim // num_heads
        self.scale = self.hidden_dim_per_head ** (-1 / 2)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            if self.is_cross_attn:
                raise CrossAttentionError(
                    "`context` must be provided since this is a cross attention!"
                )
            context = x

        query = self.transform_to_multi_head(self.q_proj(x))
        key = self.transform_to_multi_head(self.k_proj(context))
        value = self.transform_to_multi_head(self.v_proj(context))

        # Flash attention
        attn_out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)

        return self.flatten_from_multi_head(attn_out)

    def transform_to_multi_head(self, x: torch.Tensor) -> torch.Tensor:
        ret = rearrange(
            x,
            pattern="b t (n_h d_h) -> b n_h t d_h",
            n_h=self.num_heads,
            d_h=self.hidden_dim_per_head,
        )
        return ret

    def flatten_from_multi_head(self, x: torch.Tensor) -> torch.Tensor:
        ret = rearrange(x, pattern="b n_h t d_h -> b t (n_h d_h)")
        return ret


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
