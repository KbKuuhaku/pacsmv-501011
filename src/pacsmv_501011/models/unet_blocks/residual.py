import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # (w - 3) + 1 + 2 x 1 = w
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.time_lin = nn.Linear(in_features=time_dim, out_features=out_channels)
        self.gnorm = nn.GroupNorm(32, num_channels=out_channels)

        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        conv_out = self.conv(x)  # (B, D, H, W)
        t_out = self.time_lin(t).reshape(B, -1, 1, 1)  # expand (B, D) to (B, D, 1, 1)

        out = self.silu(self.gnorm(conv_out + t_out))

        # Residual connection
        ret = conv_out + out

        return ret
