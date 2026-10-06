"""Dilated and Depth-Separable Convolution stack, used inside the
stochastic duration predictor and ConvFlow's conditioner network."""
import torch
from torch import nn
from torch.nn import functional as F

from .normalization import LayerNorm


class DDSConv(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int, n_layers: int, p_dropout: float = 0.0
    ):
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None) -> torch.Tensor:
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = F.gelu(self.norms_1[i](y))
            y = self.convs_1x1[i](y)
            y = F.gelu(self.norms_2[i](y))
            x = x + self.drop(y)
        return x * x_mask
