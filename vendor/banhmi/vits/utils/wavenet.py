"""WaveNet-style dilated-conv stack with gated activations, used as the
conditioner network inside PosteriorEncoder and ResidualCouplingLayer.
"""
import torch
from torch import nn

from .commons import fused_add_tanh_sigmoid_multiply


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.in_layers.append(nn.utils.weight_norm(in_layer, name="weight"))

            # Last layer only needs a residual OR skip output, not both.
            res_skip_channels = (
                hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            )
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            self.res_skip_layers.append(nn.utils.weight_norm(res_skip_layer, name="weight"))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor))
            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            nn.utils.remove_weight_norm(layer)
