"""VITS2-style time-step-wise duration discriminator.

Scores each phoneme's predicted duration as real/fake, conditioned on the
text-encoder hidden state at that position. Training-only: discarded
before ONNX export, so it adds zero cost to the deployed model.
"""
import torch
from torch import nn

from ..utils.normalization import LayerNorm


class DurationDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = LayerNorm(filter_channels)
        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def _forward_probability(self, x, x_mask, dur):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.drop(self.pre_out_norm_1(torch.relu(self.pre_out_conv_1(x * x_mask))))
        x = self.drop(self.pre_out_norm_2(torch.relu(self.pre_out_conv_2(x * x_mask))))
        x = (x * x_mask).transpose(1, 2)
        return self.output_layer(x)  # [B, T, 1]

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, dur_r: torch.Tensor, dur_hat: torch.Tensor, g=None):
        x = torch.detach(x)
        if g is not None:
            x = x + self.cond(torch.detach(g))
        x = self.drop(self.norm_1(torch.relu(self.conv_1(x * x_mask))))
        x = self.drop(self.norm_2(torch.relu(self.conv_2(x * x_mask))))

        prob_real = self._forward_probability(x, x_mask, dur_r)
        prob_fake = self._forward_probability(x, x_mask, dur_hat)
        return [prob_real], [prob_fake]
