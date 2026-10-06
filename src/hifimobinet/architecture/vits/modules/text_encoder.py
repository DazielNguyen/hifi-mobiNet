"""Phoneme-id sequence -> hidden states + prior distribution (m, logs).

The prior (m, logs) parameterizes a diagonal Gaussian over the latent
space that the normalizing flow maps to/from -- this is the "conditional"
half of VITS's conditional VAE.
"""
import math

import torch
from torch import nn

from ..utils import commons
from ..utils.attention import Encoder


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).type_as(x)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask
