"""Transformer self-attention stack used by TextEncoder (and, as a small
extra context pass, TransformerCouplingLayer in flow_block.py).

Upstream's attentions.py also defines a `Decoder` class (cross-attention
transformer decoder) -- dropped here since nothing in the training pipeline
calls it: VITS's decoder is the flow-based Generator, not a transformer:
this class is leftover from an earlier Transformer-TTS-style design.
"""
import math
import typing

import torch
from torch import nn
from torch.nn import functional as F

from .normalization import LayerNorm


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            y = self.drop(attn(x, x, attn_mask))
            x = norm1(x + y)
            y = self.drop(ffn(x, x_mask))
            x = norm2(x + y)
        return x * x_mask


class MultiHeadAttention(nn.Module):
    """Self/cross attention with T5-style relative position embeddings
    (window_size caps how far the relative-position table extends)."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: typing.Optional[int] = None,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask=None) -> torch.Tensor:
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        out = self._attention(q, k, v, mask=attn_mask)
        return self.conv_o(out)

    def _attention(self, query, key, value, mask=None):
        # [b, d, t] -> [b, n_h, t, d_k]
        b, _d, t_s, t_t = key.size(0), key.size(1), key.size(2), query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_rel = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = torch.matmul(
                query / math.sqrt(self.k_channels), key_rel.unsqueeze(0).transpose(-2, -1)
            )
            scores = scores + self._relative_to_absolute(rel_logits)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = self._absolute_to_relative(p_attn)
            value_rel = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + torch.matmul(relative_weights, value_rel.unsqueeze(0))

        # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output.transpose(2, 3).contiguous().view(b, -1, t_t)

    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length: int):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            relative_embeddings = F.pad(
                relative_embeddings, (0, 0, pad_length, pad_length, 0, 0)
            )
        return relative_embeddings[:, slice_start:slice_end]

    @staticmethod
    def _relative_to_absolute(x: torch.Tensor) -> torch.Tensor:
        """x: [b, h, l, 2l-1] -> [b, h, l, l]"""
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, 1, 0, 0, 0, 0, 0, 0))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, (0, length - 1, 0, 0, 0, 0))
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_to_relative(x: torch.Tensor) -> torch.Tensor:
        """x: [b, h, l, l] -> [b, h, l, 2l-1]"""
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, length - 1, 0, 0, 0, 0, 0, 0))
        x_flat = x.view([batch, heads, length * length + length * (length - 1)])
        x_flat = F.pad(x_flat, (length, 0, 0, 0, 0, 0))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self._same_padding(x * x_mask))
        x = self.drop(torch.relu(x))
        x = self.conv_2(self._same_padding(x * x_mask))
        return x * x_mask

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return F.pad(x, (pad_l, pad_r, 0, 0, 0, 0))
