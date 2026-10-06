"""Predicts a per-phoneme log-F0 (pitch) value from the text encoder's hidden
state -- same input/architecture shape as DurationPredictor. Training-only
target comes from averaging the ground-truth per-frame F0 (extracted at
preprocess time by banhmi_train/preprocess/norm_audio.py::cache_f0) within
each phoneme's MAS-aligned frame range; see SynthesizerTrn.forward()/infer()
for how this predictor's output actually conditions the decoder (and why it's
the ground-truth average re-expanded through the same hard alignment at both
training and inference time, not the raw per-frame contour -- see this
project's own commit/PR notes on the EdgeTTS F0 train/inference mismatch this
was written to avoid).
"""
import torch
from torch import nn

from ..utils.normalization import LayerNorm


class F0Predictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm_1(torch.relu(self.conv_1(x * x_mask))))
        x = self.drop(self.norm_2(torch.relu(self.conv_2(x * x_mask))))
        return self.proj(x * x_mask) * x_mask  # log-F0 per phoneme, [B, 1, T_text]
