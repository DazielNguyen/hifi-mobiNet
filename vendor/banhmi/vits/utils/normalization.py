"""Normalization / activation building blocks shared by several components."""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """Channel-first LayerNorm: normalizes over the channel dim of a
    [batch, channels, time] tensor (torch's built-in LayerNorm normalizes
    over the last dim, so channels get transposed there and back)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class LeakyReLUActivation(nn.Module):
    """Vanilla HiFi-GAN/VITS activation (no Snake at all), matching the
    signature of SnakeBeta so it can be used as a drop-in `activation_cls`
    -- used by the Generator with the HiFi-GAN resblocks ("1"/"2").
    """

    _SLOPE = 0.1

    def __init__(self, channels: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self._SLOPE)


class SnakeBeta(nn.Module):
    """Snake-Beta activation (later BigVGAN revision): x + (1/beta) *
    sin(alpha * x)^2, with alpha (frequency) and beta (amplitude) as two
    *separate* learnable parameters instead of Snake1d's single alpha
    reused for both roles.

    Not present in EdgeTTS (which uses plain Snake1d) -- this is a
    BanhmiTTS-only addition, used by the Generator with the
    inverted-residual resblocks ("mb"/"mrf").
    """

    # Bounds alpha/beta to [_min, _max] instead of BigVGAN's usual
    # log-space (exp()) reparameterization -- see the TEMPORARY EXPERIMENT
    # note below. A hard abs()+clamp() floor has a failure mode log-space
    # doesn't: right at the floor, d(1/beta)/dbeta = -1/beta^2, so a floor
    # as small as the original 1e-9 turns a single optimizer step that
    # pushes beta toward 0 into a ~1e18-magnitude gradient -- enough to
    # overflow to Inf under fp16/bf16 autocast well before any global
    # grad-norm clip sees it, and Inf/NaN then propagates through every
    # downstream layer for the rest of training. 1e-2 keeps 1/beta (and
    # its gradient) bounded to a still-generous but finite dynamic range;
    # the upper clamp guards the symmetric case where alpha grows
    # unbounded and sin(alpha*x)'s gradient (x*cos(alpha*x)) blows up.
    _min = 1e-2
    _max = 1e2

    def __init__(self, channels: int):
        super().__init__()
        # TEMPORARY EXPERIMENT (2026-08-26): swapped to EdgeTTS's linear-
        # space abs()+clamp() reparameterization (piper_train/vits/modules.py)
        # instead of this file's previous log-space exp() one, to test
        # whether the reparameterization difference explains a measured
        # UTMOS gap. Revert to log_alpha/log_beta + exp() after the test.
        # The clamp bounds above were tightened from the original 1e-9
        # floor (NaN-prone, see docstring) without reverting the
        # experiment itself.
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.abs().clamp(min=self._min, max=self._max)
        beta = self.beta.abs().clamp(min=self._min, max=self._max)
        return x + (1.0 / beta) * torch.sin(alpha * x) ** 2
