"""Adversarial discriminators. Two families, both scoring real vs.
generated waveforms and both used together during training (their losses
are summed in the training loop):

- MultiPeriodDiscriminator (original HiFi-GAN/VITS): reshapes the waveform
  into 2D at several fixed periods, so each sub-discriminator specializes
  in one periodicity.
- MultiResolutionDiscriminator (UnivNet-style, EdgeTTS addition beyond
  vanilla Piper): scores the STFT magnitude at several time-frequency
  resolutions instead of the raw waveform.

Training-only: none of this is part of the exported/ONNX inference model.
"""
import typing

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from ..utils.commons import get_padding

_LRELU_SLOPE = 0.1


class DiscriminatorS(nn.Module):
    """Scores the raw waveform directly (no period reshaping)."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), _LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorP(nn.Module):
    """Reshapes the waveform to 2D with row length `period` and convolves
    over that grid, so it specializes in periodic structure at that period."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        channels = (1, 32, 128, 512, 1024, 1024)
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        channels[i],
                        channels[i + 1],
                        (kernel_size, 1),
                        (stride, 1) if i < 4 else 1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for i in range(5)
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t += pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = F.leaky_relu(layer(x), _LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    _PERIODS = (2, 3, 5, 7, 11)

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in self._PERIODS]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return _run_discriminators(self.discriminators, y, y_hat)


class DiscriminatorR(nn.Module):
    """Single-resolution STFT-magnitude discriminator (UnivNet)."""

    def __init__(self, resolution: typing.Tuple[int, int, int], use_spectral_norm: bool = False):
        super().__init__()
        self.resolution = resolution
        _, _, win_length = resolution
        self.register_buffer("hann_window", torch.hann_window(win_length))
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, 32, (3, 9), padding=(1, 4))),
                norm_f(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        x = x.squeeze(1)
        pad = (n_fft - hop_length) // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        # cuFFT doesn't support half/bf16 input, so this must stay fp32 even
        # when the rest of the discriminator runs under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.stft(
                x.float(), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=self.hann_window, center=False, return_complex=True,
            )
        return torch.abs(spec).unsqueeze(1)  # [B, 1, Freq, Frames]

    def forward(self, x: torch.Tensor):
        fmap = []
        mag = self._spectrogram(x)
        for layer in self.convs:
            mag = F.leaky_relu(layer(mag), _LRELU_SLOPE)
            fmap.append(mag)
        mag = self.conv_post(mag)
        fmap.append(mag)
        return torch.flatten(mag, 1, -1), fmap


class MultiResolutionDiscriminator(nn.Module):
    _DEFAULT_RESOLUTIONS = ((512, 50, 240), (1024, 120, 600), (2048, 240, 1200))

    def __init__(
        self,
        resolutions: typing.Optional[typing.List[typing.Tuple[int, int, int]]] = None,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        resolutions = resolutions or self._DEFAULT_RESOLUTIONS
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(r, use_spectral_norm) for r in resolutions]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return _run_discriminators(self.discriminators, y, y_hat)


def _run_discriminators(discriminators, y: torch.Tensor, y_hat: torch.Tensor):
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
    for d in discriminators:
        y_d_r, fmap_r = d(y)
        y_d_g, fmap_g = d(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)
    return y_d_rs, y_d_gs, fmap_rs, fmap_gs
