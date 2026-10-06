"""HiFi-GAN-style residual dilated-conv blocks used inside Generator's
upsampling stack. `activation_cls` is pluggable (SnakeBeta/LeakyReLUActivation,
see Generator) so this block's activation always matches the rest of the
Generator's choice.
"""
import typing

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .commons import get_padding, init_weights
from .normalization import SnakeBeta

ActivationCls = typing.Callable[[int], nn.Module]


class ResBlock1(nn.Module):
    """Three (dilation, dilation=1) conv pairs, each gated by Snake."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: typing.Tuple[int, ...] = (1, 3, 5),
        activation_cls: ActivationCls = SnakeBeta,
    ):
        super().__init__()
        self.snakes1 = nn.ModuleList([activation_cls(channels) for _ in dilation])
        self.snakes2 = nn.ModuleList([activation_cls(channels) for _ in dilation])
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        for snake1, c1, snake2, c2 in zip(self.snakes1, self.convs1, self.snakes2, self.convs2):
            xt = snake1(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = snake2(c1(xt))
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in (*self.convs1, *self.convs2):
            remove_weight_norm(layer)


class ResBlock2(nn.Module):
    """Two-conv-pair variant of ResBlock1 (fewer dilations, lighter)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: typing.Tuple[int, ...] = (1, 3),
        activation_cls: ActivationCls = SnakeBeta,
    ):
        super().__init__()
        self.snakes = nn.ModuleList([activation_cls(channels) for _ in dilation])
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        for snake, conv in zip(self.snakes, self.convs):
            xt = snake(x)
            if x_mask is not None:
                xt = xt * x_mask
            x = conv(xt) + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class ResBlockInverted(nn.Module):
    """Inverted-residual / MBConv-style block (MobileNetV2, Sandler et al.
    2018 -- also the shape of Conformer's convolution module): pointwise
    (expand) -> depthwise -> pointwise (project), single residual around
    the whole block, replacing ResBlock1/ResBlock2's parallel dilated-conv
    branches entirely (one block per stage, not multiple summed branches).

    Two deliberate departures from the two blocks above, both novel to
    this direction (not MarGan): weight_norm instead of MobileNetV2's
    BatchNorm (matches every other conv in this Generator, and avoids
    BatchNorm's batch-size sensitivity during GAN training), and the
    project's own SnakeBeta instead of ReLU6.

    Linear bottleneck (MobileNetV2's own term): the last pointwise conv
    has NO activation after it -- the paper's finding that nonlinearities
    destroy information when applied to an already-projected-down (here:
    back to `channels`, not the expanded `channels*expansion`) representation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        activation_cls: ActivationCls = SnakeBeta,
        expansion: int = 2,
    ):
        super().__init__()
        hidden = channels * expansion
        self.pw_expand = weight_norm(Conv1d(channels, hidden, 1))
        self.act1 = activation_cls(hidden)
        self.dw = weight_norm(
            Conv1d(hidden, hidden, kernel_size, dilation=dilation, groups=hidden, padding=get_padding(kernel_size, dilation))
        )
        self.act2 = activation_cls(hidden)
        self.pw_project = weight_norm(Conv1d(hidden, channels, 1))  # linear bottleneck, no activation after
        for layer in (self.pw_expand, self.dw, self.pw_project):
            layer.apply(init_weights)
        # Zero-init pw_project so the block starts as an identity no-op
        # (residual passthrough), not a random perturbation from step 0 --
        # same graft-without-disturbing convention already used in this
        # project (MarGan's Stage A/B, flow_block.py's zero-initialized
        # `post` layer). weight_norm splits weight into direction
        # (weight_v) and magnitude (weight_g) -- zeroing weight_g alone
        # makes the effective weight exactly zero regardless of weight_v.
        self.pw_project.weight_g.data.zero_()
        self.pw_project.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
        xt = self.pw_expand(x)
        xt = self.act1(xt)
        if x_mask is not None:
            xt = xt * x_mask
        xt = self.dw(xt)
        xt = self.act2(xt)
        if x_mask is not None:
            xt = xt * x_mask
        xt = self.pw_project(xt)
        x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in (self.pw_expand, self.dw, self.pw_project):
            remove_weight_norm(layer)
