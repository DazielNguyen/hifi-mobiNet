"""Invertible flow steps used to build up the normalizing-flow prior
(ResidualCouplingBlock in flow_block.py) and the stochastic duration
predictor's two flow stacks.

Each flow's forward() returns (output, logdet) when reverse=False and just
output when reverse=True -- the log-determinant of the Jacobian is only
needed for the forward (density-evaluation) direction.
"""
import math

import torch
from torch import nn

from .dds_conv import DDSConv
from .transforms import piecewise_rational_quadratic_transform


class Log(nn.Module):
    # Shared with ElementwiseAffine's bound below -- exp(10) ~= 2.2e4, still
    # far more than any legitimate value this flow produces (log-duration
    # of 10 would mean a duration of ~22000 frames for one phoneme), but
    # tight enough that a value passing this guard can't itself explode a
    # downstream Conv1d stack (the previous bound of 30 => exp(30) ~= 1e13
    # was observed to survive this guard intact and then blow up further
    # downstream, causing a 3rd StochasticDurationPredictor collapse despite
    # both this guard and gradient clipping being active).
    _EXP_CLAMP_MAX = 10.0

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        # reverse=True runs every training step too (SynthesizerTrn.forward's
        # duration-discriminator sample, StochasticDurationPredictor._sample),
        # not just at inference -- an ordinary-looking upstream value here can
        # still exp() into inf under bf16/fp32 if training pushes x large
        # before this flow stabilizes. Same failure family as the spline
        # transform's unguarded division/log (see transforms.py's _EPS) --
        # ported unchanged from upstream Piper/EdgeTTS, which has this same
        # gap (piper_train/vits/modules.py's Log/ElementwiseAffine).
        return torch.exp(x.clamp(max=self._EXP_CLAMP_MAX)) * x_mask


class Flip(nn.Module):
    def forward(self, x: torch.Tensor, *args, reverse: bool = False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0)).type_as(x)
        return x


class ElementwiseAffine(nn.Module):
    # self.logs is an unconstrained learnable parameter -- nothing bounds how
    # far gradient descent can push it. Both branches below exponentiate it
    # (or its negation), so an unclamped logs that drifts too far either way
    # overflows exp() to inf under bf16/fp32, which then turns into NaN the
    # moment it multiplies a zero (e.g. via x_mask's padding). Observed in
    # practice: a baseline (non-Vocos) run's dp.flows/post_flows collapsed to
    # NaN across every model_g/model_d parameter within a few epochs after a
    # resume, root-caused to here -- not present in upstream Piper/EdgeTTS
    # either (piper_train/vits/modules.py has the identical unguarded code),
    # so this is a latent bug inherited from there, not a regression.
    # +-10 keeps exp(logs)/exp(-logs) within [~4.5e-5, ~2.2e4] -- wide enough
    # to never bind a legitimately-learned scale, per Log's clamp above.
    _LOGS_CLAMP = 10.0

    def __init__(self, channels: int):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False, **kwargs):
        logs = self.logs.clamp(min=-self._LOGS_CLAMP, max=self._LOGS_CLAMP)
        if not reverse:
            y = (self.m + torch.exp(logs) * x) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
            return y, logdet
        return (x - self.m) * torch.exp(-logs) * x_mask


class ConvFlow(nn.Module):
    """Coupling layer whose affine-transform-predictor output instead
    parameterizes a rational-quadratic spline (a strictly more expressive,
    still-invertible transform of x1)."""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
    ):
        super().__init__()
        self.half_channels = in_channels // 2
        self.filter_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None, reverse: bool = False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.convs(self.pre(x0), x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, c*?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        return x
