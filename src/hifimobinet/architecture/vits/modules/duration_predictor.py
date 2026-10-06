"""Predicts how many output frames each phoneme should last, via a small
normalizing flow over duration itself (StochasticDurationPredictor),
trained by maximum likelihood (matches how the rest of VITS is trained) and
sampled from at inference time -- this is VITS's actual default and what
every current Piper voice uses. Always used here; the deterministic
regression-based DurationPredictor from VITS's predecessor (Glow-TTS) is
not implemented since nothing in this codebase ever selects it.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.dds_conv import DDSConv
from ..utils.flows import ConvFlow, ElementwiseAffine, Flip, Log


class StochasticDurationPredictor(nn.Module):
    """Two coupled flow stacks: `post_flows` learns q(u|w,x) (a variational
    posterior over an auxiliary variable u used to make discrete-ish
    duration counts fit a continuous flow), `flows` is the actual
    duration-density flow conditioned on the text encoding x. Trained by
    maximizing the resulting ELBO (forward() returns its negative)."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        filter_channels = in_channels  # matches upstream; kept as an alias for readability below

        self.log_flow = Log()
        self.flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w=None,
        g=None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ):
        # Normalizing flows are precision-sensitive in a way the rest of this
        # model is not: the spline transform's bin widths come from a
        # cumsum-then-difference, and its log-determinant terms divide by
        # quantities that are only analytically bounded away from zero. Under
        # bf16's ~7-bit mantissa those can round to exactly 0 / cancel, which
        # historically produced a NaN that permanently poisoned dp's own
        # parameters (dec/enc_p stayed healthy, so val_loss_mel kept improving
        # and hid it). dp is tiny next to dec, so forcing fp32 here costs very
        # little and removes that whole class of failure at its source --
        # applied inside forward() so every call site (NLL, the duration
        # discriminator's reverse sample, and infer()) gets it automatically.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(x, x_mask, w, g, reverse, noise_scale)

    def _forward_fp32(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w,
        g,
        reverse: bool,
        noise_scale: float,
    ):
        x = x.float()
        x_mask = x_mask.float()
        if w is not None:
            w = w.float()
        if g is not None:
            g = g.float()

        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(torch.detach(g))
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            return self._forward_nll(x, x_mask, w)
        return self._sample(x, x_mask, noise_scale)

    def _forward_nll(self, x: torch.Tensor, x_mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        assert w is not None

        logdet_tot_q = 0.0
        h_w = self.post_convs(self.post_pre(w), x_mask)
        h_w = self.post_proj(h_w) * x_mask
        e_q = torch.randn(w.size(0), 2, w.size(2)).type_as(x) * x_mask
        z_q = e_q
        for flow in self.post_flows:
            z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
            logdet_tot_q += logdet_q
        z_u, z1 = torch.split(z_q, [1, 1], 1)
        u = torch.sigmoid(z_u) * x_mask
        z0 = (w - u) * x_mask
        logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
        logq = (
            torch.sum(-0.5 * (math.log(2 * math.pi) + e_q**2) * x_mask, [1, 2]) - logdet_tot_q
        )

        logdet_tot = 0.0
        z0, logdet = self.log_flow(z0, x_mask)
        logdet_tot += logdet
        z = torch.cat([z0, z1], 1)
        for flow in self.flows:
            z, logdet = flow(z, x_mask, g=x, reverse=False)
            logdet_tot += logdet
        nll = torch.sum(0.5 * (math.log(2 * math.pi) + z**2) * x_mask, [1, 2]) - logdet_tot
        return nll + logq  # [b]

    def _sample(self, x: torch.Tensor, x_mask: torch.Tensor, noise_scale: float) -> torch.Tensor:
        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]  # skip the redundant last Flip
        z = torch.randn(x.size(0), 2, x.size(2)).type_as(x) * noise_scale
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)
        z0, _z1 = torch.split(z, [1, 1], 1)
        return z0  # log-duration
