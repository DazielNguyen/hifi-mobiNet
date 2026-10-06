"""The normalizing flow that maps between the posterior's latent z and the
text-conditioned prior's z_p, built from alternating coupling layers and
channel flips (the flip is what lets *both* halves of the channels get
transformed across the whole stack, since each individual coupling layer
only ever transforms one half).
"""
import torch
from torch import nn

from ..utils.flows import Flip
from ..utils.wavenet import WN


class TransformerCouplingLayer(nn.Module):
    """VITS2-style residual coupling layer: a WaveNet-style conv stack
    predicts an affine transform of one half of the channels from the
    other half, with a self-attention pass added for global context.

    Invertibility is unaffected: x1 is still an invertible affine function
    of x0 alone; making the *function that computes the affine params*
    more expressive doesn't change that x0 passes through unchanged.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_heads: int = 2,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        # Local import: attention.py doesn't need to know about flow_block.py,
        # only this file needs Encoder, so keep the dependency one-directional.
        from ..utils.attention import Encoder

        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.attn = Encoder(
            hidden_channels, hidden_channels * 2, n_heads=n_heads, n_layers=1, kernel_size=1, p_dropout=p_dropout
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None, reverse: bool = False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        # Attention before WaveNet -- matches EdgeTTS's own TransformerCouplingLayer
        # and the VITS2 reference implementation's ResidualCouplingTransformersLayer2
        # (github.com/p0p4k/vits2_pytorch), verified directly against both.
        h = h + self.attn(h, x_mask)
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if self.mean_only:
            m, logs = stats, torch.zeros_like(stats)
        else:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            logdet = torch.sum(logs, [1, 2])
            return torch.cat([x0, x1], 1), logdet

        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        n_heads: int = 2,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    n_heads=n_heads,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None, reverse: bool = False):
        flows = self.flows if not reverse else reversed(self.flows)
        for flow in flows:
            if not reverse:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
            else:
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for flow in self.flows:
            if hasattr(flow, "remove_weight_norm"):
                flow.remove_weight_norm()
