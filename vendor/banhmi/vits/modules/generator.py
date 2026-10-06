"""HiFi-GAN-style vocoder decoder: upsamples the flow's latent sequence
(one vector per output frame) directly to a raw waveform via transposed
convolutions + residual blocks. The activation follows the resblock type:
LeakyReLU for the HiFi-GAN resblocks ("1"/"2"), SnakeBeta for the
inverted-residual ones ("mb"/"mrf").
"""
import typing

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from ..utils.commons import init_weights
from ..utils.normalization import LeakyReLUActivation, SnakeBeta
from ..utils.resblocks import ResBlock1, ResBlock2, ResBlockInverted

# The inverted-residual resblocks were designed and trained as one bundle
# with SnakeBeta activations and the MRD discriminator (see VitsModel); the
# HiFi-GAN resblocks use LeakyReLU and no MRD. Every released checkpoint
# follows this pairing, so it is fixed here rather than exposed as flags.
INVERTED_RESBLOCKS = ("mb", "mrf")


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: typing.Tuple[int, ...],
        resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
        upsample_rates: typing.Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: typing.Tuple[int, ...],
        gin_channels: int = 0,
        mb_expansion: typing.Union[int, typing.Tuple[int, ...]] = 6,
        mb_kernel_size: int = 3,
        mb_blocks_per_stage: int = 2,
        mb_mrf_kernel_sizes: typing.Tuple[int, ...] = (3, 5, 7),
    ):
        super().__init__()
        # "mb" -- new novelty direction (MobileNetV2-style inverted-residual
        # blocks, see ResBlockInverted's docstring), validated by overfit-test
        # sweep in test_function/compare_generators.py before wiring here:
        # expansion=6/kernel=3/blocks_per_stage=2 (true MobileNetV2 defaults)
        # beat both resblock "1" and "2" at a matched parameter budget, and
        # zero-initializing ResBlockInverted's last pointwise conv (so each
        # block starts as an identity no-op) was what actually fixed training
        # instability -- not a smaller learning rate, which only masked it.
        # This path is fully separate from "1"/"2" below (own attribute,
        # own forward()/remove_weight_norm() branch) so existing "1"/"2"
        # checkpoints' state_dict keys (`resblocks.*`) are untouched.
        self.resblock_mode = resblock
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        activation_cls = SnakeBeta if resblock in INVERTED_RESBLOCKS else LeakyReLUActivation

        self.ups = nn.ModuleList()
        self.pre_up_snakes = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.pre_up_snakes.append(activation_cls(upsample_initial_channel // (2**i)))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        if resblock == "mb":
            # mb_expansion can be a single int (same expansion at every
            # stage, the original config) or a per-stage tuple -- added so
            # the last stage (T=8192, the raw-sample-rate stage) can use a
            # smaller expansion than the earlier stages (T=256/2048): a
            # profiling pass (see docs -- RTF was ~3.7x baseline at uniform
            # expansion=6) found the expand/project pointwise convs' cost
            # scales with expansion AND with T, so the same expansion is far
            # more expensive at the last stage than the first. Earlier
            # stages' T is small enough that keeping expansion=6 there stays
            # cheap even though it's the same ratio.
            if isinstance(mb_expansion, int):
                mb_expansion = (mb_expansion,) * len(self.ups)
            assert len(mb_expansion) == len(self.ups), (
                f"mb_expansion must be an int or a tuple with one entry per upsample stage "
                f"({len(self.ups)}), got {mb_expansion}"
            )
            self.resblocks = None
            self.mb_blocks = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                self.mb_blocks.append(
                    nn.ModuleList(
                        ResBlockInverted(ch, kernel_size=mb_kernel_size, activation_cls=activation_cls, expansion=mb_expansion[i])
                        for _ in range(mb_blocks_per_stage)
                    )
                )
        elif resblock == "mrf":
            # HiFi-GAN's real MRF mechanism (N branches run in PARALLEL on
            # the same input, summed then divided by N -- exactly what the
            # "else" branch below does for resblock "1"/"2") but each
            # branch is a ResBlockInverted instead of ResBlock1/2, one
            # branch per entry in mb_mrf_kernel_sizes. Validated by
            # overfit-test sweep (test_function/compare_generators.py's
            # `_HiFiGANInvertedMRF`) against "mb" sequential-stacking above,
            # across 4 per-stage expansion schedules (docs/
            # mbconv_resblock_report.md SS6): at expansion=(1,1,1) this is
            # the only MRF schedule where sequential doesn't clearly win on
            # both loss_mel and speed -- MRF(1,1,1) trades a small amount
            # of speed (still faster than plain resblock "2" in isolated
            # generator benchmarks, though those were noisy -- see report)
            # against sequential(1,1,1) for a chance at the multi-receptive-
            # field mechanism HiFi-GAN's own design actually uses.
            if isinstance(mb_expansion, int):
                mb_expansion = (mb_expansion,) * len(self.ups)
            assert len(mb_expansion) == len(self.ups), (
                f"mb_expansion must be an int or a tuple with one entry per upsample stage "
                f"({len(self.ups)}), got {mb_expansion}"
            )
            self.resblocks = None
            self.mb_blocks = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                self.mb_blocks.append(
                    nn.ModuleList(
                        ResBlockInverted(ch, kernel_size=ks, activation_cls=activation_cls, expansion=mb_expansion[i])
                        for ks in mb_mrf_kernel_sizes
                    )
                )
        else:
            resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
            self.mb_blocks = None
            self.resblocks = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                    self.resblocks.append(resblock_cls(ch, k, d, activation_cls=activation_cls))

        self.final_snake = activation_cls(ch)
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g=None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = up(self.pre_up_snakes[i](x))
            if self.resblock_mode == "mb":
                for block in self.mb_blocks[i]:
                    x = block(x)
            elif self.resblock_mode == "mrf":
                branches = self.mb_blocks[i]
                x = sum(branch(x) for branch in branches) / len(branches)
            else:
                xs = sum(
                    self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels)
                )
                x = xs / self.num_kernels

        x = self.conv_post(self.final_snake(x))
        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        if self.resblock_mode in ("mb", "mrf"):
            for stage in self.mb_blocks:
                for block in stage:
                    block.remove_weight_norm()
        else:
            for layer in self.resblocks:
                layer.remove_weight_norm()
