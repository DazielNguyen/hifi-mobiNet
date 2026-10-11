"""Piper-ParIR / Piper-SeqIR: the Piper_no_VITS2_cpn backbone (EdgeTTS Config A)
with the decoder bundle of the released Parallel-IR / Sequential-IR models.

Everything except the waveform decoder and the extra discriminator is the
unchanged Config A module (``PiperNoVits2Module`` -> EdgeTTS ``VitsModel``):
text encoder, posterior encoder, residual-coupling flow, stochastic duration
predictor, plain MAS, multi-period discriminator, losses, manual optimization,
optimizers and schedulers, data, validation and the checkpoint rule.
``use_vits2``, ``use_bigvgan`` and ``use_f0`` stay false.

The decoder bundle is the one the released IR checkpoints were trained with
(``src/hifimobinet/architecture``, relocated BanhmiTTS code; checkpoint
hyper-parameters: ``resblock`` "mrf"/"mb", ``mb_expansion`` (1, 1, 1),
``use_snake`` and ``use_mrd`` true):

- ``model_g.dec`` is that ``Generator`` with ``resblock`` "mrf" (three parallel
  ``ResBlockInverted`` branches per stage, kernels 3/5/7, averaged) or "mb"
  (two sequential ``ResBlockInverted`` blocks per stage, kernel 3). The
  generator uses SnakeBeta for every activation in these modes
  (``INVERTED_RESBLOCKS``); upsampling sizes are Config A's.
- ``model_d_mrd`` is that ``MultiResolutionDiscriminator`` (three STFT
  resolutions). EdgeTTS's own ``training_step_g``/``training_step_d`` and
  ``configure_optimizers`` already handle a non-None ``model_d_mrd``: the
  generator loss adds ``loss_gen_mrd`` and the MRD feature-matching loss
  (weight 1), the discriminator loss adds ``loss_disc_mrd``, and the MRD
  parameters join the discriminator optimizer. These are the terms and weights
  of the IR runs (BanhmiTTS ``training.py``: ``mrd_coeff`` 1.0 without Vocos);
  the LSGAN loss functions of both code bases are the same formulas.

EdgeTTS's ``use_bigvgan`` is deliberately not used: it would build EdgeTTS's own
Snake ResBlock2 decoder and EdgeTTS's MRD, not the released IR implementation.

Construction order: EdgeTTS first builds its Config A synthesizer (including a
ResBlock2 decoder, because its ``Generator`` treats any ``resblock`` other than
"1" as "2") and the MPD; the IR decoder then replaces that decoder and the MRD
is added. Under the same seed every other initial weight therefore equals the
Config A run's initial weights.
"""
from __future__ import annotations

from ..architecture.vits.modules.discriminators import MultiResolutionDiscriminator
from ..architecture.vits.modules.generator import INVERTED_RESBLOCKS, Generator
from .piper_module import PiperNoVits2Module

IR_KEYS = {"mrf": ("mb_expansion", "mb_mrf_kernel_sizes"),
           "mb": ("mb_expansion", "mb_kernel_size", "mb_blocks_per_stage")}


class IRDecoder(Generator):
    """The IR ``Generator`` behind EdgeTTS's decoder call ``dec(z, g=g, f0=f0)``.

    Same parameters and forward as ``Generator`` (state-dict names stay
    ``dec.*``); ``f0`` is accepted only as None because the IR decoder has no
    F0 conditioning and Config A never produces F0.
    """

    def forward(self, x, g=None, f0=None):
        if f0 is not None:
            raise ValueError("the IR decoder has no F0 conditioning; use_f0 must stay false")
        return super().forward(x, g=g)


class PiperIRNoVits2Module(PiperNoVits2Module):
    family = "edgetts"

    def __init__(self, num_symbols: int, **kwargs):
        resblock = kwargs.get("resblock")
        if resblock not in INVERTED_RESBLOCKS:
            raise ValueError(f"IR decoder needs resblock in {INVERTED_RESBLOCKS}, got {resblock!r}")
        if kwargs.get("use_mrd") is not True:
            raise ValueError("the IR decoder bundle includes the MRD discriminator; use_mrd must be true")
        missing = [k for k in IR_KEYS[resblock] if k not in kwargs]
        if missing:
            raise ValueError(f"missing IR decoder settings for resblock {resblock!r}: {missing}")
        # All keys go to EdgeTTS's **kwargs as well, so they are saved in hparams.
        super().__init__(num_symbols=num_symbols, **kwargs)
        hp = self.hparams
        extra = {"mb_expansion": tuple(hp.mb_expansion)}
        if resblock == "mb":
            extra.update(mb_kernel_size=int(hp.mb_kernel_size), mb_blocks_per_stage=int(hp.mb_blocks_per_stage))
        else:
            extra.update(mb_mrf_kernel_sizes=tuple(hp.mb_mrf_kernel_sizes))
        self.model_g.dec = IRDecoder(
            hp.inter_channels, resblock, tuple(hp.resblock_kernel_sizes),
            tuple(tuple(d) for d in hp.resblock_dilation_sizes), tuple(hp.upsample_rates),
            hp.upsample_initial_channel, tuple(hp.upsample_kernel_sizes), gin_channels=hp.gin_channels, **extra,
        )
        self.model_d_mrd = MultiResolutionDiscriminator(use_spectral_norm=hp.use_spectral_norm)
