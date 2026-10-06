"""SynthesizerTrn: the top-level generator model (text -> audio), tying
together the text encoder, posterior encoder, normalizing flow, duration
predictor, and HiFi-GAN decoder described in the other vits/ files.

`voice_conversion()` (upstream: encode a source speaker's audio, re-decode
with a different speaker's embedding) is dropped here -- it's multi-speaker
-only functionality (asserts n_speakers > 1) that can never run for this
project's permanently-single-speaker scope, unlike the gin_channels/g=None
plumbing elsewhere which is exercised (as a no-op) on every single-speaker
forward pass too.
"""
import logging
import math
import typing

import torch
from torch import nn

from ...Vocos.generator import VocosGenerator
from ..utils import commons
from ..utils.monotonic_align import maximum_path
from .duration_predictor import StochasticDurationPredictor
from .f0_predictor import F0Predictor
from .flow_block import ResidualCouplingBlock
from .generator import Generator
from .posterior_encoder import PosteriorEncoder
from .text_encoder import TextEncoder

_LOGGER = logging.getLogger(__name__)


class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: typing.Tuple[int, ...],
        resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
        upsample_rates: typing.Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: typing.Tuple[int, ...],
        n_speakers: int = 1,
        gin_channels: int = 0,
        posterior_encoder_kernel_size: int = 5,
        posterior_encoder_dilation_rate: int = 1,
        posterior_encoder_layers: int = 16,
        flow_kernel_size: int = 5,
        flow_dilation_rate: int = 1,
        flow_n_flows: int = 4,
        # Per-stage expansion ratio for resblock="mb" (ResBlockInverted) --
        # see Generator's own docstring/comment for why this needs to vary
        # by stage: T grows 8x/8x/4x across the 3 upsample stages while
        # channels shrink, so the same expansion costs far more compute at
        # the last stage (T=8192) than the first (T=256). Silently unused
        # unless resblock="mb".
        mb_expansion: typing.Union[int, typing.Tuple[int, ...]] = 6,
        # Vocos Generator (ConvNeXt backbone + ISTFT head -- see
        # banhmi_train/Vocos/generator.py) instead of the HiFi-GAN-family
        # Generator above -- mutually exclusive with resblock/upsample_*
        # (all silently unused in this branch, kept as required
        # args above only because VitsModel always computes/passes them).
        # vocos_n_fft/vocos_hop_length must match the training pipeline's
        # own filter_length/hop_length (VitsModel's hparams) -- the ISTFT
        # head reconstructs at that exact frame rate.
        use_vocos: bool = False,
        vocos_n_fft: int = 1024,
        vocos_hop_length: int = 256,
        vocos_dim: int = 512,
        vocos_intermediate_dim: int = 1536,
        vocos_num_layers: int = 8,
        # F0Predictor + decoder F0 conditioning -- only wired into VocosGenerator
        # (use_vocos=True); the HiFi-GAN-family Generator above doesn't accept
        # an f0 kwarg at all, so this is silently a no-op without use_vocos.
        use_f0: bool = False,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.use_f0 = use_f0

        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        if use_vocos:
            self.dec = VocosGenerator(
                inter_channels,
                n_fft=vocos_n_fft,
                hop_length=vocos_hop_length,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                gin_channels=gin_channels,
                use_f0=use_f0,
            )
        else:
            self.dec = Generator(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
                mb_expansion=mb_expansion,
            )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            posterior_encoder_kernel_size,
            posterior_encoder_dilation_rate,
            posterior_encoder_layers,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            flow_kernel_size,
            flow_dilation_rate,
            n_layers=4,  # matches EdgeTTS's hardcoded WN depth per coupling layer
            n_flows=flow_n_flows,
            gin_channels=gin_channels,
        )

        self.dp = StochasticDurationPredictor(hidden_channels, 3, 0.5, 4, gin_channels=gin_channels)

        if use_f0:
            self.f0_predictor = F0Predictor(hidden_channels, 256, 3, 0.5)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def _speaker_embedding(self, sid):
        if self.n_speakers > 1:
            return self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        return None

    def forward(self, x, x_lengths, y, y_lengths, sid=None, mas_noise_scale: float = 0.0, f0=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._speaker_embedding(sid)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            attn = self._monotonic_alignment(x_mask, y_mask, z_p, m_p, logs_p, mas_noise_scale)

        w = attn.sum(2)
        logw_ = torch.log(w + 1e-6) * x_mask
        l_length = self.dp(x, x_mask, w, g=g)
        l_length = l_length / torch.sum(x_mask)
        # Sample a predicted duration (reverse mode) purely to feed the
        # duration discriminator -- same call infer() uses, does not
        # affect l_length (the SDP's own NLL loss).
        logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=1.0)

        attn_sq = attn.squeeze(1)
        # Expand the per-phoneme prior to per-frame using the alignment.
        m_p = torch.matmul(attn_sq, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn_sq, logs_p.transpose(1, 2)).transpose(1, 2)

        l_f0 = torch.zeros((), device=x.device)
        f0_frame = None
        if self.use_f0 and f0 is not None:
            # Ground-truth per-phoneme F0: average the real per-frame contour
            # (from preprocess/norm_audio.py::cache_f0) within each phoneme's
            # MAS-aligned frame range -- this is the F0Predictor's regression
            # target, same as EdgeTTS's own approach.
            f0_trunc = f0[:, : attn_sq.shape[1]].unsqueeze(1)  # [b, 1, t_t]
            phone_f0_sum = torch.matmul(f0_trunc, attn_sq)  # [b, 1, t_s]
            phone_f0 = phone_f0_sum / torch.clamp_min(w, 1.0)
            log_f0_target = torch.log(torch.clamp_min(phone_f0, 1.0)) * x_mask
            log_f0_pred = self.f0_predictor(x, x_mask)
            l_f0 = torch.sum((log_f0_pred - log_f0_target) ** 2 * x_mask) / torch.sum(x_mask)
            # Decoder conditioning: re-expand the SAME ground-truth per-phoneme
            # average back to frame-level through the identical hard alignment
            # infer() uses to expand the F0Predictor's own output -- NOT the
            # raw per-frame f0 (that's the train/inference mismatch bug in
            # upstream EdgeTTS: its decoder trains on a smooth natural F0
            # contour but only ever sees a piecewise-constant, phoneme-flat
            # "staircase" F0 at inference, since a hard alignment matrix can't
            # produce anything smoother). Using phone_f0 here instead makes
            # training and inference see the exact same signal shape.
            f0_frame = torch.matmul(attn_sq, phone_f0.transpose(1, 2)).transpose(1, 2)  # [b, 1, t_t]

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        f0_slice = None
        if f0_frame is not None:
            f0_slice = commons.slice_segments(f0_frame, ids_slice, self.segment_size)
        if isinstance(self.dec, VocosGenerator):
            o = self.dec(z_slice, g=g, f0=f0_slice)
        else:
            o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_, l_f0),
        )

    def _monotonic_alignment(self, x_mask, y_mask, z_p, m_p, logs_p, mas_noise_scale: float = 0.0):
        """Finds the most likely phoneme<->frame alignment given the
        current prior, via the negative cross-entropy between each frame's
        posterior sample and each phoneme's prior distribution.

        mas_noise_scale (VITS2's "noise-scaled MAS"): early in training the
        prior/posterior are poorly calibrated, so the argmax alignment can
        lock onto a bad path and never recover (monotonic search has no way
        to "undo" a bad early commitment). Injecting noise proportional to
        neg_cent's own spread lets a few different alignments win early on;
        the caller anneals this to 0 over training so it converges to the
        exact deterministic search vanilla VITS always used.
        """
        s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t_s]
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
        neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)  # [b, t_t, t_s]
        neg_cent3 = torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)  # [b, t_t, t_s]
        neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        if mas_noise_scale > 0:
            neg_cent = neg_cent + torch.randn_like(neg_cent) * torch.std(neg_cent) * mas_noise_scale

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        return maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        onnx_export: bool = False,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._speaker_embedding(sid)

        logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        # Same guard/bound as flows.py's Log._EXP_CLAMP_MAX: clamp before
        # exp() rather than only handling the fallout after. This exp() was
        # previously unguarded -- observed root-causing a full model_g/
        # model_d NaN collapse (StochasticDurationPredictor's reverse-mode
        # sampling, called every validation epoch via _log_audio_samples):
        # an unclamped logw let exp(logw) reach a very large but still-finite
        # value that survived this line, then blew up further downstream
        # (mask multiply, w_ceil sum, sequence_mask) into inf/NaN. Clamping
        # logw itself keeps w_ceil bounded to a generous-but-sane frame
        # count (exp(10) ~= 22000 frames) instead of merely reacting to
        # non-finite values after the fact (see the nan_to_num/degenerate
        # handling below, which stays as a second line of defense).
        w_ceil = torch.ceil(torch.exp(logw.clamp(max=10.0)) * x_mask * length_scale)
        # An undertrained/unstable duration predictor can occasionally emit a
        # non-finite logw (exp() overflow -> inf, or NaN) -> casting inf/NaN
        # to long is undefined (observed: wraps to a large negative number),
        # which then crashes commons.sequence_mask's torch.arange on a
        # negative length. nan_to_num first (clamp alone lets NaN through
        # unchanged, since comparisons against NaN are always false), then
        # clamp to a generous-but-finite frame count to keep infer() crash-safe.
        y_lengths = torch.sum(w_ceil, [1, 2])
        # Surface duration-predictor collapse the moment it happens (rather
        # than silently falling back to a near-empty 1-frame output with no
        # trace) -- non-finite or near-zero durations here mean self.dp's
        # reverse-mode sampling has degenerated, independently of how good
        # loss_mel/val_loss_mel look (those never exercise this code path).
        degenerate = ~torch.isfinite(y_lengths) | (y_lengths < 4)
        if degenerate.any():
            _LOGGER.warning(
                "StochasticDurationPredictor produced degenerate duration(s) in infer(): "
                "raw summed w_ceil=%s (finite min=%s over the non-degenerate rest) -- "
                "falling back to a 1-frame minimum for the affected batch item(s). "
                "This indicates dp's reverse-sampling has collapsed, not a Generator/mel issue.",
                y_lengths.detach().cpu().tolist(),
                y_lengths[~degenerate].min().item() if (~degenerate).any() else "n/a",
            )
        y_lengths = torch.nan_to_num(y_lengths, nan=1.0, posinf=100000.0, neginf=1.0)
        y_lengths = torch.clamp(y_lengths, min=1, max=100000).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        attn_sq = attn.squeeze(1)
        # [b, t', t] x [b, t, d] -> [b, d, t']: expand per-phoneme prior to per-frame
        m_p = torch.matmul(attn_sq, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn_sq, logs_p.transpose(1, 2)).transpose(1, 2)

        f0_frame = None
        if self.use_f0:
            log_f0_pred = self.f0_predictor(x, x_mask)  # [b, 1, t_s]
            # Same hard-alignment expansion as forward()'s f0_frame, and the
            # same clamp-before-exp guard as logw above (this exp() has the
            # identical overflow failure mode).
            f0_frame = torch.matmul(attn_sq, log_f0_pred.transpose(1, 2)).transpose(1, 2)  # [b, 1, t_t]
            f0_frame = torch.exp(f0_frame.clamp(max=10.0))

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        if isinstance(self.dec, VocosGenerator):
            # onnx_export routes ISTFTHead around torch.complex/torch.fft
            # (unsupported by torch.onnx's exporter) -- a no-op kwarg for
            # every other Generator family, which never uses either op.
            f0_out = f0_frame[:, :, :max_len] if f0_frame is not None else None
            o = self.dec((z * y_mask)[:, :, :max_len], g=g, f0=f0_out, onnx_export=onnx_export)
        else:
            o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)
