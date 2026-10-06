"""Internal baseline (HiFi-GAN ResBlock2 + VITS2 components) training module.

Adapted from vendor/banhmi/vits/training.py (BanhmiTTS 8fd475f, current
source -- not proven identical to the historical training-time source).
Generator/discriminator losses, the duration-discriminator updates,
noise-scaled MAS schedule, optimizers, schedulers, the non-finite-update
skip and the free-running health gate are kept as in that file. Adaptations
(see docs/training/piper-component-comparison.md, "Adapter diff"):

- model components come from ``hifimobinet.architecture.vits`` (the
  relocated, output-verified copy) instead of ``banhmi_train.vits``;
- the Vocos/F0 branches are removed and rejected explicitly (the selected
  recipes never enable them; their modules are outside this repository);
- datasets are injected (explicit split file, path resolution) instead of
  being loaded inside ``__init__`` with a canonical-split/random fallback;
- the spectrogram-length cache is written to the run directory instead of
  next to the source dataset;
- RNG state can be saved/restored on resume (harness option).
"""
from __future__ import annotations

import itertools
import logging
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..architecture.vits.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from ..architecture.vits.modules.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from ..architecture.vits.modules.duration_discriminator import DurationDiscriminator
from ..architecture.vits.modules.generator import INVERTED_RESBLOCKS
from ..architecture.vits.modules.synthesizer import SynthesizerTrn
from ..architecture.vits.utils.commons import slice_segments
from .harness import HarnessMixin, free_running_healthy
from .vendor import vendor_module

_LOGGER = logging.getLogger(__name__)
_N_SPEAKERS = 1
_GIN_CHANNELS = 0


class BaselineVits2Module(HarnessMixin, pl.LightningModule):
    family = "banhmi"

    def __init__(
        self,
        num_symbols: int,
        resblock: str,
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        filter_length: int,
        hop_length: int,
        win_length: int,
        mel_channels: int,
        sample_rate: int,
        mel_fmin: float,
        mel_fmax: Optional[float],
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        use_spectral_norm: bool,
        segment_size: int,
        posterior_encoder_kernel_size: int,
        posterior_encoder_dilation_rate: int,
        posterior_encoder_layers: int,
        flow_kernel_size: int,
        flow_dilation_rate: int,
        flow_n_flows: int,
        mas_noise_scale_initial: float,
        mas_noise_scale_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
        lr_decay: float,
        c_mel: float,
        c_kl: float,
        batch_size: int,
        sampler: str,
        num_workers: int,
        max_phoneme_ids: Optional[int],
        num_audio_samples: int,
        skip_nonfinite_updates: bool,
        dp_health_gate: bool,
        save_rng_state: bool,
        seed: int,
        mb_expansion=6,
    ):
        super().__init__()
        self.save_hyperparameters()
        if resblock in INVERTED_RESBLOCKS:
            raise ValueError("This training model is the ResBlock2 baseline; inverted-residual decoders are not selected here")

        self.model_g = SynthesizerTrn(
            n_vocab=num_symbols,
            spec_channels=filter_length // 2 + 1,
            segment_size=segment_size // hop_length,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            mb_expansion=mb_expansion,
            n_speakers=_N_SPEAKERS,
            gin_channels=_GIN_CHANNELS,
            posterior_encoder_kernel_size=posterior_encoder_kernel_size,
            posterior_encoder_dilation_rate=posterior_encoder_dilation_rate,
            posterior_encoder_layers=posterior_encoder_layers,
            flow_kernel_size=flow_kernel_size,
            flow_dilation_rate=flow_dilation_rate,
            flow_n_flows=flow_n_flows,
            use_vocos=False,
            use_f0=False,
        )
        self.model_d = MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm)
        # The original ties MRD to inverted-residual decoders; never built for ResBlock2.
        self.model_d_mrd = None
        self.model_d_dur = DurationDiscriminator(
            in_channels=hidden_channels,
            filter_channels=hidden_channels,
            kernel_size=3,
            p_dropout=p_dropout,
            gin_channels=_GIN_CHANNELS,
        )

        self._data = None
        self._train_batch_sampler = None
        self._y = None
        self._y_hat = None
        self._dur_x = self._dur_mask = self._dur_real = self._dur_fake = None
        self._last_loss_mel = None
        self._last_batch_fingerprint = None
        mel = vendor_module("banhmi", "mel_processing")
        self._mel_spectrogram_torch = mel.mel_spectrogram_torch
        self._spec_to_mel_torch = mel.spec_to_mel_torch

    # ------------------------------------------------------------------ data
    def attach_data(self, data, length_fn) -> None:
        """``data``: training.data.TrainingData; ``length_fn(subset)`` -> lengths."""
        if data.family != self.family:
            raise ValueError(f"{type(self).__name__} needs {self.family!r} data, got {data.family!r}")
        self._data = data
        self._length_fn = length_fn
        n = min(self.hparams.num_audio_samples, len(data.val))
        self._audio_sample_dataset = torch.utils.data.Subset(data.val, range(n))

    def forward(self, text, text_lengths, scales, sid=None):
        audio, *_ = self.model_g.infer(
            text, text_lengths, noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2]
        )
        return audio

    def train_dataloader(self):
        collate = self._data.collate(self.hparams.segment_size)
        if self.hparams.sampler == "upstream_sequential":
            return DataLoader(self._data.train, collate_fn=collate, batch_size=self.hparams.batch_size,
                              num_workers=self.hparams.num_workers)
        world_size, rank = 1, 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
        sampler_cls = vendor_module("banhmi", "vits.length_bucket_sampler").LengthBucketBatchSampler
        batch_sampler = sampler_cls(
            self._length_fn(self._data.train),
            batch_size=self.hparams.batch_size,
            boundaries=(150, 250, 350, 450, 600, 800, 1000),
            seed=self.hparams.seed,
            num_replicas=world_size,
            rank=rank,
        )
        batch_sampler.set_epoch(self.current_epoch)
        self._train_batch_sampler = batch_sampler
        return DataLoader(self._data.train, collate_fn=collate, num_workers=self.hparams.num_workers,
                          batch_sampler=batch_sampler, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)

    def on_train_epoch_start(self) -> None:
        if self._train_batch_sampler is not None:
            self._train_batch_sampler.set_epoch(self.current_epoch)

    def val_dataloader(self):
        return DataLoader(self._data.val, collate_fn=self._data.collate(self.hparams.segment_size),
                          num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
                          pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    # -------------------------------------------------------------- training
    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            self._last_batch_fingerprint = (int(self.current_epoch), int(batch_idx), batch.phoneme_lengths.tolist())
            return self.training_step_g(batch)
        return self.training_step_d(batch)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, **kwargs):
        if self.hparams.skip_nonfinite_updates:
            self.nonfinite_skipping_step(optimizer, optimizer_idx, optimizer_closure, self._last_batch_fingerprint)
        else:
            optimizer.step(closure=optimizer_closure)

    def _current_mas_noise_scale(self) -> float:
        scale = self.hparams.mas_noise_scale_initial - self.global_step * self.hparams.mas_noise_scale_decay
        return max(scale, 0.0)

    def training_step_g(self, batch):
        h = self.hparams
        x, x_lengths = batch.phoneme_ids, batch.phoneme_lengths
        y, spec, spec_lengths = batch.audios, batch.spectrograms, batch.spectrogram_lengths

        mas_noise_scale = self._current_mas_noise_scale()
        (
            y_hat, l_length, _attn, ids_slice, x_mask, z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
            (x_hidden, logw, logw_, l_f0),
        ) = self.model_g(x, x_lengths, spec, spec_lengths, mas_noise_scale=mas_noise_scale, f0=None)
        self._y_hat = y_hat
        self.log("mas_noise_scale", mas_noise_scale)
        self._dur_x, self._dur_mask, self._dur_real, self._dur_fake = x_hidden, x_mask, logw_, logw

        mel = self._spec_to_mel_torch(spec, h.filter_length, h.mel_channels, h.sample_rate, h.mel_fmin, h.mel_fmax)
        y_mel = slice_segments(mel, ids_slice, h.segment_size // h.hop_length)
        y_hat_mel = self._mel_spectrogram_torch(
            y_hat.squeeze(1), h.filter_length, h.mel_channels, h.sample_rate, h.hop_length, h.win_length,
            h.mel_fmin, h.mel_fmax,
        )
        y = slice_segments(y, ids_slice * h.hop_length, h.segment_size)
        self._y = y

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)

        with autocast(self.device.type, enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * h.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * h.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            # l_f0 is the synthesizer's constant zero when F0 is disabled.
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + l_f0

            self._last_loss_mel = loss_mel.detach()
            self.log("loss_mel", loss_mel)
            self.log("loss_kl", loss_kl)
            self.log("loss_dur", loss_dur)
            self.log("loss_gen", loss_gen)
            self.log("loss_fm", loss_fm)

            _dur_probs_r, dur_probs_hat = self.model_d_dur(x_hidden, x_mask, logw_, logw)
            loss_dur_gen, _ = generator_loss(dur_probs_hat)
            loss_gen_all = loss_gen_all + loss_dur_gen
            self.log("loss_dur_gen", loss_dur_gen)
            self.log("loss_gen_all", loss_gen_all)
            return loss_gen_all

    def training_step_d(self, batch):
        y, y_hat = self._y, self._y_hat
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())
        with autocast(self.device.type, enabled=False):
            loss_disc, *_ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
            self.log("loss_disc", loss_disc)
            dur_probs_r, dur_probs_hat = self.model_d_dur(
                self._dur_x.detach(), self._dur_mask, self._dur_real.detach(), self._dur_fake.detach()
            )
            loss_disc_dur, *_ = discriminator_loss(dur_probs_r, dur_probs_hat)
            loss_disc_all = loss_disc_all + loss_disc_dur
            self.log("loss_disc_dur", loss_disc_dur)
            self.log("loss_disc_all", loss_disc_all)
            return loss_disc_all

    def validation_step(self, batch, batch_idx: int):
        val_loss = self.training_step_g(batch) + self.training_step_d(batch)
        self.log("val_loss", val_loss, sync_dist=True)
        healthy = free_running_healthy(self, self._audio_sample_dataset, self.hparams.sample_rate)
        val_loss_mel = self._last_loss_mel
        if self.hparams.dp_health_gate and not healthy:
            _LOGGER.warning("Reporting val_loss_mel=inf: free-running inference is degenerate")
            val_loss_mel = torch.tensor(float("inf"), device=self.device)
        self.log("val_loss_mel", val_loss_mel, sync_dist=True)
        return val_loss

    @staticmethod
    def _make_adamw(params, **kwargs) -> torch.optim.AdamW:
        try:
            return torch.optim.AdamW(params, fused=torch.cuda.is_available(), **kwargs)
        except TypeError:
            return torch.optim.AdamW(params, **kwargs)

    def configure_optimizers(self):
        h = self.hparams
        discriminators = [self.model_d, self.model_d_mrd, self.model_d_dur]
        discriminator_params = itertools.chain(*(d.parameters() for d in discriminators if d is not None))
        optimizers = [
            self._make_adamw(self.model_g.parameters(), lr=h.learning_rate, betas=h.betas, eps=h.eps),
            self._make_adamw(discriminator_params, lr=h.learning_rate, betas=h.betas, eps=h.eps),
        ]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=h.lr_decay),
            torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=h.lr_decay),
        ]
        return optimizers, schedulers
