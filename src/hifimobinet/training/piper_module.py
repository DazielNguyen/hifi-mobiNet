"""Piper_no_VITS2_cpn: upstream Piper VITS training graph, trained from scratch
inside this project's data/checkpoint harness.

Subclasses the unmodified upstream ``VitsModel`` (vendor/piper/vits/lightning.py,
rhasspy/piper 73c04d81d5590ecc46e522de3601ce7fb29fc2be). Inherited unchanged:
``SynthesizerTrn`` (text encoder, posterior encoder, WaveNet coupling flow,
stochastic duration predictor, plain MAS, HiFi-GAN generator),
``MultiPeriodDiscriminator``, losses, ``training_step``, ``training_step_d``,
``forward`` and ``configure_optimizers``.

Overridden for the project harness (documented in
docs/training/piper-component-comparison.md):

- dataset loading/dataloaders: explicit split file and project dataset instead
  of ``random_split`` with ``validation_split``/``num_test_examples``;
- ``training_step_g``: identical arithmetic and return value; additionally
  logs the individual loss terms and keeps ``loss_mel`` for validation;
- ``validation_step``: the upstream ``val_loss`` plus ``val_loss_mel`` for
  checkpoint selection; audio examples come from validation utterances
  (upstream used its 5 held-out "test" utterances);
- optional harness policies (non-finite update skip, free-running health
  gate, RNG state), each explicit in the config.
"""
from __future__ import annotations

import logging

import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .harness import HarnessMixin, free_running_healthy
from .vendor import vendor_module

_LOGGER = logging.getLogger(__name__)

_lightning = vendor_module("piper_vits", "lightning")
_commons = vendor_module("piper_vits", "commons")
_losses = vendor_module("piper_vits", "losses")
_mel = vendor_module("piper_vits", "mel_processing")

UpstreamVitsModel = _lightning.VitsModel


class PiperNoVits2Module(HarnessMixin, UpstreamVitsModel):
    family = "piper"

    def __init__(self, num_symbols: int, sampler: str, num_audio_samples: int,
                 skip_nonfinite_updates: bool, dp_health_gate: bool, save_rng_state: bool, **upstream_kwargs):
        # dataset=None: upstream _load_datasets() returns early; data is attached explicitly.
        # Extra harness keys travel through upstream's **kwargs into hparams.
        super().__init__(
            num_symbols=num_symbols, num_speakers=1, dataset=None, sampler=sampler,
            num_audio_samples=num_audio_samples, skip_nonfinite_updates=skip_nonfinite_updates,
            dp_health_gate=dp_health_gate, save_rng_state=save_rng_state, **upstream_kwargs,
        )
        self._data = None
        self._train_batch_sampler = None
        self._last_loss_mel = None
        self._last_batch_fingerprint = None

    # ------------------------------------------------------------------ data
    def attach_data(self, data, length_fn) -> None:
        if data.family != self.family:
            raise ValueError(f"{type(self).__name__} needs {self.family!r} data, got {data.family!r}")
        self._data = data
        self._length_fn = length_fn
        self._train_dataset, self._val_dataset = data.train, data.val
        self._test_dataset = None  # never exposed to training/validation
        n = min(self.hparams.num_audio_samples, len(data.val))
        self._audio_sample_dataset = Subset(data.val, range(n))

    def train_dataloader(self):
        collate = self._data.collate(self.hparams.segment_size)
        if self.hparams.sampler == "upstream_sequential":
            # Upstream Piper's own DataLoader: fixed order, no shuffling.
            return DataLoader(self._data.train, collate_fn=collate, num_workers=self.hparams.num_workers,
                              batch_size=self.hparams.batch_size)
        world_size, rank = 1, 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
        sampler_cls = vendor_module("banhmi", "vits.length_bucket_sampler").LengthBucketBatchSampler
        batch_sampler = sampler_cls(
            self._length_fn(self._data.train), batch_size=self.hparams.batch_size,
            boundaries=(150, 250, 350, 450, 600, 800, 1000), seed=self.hparams.seed,
            num_replicas=world_size, rank=rank,
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
                          num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        raise RuntimeError("The held-out test split is not used by this training harness")

    # -------------------------------------------------------------- training
    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            self._last_batch_fingerprint = (int(self.current_epoch), int(batch_idx), batch.phoneme_lengths.tolist())
        return super().training_step(batch, batch_idx, optimizer_idx)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, **kwargs):
        if self.hparams.skip_nonfinite_updates:
            self.nonfinite_skipping_step(optimizer, optimizer_idx, optimizer_closure, self._last_batch_fingerprint)
        else:
            optimizer.step(closure=optimizer_closure)

    def training_step_g(self, batch):
        # Upstream body (vendor/piper/vits/lightning.py VitsModel.training_step_g)
        # with added self.log calls and the loss_mel stash; arithmetic unchanged.
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
        )
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            _x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
        ) = self.model_g(x, x_lengths, spec, spec_lengths, speaker_ids)
        self._y_hat = y_hat

        mel = _mel.spec_to_mel_torch(
            spec, self.hparams.filter_length, self.hparams.mel_channels, self.hparams.sample_rate,
            self.hparams.mel_fmin, self.hparams.mel_fmax,
        )
        y_mel = _commons.slice_segments(mel, ids_slice, self.hparams.segment_size // self.hparams.hop_length)
        y_hat_mel = _mel.mel_spectrogram_torch(
            y_hat.squeeze(1), self.hparams.filter_length, self.hparams.mel_channels, self.hparams.sample_rate,
            self.hparams.hop_length, self.hparams.win_length, self.hparams.mel_fmin, self.hparams.mel_fmax,
        )
        y = _commons.slice_segments(y, ids_slice * self.hparams.hop_length, self.hparams.segment_size)
        self._y = y

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)

        with autocast(self.device.type, enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = _losses.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_fm = _losses.feature_loss(fmap_r, fmap_g)
            loss_gen, _losses_gen = _losses.generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            self._last_loss_mel = loss_mel.detach()
            self.log("loss_mel", loss_mel)
            self.log("loss_kl", loss_kl)
            self.log("loss_dur", loss_dur)
            self.log("loss_gen", loss_gen)
            self.log("loss_fm", loss_fm)
            self.log("loss_gen_all", loss_gen_all)

            return loss_gen_all

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
