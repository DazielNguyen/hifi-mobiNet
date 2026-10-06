"""Piper_no_VITS2_cpn: EdgeTTS Config A (vanilla Piper, all component flags off),
trained from scratch in this project's data/split/checkpoint harness.

Subclasses the unmodified EdgeTTS ``VitsModel`` (vendor/edgetts/vits/lightning.py,
EdgeTTS a73a897, a fork of rhasspy/piper 73c04d81). With ``use_bigvgan``,
``use_vits2`` and ``use_f0`` all false it builds Piper's WaveNet coupling flow,
stochastic duration predictor, plain MAS, LeakyReLU ResBlock2 generator and MPD,
with no duration discriminator, MRD, Snake activation or F0 branch. Inherited
unchanged: the model, losses, manual-optimization ``training_step`` (generator
then discriminator, gradient-norm clipping by ``grad_clip``),
``on_train_epoch_end`` (per-epoch ExponentialLR steps), ``configure_optimizers``,
``training_step_g``/``training_step_d`` and ``forward``.

Overridden for the project harness (docs/training/piper-component-comparison.md):

- data: the shared explicit split and dataset instead of ``random_split``;
  batches carry no F0 (Config A never reads it);
- ``validation_step``: same ``val_loss`` / ``val_loss_mel`` values, logged with
  ``sync_dist`` like the baseline; audio examples from validation utterances
  (EdgeTTS used its held-out test split for them);
- RNG state in checkpoints and Lightning-1.7 scheduler compatibility (harness).
"""
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, Subset

from .harness import HarnessMixin, free_running_healthy
from .vendor import vendor_module

_LOGGER = logging.getLogger(__name__)

EdgeTTSVitsModel = vendor_module("edgetts_vits", "lightning").VitsModel
CONFIG_A_FLAGS = ("use_bigvgan", "use_vits2", "use_f0")


class PiperNoVits2Module(HarnessMixin, EdgeTTSVitsModel):
    family = "edgetts"

    def __init__(self, num_symbols: int, sampler: str, num_audio_samples: int,
                 skip_nonfinite_updates: bool, dp_health_gate: bool, save_rng_state: bool, **edgetts_kwargs):
        on = [flag for flag in CONFIG_A_FLAGS if edgetts_kwargs.get(flag)]
        if on:
            raise ValueError(f"Piper_no_VITS2_cpn is EdgeTTS Config A; these flags must be false: {on}")
        if skip_nonfinite_updates:
            raise ValueError("EdgeTTS uses manual optimization; the non-finite update skip is not available")
        # dataset=None: EdgeTTS _load_datasets() returns early; data is attached explicitly.
        super().__init__(
            num_symbols=num_symbols, num_speakers=1, dataset=None, sampler=sampler,
            num_audio_samples=num_audio_samples, skip_nonfinite_updates=skip_nonfinite_updates,
            dp_health_gate=dp_health_gate, save_rng_state=save_rng_state, **edgetts_kwargs,
        )
        self._data = None
        self._train_batch_sampler = None

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
            # EdgeTTS/Piper's own loader: fixed order, no shuffling.
            return DataLoader(self._data.train, collate_fn=collate, num_workers=self.hparams.num_workers,
                              batch_size=self.hparams.batch_size, pin_memory=True,
                              persistent_workers=self.hparams.num_workers > 0)
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
                          num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
                          pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    def test_dataloader(self):
        raise RuntimeError("The held-out test split is not used by this training harness")

    # ------------------------------------------------------------ validation
    def validation_step(self, batch, batch_idx: int):
        loss_gen_all, loss_mel = self.training_step_g(batch)
        val_loss = loss_gen_all + self.training_step_d(batch)
        val_loss_mel = loss_mel
        if self.hparams.dp_health_gate:
            if not free_running_healthy(self, self._audio_sample_dataset, self.hparams.sample_rate):
                val_loss_mel = torch.tensor(float("inf"), device=self.device)
        elif batch_idx == 0:
            self._log_audio_examples()
        self.log("val_loss", val_loss, sync_dist=True)
        self.log("val_loss_mel", val_loss_mel, sync_dist=True)
        return val_loss

    def _log_audio_examples(self) -> None:
        # EdgeTTS behaviour (vendor lightning.py validation_step): synthesize a
        # few utterances on the first validation batch and skip any that fail
        # early in training; here from validation utterances, not the test split.
        for idx, utt in enumerate(self._audio_sample_dataset):
            try:
                text = utt.phoneme_ids.unsqueeze(0).to(self.device)
                lengths = torch.LongTensor([len(utt.phoneme_ids)]).to(self.device)
                audio = self(text, lengths, [0.667, 1.0, 0.8]).detach()
                audio = audio * (1.0 / max(0.01, abs(audio.max())))
                if self.logger is not None:
                    self.logger.experiment.add_audio(utt.text or str(idx), audio, sample_rate=self.hparams.sample_rate)
            except Exception:  # noqa: BLE001 - mirrors EdgeTTS; failures are expected early in training
                _LOGGER.debug("Audio example %d failed", idx, exc_info=True)
