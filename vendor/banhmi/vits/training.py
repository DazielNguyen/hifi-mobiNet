"""PyTorch Lightning training loop for the single-speaker VITS model.

Single-speaker is not a runtime option here: banhmi_train.preprocess's
config.json has no `num_speakers`/`speaker_id_map` fields at all (removed
since this project permanently targets single-speaker LJSpeech), so
n_speakers=1 / gin_channels=0 are hardcoded rather than read from anywhere
-- there is no other value they could ever take.
"""
import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from ..Vocos.discriminators import (
    MultiPeriodDiscriminator as VocosMultiPeriodDiscriminator,
    MultiResolutionDiscriminator as VocosMultiResolutionDiscriminator,
)
from ..Vocos.losses import (
    discriminator_loss as vocos_discriminator_loss,
    feature_loss as vocos_feature_loss,
    generator_loss as vocos_generator_loss,
)
from .dataset import Batch, UtteranceCollate, VitsDataset
from .length_bucket_sampler import LengthBucketBatchSampler
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .modules.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from .modules.duration_discriminator import DurationDiscriminator
from .modules.generator import INVERTED_RESBLOCKS
from .modules.synthesizer import SynthesizerTrn
from .utils.commons import slice_segments

_LOGGER = logging.getLogger("banhmi_train.vits.training")

# Dedicated seed for the train/val/test split only -- deliberately separate
# from --seed (which governs weight init/dropout/etc, and can legitimately
# differ run to run) so every architecture ever trained on this dataset
# gets the exact same split. See VitsModel._load_dataset's use of this.
_DATASET_SPLIT_SEED = 20260101

_N_SPEAKERS = 1
_GIN_CHANNELS = 0


class VitsModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        # audio
        resblock: str = "2",
        resblock_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = ((1, 2), (2, 6), (3, 12)),
        upsample_rates: Tuple[int, ...] = (8, 8, 4),
        upsample_initial_channel: int = 256,
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 8),
        # Per-stage expansion ratio for resblock="mb" (ResBlockInverted) --
        # e.g. (4,4,1) or (6,6,1) to keep the last upsample stage (T=8192,
        # the most compute-exposed) cheap while earlier stages (much
        # smaller T) can afford a higher ratio. Silently unused unless
        # resblock="mb".
        mb_expansion: Union[int, Tuple[int, ...]] = 6,
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        use_spectral_norm: bool = False,
        segment_size: int = 8192,
        # PosteriorEncoder is training-only (discarded before ONNX export),
        # so its size only trades off training speed/memory against how
        # good a training signal it gives the flow -- never inference cost.
        # 16 matches EdgeTTS/vanilla Piper's own hardcoded WaveNet depth.
        posterior_encoder_kernel_size: int = 5,
        posterior_encoder_dilation_rate: int = 1,
        posterior_encoder_layers: int = 16,
        # Flow (ResidualCouplingBlock): part of the deployed/exported model,
        # so these trade off real inference speed/quality, unlike the
        # PosteriorEncoder knobs above.
        # Flow's transformer-in-flow coupling (VITS2), the duration
        # discriminator, and noise-scaled MAS below are baked permanently
        # into the architecture -- not configurable -- since BanhmiTTS
        # builds on the full VITS2 recipe unconditionally rather than
        # replicating EdgeTTS's opt-in --use-vits2 toggle. Noise-scaled MAS
        # is training-only (the alignment search never runs at inference --
        # infer() uses the duration predictor + generate_path instead),
        # decays linearly to 0 over `global_step`, so it only affects how
        # training explores alignments early on, not the final converged
        # behavior.
        flow_kernel_size: int = 5,
        flow_dilation_rate: int = 1,
        flow_n_flows: int = 4,
        mas_noise_scale_initial: float = 0.01,
        mas_noise_scale_decay: float = 2e-6,
        # Vocos Generator (ConvNeXt backbone + ISTFT head, no time-domain
        # upsampling -- banhmi_train/Vocos/) + Vocos's own discriminators
        # (5-branch MPD without DiscriminatorS + DAC-style multi-band MRD)
        # instead of the HiFi-GAN-family Generator/discriminators -- see
        # configs/banhmi_vocos.yaml.
        # vocos_n_fft/vocos_hop_length aren't separate knobs here: the
        # ISTFT head is built from this model's own filter_length/
        # hop_length below, since it must match the rest of the mel/spec
        # pipeline exactly.
        use_vocos: bool = False,
        vocos_dim: int = 512,
        vocos_intermediate_dim: int = 1536,
        vocos_num_layers: int = 8,
        # vocos/experiment.py's training_step: loss_mp/loss_mrd (and their
        # generator-side counterparts) are each averaged over their own
        # sub-discriminator count, THEN combined as
        # `loss_mp + mrd_loss_coeff * loss_mrd` -- the class's own default
        # is 1.0, but the actual released 24kHz checkpoint's config
        # (configs/vocos.yaml) overrides it to 0.1, which is what produced
        # their published results, so that's the default here too. Only
        # applied when use_vocos=True -- the HiFi-GAN-family MRD keeps its
        # original equal-weight sum.
        vocos_mrd_loss_coeff: float = 0.1,
        # training
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 1,
        lr_decay: float = 0.999875,
        c_mel: int = 45,
        c_kl: float = 1.0,
        num_workers: int = 1,
        seed: int = 1234,
        # A real 3-way split: `test` is held out entirely until after
        # training finishes (see test_step/trainer.test), never touched for
        # monitoring or tuning decisions. Fixed counts, not fractions --
        # LJSpeech's size is permanently known (13,100 utterances), so a
        # hardcoded absolute split is as valid as a percentage and doesn't
        # silently balloon/shrink if the dataset ever changes size.
        # num_audio_samples is unrelated -- just a handful of *validation*
        # utterances synthesized and logged as listenable audio during
        # training, not a metric of any kind.
        num_val_examples: int = 100,
        num_test_examples: int = 500,
        num_audio_samples: int = 5,
        max_phoneme_ids: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_g = SynthesizerTrn(
            n_vocab=self.hparams.num_symbols,
            spec_channels=self.hparams.filter_length // 2 + 1,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            mb_expansion=self.hparams.mb_expansion,
            n_speakers=_N_SPEAKERS,
            gin_channels=_GIN_CHANNELS,
            posterior_encoder_kernel_size=self.hparams.posterior_encoder_kernel_size,
            posterior_encoder_dilation_rate=self.hparams.posterior_encoder_dilation_rate,
            posterior_encoder_layers=self.hparams.posterior_encoder_layers,
            flow_kernel_size=self.hparams.flow_kernel_size,
            flow_dilation_rate=self.hparams.flow_dilation_rate,
            flow_n_flows=self.hparams.flow_n_flows,
            use_vocos=self.hparams.use_vocos,
            vocos_n_fft=self.hparams.filter_length,
            vocos_hop_length=self.hparams.hop_length,
            vocos_dim=self.hparams.vocos_dim,
            vocos_intermediate_dim=self.hparams.vocos_intermediate_dim,
            vocos_num_layers=self.hparams.vocos_num_layers,
            use_f0=self.hparams.use_f0,
        )
        if self.hparams.use_vocos:
            self.model_d = VocosMultiPeriodDiscriminator()
            self.model_d_mrd = VocosMultiResolutionDiscriminator()
        else:
            self.model_d = MultiPeriodDiscriminator(use_spectral_norm=self.hparams.use_spectral_norm)
            self.model_d_mrd = (
                MultiResolutionDiscriminator(use_spectral_norm=self.hparams.use_spectral_norm)
                if self.hparams.resblock in INVERTED_RESBLOCKS
                else None
            )
        self.model_d_dur = DurationDiscriminator(
            in_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.hidden_channels,
            kernel_size=3,
            p_dropout=self.hparams.p_dropout,
            gin_channels=_GIN_CHANNELS,
        )

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._audio_sample_dataset: Optional[Dataset] = None
        self._train_batch_sampler: Optional[LengthBucketBatchSampler] = None
        self._load_datasets(num_val_examples, num_test_examples, num_audio_samples, max_phoneme_ids)

        # State kept between the generator and discriminator optimizer steps
        self._y = None
        self._y_hat = None
        self._dur_x = None
        self._dur_mask = None
        self._dur_real = None
        self._dur_fake = None
        self._last_loss_mel = None
        # See optimizer_step(): counters for steps discarded because their
        # gradients weren't finite, plus a fingerprint of the last batch so a
        # discarded step can be traced back to the data that produced it.
        self._skipped_steps = 0
        self._total_steps = 0
        self._last_batch_fingerprint = None

    def _load_datasets(
        self,
        num_val_examples: int,
        num_test_examples: int,
        num_audio_samples: int,
        max_phoneme_ids: Optional[int],
    ):
        if not self.hparams.dataset:
            _LOGGER.debug("No dataset to load")
            return

        full_dataset = VitsDataset(self.hparams.dataset[0], max_phoneme_ids=max_phoneme_ids)
        train_size = len(full_dataset) - num_val_examples - num_test_examples
        if train_size <= 0:
            raise ValueError(
                f"num_val_examples ({num_val_examples}) + num_test_examples ({num_test_examples}) "
                f">= dataset size ({len(full_dataset)}) -- nothing left to train on"
            )

        # Fixed, architecture-independent generator for the split -- using
        # the ambient global RNG here (seeded once via --seed, then advanced
        # by however many random draws model weight-init consumed) made the
        # train/val/test split silently depend on the model's own param
        # count: two architectures with the same --seed but different
        # numbers of randomly-initialized weights reach this call with the
        # global RNG in different states, producing DIFFERENT splits (and
        # therefore train/test leakage when one model's "held-out" test
        # utterances turn out to be in the other's training set -- see
        # docs/mrf_full_investigation_report.md SS8.2). A dedicated
        # generator seeded with a constant, independent of --seed and of
        # anything consumed before this line, makes the split identical
        # across every architecture trained on this dataset from now on.
        #
        # Preferred path: a canonical_split.json next to the dataset, listing
        # each utterance's audio_norm_path per partition -- lets a NEW
        # architecture reuse an EXISTING model's exact already-trained split
        # (e.g. baseline's) without retraining that existing model, which
        # the generator-seed fallback below cannot do (a fixed seed only
        # guarantees new runs match EACH OTHER, not a specific past run that
        # was itself produced by the old architecture-dependent mechanism).
        canonical_split_path = Path(self.hparams.dataset[0]).parent / "canonical_split.json"
        if canonical_split_path.exists():
            with open(canonical_split_path, "r", encoding="utf-8") as split_file:
                canonical_split = json.load(split_file)
            path_to_index = {str(u.audio_norm_path): i for i, u in enumerate(full_dataset.utterances)}
            train_idx = [path_to_index[p] for p in canonical_split["train"]]
            val_idx = [path_to_index[p] for p in canonical_split["val"]]
            test_idx = [path_to_index[p] for p in canonical_split["test"]]
            self._train_dataset = Subset(full_dataset, train_idx)
            self._val_dataset = Subset(full_dataset, val_idx)
            self._test_dataset = Subset(full_dataset, test_idx)
            _LOGGER.info("Loaded canonical train/val/test split from %s", canonical_split_path)
        else:
            split_generator = torch.Generator().manual_seed(_DATASET_SPLIT_SEED)
            self._train_dataset, self._val_dataset, self._test_dataset = random_split(
                full_dataset, [train_size, num_val_examples, num_test_examples], generator=split_generator
            )
        # A handful of *validation* utterances (never the held-out test set)
        # purely for logging listenable audio samples during training.
        num_audio_samples = min(num_audio_samples, len(self._val_dataset))
        self._audio_sample_dataset = Subset(self._val_dataset, range(num_audio_samples))

    def forward(self, text, text_lengths, scales):
        noise_scale, length_scale, noise_scale_w = scales[0], scales[1], scales[2]
        audio, *_ = self.model_g.infer(
            text, text_lengths, noise_scale=noise_scale, length_scale=length_scale, noise_scale_w=noise_scale_w
        )
        return audio

    def _dataloader(self, dataset):
        return DataLoader(
            dataset,
            collate_fn=UtteranceCollate(segment_size=self.hparams.segment_size),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def _spectrogram_lengths(self, dataset: Subset) -> List[int]:
        """Cached spectrogram frame count per item in a VitsDataset Subset.

        Phoneme-id count was tried first as a bucketing key (cheap: already
        in memory, no I/O) but measured WORSE than no bucketing at all --
        14.2 samples/s (bucketed) vs 37.8 (unbucketed), both with
        cudnn.benchmark=True. Root cause: phoneme count only stabilizes
        TextEncoder/StochasticDurationPredictor shapes. The PosteriorEncoder
        (a 16-layer WaveNet -- the single largest non-Generator compute
        block) and flow run on the *full* spectrogram/latent length before
        SynthesizerTrn.forward() crops a fixed segment_size window out of it;
        phoneme count is too weak a proxy for spectrogram length (different
        speaking rates, pauses, etc.) to stabilize that shape. Loading each
        cached .spec.pt just for its frame count (matches banhmi_tts's own
        LengthBucketBatchSampler, which buckets by spectrogram_lengths) is a
        one-time cost at dataloader setup -- EXCEPT it isn't quite one-time
        as originally written: train_dataloader() runs independently on
        every DDP rank, so with a full 13,100-utterance dataset (12,500
        after held-out val/test) this took long enough (~13 min, ~14GB of
        .spec.pt reads per rank -- confirmed via /proc/<pid>/io) and drifted
        enough between ranks that NCCL's collective-op watchdog aborted with
        SIGABRT before both ranks reached the same barrier. Caching to a
        JSON file next to dataset.jsonl fixes the repeat-run cost; a cache
        pre-warmed by a quick single-process pass before launching the real
        multi-rank job avoids the first-run rank skew entirely (see
        scripts/warm_length_cache.py in the training runbook).
        """
        base = dataset.dataset
        dataset_path = Path(self.hparams.dataset[0])
        cache_path = dataset_path.parent / ".spectrogram_lengths_cache.json"

        cache: dict = {}
        if cache_path.is_file():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cache = {}

        lengths = []
        dirty = False
        for index in dataset.indices:
            spec_path = str(base.utterances[index].audio_spec_path)
            cached_length = cache.get(spec_path)
            if cached_length is None:
                spec = torch.load(spec_path, map_location="cpu")
                cached_length = int(spec.shape[-1])
                cache[spec_path] = cached_length
                dirty = True
            lengths.append(cached_length)

        if dirty:
            try:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            except OSError:
                pass  # best-effort; a failed write just means no speedup next run

        return lengths

    def train_dataloader(self):
        # Length-bucketed instead of plain batch_size=: keeps padded batch
        # shapes stable across steps so cudnn.benchmark (see train.py) can
        # actually reuse its cached algorithm instead of re-searching almost
        # every step. This sampler shards across ranks itself (num_replicas/
        # rank below), so DDP's automatic sampler replacement must be off --
        # train.py forces `replace_sampler_ddp=False` for exactly this.
        world_size, rank = 1, 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        batch_sampler = LengthBucketBatchSampler(
            self._spectrogram_lengths(self._train_dataset),
            batch_size=self.hparams.batch_size,
            boundaries=(150, 250, 350, 450, 600, 800, 1000),
            seed=self.hparams.seed,
            num_replicas=world_size,
            rank=rank,
        )
        batch_sampler.set_epoch(self.current_epoch)
        # PL calls train_dataloader() once (not per epoch) by default, so the
        # sampler's own epoch tracking would otherwise stay frozen at 0 --
        # reshuffle it explicitly at the start of every subsequent epoch.
        self._train_batch_sampler = batch_sampler
        return DataLoader(
            self._train_dataset,
            collate_fn=UtteranceCollate(segment_size=self.hparams.segment_size),
            num_workers=self.hparams.num_workers,
            batch_sampler=batch_sampler,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def on_train_epoch_start(self) -> None:
        if self._train_batch_sampler is not None:
            self._train_batch_sampler.set_epoch(self.current_epoch)

    def val_dataloader(self):
        return self._dataloader(self._val_dataset)

    def test_dataloader(self):
        return self._dataloader(self._test_dataset)

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            self._last_batch_fingerprint = (
                int(self.current_epoch),
                int(batch_idx),
                batch.phoneme_lengths.tolist(),
            )
            return self.training_step_g(batch)
        return self.training_step_d(batch)

    # Fraction of discarded steps above which the run is treated as genuinely
    # diverging rather than as having hit isolated bad batches. The observed
    # real-world rate for the dp instability this guards against is on the
    # order of one step in 10^5, so anything near 1% means something else is
    # wrong and must be surfaced, not silently absorbed.
    _SKIP_RATE_ALARM = 0.01

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, **kwargs
    ):
        """Discards an update whose gradients aren't finite instead of letting
        it reach the optimizer.

        This is the safety net `torch.cuda.amp.GradScaler` provides for free
        under fp16 (check for inf/NaN, skip the step, carry on) and which
        bf16 training silently does without, since bf16 needs no loss scaling
        and therefore gets no scaler. Without it, one non-finite gradient is
        permanent: AdamW writes NaN into the parameters, and every subsequent
        forward pass through those parameters is NaN forever after. That is
        exactly how StochasticDurationPredictor kept dying here -- and because
        dp's gradients don't reach dec/enc_p, loss_mel went on improving and
        hid the damage for hundreds of epochs.

        Skipping costs one finite-check per parameter per step and throws away
        only the genuinely broken batch; the parameters keep their last good
        values and training continues from there.

        Under DDP this stays consistent across ranks without extra
        synchronisation: gradients are all-reduced during backward, and a NaN
        on any rank propagates through that averaging, so every rank sees the
        same non-finite gradients and independently reaches the same decision.
        """
        optimizer_closure()  # forward + backward; populates .grad

        nonfinite = False
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    nonfinite = True
                    break
            if nonfinite:
                break

        if optimizer_idx == 0:
            self._total_steps += 1

        if not nonfinite:
            optimizer.step()
            return

        optimizer.zero_grad(set_to_none=True)
        self._skipped_steps += 1
        skip_rate = self._skipped_steps / max(self._total_steps, 1)
        _LOGGER.warning(
            "Discarded optimizer step (optimizer_idx=%d): non-finite gradients. "
            "Parameters left untouched, batch dropped. Skipped %d of %d steps so far "
            "(%.4f%%). Batch fingerprint (epoch, batch_idx, phoneme_lengths)=%r",
            optimizer_idx, self._skipped_steps, self._total_steps, 100 * skip_rate,
            self._last_batch_fingerprint,
        )
        if skip_rate > self._SKIP_RATE_ALARM and self._total_steps > 200:
            _LOGGER.error(
                "Discarded-step rate %.2f%% exceeds %.2f%% -- this is no longer isolated "
                "bad batches; the run is likely diverging and needs investigation, not "
                "more skipping.",
                100 * skip_rate, 100 * self._SKIP_RATE_ALARM,
            )
        self.log("skipped_steps", float(self._skipped_steps))

    def _current_mas_noise_scale(self) -> float:
        scale = self.hparams.mas_noise_scale_initial - self.global_step * self.hparams.mas_noise_scale_decay
        return max(scale, 0.0)

    def _gan_loss_fns(self):
        """Selects which discriminator/generator/feature-matching loss
        formulas to use against model_d/model_d_mrd -- Vocos's own hinge
        loss (Vocos/losses.py) when use_vocos, else the project's LSGAN
        default (.losses). model_d_dur always uses the LSGAN functions
        directly (imported at module level), regardless of this."""
        if self.hparams.use_vocos:
            return vocos_discriminator_loss, vocos_generator_loss, vocos_feature_loss
        return discriminator_loss, generator_loss, feature_loss

    def training_step_g(self, batch: Batch):
        x, x_lengths = batch.phoneme_ids, batch.phoneme_lengths
        y, spec, spec_lengths = batch.audios, batch.spectrograms, batch.spectrogram_lengths

        mas_noise_scale = self._current_mas_noise_scale()
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
            (x_hidden, logw, logw_, l_f0),
        ) = self.model_g(x, x_lengths, spec, spec_lengths, mas_noise_scale=mas_noise_scale, f0=batch.f0s)
        self._y_hat = y_hat
        self.log("mas_noise_scale", mas_noise_scale)

        # Saved for training_step_d's duration discriminator
        self._dur_x, self._dur_mask, self._dur_real, self._dur_fake = x_hidden, x_mask, logw_, logw

        mel = spec_to_mel_torch(
            spec, self.hparams.filter_length, self.hparams.mel_channels,
            self.hparams.sample_rate, self.hparams.mel_fmin, self.hparams.mel_fmax,
        )
        y_mel = slice_segments(mel, ids_slice, self.hparams.segment_size // self.hparams.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), self.hparams.filter_length, self.hparams.mel_channels,
            self.hparams.sample_rate, self.hparams.hop_length, self.hparams.win_length,
            self.hparams.mel_fmin, self.hparams.mel_fmax,
        )
        y = slice_segments(y, ids_slice * self.hparams.hop_length, self.hparams.segment_size)
        self._y = y

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)
        _disc_loss_fn, gen_loss_fn, feat_loss_fn = self._gan_loss_fns()

        with autocast(self.device.type, enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_fm = feat_loss_fn(fmap_r, fmap_g)
            loss_gen, _ = gen_loss_fn(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + l_f0
            self.log("loss_f0", l_f0)

            # Stashed so validation_step can log a *validation-only*
            # "val_loss_mel" (see below) -- self.log("loss_mel", ...) alone
            # mixes training-step and validation-step calls into one epoch
            # average, which is a noisier/less meaningful "best model" signal
            # than a clean non-adversarial validation reconstruction loss
            # (matches EdgeTTS's own val_loss_mel).
            self._last_loss_mel = loss_mel.detach()
            self.log("loss_mel", loss_mel)
            self.log("loss_kl", loss_kl)
            self.log("loss_dur", loss_dur)
            self.log("loss_gen", loss_gen)
            self.log("loss_fm", loss_fm)

            if self.model_d_mrd is not None:
                _y_d_hat_r_mrd, y_d_hat_g_mrd, fmap_r_mrd, fmap_g_mrd = self.model_d_mrd(y, y_hat)
                loss_fm_mrd = feat_loss_fn(fmap_r_mrd, fmap_g_mrd)
                loss_gen_mrd, _ = gen_loss_fn(y_d_hat_g_mrd)
                # Only Vocos's own recipe down-weights MRD's contribution
                # (vocos/experiment.py: loss_gen_mp + mrd_loss_coeff *
                # loss_gen_mrd) -- the HiFi-GAN-family MRD keeps its
                # original equal-weight sum (coeff=1.0).
                mrd_coeff = self.hparams.vocos_mrd_loss_coeff if self.hparams.use_vocos else 1.0
                loss_gen_all = loss_gen_all + mrd_coeff * (loss_gen_mrd + loss_fm_mrd)
                self.log("loss_gen_mrd", loss_gen_mrd)

            _dur_probs_r, dur_probs_hat = self.model_d_dur(x_hidden, x_mask, logw_, logw)
            loss_dur_gen, _ = generator_loss(dur_probs_hat)
            loss_gen_all = loss_gen_all + loss_dur_gen
            self.log("loss_dur_gen", loss_dur_gen)

            self.log("loss_gen_all", loss_gen_all)
            return loss_gen_all

    def training_step_d(self, batch: Batch):
        y, y_hat = self._y, self._y_hat
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())
        disc_loss_fn, _gen_loss_fn, _feat_loss_fn = self._gan_loss_fns()

        with autocast(self.device.type, enabled=False):
            loss_disc, *_ = disc_loss_fn(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
            self.log("loss_disc", loss_disc)

            if self.model_d_mrd is not None:
                y_d_hat_r_mrd, y_d_hat_g_mrd, _, _ = self.model_d_mrd(y, y_hat.detach())
                loss_disc_mrd, *_ = disc_loss_fn(y_d_hat_r_mrd, y_d_hat_g_mrd)
                mrd_coeff = self.hparams.vocos_mrd_loss_coeff if self.hparams.use_vocos else 1.0
                loss_disc_all = loss_disc_all + mrd_coeff * loss_disc_mrd
                self.log("loss_disc_mrd", loss_disc_mrd)

            dur_probs_r, dur_probs_hat = self.model_d_dur(
                self._dur_x.detach(), self._dur_mask, self._dur_real.detach(), self._dur_fake.detach()
            )
            loss_disc_dur, *_ = discriminator_loss(dur_probs_r, dur_probs_hat)
            loss_disc_all = loss_disc_all + loss_disc_dur
            self.log("loss_disc_dur", loss_disc_dur)

            self.log("loss_disc_all", loss_disc_all)
            return loss_disc_all

    def validation_step(self, batch: Batch, batch_idx: int):
        val_loss = self.training_step_g(batch) + self.training_step_d(batch)
        # sync_dist=True: only 1 log call/epoch here (unlike the per-step
        # training_step_g/d logs above, where syncing every step would add
        # real overhead) -- without it, under DDP each rank's local value
        # gets logged/compared independently, so ModelCheckpoint(monitor=
        # "val_loss_mel") can pick a "best" that isn't actually the best
        # across ranks, and the TensorBoard curve can show values the saved
        # checkpoint never actually reflects. Cheap here; correctness-critical.
        self.log("val_loss", val_loss, sync_dist=True)
        # Free-running inference health has to be established BEFORE
        # val_loss_mel is logged, because it decides what gets logged.
        dp_healthy = self._log_audio_samples(self._audio_sample_dataset)
        # Non-adversarial reconstruction loss, logged only from this method
        # (unlike "loss_mel", which also gets logged from real training
        # steps) -- a clean, low-noise "best model" signal for
        # ModelCheckpoint(monitor="val_loss_mel", ...) to track, matching
        # EdgeTTS's own val_loss_mel.
        #
        # val_loss_mel is teacher-forced: it exercises dec/enc_p but never
        # dp's reverse sampling, so a model whose dp has collapsed still
        # posts an improving val_loss_mel and ModelCheckpoint happily
        # promotes it to "best", overwriting the genuinely-good checkpoints
        # until none are left (observed: an entire run's clean checkpoints
        # deleted this way, leaving nothing to resume from). Reporting +inf
        # for an unusable model keeps "best" pointing at the last checkpoint
        # that could actually synthesise speech.
        val_loss_mel = self._last_loss_mel
        if not dp_healthy:
            _LOGGER.warning(
                "Reporting val_loss_mel=inf for this epoch: free-running inference is "
                "degenerate (collapsed StochasticDurationPredictor), so this checkpoint "
                "must not be promoted over an earlier usable one."
            )
            val_loss_mel = torch.tensor(float("inf"), device=self.device)
        self.log("val_loss_mel", val_loss_mel, sync_dist=True)
        return val_loss

    def test_step(self, batch: Batch, batch_idx: int):
        """Runs once, after training finishes (see train.py's trainer.test()
        call) -- on data the training process never touched for monitoring
        or tuning, unlike val_loss above which is checked throughout
        training and can indirectly influence decisions (e.g. when to stop).
        """
        test_loss = self.training_step_g(batch) + self.training_step_d(batch)
        self.log("test_loss", test_loss, sync_dist=True)
        return test_loss

    def _log_audio_samples(self, dataset) -> bool:
        # A real utterance should produce at least ~0.2s of audio; anything
        # shorter (e.g. the dp.reverse-collapse 1-frame fallback in
        # SynthesizerTrn.infer(), see that file's own warning) means free-
        # running inference is broken even though val_loss_mel (teacher-
        # forced, never exercises this path) can still look fine. Surfacing
        # it here catches it live instead of only discovering it much later
        # via a manual listening test.
        min_reasonable_samples = int(0.2 * self.hparams.sample_rate)
        # Returned to validation_step, which uses it to decide whether this
        # epoch's checkpoint is eligible to become "best" at all.
        healthy = True
        for utt_idx, utt in enumerate(dataset):
            tag = utt.text or str(utt_idx)
            text = utt.phoneme_ids.unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(utt.phoneme_ids)]).to(self.device)
            scales = [0.667, 1.0, 0.8]
            audio = self(text, text_lengths, scales).detach()
            if audio.shape[-1] < min_reasonable_samples or not torch.isfinite(audio).all():
                healthy = False
                _LOGGER.warning(
                    "_log_audio_samples: utt %r produced only %d samples (%.3fs) -- "
                    "suspiciously short, likely a collapsed StochasticDurationPredictor "
                    "rather than a real synthesized utterance.",
                    tag, audio.shape[-1], audio.shape[-1] / self.hparams.sample_rate,
                )
            audio = audio * (1.0 / max(0.01, abs(audio.max())))
            self.logger.experiment.add_audio(tag, audio, sample_rate=self.hparams.sample_rate)
        return healthy

    @staticmethod
    def _make_adamw(params, **kwargs) -> torch.optim.AdamW:
        # fused=True dispatches to a single fused CUDA kernel for the whole
        # parameter-update step instead of one kernel launch per tensor --
        # meaningful here with 72M+ params split across two optimizers.
        # Only on torch>=2.0 (the `fused` kwarg doesn't exist before that);
        # falls back silently on anything else (older torch, CPU-only, etc).
        try:
            return torch.optim.AdamW(params, fused=torch.cuda.is_available(), **kwargs)
        except TypeError:
            return torch.optim.AdamW(params, **kwargs)

    def configure_optimizers(self):
        discriminators = [self.model_d, self.model_d_mrd, self.model_d_dur]
        discriminator_params = itertools.chain(
            *(d.parameters() for d in discriminators if d is not None)
        )
        optimizers = [
            self._make_adamw(
                self.model_g.parameters(), lr=self.hparams.learning_rate,
                betas=self.hparams.betas, eps=self.hparams.eps,
            ),
            self._make_adamw(
                discriminator_params, lr=self.hparams.learning_rate,
                betas=self.hparams.betas, eps=self.hparams.eps,
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=self.hparams.lr_decay),
            torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=self.hparams.lr_decay),
        ]
        return optimizers, schedulers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VitsModel")
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument(
            "--num-workers", type=int, default=1,
            help="Dataloader worker processes -- pure data-loading parallelism, no effect on model quality",
        )
        parser.add_argument(
            "--num-val-examples", type=int, default=100,
            help="Utterances checked periodically during training (can indirectly influence when you stop) -- matches EdgeTTS's own default",
        )
        parser.add_argument(
            "--num-test-examples", type=int, default=500,
            help="Utterances held out entirely until trainer.test() runs after training finishes -- never touched during training",
        )
        parser.add_argument(
            "--num-audio-samples", type=int, default=5,
            help="How many *validation* utterances to synthesize and log as listenable audio during training (not a metric)",
        )
        parser.add_argument(
            "--max-phoneme-ids", type=int,
            help="Exclude utterances with phoneme id lists longer than this",
        )
        parser.add_argument("--hidden-channels", type=int, default=192)
        parser.add_argument("--inter-channels", type=int, default=192)
        parser.add_argument("--filter-channels", type=int, default=768)
        parser.add_argument("--n-layers", type=int, default=6)
        parser.add_argument("--n-heads", type=int, default=2)
        parser.add_argument("--c-mel", type=int, default=45)
        parser.add_argument("--c-kl", type=float, default=1.0)
        parser.add_argument(
            "--posterior-encoder-layers", type=int, default=16,
            help="PosteriorEncoder WaveNet depth -- training-only cost, no effect on the exported model",
        )
        parser.add_argument("--posterior-encoder-kernel-size", type=int, default=5)
        parser.add_argument("--posterior-encoder-dilation-rate", type=int, default=1)
        parser.add_argument(
            "--flow-n-flows", type=int, default=4,
            help="Depth of the normalizing-flow stack -- part of the exported model, affects real inference cost",
        )
        parser.add_argument("--flow-kernel-size", type=int, default=5)
        parser.add_argument("--flow-dilation-rate", type=int, default=1)
        parser.add_argument("--mas-noise-scale-initial", type=float, default=0.01)
        parser.add_argument("--mas-noise-scale-decay", type=float, default=2e-6)
        parser.add_argument(
            "--use-vocos", action=argparse.BooleanOptionalAction, default=False,
            help="Vocos Generator (ConvNeXt backbone + ISTFT head, no time-domain upsampling) + Vocos's own discriminators (5-branch MPD, no DiscriminatorS + DAC-style multi-band MRD) instead of the HiFi-GAN-family Generator/discriminators -- see configs/banhmi_vocos.yaml",
        )
        parser.add_argument("--vocos-dim", type=int, default=512, help="Vocos ConvNeXt backbone width, only used with --use-vocos")
        parser.add_argument("--vocos-intermediate-dim", type=int, default=1536, help="Vocos ConvNeXt block MLP width, only used with --use-vocos")
        parser.add_argument("--vocos-num-layers", type=int, default=8, help="Number of Vocos ConvNeXt blocks, only used with --use-vocos")
        parser.add_argument(
            "--vocos-mrd-loss-coeff", type=float, default=0.1,
            help="Weight on MRD's adversarial+feature-matching loss terms relative to MPD's, only used with --use-vocos (matches Vocos's own released config, not the class default of 1.0)",
        )
        parser.add_argument(
            "--use-f0", action=argparse.BooleanOptionalAction, default=False,
            help="F0Predictor + decoder F0 conditioning, wired into the Vocos generator only (only meaningful with --use-vocos). This project's own bug-fixed pathway -- trains the decoder on the ground-truth per-phoneme F0 average re-expanded through the same hard alignment inference uses, instead of the raw per-frame F0 upstream EdgeTTS trains on, which creates a train/inference mismatch (decoder never sees the smooth signal it trained on at inference, only a piecewise-constant one) -- see f0_predictor.py/synthesizer.py",
        )
        return parent_parser
