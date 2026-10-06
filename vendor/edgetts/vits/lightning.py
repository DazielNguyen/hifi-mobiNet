import itertools
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .commons import slice_segments
from .dataset import Batch, PiperDataset, UtteranceCollate
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    SynthesizerTrn,
)

_LOGGER = logging.getLogger("vits.lightning")


class VitsModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        num_speakers: int,
        # audio
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=(
            (1, 2),
            (2, 6),
            (3, 12),
        ),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        sample_bytes: int = 2,
        channels: int = 1,
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
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = True,
        use_bigvgan: bool = False,   # SnakeBeta activation + MRD discriminator
        use_vits2: bool = False,     # transformer coupling flows + duration discriminator + noised MAS
        use_f0: bool = False,        # F0 predictor + decoder conditioning
        segment_size: int = 8192,
        # training
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 1,
        lr_decay: float = 0.999875,
        init_lr_ratio: float = 1.0,
        warmup_epochs: int = 0,
        c_mel: int = 45,
        c_kl: float = 1.0,
        grad_clip: Optional[float] = None,
        mas_noise_scale_decay: float = 2e-6,
        num_workers: int = 1,
        seed: int = 1234,
        num_val_examples: int = 100,
        num_test_examples: int = 500,
        max_phoneme_ids: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # GAN training needs two independent backward/step calls per batch
        # (generator, then discriminator) -- not expressible via PL's
        # automatic single-optimizer-per-call dispatch since PL 2.0 dropped
        # the old `optimizer_idx` argument.
        self.automatic_optimization = False

        if (self.hparams.num_speakers > 1) and (self.hparams.gin_channels <= 0):
            # Default gin_channels for multi-speaker model
            self.hparams.gin_channels = 512

        # Derive low-level component flags from the three high-level group flags
        _use_snake = self.hparams.use_bigvgan
        _use_mrd = self.hparams.use_bigvgan
        _use_dur_disc = self.hparams.use_vits2
        _use_transformer_flows = self.hparams.use_vits2
        _use_noised_mas = self.hparams.use_vits2

        # Set up models
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
            n_speakers=self.hparams.num_speakers,
            gin_channels=self.hparams.gin_channels,
            use_sdp=self.hparams.use_sdp,
            use_snake=_use_snake,
            use_f0=self.hparams.use_f0,
            use_transformer_flows=_use_transformer_flows,
            use_noised_mas=_use_noised_mas,
            mas_noise_scale=0.01,
            use_speaker_cond_enc=False,
        )
        self.model_d = MultiPeriodDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        )
        self.model_d_mrd = (
            MultiResolutionDiscriminator(use_spectral_norm=self.hparams.use_spectral_norm)
            if _use_mrd
            else None
        )
        self.model_d_dur = (
            DurationDiscriminator(
                in_channels=self.hparams.hidden_channels,
                filter_channels=self.hparams.hidden_channels,
                kernel_size=3,
                p_dropout=self.hparams.p_dropout,
                gin_channels=self.hparams.gin_channels,
            )
            if _use_dur_disc
            else None
        )

        # Dataset splits
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._load_datasets(num_val_examples, num_test_examples, max_phoneme_ids)

        # State kept between training optimizers
        self._y = None
        self._y_hat = None

    def _load_datasets(
        self,
        num_val_examples: int,
        num_test_examples: int,
        max_phoneme_ids: Optional[int] = None,
    ):
        if self.hparams.dataset is None:
            _LOGGER.debug("No dataset to load")
            return

        full_dataset = PiperDataset(
            self.hparams.dataset, max_phoneme_ids=max_phoneme_ids
        )
        train_set_size = len(full_dataset) - num_val_examples - num_test_examples

        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset,
            [train_set_size, num_test_examples, num_val_examples],
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

    def forward(self, text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )

        return audio

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def training_step(self, batch: Batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()

        loss_gen_all, _loss_mel = self.training_step_g(batch)
        opt_g.zero_grad()
        self.manual_backward(loss_gen_all)
        if self.hparams.grad_clip is not None:
            self.clip_gradients(opt_g, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
        opt_g.step()

        loss_disc_all = self.training_step_d(batch)
        opt_d.zero_grad()
        self.manual_backward(loss_disc_all)
        if self.hparams.grad_clip is not None:
            self.clip_gradients(opt_d, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
        opt_d.step()

        # Anneal MAS noise scale: VITS2 schedule max(initial - step * decay, 0)
        if self.hparams.use_vits2 and self.hparams.mas_noise_scale_decay > 0:
            self.model_g.mas_noise_scale = max(
                0.01 - self.global_step * self.hparams.mas_noise_scale_decay, 0.0
            )

    def on_train_epoch_end(self):
        # Automatic LR scheduler stepping is disabled along with automatic
        # optimization, so step both schedulers once per epoch here (same
        # cadence as the original PL 1.7 default for ExponentialLR).
        sched_g, sched_d = self.lr_schedulers()
        sched_g.step()
        sched_d.step()

    def training_step_g(self, batch: Batch):
        x, x_lengths, y, _, spec, spec_lengths, f0, speaker_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.f0s,
            batch.speaker_ids if batch.speaker_ids is not None else None,
        )
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
            (x_hidden, logw, logw_, l_f0),
        ) = self.model_g(x, x_lengths, spec, spec_lengths, speaker_ids, f0=f0)
        self._y_hat = y_hat

        # Save for training_step_d (duration discriminator)
        self._dur_x = x_hidden
        self._dur_mask = x_mask
        self._dur_real = logw_
        self._dur_fake = logw

        # cuFFT (torch.stft, used inside the mel functions) doesn't accept
        # half/bf16 input, so this has to run outside the autocast region
        # under mixed-precision training -- not just the loss sums below.
        with autocast(self.device.type, enabled=False):
            mel = spec_to_mel_torch(
                spec.float(),
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )
            y_mel = slice_segments(
                mel,
                ids_slice,
                self.hparams.segment_size // self.hparams.hop_length,
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.float().squeeze(1),
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.hop_length,
                self.hparams.win_length,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )
        y = slice_segments(
            y,
            ids_slice * self.hparams.hop_length,
            self.hparams.segment_size,
        )  # slice

        # Save for training_step_d
        self._y = y

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen_mrd = torch.zeros(1, device=y.device)
        loss_dur_gen = torch.zeros(1, device=y.device)

        if self.model_d_mrd is not None:
            _y_d_hat_r_mrd, y_d_hat_g_mrd, fmap_r_mrd, fmap_g_mrd = self.model_d_mrd(y, y_hat)
            loss_fm = loss_fm + feature_loss(fmap_r_mrd, fmap_g_mrd)
            loss_gen_mrd, _ = generator_loss(y_d_hat_g_mrd)

        if self.model_d_dur is not None:
            _dur_probs_r, dur_probs_hat = self.model_d_dur(x_hidden, x_mask, logw_, logw)
            loss_dur_gen, _ = generator_loss(dur_probs_hat)

        with autocast(self.device.type, enabled=False):
            # Generator loss
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_gen, _losses_gen = generator_loss(y_d_hat_g)

            loss_gen_all = (
                loss_gen
                + loss_gen_mrd
                + loss_fm
                + loss_mel
                + loss_dur
                + loss_kl
                + loss_dur_gen
                + l_f0
            )

            self.log("loss_gen_all", loss_gen_all)
            self.log("loss_mel", loss_mel)
            self.log("loss_kl", loss_kl)
            self.log("loss_dur", loss_dur)
            self.log("loss_gen", loss_gen)
            self.log("loss_gen_mrd", loss_gen_mrd)
            self.log("loss_dur_gen", loss_dur_gen)
            self.log("loss_fm", loss_fm)
            self.log("loss_f0", l_f0)

            return loss_gen_all, loss_mel

    def training_step_d(self, batch: Batch):
        # From training_step_g
        y = self._y
        y_hat = self._y_hat
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())

        loss_disc_mrd = torch.zeros(1, device=y.device)
        loss_disc_dur = torch.zeros(1, device=y.device)

        if self.model_d_mrd is not None:
            y_d_hat_r_mrd, y_d_hat_g_mrd, _, _ = self.model_d_mrd(y, y_hat.detach())
            loss_disc_mrd, _, _ = discriminator_loss(y_d_hat_r_mrd, y_d_hat_g_mrd)

        if self.model_d_dur is not None:
            dur_probs_r, dur_probs_hat = self.model_d_dur(
                self._dur_x.detach(),
                self._dur_mask,
                self._dur_real.detach(),
                self._dur_fake.detach(),
            )
            loss_disc_dur, _, _ = discriminator_loss(dur_probs_r, dur_probs_hat)

        with autocast(self.device.type, enabled=False):
            loss_disc, _losses_disc_r, _losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc + loss_disc_mrd + loss_disc_dur

            self.log("loss_disc_all", loss_disc_all)
            self.log("loss_disc", loss_disc)
            self.log("loss_disc_mrd", loss_disc_mrd)
            self.log("loss_disc_dur", loss_disc_dur)

            return loss_disc_all

    def validation_step(self, batch: Batch, batch_idx: int):
        loss_gen_all, loss_mel = self.training_step_g(batch)
        val_loss = loss_gen_all + self.training_step_d(batch)
        self.log("val_loss", val_loss)
        # Non-adversarial reconstruction loss -- unlike val_loss (which mixes
        # in GAN terms that don't track audio quality monotonically), this is
        # a sane "best model" signal for ModelCheckpoint to monitor.
        self.log("val_loss_mel", loss_mel)

        # Generate audio examples — only on the first val batch to avoid
        # running num_val_batches × num_test_examples synthesis calls per epoch.
        if batch_idx == 0:
            for utt_idx, test_utt in enumerate(list(self._test_dataset)[:5]):
                try:
                    text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
                    text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(self.device)
                    scales = [0.667, 1.0, 0.8]
                    sid = (
                        test_utt.speaker_id.to(self.device)
                        if test_utt.speaker_id is not None
                        else None
                    )
                    test_audio = self(text, text_lengths, scales, sid=sid).detach()

                    # Scale to make louder in [-1, 1]
                    test_audio = test_audio * (1.0 / max(0.01, abs(test_audio.max())))

                    tag = test_utt.text or str(utt_idx)
                    self.logger.experiment.add_audio(
                        tag, test_audio, sample_rate=self.hparams.sample_rate
                    )
                except Exception:
                    # Synthesis can fail early in training (NaN in bfloat16 from
                    # unstable model state); skip the audio log rather than crash.
                    pass

        return val_loss

    def configure_optimizers(self):
        disc_param_groups = [self.model_d.parameters()]
        if self.model_d_mrd is not None:
            disc_param_groups.append(self.model_d_mrd.parameters())
        if self.model_d_dur is not None:
            disc_param_groups.append(self.model_d_dur.parameters())
        discriminator_params = itertools.chain(*disc_param_groups)
        optimizers = [
            torch.optim.AdamW(
                self.model_g.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ),
            torch.optim.AdamW(
                discriminator_params,
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[0], gamma=self.hparams.lr_decay
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[1], gamma=self.hparams.lr_decay
            ),
        ]

        return optimizers, schedulers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VitsModel")
        # Default tuned for a 12 GB GPU (RTX 4070 Ti) with segment_size=8192
        # and the added MRD/duration-discriminator VRAM overhead -- raise if
        # memory headroom allows, lower on OOM.
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="DataLoader workers per GPU process (default tuned for a 20-core/28-thread CPU)",
        )
        parser.add_argument("--num-val-examples", type=int, default=100)
        parser.add_argument("--num-test-examples", type=int, default=500)
        parser.add_argument(
            "--max-phoneme-ids",
            type=int,
            default=400,
            help="Exclude utterances with phoneme id lists longer than this",
        )
        #
        parser.add_argument("--hidden-channels", type=int, default=192)
        parser.add_argument("--inter-channels", type=int, default=192)
        parser.add_argument("--filter-channels", type=int, default=768)
        parser.add_argument("--n-layers", type=int, default=6)
        parser.add_argument("--n-heads", type=int, default=2)
        parser.add_argument("--grad-clip", type=float, default=None,
                            help="Max gradient norm for clipping (None = disabled)")
        # Three high-level architecture flags (default=False = vanilla VITS baseline)
        parser.add_argument("--use-bigvgan", type=lambda x: x.lower() != "false", default=False,
                            help="Enable BigVGAN components: SnakeBeta activation + MRD discriminator")
        parser.add_argument("--use-vits2", type=lambda x: x.lower() != "false", default=False,
                            help="Enable VITS2 components: transformer flows + duration discriminator + noised MAS")
        parser.add_argument("--use-f0", type=lambda x: x.lower() != "false", default=False,
                            help="Enable F0 predictor and decoder F0 conditioning")
        parser.add_argument("--mas-noise-scale-decay", type=float, default=2e-6,
                            help="Per-step linear decay of MAS noise scale (0 = no annealing). Default 2e-6 matches VITS2.")
        #
        return parent_parser
