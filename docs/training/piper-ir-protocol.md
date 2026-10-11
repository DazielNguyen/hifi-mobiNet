# Protocol: Piper-ParIR and Piper-SeqIR on the Config A backbone

Status: set up on 2026-10-11, before either long run.

Purpose: train the released Parallel-IR and Sequential-IR decoder bundles on the
backbone, data and recipe of the finished `Piper_no_VITS2_cpn` run (Piper-ConfigA).
The two runs go one after the other.

| Training ID | Short name | Config |
|---|---|---|
| `Piper_ParIR_no_VITS2_cpn` | Piper-ParIR | `configs/training/piper-parir-no-vits2-cpn.yaml` |
| `Piper_SeqIR_no_VITS2_cpn` | Piper-SeqIR | `configs/training/piper-seqir-no-vits2-cpn.yaml` |

## 1. Config A reference, verified from its records

These values come from the run itself:
- Config A's `configs/training/piper-no-vits2-cpn.yaml`, with SHA-256 `2d9c1f47…` equal to its `run_record.json`;
- its Lightning `hparams.yaml`;
- the launch commit `d0c20d8` source.

They are not taken from the name "Piper No Vits".

| Item | Config A value |
|---|---|
| Frontend | Precomputed phoneme IDs in `dataset.jsonl` (espeak `en-us`, `num_symbols` 256, 159 used); blank `_`=0 between every phoneme, `^`=1, `$`=2 |
| Text encoder | Relative-position Transformer, 6 layers, 2 heads, 192/768 |
| Posterior encoder | EdgeTTS/Piper WaveNet, 16 layers, kernel 5, on 513-bin linear spectrograms |
| Flow | `ResidualCouplingLayer` (WaveNet, no attention) ×4 with Flip, mean-only |
| Duration | Stochastic duration predictor (`use_sdp` true) |
| MAS | Plain. `use_noised_mas = use_vits2 = false`; the only MAS noise draw sits under `if self.use_noised_mas:` (`vendor/edgetts/vits/models.py`, tested) |
| VITS2 components | Transformer-in-flow, duration discriminator and noise-scaled MAS all off (`use_vits2` false) |
| Data / split | `dataset.jsonl` SHA-256 `f4a72ae0…295b`; canonical split `e678cf43…ae9d` (12,500 / 100 / 500); staged in-repo copy |
| Optimizer | AdamW lr 2e-4, betas (0.8, 0.99), eps 1e-9; ExponentialLR 0.999875 per epoch |
| Losses | gen + fm + 45·mel + dur + kl; EdgeTTS manual optimization (G step, then D step) |
| Batch / sampler / workers | 16 per GPU, length-bucket (seed 1234), 1 worker, no phoneme-length filter |
| Precision / clipping / seed | bf16 AMP; no gradient clipping (`grad_clip` and `gradient_clip_val` null); seed 1234 |
| Devices / budget | 2 GPUs DDP; 1,500 epochs (392 batches per epoch per rank) |
| Validation / selection | `val_loss_mel` on the 100 validation utterances every epoch. Top-3 by `val_loss_mel` plus `last.ckpt`. Evaluate the lowest-`val_loss_mel` checkpoint |

## 2. Decoder bundle, as implemented and as released

The released IR checkpoints were trained with `resblock` "mrf" / "mb", `mb_expansion`
(1, 1, 1), `use_snake` and `use_mrd` true (`configs/parallel-ir.json`,
`configs/sequential-ir.json`, from checkpoint metadata). Q05 loaded both
checkpoints strictly into the repository `Generator` with its defaults
(`mb_kernel_size` 3, `mb_blocks_per_stage` 2, `mb_mrf_kernel_sizes` 3/5/7), which
confirms these values at tensor-shape level.

Implementation (`src/hifimobinet/training/piper_ir_module.py`; repository code, not re-designed):

- **Block (`ResBlockInverted`).** Pointwise expand → SnakeBeta → depthwise (kernel *k*) → SnakeBeta → pointwise project, with no activation after it.
  - One residual around the block.
  - Weight norm on all three convolutions.
  - `pw_project` is zero-initialised, so each block starts as the identity.
  - The pointwise expand is kept at expansion 1.
- **ParIR (`resblock` "mrf").** Per upsampling stage, three branches with kernels 3/5/7 run in parallel on the same input; their outputs are summed and divided by 3.
- **SeqIR (`resblock` "mb").** Per stage, two blocks with kernel 3 in sequence.
- **Generator.** SnakeBeta replaces LeakyReLU everywhere in the generator: before each upsampling layer and before `conv_post`. Upsampling 8/8/4, kernels 16/16/8, initial channels 256, `conv_post` without bias, `tanh` output.
- **MRD.** Three STFT resolutions ((512, 50, 240), (1024, 120, 600), (2048, 240, 1200)), UnivNet-style.
  - EdgeTTS's unchanged `training_step_g` adds `loss_gen_mrd` and the MRD feature loss (weight 1).
  - `training_step_d` adds `loss_disc_mrd`.
  - `configure_optimizers` puts the MRD into the discriminator optimizer.
  - These are the terms and weights of the IR runs. The LSGAN formulas of the two code bases are identical.

**Flags actually used:**
- `use_vits2`, `use_bigvgan` and `use_f0` stay false.
  - `use_bigvgan` is not used because it would build EdgeTTS's own Snake ResBlock2 and EdgeTTS's MRD, not the released IR implementation.
- The IR settings are separate config keys that only the IR models accept.
- `use_mrd: true` is explicit and required.
- The Config A model is unchanged; its tests still pass.

**Initialization:**
- EdgeTTS first builds its Config A synthesizer and MPD; then the IR decoder replaces the ResBlock2 decoder, and the MRD is added.
- Under seed 1234, every non-decoder initial weight (text encoder, posterior encoder, flow, SDP and MPD) therefore equals Config A's initial weights (tested).

## 3. Config diff: ConfigA → ParIR → SeqIR

All other keys are identical; this is tested against the Config A file.

| Item | Piper-ConfigA | Piper-ParIR | Piper-SeqIR |
|---|---|---|---|
| `resblock` | "2" (HiFi-GAN ResBlock2) | "mrf" | "mb" |
| Residual blocks per stage | 3 parallel ResBlock2 (k 3/5/7, dilations [1,2],[2,6],[3,12]), averaged | 3 parallel IR branches (dw k 3/5/7), averaged | 2 sequential IR blocks (dw k 3) |
| Expansion | — | 1, 1, 1 | 1, 1, 1 |
| Generator activation | LeakyReLU 0.1 | SnakeBeta | SnakeBeta |
| Discriminators | MPD | MPD + MRD | MPD + MRD |
| Adversarial / FM loss terms | MPD | MPD + MRD (weight 1) | MPD + MRD (weight 1) |
| Decoder parameters (weight-norm form) | 1,664,768 | 1,157,024 | 1,109,760 |
| Generator without decoder | 21,999,024 | 21,999,024 | 21,999,024 |
| MPD / MRD parameters | 46,747,132 / — | 46,747,132 / 280,902 | 46,747,132 / 280,902 |
| Everything else in section 1 | Config A | same | same |

## 4. What the comparisons are and are not

- **ConfigA vs ParIR/SeqIR is a decoder-bundle comparison, not a topology-only comparison.**
  - The IR runs change the residual blocks and the generator activation (SnakeBeta).
  - They also add the MRD discriminator and its loss terms.
  - These cannot be separated in these runs.
- **ParIR vs SeqIR is not topology-only either.** Besides parallel versus sequential arrangement, the number of blocks per stage (3 vs 2) and the depthwise kernels (3/5/7 vs 3) differ.
- **Each model has one training run with seed 1234.** Sentence-level tests later will not cover training variation.
- **These runs differ from the released Parallel-IR / Sequential-IR runs.** The backbone and code base are different: Config A without VITS2 components and EdgeTTS manual optimization, versus the BanhmiTTS VITS2 baseline.

## 5. Rules for this round

- **Nothing outside the config is changed:** from scratch, the same seed, no sweep of blocks, kernels or expansion.
- **No selection on held-out data:** Harvard-720 and the test split are not used for selection or configuration.
- **One run at a time:** the two runs never train at the same time. A controller runs them one after the other (`docs/training/piper-ir-runbook.md`).
- **Evaluation is a separate step:** Harvard, MOS, ONNX/PTQ and edge benchmarks are requested separately later.
