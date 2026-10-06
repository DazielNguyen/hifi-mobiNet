# Piper / EdgeTTS Config A audit and component comparison

Scope: training recipes added on 2026-10-06. The released checkpoints
(`baseline-resblock2`, `parallel-ir`, `sequential-ir`, `piper-original`) are not
retrained, renamed or overwritten.

| Training ID | Meaning |
|---|---|
| `baseline-resblock2-vits2` | Internal baseline recipe: HiFi-GAN ResBlock2 decoder plus the project's VITS2 components, current BanhmiTTS training code |
| `Piper_no_VITS2_cpn` | The hifi-mobiNet baseline **without** the VITS2 components: **EdgeTTS Config A** (vanilla Piper: BigVGAN, VITS2 and F0 flags all off), trained from scratch on the baseline's data and split with the SEQ/MRF training setup |
| `piper-original` | Published Piper lj-medium checkpoint (external, unchanged; not produced here) |

Purpose (author, 2026-10-06): `Piper_no_VITS2_cpn` is the reference for asking
whether adding the VITS2 components to Piper moves results in a positive or a
negative direction. Its place in the paper is not decided.

## 1. Source identity

| Check | Result |
|---|---|
| Code used | EdgeTTS repository (remote `MardeusVN/PitchFlowNet`), commit `a73a897`, working tree clean; 22 files vendored unchanged under `vendor/edgetts/` from Git objects (`docs/source-map.json`, `source_repo: EdgeTTS`) |
| Fork base | EdgeTTS contains the full Piper history; its last upstream commit is rhasspy/piper `73c04d81d5590ecc46e522de3601ce7fb29fc2be` (2025-08-26), with 0 upstream commits between it and the fork. That commit is the archived rhasspy repository's final `master` (GitHub API, read 2026-10-06), not the newer `piper1-gpl` project, which is not used |
| EdgeTTS changes on top of 73c04d8 | 6 files, +801/−108 lines (`lightning.py`, `models.py`, `modules.py`, `__main__.py`, `dataset.py`, `transforms.py`); `attentions`, `commons`, `losses`, `mel_processing`, `monotonic_align` unchanged |
| Config A definition | `vendor/edgetts/configs/baseline-vits.yaml` ("vanilla Piper without any extensions"; all three flags false) and `docs/ablation_study_plan.md` variant 0 ("Baseline (vanilla VITS, 73c04d8)") |
| License | Piper MIT ("Copyright (c) 2022 Michael Hansen", `vendor/edgetts/LICENSE.md`); EdgeTTS changes owned by the project authors |

`vendor/piper/` (rhasspy 73c04d8) remains as the upstream reference for this
audit; it is no longer used for training.

### Historical EdgeTTS Config A run (read-only finding)

The workstation holds EdgeTTS's own Config A run (`02_Baseline_VanillaVITS`,
2026-07-09 → 07-13, 1,100 epochs, torch 2.12.1, Lightning 2.6.5, hparams:
bf16-mixed, 2 GPUs, batch 16, `grad_clip` 1.0, `max_phoneme_ids` 400, seed 1234,
all flags false). A static key scan of its checkpoints (no unpickling) finds
**36 SnakeBeta parameters inside the 9 decoder ResBlocks** (no flow attention,
no duration discriminator, no MRD). At that time EdgeTTS's ResBlocks always
built Snake activations; commit `8f07b63` (2026-07-25) added the `use_snake`
switch. So the historical "vanilla" run was not fully vanilla. The vendored
HEAD code builds LeakyReLU ResBlocks when `use_bigvgan` is false, which the
smoke test confirms (no Snake module in the model).

## 2. Component table

Locators: **B** = `src/hifimobinet/architecture/vits/` (relocated BanhmiTTS
model code, output-verified against `vendor/banhmi/`), **BT** =
`vendor/banhmi/vits/training.py`, **E** = `vendor/edgetts/vits/` (EdgeTTS
a73a897), **P** = `vendor/piper/vits/` (rhasspy 73c04d8, reference).

| Component | Baseline (VITS2) | Piper_no_VITS2_cpn (EdgeTTS Config A) | Evidence | Action | Limit |
|---|---|---|---|---|---|
| Text encoder | Relative-position MHA, 6 layers, 2 heads, FFN 768 | Same structure (Piper) | B `modules/text_encoder.py:35`; E `models.py:206` | Keep attention (not a VITS2 addition) | B refactored, not byte-identical |
| Posterior encoder | WN 16 layers, kernel 5 | Same (hard-coded) | B `modules/posterior_encoder.py`; E `models.py:395` | Keep | — |
| Flow coupling | `TransformerCouplingLayer` (attention before WN) | `ResidualCouplingLayer` (WN only) via `use_transformer_flows=False` | B `modules/flow_block.py:14-74`; E `models.py:358-373`, `modules.py:443` | **Removed (VITS2)** | — |
| Flow count / Flip | 4 couplings + Flip, WN 4 layers, mean-only | Same | B `flow_block.py:80-116`; E `models.py:333` | Keep | — |
| Stochastic duration predictor | SDP + BanhmiTTS guards (log/logs clamps, spline eps) | Piper SDP; EdgeTTS adds only the discriminant clamp | B `utils/flows.py:27-42,74`, `utils/transforms.py:155-238`; E `models.py:14`, `transforms.py:175` | **Keep SDP** | Guards differ (not VITS2) |
| Extra reverse SDP sample per step | Yes (feeds the duration discriminator) | Yes, still computed but unused (no duration discriminator) | B `synthesizer.py:172`; E `models.py:1040` | Kept as EdgeTTS does | Consumes RNG only; no loss term |
| Duration discriminator | Present; loss in G and D; parameters in D optimizer | `model_d_dur = None`; logged duration losses are exactly 0 | B `modules/duration_discriminator.py`, BT `584-587,609-614,710`; E `lightning.py:143-153` | **Removed (VITS2)** | — |
| MAS | Cython MAS | Same algorithm (Piper source) | B `synthesizer.py:222-246`; E `monotonic_align/` | Keep | Built with Cython 3.2.9 |
| Noise-scaled MAS | Scale 0.01, −2e-6 per global step | `use_noised_mas=False` | B `synthesizer.py:242-243`; E `models.py:1011-1014` | **Removed (VITS2)** | — |
| Decoder ResBlock2 / upsampling | 8/8/4, k 3/5/7, branch average | Same | B `generator.py:163-166`; E `models.py:526` | Keep | — |
| Activations | LeakyReLU 0.1 incl. before `conv_post` | LeakyReLU 0.1 incl. before `conv_post` (EdgeTTS; Piper 73c04d8 used 0.01 there) | B `normalization.py:31`; E `models.py:452,517,530`; P `models.py:364` | Keep | Matches baseline |
| Snake / MRD / F0 | Not used | Not built (`use_bigvgan`, `use_f0` false) | E `lightning.py:101-153` | Off | — |
| Waveform discriminators | MPD + DiscriminatorS | Same | B `discriminators.py:97`; E `models.py:661` | Keep | — |
| Losses | gen + fm + 45·mel + dur + kl (+ duration-adversarial) | gen + fm + 45·mel + dur + kl (component terms 0) | BT `549-617`; E `lightning.py:271-415` | Duration-adversarial term absent | — |
| Optimization | Lightning automatic optimization, 2 AdamW (fused on CUDA); released run: clip null then 1.0 | Manual optimization: G step then D step, 2 AdamW (not fused); `grad_clip` null here, as in SEQ/MRF (EdgeTTS's Config A run: 1.0) | BT `697-728`; E `lightning.py:94,240-255,455-485` | Keep EdgeTTS | Same math; framework path differs |
| LR schedule | ExponentialLR 0.999875 per epoch (Lightning-stepped) | Same, stepped in `on_train_epoch_end` | E `lightning.py:263-269` | Keep | — |
| Precision | bf16 (STFT in fp32) | bf16-mixed (mel/STFT outside autocast) | E `lightning.py:303,351,404` | bf16 | — |

## 3. What the new model is

- **Faithful to EdgeTTS Config A:** model, losses, manual-optimization training
  step, schedulers and optimizers are EdgeTTS's own objects
  (`PiperNoVits2Module` subclasses EdgeTTS `VitsModel`; a test asserts those
  methods are inherited unchanged). Config values follow Config A.
- **Adaptation to the project:** dataset/phonemizer/split shared with the
  baseline, batches without F0, length-bucket sampler, validation logging with
  `sync_dist`, audio examples from validation utterances, RNG in checkpoints,
  Lightning 1.7.7 instead of 2.6.5 (section 5).
- **Ablation reading:** against the baseline, the VITS2 package (Transformer
  flow, duration discriminator, noise-scaled MAS) is removed as a whole, but
  the two models also come from different codebases (SDP guards, AdamW kernel,
  automatic vs manual optimization). It is **not** a clean single-component
  ablation. The cleanest test of the VITS2 package inside one codebase would be
  EdgeTTS Config A versus EdgeTTS Config C (`use_vits2=true` only,
  `vendor/edgetts/configs/03_Config_C_VITS2.yaml`) trained identically; that
  pair is not set up here (author decision).

## 4. Adapter diff (`src/hifimobinet/training/piper_module.py`)

| EdgeTTS behaviour | Harness behaviour | Why |
|---|---|---|
| `random_split(seed)` into train/test/val | Canonical split file (12,500 / 100 / 500), test never loaded | Same split as the baseline |
| `PiperDataset` requires `audio_f0_path` and loads F0 for every row | Rows read like the baseline; `Batch.f0s=None` | Config A never reads F0; tested |
| Train loader in fixed order | Length-bucket sampler (`upstream_sequential` reproduces EdgeTTS) | Same sampler as the baseline |
| `validation_step` logs `val_loss`/`val_loss_mel` per rank; audio from 5 test utterances | Same values with `sync_dist=True`; audio from validation utterances | Consistent DDP selection; test untouched |
| `ModelCheckpoint`: best (top-1 by `val_loss_mel`) + rolling last | Top-3 by `val_loss_mel` + `last.ckpt` (baseline rule) | Best + last as decided; extra two kept for audit |
| Lightning 2.6.5 | Lightning 1.7.7 with `lr_scheduler_step` hook (never called in manual mode, needed for Lightning 1.7's scheduler check) | One environment for both models |
| No RNG in checkpoints | RNG saved as tensors | Resume fidelity |

## 5. Environment findings

- EdgeTTS pins `pytorch-lightning>=2.5,<3` and `torch>=2.1`; its Config A ran on
  torch 2.12.1 / Lightning 2.6.5. The harness runs EdgeTTS's LightningModule on
  torch 2.13.0 / Lightning 1.7.7, the same environment as the baseline. Manual
  optimization APIs used by EdgeTTS (`optimizers()`, `manual_backward`,
  `clip_gradients`, `lr_schedulers()`) exist in Lightning 1.7.7; global steps
  count both optimizer steps in both versions (EdgeTTS run: step 860,200 after
  1,100 epochs of 391 batches).
- Lightning 1.7.7 rejects torch-2 `ExponentialLR` unless `lr_scheduler_step` is
  overridden; the harness overrides it with Lightning's own default body. The
  existing BanhmiTTS interpreter has the same versions and the same failure.
- Checkpoints written by the harness load with `torch.load(weights_only=True)`.
