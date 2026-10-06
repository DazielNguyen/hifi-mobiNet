# Piper reference audit and component comparison

Scope: training recipes added on 2026-10-06. The released checkpoints
(`baseline-resblock2`, `parallel-ir`, `sequential-ir`, `piper-original`) are not
retrained, renamed or overwritten.

| Training ID | Meaning |
|---|---|
| `baseline-resblock2-vits2` | Internal baseline recipe: HiFi-GAN ResBlock2 decoder plus the project's VITS2 components, run with the current BanhmiTTS training code |
| `Piper_no_VITS2_cpn` | New model: the verified upstream Piper VITS training graph without VITS2 components, trained from scratch in this project's harness |
| `piper-original` | Published Piper lj-medium checkpoint (external, unchanged; not produced here) |

## 1. Upstream identity

| Check | Result |
|---|---|
| Repository | `https://github.com/rhasspy/piper` (local remote `origin`) |
| Commit | `73c04d81d5590ecc46e522de3601ce7fb29fc2be`, 2025-08-26, "Add test sentences" |
| Upstream status (GitHub API, read 2026-10-06) | Repository archived; `master` and `HEAD` both point to this commit, so it is the final upstream revision |
| Local object | `git cat-file -t` = commit; the Windows clone's working tree is dirty only in the generated `monotonic_align/core.c` and an untracked build directory (neither imported) |
| Second clone (WSL) | Same HEAD, clean; blob IDs of `vits/models.py` and `__main__.py` equal the Windows clone's |
| Vendored files | All 13 `vendor/piper/vits/*.py` blobs equal `git rev-parse 73c04d8:<path>`; 7 further files imported in this change come from Git objects at the same commit (`docs/source-map.json`, entries with `source_file_state: git_object_at_pinned_commit`) |
| License | MIT, "Copyright (c) 2022 Michael Hansen" (`licenses/Piper-MIT.txt`) |

Limits: this proves which upstream revision is vendored. It does not prove that
the published `piper-original` checkpoint was trained with this revision, nor
that BanhmiTTS derives from it (the author reports an independent
re-implementation).

## 2. Component table

Locators: **B** = `src/hifimobinet/architecture/vits/` (relocated BanhmiTTS model
code, output-verified against `vendor/banhmi/`), **BT** =
`vendor/banhmi/vits/training.py`, **P** = `vendor/piper/vits/` (upstream
73c04d8), **PT** = `vendor/piper/vits/lightning.py`.

| Component | Baseline (VITS2) | Piper reference | Piper_no_VITS2_cpn | Evidence | Action | Limit |
|---|---|---|---|---|---|---|
| Text encoder | Relative-position multi-head self-attention, 6 layers, 2 heads, FFN 768, 192 ch | Same structure | Upstream, unchanged | B `modules/text_encoder.py:35`, `utils/attention.py:19,71`; P `models.py:168-210`, `attentions.py` | Keep attention (not a VITS2 addition) | B is refactored code, not byte-identical to P |
| Posterior encoder | WN, 16 layers, kernel 5 (configurable) | WN, 16 layers, kernel 5 (hard-coded) | Upstream | B `modules/posterior_encoder.py`; P `models.py:592-600` | Keep | — |
| Flow coupling | `TransformerCouplingLayer`: 1-layer attention encoder added before WN (VITS2) | `ResidualCouplingLayer`: WN only | Upstream WN coupling, no attention | B `modules/flow_block.py:14-74` (attention 47-49, 60); P `models.py:212-255`, `modules.py:420-460` | **Removed (VITS2)** | — |
| Flow WN / Flip / count | WN 4 layers, mean-only, 4 couplings + Flip | Same | Upstream | B `flow_block.py:80-116`; P `models.py:232-245` | Keep | — |
| Stochastic duration predictor | SDP, filter 192, 4 flows, dropout 0.5; extra numerical guards (log clamp, logs clamp, spline eps) | SDP, same sizes, no guards | Upstream SDP, no guards | B `modules/duration_predictor.py:19-55`, `utils/flows.py:27-42,74`, `utils/transforms.py:155-238`; P `models.py:14-118`, `transforms.py` | **Keep SDP** (not removed with the duration discriminator) | Guards are BanhmiTTS additions, not VITS2; absent in Piper |
| SDP reverse sample during training | Extra `dp(reverse=True)` each step to feed the duration discriminator | Not called | Not called | B `synthesizer.py:172` | Removed with the duration discriminator | — |
| Duration discriminator | Present; its adversarial loss is added to the generator loss; its parameters are in the discriminator optimizer | Absent | Absent | B `modules/duration_discriminator.py`; BT `584-587`, `609-614`, `710`; smoke check `no_vits2_modules_or_optimizer_groups` | **Removed (VITS2)** incl. loss terms and optimizer group | — |
| MAS | Cython MAS on negative cross-entropy | Same algorithm (refactored Cython source) | Upstream `core.pyx` built as upstream lays it out | B `synthesizer.py:222-246`; P `models.py:628-650`, `monotonic_align/` | Keep, upstream build | Built with Cython 3.2.9, `language_level=3`; upstream pinned Cython < 1 |
| Noise-scaled MAS | Noise `std(neg_cent) * scale`, scale 0.01 decaying 2e-6 per global step to 0 | Absent | Absent; `forward()` has no noise argument | B `synthesizer.py:242-243`; BT `498-500` | **Removed (VITS2)** | — |
| Decoder (ResBlock2) | 3 stages 8/8/4, kernels 3/5/7, dilations (1,2),(2,6),(3,12); branch **average** | Same; branch average | Upstream | B `modules/generator.py:163-166`; P `models.py:356-363` | Keep | — |
| Upsampling | ConvTranspose 16/16/8, 256 initial channels, weight norm | Same | Upstream | B `generator.py`; P `models.py:320-332` | Keep | — |
| Activations | LeakyReLU 0.1 everywhere, including before `conv_post` | LeakyReLU 0.1, but **0.01** before `conv_post` | Upstream (0.01) | B `utils/normalization.py:31`, `generator.py:142,168`; P `models.py:354,364` | Keep upstream | Known difference (J-E018) |
| Waveform discriminators | MPD (periods 2,3,5,7,11) + scale discriminator; MRD never built for ResBlock2 | Same MPD + DiscriminatorS | Upstream | B `modules/discriminators.py:97-108`; P `models.py:495-520` | Keep | — |
| Generator loss | gen + fm + 45·mel + dur + kl (+ duration-adversarial) | gen + fm + 45·mel + dur + kl | Upstream | BT `549-590`; PT `251-263` | Duration-adversarial term removed | — |
| Discriminator loss | LSGAN MPD (+ duration discriminator) | LSGAN MPD | Upstream | BT `592-617`; PT `265-280` | Duration term removed | — |
| Optimizers / schedulers | AdamW(2e-4, 0.8/0.99, 1e-9) ×2, `fused` on CUDA; ExponentialLR 0.999875 per epoch | AdamW ×2 (not fused); ExponentialLR | Upstream `configure_optimizers` | BT `697-728`; PT `308-332` | Keep upstream | Fused vs non-fused AdamW kernel differs numerically, not mathematically |
| Precision | bf16 (STFT forced to fp32 inside) | fp32 (published checkpoint metadata: 32) | **fp32** | B `vendor/banhmi/mel_processing.py:37`; P `mel_processing.py:120`; probe: bf16 raises `cuFFT doesn't support tensor of type: BFloat16` | fp32 | Unavoidable difference without editing upstream code |

## 3. What the new model is

- **Faithful part:** model graph, forward/infer, losses, generator/discriminator
  steps and optimizer/scheduler construction are the upstream objects, imported
  unmodified (`PiperNoVits2Module` subclasses upstream `VitsModel`). A unit test
  checks that the harness's `training_step_g` returns a bit-identical loss to
  upstream's for the same input and RNG seed.
- **Adaptation to the project:** dataset, phonemizer and split, dataloaders,
  validation metric, checkpoint selection and logging (section 4).
- **Ablation reading:** relative to the baseline it removes the whole VITS2
  package (Transformer flow, duration discriminator, noise-scaled MAS) at once,
  and also differs in BanhmiTTS-specific additions (SDP numerical guards,
  final LeakyReLU slope, fused AdamW) and in precision. It is **not** a clean
  single-component ablation; no per-component effect can be attributed.

## 4. Adapter diff

`src/hifimobinet/training/piper_module.py` (over upstream `VitsModel`):

| Upstream behaviour | Harness behaviour | Why |
|---|---|---|
| `random_split` of the dataset by `validation_split` and `num_test_examples` with the global RNG | Explicit split file (canonical 12,500 / 100 / 500), test partition never loaded | Same split as the baseline; no test leakage |
| Train `DataLoader` without shuffling | `length_bucket` sampler (BanhmiTTS) in the matched config; `upstream_sequential` available | Match the baseline harness |
| `validation_step` logs `val_loss` and synthesizes the 5 held-out "test" utterances | Same `val_loss`, plus `val_loss_mel`; audio from validation utterances | Checkpoint selection identical to the baseline; held-out test untouched |
| `training_step_g` | Identical arithmetic; adds per-term `self.log` and keeps `loss_mel` | Needed for `val_loss_mel` |
| `optimizer_step` (Lightning default) | Optional BanhmiTTS non-finite skip (config) | Harness parity |
| Lightning 1.7 rejects torch-2 `ExponentialLR` | `lr_scheduler_step` overridden with Lightning's own default body | Version compatibility (section 5) |
| No RNG in checkpoints | Python/NumPy/torch/CUDA RNG saved as tensors; restored single-process only | Resume fidelity |
| `--checkpoint-epochs` → `ModelCheckpoint(every_n_epochs)` | Top-3 by `val_loss_mel` + `last.ckpt` in a fixed directory | Same rule as the baseline |

`src/hifimobinet/training/baseline_module.py` (over `vendor/banhmi/vits/training.py`):
model code from the relocated package; Vocos/F0/MRD branches removed and rejected;
datasets injected instead of loaded in `__init__` (no random-split fallback);
length cache written to the run directory, not next to the source data;
`loss_f0`/`loss_gen_mrd` logs dropped (constant zero / never built); non-finite
skip and health gate made explicit config switches (both on); RNG state option;
scheduler hook as above.

Data path (both models): the project's preprocessed LJSpeech (`dataset.jsonl`
SHA-256 `f4a72ae0…295b`, 13,100 rows), `banhmi_phonemize` en-us IDs (256-symbol
table) and cached linear spectrograms computed like Piper's
(`n_fft` 1024, hop 256, Hann, reflect padding, `center=False`). Unlike upstream
Piper preprocessing there is **no Silero VAD trimming**. All 13,100 stored ID
sequences were reproduced exactly by re-phonemizing the texts with the original
frontend (default casing). Piper's own `piper-phonemize` ID table was not
compared; the new model learns its embeddings from these IDs, so it does not
claim Piper's frontend.

## 5. Training-environment findings

- `requirements.txt` at 73c04d8 pins `torch>=1.11,<2` and `pytorch-lightning~=1.7.0`.
  The harness uses torch 2.13.0 with Lightning 1.7.7 (versions of the existing
  workstation interpreter). With these, Lightning rejects `ExponentialLR`
  (`isinstance(..., _LRScheduler)` is false since torch 2.0). The existing
  BanhmiTTS interpreter has the same versions and the same failure, so the
  current BanhmiTTS training code cannot run unchanged there; the historical
  training environment must have differed (version not recorded).
- With torch ≥ 2.6, Lightning 1.7 resumes through `torch.load(weights_only=True)`.
  Checkpoints written by this harness contain only tensors and plain values and
  load that way (smoke check `checkpoint_loads_weights_only`).
