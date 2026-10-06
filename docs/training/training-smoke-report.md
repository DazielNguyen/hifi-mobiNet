# Training smoke report (2026-10-06)

Functional checks of the training paths. **Not research results**: the runs
use 16 utterances and at most 24 optimizer updates per run; no quality judgement
is made from losses or WAVs. Commit-safe per-run data:
[training-smoke-results.json](training-smoke-results.json). Raw reports, logs,
checkpoints and WAVs stay in local run storage.

## Setup

| Item | Value |
|---|---|
| Host | Ubuntu 24.04.4 LTS on WSL2, RTX 4070 Ti (1 of 2 used; 2 for the DDP check), driver 596.36 |
| Environment | New venv: Python 3.10.20, torch 2.13.0+cu130, Lightning 1.7.7 ([environment-record.json](environment-record.json)) |
| Data | 16 canonical-train + 4 canonical-validation utterances copied into a new subset; **no test or Harvard item**; no F0 |
| Smoke overrides | batch 4, 2 workers, 2 audio samples (recorded per run) |
| Procedure | Phase A trains one epoch from scratch and checkpoints; phase B resumes from `last.ckpt` for one more epoch; one PyTorch inference WAV |

## Results (current implementation)

| Check | baseline-resblock2-vits2 (bf16) | Piper_no_VITS2_cpn = EdgeTTS Config A (bf16) |
|---|---|---|
| 1 Preprocessing / IDs | pass | pass |
| 2 Batch shapes, lengths, padding; batches carry no F0 | pass | pass |
| 3 Real forward/backward | pass | pass |
| 4 Finite losses and gradients | pass | pass |
| 5 Generator and waveform discriminator updated | pass (745/745, 111/111 tensors) | pass (673/673, 111/111) |
| 6 Duration discriminator updated | pass (20/20 tensors) | not applicable |
| 7 VITS2 present (baseline) / absent (new model): Transformer flow, duration discriminator, noised MAS; also no Snake, MRD, F0 | pass | pass; component losses exactly 0, no MAS-noise schedule |
| 8 Resume: model, optimizer, scheduler, global step, torch RNG | pass | pass |
| 8b Checkpoint loads with `torch.load(weights_only=True)` | pass | pass |
| 9 Validation leaves weights unchanged | pass | pass |
| 10 PyTorch inference WAV written, finite | pass | pass (4.73 s) |
| Optimizer updates in the run | 24 | 24 |

Additional checks:

- Phoneme IDs: all 13,100 dataset rows reproduced by re-phonemizing with the
  original frontend (Windows build, default casing).
- Unit tests assert that `PiperNoVits2Module` inherits EdgeTTS's
  `training_step`, `training_step_g`, `training_step_d`, `configure_optimizers`,
  `on_train_epoch_end` and `forward` unchanged, that its batches equal EdgeTTS's
  own collate on every non-F0 field, and that Config A flags cannot be enabled.
- Two-GPU DDP through `hifimobinet.training.train` from a clean clone: both
  models initialised both ranks, completed one batch (global step 2), wrote
  weights-only-loadable checkpoints and a run record on a clean tree (handoff
  section 6).

### Memory probe (no optimizer step; weights verified unchanged)

16 longest training utterances by phoneme count (395 IDs, 864 frames), one
generator and one discriminator forward/backward, batch 16:

| Model / precision | Peak allocated (of 11.99 GiB) |
|---|---|
| Piper_no_VITS2_cpn (EdgeTTS Config A), bf16 | 5.47 GiB |
| baseline-resblock2-vits2, bf16 | 6.93 GiB |

## Failures found and fixed

| Run | Original error | Root cause | Fix |
|---|---|---|---|
| smoke-baseline-1 | `MisconfigurationException: The provided lr scheduler ExponentialLR doesn't follow PyTorch's LRScheduler API` (no update made) | Lightning 1.7.7 checks `isinstance(s, torch.optim.lr_scheduler._LRScheduler)`; since torch 2.0 `ExponentialLR` derives from `LRScheduler`. The existing workstation interpreter shows the same result | `lr_scheduler_step` override with Lightning 1.7.7's default body; unit-tested |
| smoke-baseline-2 | Resume: `UnpicklingError: Weights only load failed ... numpy.core.multiarray._reconstruct` (phase A: 24 updates, checks passed) | NumPy RNG state array in the checkpoint; Lightning 1.7 resumes with `torch.load`, default `weights_only=True` on torch ≥ 2.6 | RNG state as tensors/plain values; static pickle scan; round-trip test |
| clean-clone DDP (2 GPUs) | Rank 0 hung in process-group initialisation; rank 1 exited with "run-dir is not empty" (no update made) | Lightning 1.7 re-runs the script for ranks ≥ 1; the run-directory guard ran there after rank 0 had populated it | Guard and run records only in the launcher process; regression test |

## Superseded implementation

Earlier the same day `Piper_no_VITS2_cpn` subclassed rhasspy/piper 73c04d8
directly (fp32, since that code fails under bf16 with `cuFFT doesn't support
tensor of type: BFloat16`). It passed the same checks (24 + 16 smoke updates,
2 DDP updates) and was replaced on the author's request by EdgeTTS Config A,
which is the same Piper base plus EdgeTTS's bf16-safe training loop.

Update accounting: baseline 0 + 24 + 24 + 2 (DDP) = 50. Piper_no_VITS2_cpn:
EdgeTTS implementation 24 + 2 (DDP) = 26; superseded implementation 42. Probes
made no updates.

## Not checked

- DDP resume and long-run behaviour (stability, NaN rate, convergence, speed).
- RNG restore under DDP (deliberately not restored; rank 0's state only).
- CUDA/NumPy/Python RNG equality after resume (restored, only torch CPU compared).
- Linux build of the native frontend; preprocessing from raw LJSpeech audio.
- EdgeTTS code under its own Lightning 2.6.5 environment (not used here).

## Test status

Ubuntu: `tests/test_training.py` 23 passed (development venv and clean clone).
Windows full suite: training tests skip there (no PyYAML/Lightning); 2
pre-existing failures in `test_huggingface_bootstrap.py` (`WinError 1314`, no
symlink privilege) also fail on a clean clone of the earlier HEAD.
