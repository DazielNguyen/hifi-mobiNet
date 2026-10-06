# Training smoke report (2026-10-06)

Functional checks of the training paths. **Not research results**: the runs
use 16 utterances and at most 24 optimizer updates per run; no quality judgement
is made from losses or WAVs. Commit-safe per-run data:
[training-smoke-results.json](training-smoke-results.json). Raw reports, logs,
checkpoints and WAVs stay in local run storage.

## Setup

| Item | Value |
|---|---|
| Host | Ubuntu 24.04.4 LTS on WSL2, RTX 4070 Ti (1 of 2 used), driver 596.36 |
| Environment | New venv: Python 3.10.20, torch 2.13.0+cu130, Lightning 1.7.7 ([environment-record.json](environment-record.json)) |
| Data | 16 canonical-train + 4 canonical-validation utterances copied into a new subset; **no test or Harvard item** |
| Smoke overrides | batch 4, 2 workers, 2 audio samples (recorded per run) |
| Procedure | Phase A trains from scratch and checkpoints; phase B resumes from `last.ckpt` and trains one more epoch; one PyTorch inference WAV |

## Results

| Check | baseline-resblock2-vits2 | Piper_no_VITS2_cpn (matched) | Piper_no_VITS2_cpn (upstream harness) |
|---|---|---|---|
| 1 Preprocessing / IDs | pass | pass | pass |
| 2 Batch shapes, lengths, padding (`spec_len == audio_len // 256`, zero padding) | pass | pass | pass |
| 3 Real forward/backward | pass | pass | pass |
| 4 Finite losses and gradients | pass | pass | pass |
| 5 Generator and waveform discriminator updated | pass (745/745, 111/111 tensors) | pass | pass |
| 6 Duration discriminator updated | pass (20/20 tensors) | not applicable | not applicable |
| 7 VITS2 present (baseline) / absent (new model): modules, flow attention, MAS-noise argument, losses, optimizer groups | pass | pass | pass |
| 8 Resume: model, optimizer, scheduler, global step, torch RNG | pass | pass | pass |
| 8b Checkpoint loads with `torch.load(weights_only=True)` | pass | pass | pass |
| 9 Validation leaves weights unchanged | pass | pass | pass |
| 10 PyTorch inference WAV written, finite | pass | pass (4.96 s) | pass |
| Optimizer updates in final run | 24 | 24 | 16 |

Phoneme IDs were additionally checked against the original frontend for all
13,100 dataset rows: 13,100 matched (Windows frontend build; default casing).
A unit test showed `PiperNoVits2Module.training_step_g` returns a bit-identical
loss to upstream `VitsModel.training_step_g` for the same batch and seed.

### Memory probe (no optimizer step; weights verified unchanged)

16 longest training utterances by phoneme count (395 IDs, 864 frames), one
generator and one discriminator forward/backward:

| Model / precision | Peak allocated | Result |
|---|---|---|
| Piper_no_VITS2_cpn, fp32, batch 16 | 5.78 GiB of 11.99 | ok |
| Piper_no_VITS2_cpn, bf16, batch 16 | — | `RuntimeError: cuFFT doesn't support tensor of type: BFloat16` at upstream `mel_processing.py:120` |
| baseline-resblock2-vits2, bf16, batch 16 | 6.93 GiB of 11.99 | ok |

## Failures found and fixed

| Run | Original error | Root cause | Fix |
|---|---|---|---|
| smoke-baseline-1 | `MisconfigurationException: The provided lr scheduler ExponentialLR doesn't follow PyTorch's LRScheduler API` (before any update) | Lightning 1.7.7 validates schedulers with `isinstance(s, torch.optim.lr_scheduler._LRScheduler)`; since torch 2.0 `ExponentialLR` derives from `LRScheduler`. The existing workstation interpreter (same versions) shows the same `False`. | Override `lr_scheduler_step` with Lightning 1.7.7's own default body; unit-tested |
| smoke-baseline-2 | Resume: `UnpicklingError: Weights only load failed ... numpy.core.multiarray._reconstruct` (phase A: 24 updates, all checks passed) | The harness stored NumPy's RNG state array in the checkpoint; Lightning 1.7 resumes through `torch.load`, whose default is `weights_only=True` on torch ≥ 2.6 | RNG state stored as tensors/plain values; static pickle scan confirmed NumPy was the only disallowed global; unit test for weights-only round trip |

Update accounting across all attempts: baseline 0 + 24 + 24 = 48; Piper
24 + 16 = 40 (limit 50 per model). Probes made no updates.

## Not checked

- Multi-GPU (DDP) training through `hifimobinet.training.train` — see the
  clean-clone record in [next-run.md](next-run.md) for the status of this check.
- RNG restore under DDP (deliberately not restored; rank 0's state only).
- CUDA/NumPy/Python RNG equality after resume (restored, only torch CPU compared).
- Baseline in fp32; Piper in bf16 (unsupported without modifying upstream).
- Linux build of the native frontend; preprocessing from raw LJSpeech audio.
- Long-run stability (SDP collapse, NaN rate), convergence, quality, speed.

## Pre-existing test status

`python -m pytest` on Windows: 42 passed, 1 skipped (training tests need PyYAML /
Lightning there), 2 failed in `test_huggingface_bootstrap.py` with
`WinError 1314` (no symlink privilege). The same two tests fail identically on
a clean clone of the previous HEAD, so they are an environment limitation, not a
regression. On Ubuntu: `tests/test_training.py` 18 passed.
