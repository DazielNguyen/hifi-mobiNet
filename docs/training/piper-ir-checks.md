# Pre-run checks: Piper-ParIR and Piper-SeqIR (2026-10-11)

These are functional checks run before the long runs. **They are not research results.**

Setup:
- training clone on Ubuntu 24.04 / WSL2;
- the Piper_no_VITS2_cpn venv (Python 3.10.20, torch 2.13.0+cu130, Lightning 1.7.7);
- 2× RTX 4070 Ti.

Raw reports, logs and checkpoints stay in ignored local storage (`training_output/`).

## Unit tests

| File | Result | What it checks |
|---|---|---|
| `tests/test_training_ir.py` (commit `b1c3d48`) | 26 passed | Config rejections; module structure (VITS2/F0 absent, `use_noised_mas` false, IR blocks and kernels, pointwise expand at expansion 1, SnakeBeta only in the decoder, three-resolution MRD in the discriminator optimizer only); decoder state-dict names and shapes equal to the released IR decoders; non-decoder initial weights equal to Config A under seed 1234; one generator/discriminator loss with finite values, non-zero MRD terms and zero duration/F0 terms; output length 256 × frames; F0 guard; MAS-noise source gate; Config A module unchanged |
| `tests/test_training_ir_configs.py` (`018bdc8`) | 2 passed | Each shipped IR config equals the Config A file plus the decoder bundle |
| `tests/test_training.py` | 26 passed | Existing Config A / baseline tests, unchanged |
| `tests/test_run_sequence.py` (`6451b23`) | 12 passed | Controller order and skipping, stop after a failed run, no automatic restart, lost process becoming `needs_recovery`, live-orphan refusal, resume preconditions, checkpoint selection, atomic status |

Training-clone run at `018bdc8`: 54 passed. Windows (no `yaml` or MAS): the training tests are skipped. The two Windows symlink tests in `test_huggingface_bootstrap.py` fail as before (no symlink privilege).

## Functional smoke test

Run with `hifimobinet.training.smoke` at `018bdc8` (clean tree).
- Data: the 16-train / 4-validation smoke subset; 1 GPU.
- Settings: batch 4; phase A trains one epoch from scratch, phase B resumes for one more; 24 optimizer updates.

| Check | Piper-ParIR | Piper-SeqIR |
|---|---|---|
| Batch shapes, lengths, padding; no F0 | pass | pass |
| Forward/backward, finite losses and gradients | pass | pass |
| Optimizer coverage: generator / discriminators (MPD + MRD) | pass | pass |
| Parameters updated: generator / MPD / MRD tensors | 744/744, 111/111, 54/54 | 705/705, 111/111, 54/54 |
| VITS2 absent (no Transformer flow, duration discriminator or noised MAS; no F0) | pass | pass |
| IR bundle | `IRDecoder` "mrf", 3 × (3, 5, 7) at expansion 1, SnakeBeta only, MRD (512, 50, 240) / (1024, 120, 600) / (2048, 240, 1200) | `IRDecoder` "mb", 3 × (3, 3) at expansion 1, SnakeBeta only, same MRD |
| Component losses: duration and F0 terms 0, MRD terms non-zero, no MAS-noise schedule | pass (gen 2.79 / disc 2.74) | pass (2.57 / 2.44) |
| Validation leaves weights unchanged | pass | pass |
| Resume: model, optimizer, scheduler, global step, torch RNG | pass | pass |
| `last.ckpt` loads with `weights_only=True` | pass | pass |
| PyTorch inference WAV, finite | pass (4.48 s) | pass (4.47 s) |

## Memory probe

Settings:
- batch 16 of the 16 longest training utterances (395 IDs, 864 frames);
- bf16;
- one generator and one discriminator forward/backward, with no optimizer step and weights verified unchanged.

| Model | Peak allocated (of 11.99 GiB) |
|---|---|
| Piper_no_VITS2_cpn (Config A, 2026-10-06) | 5.47 GiB |
| Piper-ParIR | 7.44 GiB |
| Piper-SeqIR | 7.25 GiB |

## Two-GPU DDP through `train.py`, including resume

Run at `018bdc8` on the smoke subset:
- first `--devices 2 --strategy ddp --limit_train_batches 1 --limit_val_batches 1 --max_epochs 1`;
- then `--resume last.ckpt --max_epochs 2`.

For both models:
- Both runs exited with code 0, and both ranks initialised.
- After the resume, `last.ckpt` is at epoch 1, global step 4.
- `run_record.json` (commit `018bdc8`, clean) and a `resume_record_*.json` were written.
- The checkpoints hold the MRD (57 keys) and the IR decoder blocks.

This is the first DDP resume exercised in this repository. Under DDP, the RNG state is not restored, by design.

## Controller, end to end (`6451b23`)

The two runs used a temporary plan with the real plan's runs and expectations. Only the data (smoke subset with its own hashes) and the budget (1 batch per epoch) differed.

1. **Normal sequence** (2 epochs each):
   - Preflight passed. ParIR started from scratch and exited 0, then was validated and completed.
   - SeqIR then started by itself and completed.
   - The final training report and the selected-checkpoint manifests were written.
2. **Interruption and recovery** (12 epochs each):
   - The ParIR tmux session was killed after its first `last.ckpt`. `status` then showed `running` with no live process.
   - `start` marked ParIR `needs_recovery` and stopped without starting SeqIR.
   - `resume --run parir --note …` continued from `last.ckpt`, which wrote a `resume_record`. ParIR validated: epoch 11, global step 24 as expected, 12 validation points.
   - SeqIR then started by itself and completed.

## WSL instance lifetime (finding)

The WSL distro shuts down when no `wsl.exe` client is connected. On 2026-10-11, `/init` restarted between separate commands.

A tmux session alone therefore does not keep a run alive. The runbook requires a connected client: an open terminal, or a detached keep-alive process (`docs/training/piper-ir-runbook.md`). The Config A run lasted four days with a terminal attached.
