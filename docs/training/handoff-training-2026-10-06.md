# Handoff: training harness and Piper_no_VITS2_cpn (2026-10-06)

For continuing on the Mac and for the journal audit. Everything here is local:
no push, upload, export, quantization or long training was performed.

## 1. What exists now

| Item | Location |
|---|---|
| Source audit (EdgeTTS Config A, Piper base), component table, adapter diff | [piper-component-comparison.md](piper-component-comparison.md) |
| Ubuntu setup from a clean clone | [training-setup-ubuntu.md](training-setup-ubuntu.md) |
| Comparison protocol (settings fixed by the author) | [piper-no-vits2-protocol.md](piper-no-vits2-protocol.md) |
| Smoke results (pass / fail / not checked) | [training-smoke-report.md](training-smoke-report.md), [training-smoke-results.json](training-smoke-results.json) |
| Long-run commands | [next-run.md](next-run.md) |
| Observed environment | [environment-record.json](environment-record.json) |
| Provenance of new code | [source-provenance.json](source-provenance.json); vendored files in `../source-map.json` |
| Code | `src/hifimobinet/training/`, `configs/training/`, `scripts/training/`, `requirements/`, `tests/test_training.py`, `vendor/edgetts/` |

Training clone on the Ubuntu workstation (Linux filesystem): all code, the
staged dataset (`data/ljspeech-medium`, 26,203 verified files, 22.85 GB), the
smoke subset (`data/smoke-ljs-16train-4val`) and runs (`training_output/`)
live inside it; data and runs are Git-ignored. The venv sits next to it under
`$HIFI_TRAIN_ROOT`. Older smoke runs and the clean-clone check remain in local
storage outside the clone. Exact host paths are intentionally omitted.

## 2. The new model and how it differs from the baseline

`Piper_no_VITS2_cpn` is the hifi-mobiNet baseline **without** the VITS2
components: **EdgeTTS Config A** (EdgeTTS's Piper fork a73a897, based on
rhasspy/piper 73c04d8, with `use_bigvgan`, `use_vits2` and `use_f0` off). The
released `baseline-resblock2` is Piper **with** the VITS2 components and is not
retrained. Only the model without VITS2 is trained, with the BanhmiTTS SEQ/MRF
setup (1,500 epochs, 2 GPUs × 16, bf16, seed 1234, no clipping, length-bucket
sampler, canonical split, best `val_loss_mel` + last). Purpose: see whether the
VITS2 components make the model better or worse; paper placement open.

Absent compared with the baseline: Transformer-conditioned flow couplings, the
duration discriminator (its losses and optimizer parameters) and noise-scaled
MAS. Shared: text-encoder attention, stochastic duration predictor, MAS,
ResBlock2 decoder with LeakyReLU 0.1, MPD, losses otherwise, bf16. Clipping:
none for the new run (SEQ/MRF), 1.0 in the released baseline's later phases.
Also different, not VITS2: codebase (BanhmiTTS SDP guards vs EdgeTTS's single
discriminant clamp; fused vs plain AdamW; automatic vs manual optimization;
the baseline's non-finite skip and health gate are absent in EdgeTTS). It is a
system-level comparison, not a clean single-component ablation.

## 3. Fidelity

Faithful to EdgeTTS Config A: model, losses, manual-optimization training
step with `grad_clip` 1.0, per-epoch schedulers and optimizers are inherited
unchanged from the vendored EdgeTTS `VitsModel` (test-asserted). Adapted:
shared canonical split instead of `random_split`, no F0 loading (Config A never
uses it), length-bucket sampler, `sync_dist` validation logging, audio examples
from validation utterances, top-3 + last checkpoints, Lightning 1.7.7 instead of
2.6.5, RNG in checkpoints.

Finding: EdgeTTS's own historical Config A run (July 2026) contains SnakeBeta
parameters in all decoder ResBlocks (static key scan), because EdgeTTS's
ResBlocks lacked a `use_snake` switch until 2026-07-25. That run is therefore
not vanilla; the vendored current code is.

## 4. Defects found and fixed

1. Lightning 1.7.7 rejects torch-2 `ExponentialLR` → `lr_scheduler_step` with
   Lightning's default body. The existing BanhmiTTS interpreter has the same
   versions, so the current BanhmiTTS training code cannot run unchanged there.
2. NumPy RNG state blocked `weights_only` resume on torch ≥ 2.6 → tensors only.
3. DDP: rank ≥ 1 exited on the non-empty run-directory guard and rank 0 hung →
   launcher-only guard and records.
4. EdgeTTS's dataset requires an F0 path per row although Config A never reads
   F0 → rows read like the baseline, `f0s=None`.
5. Setup notes: pip 24.0 for Lightning 1.7 metadata; background jobs in
   `wsl -e bash -c` die with the session; the Windows checkout's `.git` is owned
   by another Windows account (per-command `safe.directory`).

## 5. Checks run

- GPU smoke (1 GPU, bf16): baseline and Config A passed every functional check.
- Memory, batch 16, bf16: Config A 5.47 GiB, baseline 6.93 GiB (of 12 GiB).
- Tests: `tests/test_training.py` 24 passed on Ubuntu.
- Final config smoke inside the training clone (`24ba622`, staged data): all checks passed.
- Zero-update dry run of the long-run command on the staged full dataset: ok.
  Windows full suite: 2 pre-existing symlink-privilege failures, unrelated.
- Clean clone on the Linux filesystem: setup script succeeded, package set equal
  to the smoke venv, working tree clean after building both MAS extensions.

## 6. Clean-clone multi-GPU check

`python -m hifimobinet.training.train ... --accelerator gpu --devices 2
--strategy ddp --max_epochs 1 --limit_train_batches 1 --limit_val_batches 1`
on the smoke subset, from the clean clone:

| Model | Commit | Ranks | Fit | `global_step` | Checkpoints | Run record |
|---|---|---|---|---|---|---|
| baseline-resblock2-vits2 (bf16) | `9fc94d9` | 2/2 | completed | 2 | best + last, weights-only load ok | clean tree |
| Piper_no_VITS2_cpn, EdgeTTS Config A (bf16, clip 1.0, flags off) | `0709f1c` | 2/2 | completed | 2 | best + last, weights-only load ok | clean tree |

The first baseline attempt (at `62c7d45`) hung: defect 3. Not checked under
DDP: resume, RNG across ranks (by design not restored), throughput, long-run
NCCL stability.

## 7. Decisions

Settled by the author on 2026-10-06: compare against the released baseline
only (no Config C, no baseline retrain); SEQ/MRF training setup; length-bucket
sampler; everything inside hifi-mobiNet. Open: where the comparison appears in
the paper. Next step: start the long run with the command in
[next-run.md](next-run.md).

## 8. Commits (local, not pushed)

| Commit | Content |
|---|---|
| `5dd763b` | Upstream audit, unmodified training reference imports, component table |
| `536bf0b` | Training setup, entry point, baseline recipe |
| `20747b4` | First Piper_no_VITS2_cpn (rhasspy 73c04d8 subclass, superseded) |
| `62c7d45` | Smoke runner, tests, records, next-run guidance |
| `9fc94d9` | DDP launcher fix found by the clean-clone check |
| `4064626` | First clean-clone record and handoff |
| `0709f1c` | Piper_no_VITS2_cpn rebased on EdgeTTS Config A (vendored EdgeTTS) |
| `1014c62` | Documentation for EdgeTTS Config A |
| `24ba622` | SEQ/MRF setup, in-repo dataset staging, runs in ignored `training_output/` |
| (this commit) | Final smoke/staging records, handoff, release manifest refresh |
