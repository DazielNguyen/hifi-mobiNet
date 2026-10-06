# Handoff: training harness and Piper_no_VITS2_cpn (2026-10-06)

For continuing on the Mac and for the journal audit. Everything here is local:
no push, upload, export, quantization or long training was performed.

## 1. What exists now

| Item | Location |
|---|---|
| Upstream audit, component table, adapter diff | [piper-component-comparison.md](piper-component-comparison.md) |
| Ubuntu setup from a clean clone | [training-setup-ubuntu.md](training-setup-ubuntu.md) |
| Comparison protocol (draft, open decisions) | [piper-no-vits2-protocol.md](piper-no-vits2-protocol.md) |
| Smoke results (pass / fail / not checked) | [training-smoke-report.md](training-smoke-report.md), [training-smoke-results.json](training-smoke-results.json) |
| Long-run commands with placeholders | [next-run.md](next-run.md) |
| Observed environment | [environment-record.json](environment-record.json) |
| Provenance of new code | [source-provenance.json](source-provenance.json); vendored files in `../source-map.json` |
| Code | `src/hifimobinet/training/`, `configs/training/`, `scripts/training/`, `requirements/`, `tests/test_training.py` |

Local-only material on the Ubuntu workstation (not in Git; under
`$HIFI_TRAIN_ROOT`): the training venv, the 16+4-utterance smoke subset, raw
smoke reports/logs/checkpoints/WAVs, the memory-probe output, and the
clean-clone check (`clean-check/`). Exact host paths are intentionally omitted.

## 2. How the new model differs from the baseline

Removed with the VITS2 package: attention inside the flow couplings, the
duration discriminator (its losses and its parameters in the discriminator
optimizer, plus the extra reverse SDP sample), and noise-scaled MAS. Kept: text
encoder attention, the stochastic duration predictor, plain MAS, the ResBlock2
decoder and MPD. Also different, but not VITS2: no SDP numerical guards,
final LeakyReLU slope 0.01, non-fused AdamW, and fp32 instead of bf16. It is a
system-level comparison, not a single-component ablation.

## 3. Fidelity to upstream Piper

Faithful: model graph, forward/infer, losses, both training steps and
optimizer/scheduler construction are the unmodified upstream objects of
rhasspy/piper `73c04d8` (the archived repository's final commit); a unit test
confirms bit-identical generator loss against the upstream method. Adapted:
project dataset (no VAD trimming; `banhmi_phonemize` IDs, all 13,100 verified),
explicit canonical split instead of `random_split`, length-bucket sampler (or
upstream's unshuffled loader), `val_loss_mel` selection, optional harness
policies, Lightning 1.7 scheduler hook for torch 2, torch 2.13 instead of
upstream's `torch<2`.

## 4. Defects found and fixed

1. Lightning 1.7.7 rejects torch-2 `ExponentialLR` → `lr_scheduler_step` with
   Lightning's default body. The existing BanhmiTTS interpreter has the same
   versions, so the current BanhmiTTS training code cannot run unchanged there.
2. NumPy RNG state blocked `weights_only` resume on torch ≥ 2.6 → tensors only.
3. DDP: rank ≥ 1 exited on the non-empty run-directory guard and rank 0 hung →
   launcher-only guard and records (found by the clean-clone check).
4. Setup notes: pip must be 24.0 for Lightning 1.7 metadata; background jobs in
   `wsl -e bash -c` die with the session; the Windows checkout's `.git` is owned
   by another Windows account (use a per-command `safe.directory`).

## 5. Checks run

- GPU smoke (1 GPU): baseline and Piper_no_VITS2_cpn (two configs) passed all
  functional checks; update totals across attempts: baseline 48 (+2 in the DDP
  check), Piper 40 (+2).
- Memory probe: Piper fp32 batch 16 = 5.78 GiB, baseline bf16 batch 16 =
  6.93 GiB; Piper bf16 fails in cuFFT.
- Tests: `tests/test_training.py` 19 passed on Ubuntu (dev venv and clean clone).
  Windows full suite: 2 pre-existing symlink-privilege failures, unrelated.
- Clean clone on the Linux filesystem: `setup_ubuntu.sh` succeeded, package set
  identical to the smoke venv, working tree clean after building.
- Multi-GPU: see section 6.

## 6. Clean-clone multi-GPU check

From the clean clone at `9fc94d9`, `python -m hifimobinet.training.train` with
`--accelerator gpu --devices 2 --strategy ddp --max_epochs 1
--limit_train_batches 1 --limit_val_batches 1` on the smoke subset:

| Model | Ranks initialised | Fit | Checkpoints | `global_step` | Run record |
|---|---|---|---|---|---|
| baseline-resblock2-vits2 (bf16) | 2/2 | completed | best + last, `weights_only` load ok | 2 | commit `9fc94d9`, clean |
| Piper_no_VITS2_cpn (fp32) | 2/2 | completed | best + last, `weights_only` load ok | 2 | commit `9fc94d9`, clean |

The first attempt (at `62c7d45`) hung: defect 3 in section 4; no update was made.
Not checked under DDP: resume, RNG behaviour across ranks (by design not
restored), throughput, NCCL stability over long runs.

## 7. Open decisions (author)

1. Dataset: keep the existing preprocessed LJSpeech cache (recommended; IDs
   verified) or re-preprocess?
2. Train only Piper_no_VITS2_cpn, or retrain the baseline too for a controlled pair?
3. Budget, GPU count and checkpoint-selection rule (same for both models).
4. Precision: accept bf16 vs fp32, or run both in fp32?
5. Seeds: single seed 1234 or several?

## 8. Commits (local, not pushed)

| Commit | Content |
|---|---|
| `5dd763b` | Upstream audit, unmodified training reference imports, component table |
| `536bf0b` | Training setup, entry point, baseline recipe |
| `20747b4` | Piper_no_VITS2_cpn model, configs, protocol |
| `62c7d45` | Smoke runner, tests, records, next-run guidance |
| `9fc94d9` | DDP launcher fix found by the clean-clone check |
| (this commit) | Clean-clone record, handoff, release manifest refresh |
