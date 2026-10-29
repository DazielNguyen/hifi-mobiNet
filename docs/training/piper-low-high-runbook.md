# Piper Low and High: Low approved after reboot

Prepared on 2026-10-30 for comparison with `Piper_no_VITS2_cpn` (Medium)
and the existing ParIR/SeqIR experiments. This source document records the
approved setup before launch; live status belongs in the controller's
`training_output/piper-low-sequence-2026-10-30/status.json`.
The user has confirmed the reboot is complete and requested that all setup
changes be committed before training. Use the existing
`/home/capstone/hifimobinet-train/venv`; do not create another environment.
No boot-time task is configured.
The revised order is Low training on both GPUs (DDP), complete Low evaluation
including ONNX, and only then High. Do not train one model per GPU or queue
High immediately after Low training. Data resampling/STFT and tests of configuration, data integrity
and mocked controller behavior are preparation only; they do not run a model.

Initial setup check: **27 tests passed** in the WSL training environment.
CPU-only constructors also confirmed the Low/High decoder types and absence
of MRD, duration discriminator, Snake and VITS2 switches, without any forward,
backward or optimizer step. The fresh preflight snapshots pass data hashes,
environment, compiled MAS, GPU availability, disk and new-output checks;
`git_clean` was false in that preparation snapshot before the local commit.
Launch requires a fresh clean-tree preflight. High's actual training VRAM
requirement remains untested and is deferred until after Low's evaluation. See the
[setup verification record](piper-sizes-2026-10-30/setup-verification.json).

After reboot, the existing venv passed **29 tests** and a fresh full cache
verification (13,100 source waveforms, 26,200 derived tensors). The launcher
tests confirm that neither a successful nor a failed Low run starts High.
See [post-reboot checks](piper-sizes-2026-10-30/post-reboot-checks.json).

## What these two recipes mean

| Setting | Low, new | Medium, existing | High, new |
|---|---|---|---|
| Model ID | `Piper_Low_no_VITS2_cpn` | `Piper_no_VITS2_cpn` | `Piper_High_no_VITS2_cpn` |
| Sample rate | 16000 Hz | 22050 Hz | 22050 Hz |
| Inter / hidden / filter channels | 192 / 192 / 768 | 192 / 192 / 768 | 192 / 192 / 768 |
| Text encoder | 6 layers, 2 heads | same | same |
| Decoder | ResBlock2 | ResBlock2 | ResBlock1 |
| Decoder kernels | 3, 5, 7 | 3, 5, 7 | 3, 7, 11 |
| Decoder dilations | (1,2), (2,6), (3,12) | same | (1,3,5) for each kernel |
| Upsample rates | 8, 8, 4 | 8, 8, 4 | 8, 8, 2, 2 |
| Initial decoder channels | 256 | 256 | 512 |
| Upsample kernels | 16, 16, 8 | 16, 16, 8 | 16, 16, 4, 4 |
| Discriminator | MPD | MPD | MPD |
| VITS2 / BigVGAN / F0 / MRD / Snake | off | off | off |

Low is **not x-low**. Piper's official [voice quality definitions](https://rhasspy.github.io/piper-samples/)
and [training guide](https://github.com/rhasspy/piper/blob/master/TRAINING.md)
distinguish Low (16 kHz, Medium-sized network) from x-low (reduced network).
The vendored `vendor/piper/__main__.py` exposes `x-low`, `medium`, `high`:
Low uses the default network with 16 kHz data; x-low changes channels to
96 / 96 / 384. High's six decoder overrides above match that vendored CLI.
These are project Config A recipes built through the same EdgeTTS fork and
harness as the existing Medium, not claims of byte-identical upstream releases.

## Shared training and selection protocol

- From scratch, no external weights, no Medium checkpoint initialization.
- LJSpeech: the identical 13,100 utterances, texts and phoneme IDs, canonical
  train/validation/test membership; test data never enters training or selection.
- 1,500 epochs, 2 GPUs with DDP, bf16, batch 16 per GPU, seed 1234;
  one worker per loader, length buckets, no phoneme-length filtering.
- AdamW as inherited from EdgeTTS: learning rate 0.0002, betas 0.8/0.99,
  eps 1e-9, epoch decay 0.999875, mel loss weight 45, KL weight 1.
- No gradient clipping or skipped nonfinite updates. Generator then
  discriminator updates through the existing manual optimization implementation.
- Validate each epoch. Save the three best full checkpoints by `val_loss_mel`
  plus `last.ckpt`; select the minimum within that run, with the existing
  controller's validation rules. No early stopping or test-based selection.
- Separate run directories, controller state, logs, TensorBoard events,
  config hashes, run records and explicit resume records for each tier.

High at batch 16/GPU has **not** been verified to fit the two 12 GiB GPUs.
After Low training and its evaluation/ONNX work, establish High's feasibility
before committing to the High long run.
If it OOMs, retain the failed session and decide a revised protocol explicitly.
Do not silently reduce batch, truncate utterances, change precision or add
Lightning gradient accumulation: this harness uses manual optimization.

## Data preparation and identity

Training checkout: `/home/capstone/hifimobinet-train/hifi-mobiNet` in
`Ubuntu-24.04`. Python: `/home/capstone/hifimobinet-train/venv/bin/python`.

High reuses `data/ljspeech-medium`. Low has its own root:
`data/ljspeech-low-16000-2026-10-30`.
Inside either root the dataset directory is `training_output/ljspeech/medium`.
That historical relative name is deliberately retained so `dataset.jsonl`
and all canonical split keys stay byte-identical. **For Low, the directory
name does not describe its sample rate**: its `config.json` and tensor
provenance say 16000 Hz, and the Low plan selects the Low root explicitly.
The dataset loader rejects a config/sample-rate mismatch.

Low preparation reads each normalized mono float32 waveform tensor from the
verified Medium cache, preserving the existing trimming, gain and frontend.
It applies SciPy `resample_poly(320, 441, window=('kaiser', 5.0), padtype='constant')`
on CPU and recomputes the linear magnitude STFT using the vendored Banhmi
function: FFT/window 1024, hop 256, reflect padding 384 each side, center false.
No extra gain or clipping is applied; any FIR overshoot is counted in the record.
The raw WAV and F0 paths remain provenance fields, not new inputs to Config A.
Neither raw WAVs nor unused F0 tensors are duplicated.

The Low root contains:

- `PREPARATION_RECORD.json`: rates, method, versions, code hashes, split counts,
  length-bucket batch count and metadata hashes.
- `TENSOR_MANIFEST.jsonl`: source waveform hashes and output tensor paths,
  dimensions, byte counts and hashes for every utterance.
- `VERIFICATION.json`: a fresh full source/output hash verification after
  tensor-by-tensor readback checks.
- A new `.spectrogram_lengths_cache.json`; never use Medium's frame counts
  for Low. The controller budget is taken from Low's actual sampler count.

Preparation on 2026-10-30 completed for all 13,100 rows / 26,200 tensors,
with fresh source/output hashes verified. Small records are also preserved in
[`piper-sizes-2026-10-30/`](piper-sizes-2026-10-30/).
Low has **393** batches/epoch/rank: 589,500 generator updates, 589,500
discriminator updates and 1,179,000 Lightning global steps across 1,500 epochs.
High retains **392**: 588,000 updates per optimizer / 1,176,000 global steps.
Six resampled waveform values exceed unit amplitude (peak 1.1642507);
these are recorded and retained, not silently clipped or normalized.

Reproduction command (only for a new, empty destination; existing output is
refused rather than overwritten):

```bash
cd /home/capstone/hifimobinet-train/hifi-mobiNet
PY=/home/capstone/hifimobinet-train/venv/bin/python
"$PY" scripts/training/prepare_piper_low.py \
  --source-data-root data/ljspeech-medium \
  --output-data-root data/ljspeech-low-16000-2026-10-30 \
  --split results/manifests/canonical_split.json
```

To verify an existing cache without regenerating it, append `--verify-only`.
Any missing source/tensor, hash mismatch or split inconsistency fails the
check. An interrupted preparation retains `PREPARATION_PENDING.json`;
inspect it and use a new destination rather than deleting evidence.

## Launch after the user is ready following the reboot

The two exact controller plans are:

- `configs/training/sequences/piper-low-2026-10-30.json`
- `configs/training/sequences/piper-high-2026-10-30.json`

Before starting, commit the reviewed configuration/code locally and ensure
the Linux training checkout contains those exact changes with a clean Git
status. Keep the existing WSL training data and outputs. No push is needed.
The controller refuses a dirty checkout or mismatched data/environment.
The preparation record is evidence of preparation, not a successful runtime
preflight: GPU availability, disk and processes must be checked again at launch.

Safe preview (prints commands only, creates no controller state):

```bash
cd /home/capstone/hifimobinet-train/hifi-mobiNet
PY=/home/capstone/hifimobinet-train/venv/bin/python
"$PY" scripts/training/run_piper_sizes.py
```

**After explicit user approval**, from the Linux checkout inside a persistent
tmux session, with a Windows WSL keepalive as described in
[the existing runbook](piper-ir-runbook.md):

```bash
"$PY" scripts/training/run_piper_sizes.py --start --confirm-training
```

This launches **Low only, on both GPUs** and returns after its training
controller finishes. It does not launch High. Complete and inspect Low's
evaluation and ONNX results before explicitly using `--run high` with the
same start/confirmation flags. `--run both` is no longer supported.
The confirmation flag is an operator opt-in, not proof of conversational
approval. Never supply it on the user's behalf before that approval.

Outputs are isolated under `training_output/piper-{low,high}-no-vits2-cpn-2026-10-30`;
state is under `training_output/piper-{low,high}-sequence-2026-10-30`.
The existing controller keeps all sessions, stops after failures and never
resumes an interrupted training automatically.

```bash
# Read status; no model execution.
"$PY" scripts/training/run_sequence.py status \
  --plan configs/training/sequences/piper-low-2026-10-30.json

# Only after inspection and approval to resume:
"$PY" scripts/training/run_sequence.py resume \
  --plan configs/training/sequences/piper-low-2026-10-30.json \
  --run low --note 'describe interruption and recovery here'
```

For High, replace `low` with `high`. Resuming Low's single-run controller does
not launch High; launch High separately after Low validates. Preserve every
session's logs/events and resume records, including any limitations of DDP
RNG restoration already documented for the Medium/IR harness.

## How to compare with the current ablation

High versus Medium keeps sample rate, data tensors and training settings fixed
while changing the decoder preset. Low versus Medium is primarily a sample-rate
comparison: their parameter shapes are identical, not a smaller-network study.
Low has an 8 kHz Nyquist limit instead of 11.025 kHz. With the same 8192-sample
segment and 256-sample hop, Low uses 512 ms segments / 16 ms frames, versus
371.52 ms / 11.61 ms at 22.05 kHz. Length buckets, batch membership and possibly
steps per epoch change, so equal epochs do not imply identical optimization
trajectories or equal compute. Use the measured budget in each plan.

`val_loss_mel` selects within each run; do not rank Low versus Medium/High by
raw mel loss, because their mel frequency grids and time scales differ.
For later evaluation, use the same Harvard utterance IDs, frontend, synthesis
seed/noise settings and scorer versions as the current experiments. Preserve
native-rate audio and record any scorer-required resampling. For RTF, use each
file's actual duration/sample rate and identical hardware/thread settings.
Include WER/CER, UTMOS, listening results, parameter counts, artifact size and
latency only after measuring them. One seed supports a descriptive comparison,
not a variance estimate. ParIR/SeqIR also change activation/MRD, so they remain
decoder-bundle comparisons rather than isolated size changes.

The user has requested full Low evaluation including ONNX before High.
The training launcher does not perform those steps: they require a separate
evaluation stage after checkpoint selection, following the existing corrected
round-2 export/parity/scoring/timing protocol. Keep native 16 kHz audio and
record scorer resampling. No evaluation or export has run yet. Human listening
results must remain pending until actually collected; never mark them complete
from an automated report.
