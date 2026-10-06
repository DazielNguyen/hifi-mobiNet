# Next run: Piper_no_VITS2_cpn long training

`Piper_no_VITS2_cpn` is the hifi-mobiNet baseline **without** the VITS2
components (EdgeTTS Config A). The released `baseline-resblock2` is Piper
**with** the VITS2 components and is not retrained. Only the model without VITS2
is trained, with the setup of the BanhmiTTS SEQ/MRF runs: 1,500 epochs, 2 GPUs
DDP, batch 16 per GPU, bf16, seed 1234, no gradient clipping, canonical split,
checkpoint every epoch, best `val_loss_mel` + last.

Everything lives inside a hifi-mobiNet clone on the Linux filesystem: code,
the staged dataset (`data/`) and the run (`training_output/`). Both directories
are Git-ignored. Nothing below has been started.

## 1. One-time preparation (inside the clone)

```sh
export HIFI_TRAIN_ROOT=<directory that holds venv/>   # see training-setup-ubuntu.md
PY="$HIFI_TRAIN_ROOT/venv/bin/python"
cd <hifi-mobiNet clone>

# Copy the preprocessed dataset into data/ljspeech-medium (verified, read-only source).
"$PY" scripts/training/stage_dataset.py \
  --source-data-root <directory the original dataset.jsonl paths are relative to> \
  --source-dataset-dir <that directory>/training_output/ljspeech/medium \
  --name ljspeech-medium \
  --expect-dataset-sha256 f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b
```

## 2. Train

```sh
"$PY" -m hifimobinet.training.train \
  --model-id Piper_no_VITS2_cpn \
  --config configs/training/piper-no-vits2-cpn.yaml \
  --dataset-dir data/ljspeech-medium/training_output/ljspeech/medium \
  --data-root data/ljspeech-medium \
  --split results/manifests/canonical_split.json \
  --run-dir training_output/piper-no-vits2-cpn \
  --length-cache data/ljspeech-medium/training_output/ljspeech/medium/.spectrogram_lengths_cache.json \
  --accelerator gpu --devices 2 --strategy ddp \
  --max_epochs 1500 --checkpoint-epochs 1
```

Outputs in `training_output/piper-no-vits2-cpn/`: `run_record.json` (commit,
config/split/dataset hashes, versions), TensorBoard logs,
`checkpoints/best-epoch=<e>-val_loss_mel=<v>.ckpt` (top 3) and
`checkpoints/last.ckpt`. Evaluate the lowest-`val_loss_mel` checkpoint.

## 3. Resume

```sh
"$PY" -m hifimobinet.training.train <same arguments> \
  --resume training_output/piper-no-vits2-cpn/checkpoints/last.ckpt
```

A `resume_record_<time>.json` is added. Under DDP the RNG state is not restored
(rank 0's state only); DDP resume itself has not been exercised yet.

## 4. Before evaluating

1. Record the selected checkpoint's SHA-256.
2. Keep run records, logs and hashes with the run; copy commit-safe summaries
   into `docs/`.
3. Evaluate on test-500 / Harvard-720 with the existing scripts, against the
   released `baseline-resblock2`; never use those sets for selection; do not
   replace historical result tables.

## Status of the multi-GPU path

A one-batch, two-GPU DDP run of this entry point passed for both models from a
clean clone (handoff section 6). Watch the first epochs of the long run.
