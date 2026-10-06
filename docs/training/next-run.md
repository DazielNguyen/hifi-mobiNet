# Next run: long training commands

Nothing below has been started. Fill every `<…>` placeholder from the decisions
in [piper-no-vits2-protocol.md](piper-no-vits2-protocol.md) section 4; do not
substitute guesses.

## Environment

```sh
export HIFI_TRAIN_ROOT=<linux-filesystem directory with venv/ and checkout/>
export BANHMI_DATA_ROOT=<directory the dataset.jsonl tensor paths are relative to>
export DATASET_DIR=<directory containing dataset.jsonl and config.json>
cd "$HIFI_TRAIN_ROOT/checkout"
git log -1 --format=%H        # record; the run also writes it into run_record.json
PY="$HIFI_TRAIN_ROOT/venv/bin/python"
```

Expected identities (checked by the harness and recorded per run):
`dataset.jsonl` SHA-256 `f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b`,
split `results/manifests/canonical_split.json` SHA-256
`e678cf436075a10bb68c9a1b435f33c4a40ff1f095b7b78ac12f546c5f12ae9d`.

## Piper_no_VITS2_cpn

```sh
"$PY" -m hifimobinet.training.train \
  --model-id Piper_no_VITS2_cpn \
  --config configs/training/piper-no-vits2-cpn.yaml \
  --dataset-dir "$DATASET_DIR" --data-root "$BANHMI_DATA_ROOT" \
  --split results/manifests/canonical_split.json \
  --run-dir "$HIFI_TRAIN_ROOT/runs/<RUN_NAME_PIPER>" \
  --length-cache "$DATASET_DIR/.spectrogram_lengths_cache.json" \
  --accelerator gpu --devices <N_GPUS> <--strategy ddp if N_GPUS > 1> \
  --max_epochs <EPOCHS> --checkpoint-epochs 1
```

## Baseline (only if the controlled pair is chosen)

```sh
"$PY" -m hifimobinet.training.train \
  --model-id baseline-resblock2-vits2 \
  --config configs/training/baseline-resblock2-vits2.yaml \
  --dataset-dir "$DATASET_DIR" --data-root "$BANHMI_DATA_ROOT" \
  --split results/manifests/canonical_split.json \
  --run-dir "$HIFI_TRAIN_ROOT/runs/<RUN_NAME_BASELINE>" \
  --length-cache "$DATASET_DIR/.spectrogram_lengths_cache.json" \
  --accelerator gpu --devices <N_GPUS> <--strategy ddp if N_GPUS > 1> \
  --max_epochs <EPOCHS> --checkpoint-epochs 1
```

Use the same `<N_GPUS>`, `<EPOCHS>` and per-device batch (config) for both
models so their optimizer-update budgets are equal. Budget reference: batch 16
gives 784 batches/epoch on one GPU and 392 per rank on two GPUs; the historical
baseline was 2 × 16 for 1,500 epochs (588,000 generator updates).

`--length-cache` only reads the existing frame-count cache; new entries are
written under the run directory. Without it the first epoch reads every
spectrogram once to bucket by length (about 14 GB of reads).

## Resume

```sh
"$PY" -m hifimobinet.training.train <same arguments, same --run-dir> \
  --resume "$HIFI_TRAIN_ROOT/runs/<RUN_NAME>/checkpoints/last.ckpt"
```

A `resume_record_<time>.json` is added next to `run_record.json`. Under DDP the
RNG state is not restored (rank 0's state only), which the record notes.

## Before evaluating

1. Select one checkpoint per model by the pre-registered rule (lowest
   `val_loss_mel` among the saved top-3 unless decided otherwise).
2. Keep `run_record.json`, resume records, TensorBoard logs and checkpoint
   hashes with the run; copy commit-safe summaries into the repository.
3. Evaluate on test-500 / Harvard-720 with the existing evaluation scripts;
   never use them for selection. Do not replace historical result tables.

## Multi-GPU check status

See the clean-clone section of [handoff-training-2026-10-06.md](handoff-training-2026-10-06.md).
