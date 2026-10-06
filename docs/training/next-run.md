# Next run: long training commands

Nothing below has been started. Settings follow
[piper-no-vits2-protocol.md](piper-no-vits2-protocol.md) section 2 (1,500 epochs,
2 GPUs × batch 16, bf16, seed 1234, canonical split, best `val_loss_mel` + last).
Only the run directory name is left to choose.

## Environment

```sh
export HIFI_TRAIN_ROOT=<linux-filesystem directory with venv/ and checkout/>
export BANHMI_DATA_ROOT=<directory the dataset.jsonl tensor paths are relative to>
export DATASET_DIR=<directory containing dataset.jsonl and config.json>
cd "$HIFI_TRAIN_ROOT/checkout"
git log -1 --format=%H        # also written into run_record.json
PY="$HIFI_TRAIN_ROOT/venv/bin/python"
```

Expected identities (recorded per run): `dataset.jsonl` SHA-256
`f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b`, split
SHA-256 `e678cf436075a10bb68c9a1b435f33c4a40ff1f095b7b78ac12f546c5f12ae9d`.

## Piper_no_VITS2_cpn (EdgeTTS Config A)

```sh
"$PY" -m hifimobinet.training.train \
  --model-id Piper_no_VITS2_cpn \
  --config configs/training/piper-no-vits2-cpn.yaml \
  --dataset-dir "$DATASET_DIR" --data-root "$BANHMI_DATA_ROOT" \
  --split results/manifests/canonical_split.json \
  --run-dir "$HIFI_TRAIN_ROOT/runs/<RUN_NAME>" \
  --length-cache "$DATASET_DIR/.spectrogram_lengths_cache.json" \
  --accelerator gpu --devices 2 --strategy ddp \
  --max_epochs 1500 --checkpoint-epochs 1
```

Checkpoints: `checkpoints/best-epoch=<e>-val_loss_mel=<v>.ckpt` (top 3 by
`val_loss_mel`) and `checkpoints/last.ckpt`. Evaluate the lowest-`val_loss_mel`
checkpoint.

## Baseline retrain (only if option 2 of the protocol is chosen)

```sh
"$PY" -m hifimobinet.training.train \
  --model-id baseline-resblock2-vits2 \
  --config configs/training/baseline-resblock2-vits2.yaml \
  --dataset-dir "$DATASET_DIR" --data-root "$BANHMI_DATA_ROOT" \
  --split results/manifests/canonical_split.json \
  --run-dir "$HIFI_TRAIN_ROOT/runs/<RUN_NAME_BASELINE>" \
  --length-cache "$DATASET_DIR/.spectrogram_lengths_cache.json" \
  --accelerator gpu --devices 2 --strategy ddp \
  --max_epochs 1500 --checkpoint-epochs 1
```

`--length-cache` only reads the existing frame-count cache; new entries go
into the run directory. Without it the first epoch reads every spectrogram once.

## Resume

```sh
"$PY" -m hifimobinet.training.train <same arguments, same --run-dir> \
  --resume "$HIFI_TRAIN_ROOT/runs/<RUN_NAME>/checkpoints/last.ckpt"
```

A `resume_record_<time>.json` is added. Under DDP the RNG state is not restored
(rank 0's state only). DDP resume itself has not yet been exercised.

## Before evaluating

1. Select the lowest-`val_loss_mel` checkpoint; record its SHA-256.
2. Keep run records, TensorBoard logs and checkpoint hashes with the run; copy
   commit-safe summaries into the repository.
3. Evaluate on test-500 / Harvard-720 with the existing scripts; never use them
   for selection; do not replace historical result tables.

## Multi-GPU check status

A one-batch, two-GPU DDP run of this entry point passed for both models from a
clean clone (handoff section 6). Resume under DDP and long-run stability remain
unchecked; watch the first epochs of the long run.
