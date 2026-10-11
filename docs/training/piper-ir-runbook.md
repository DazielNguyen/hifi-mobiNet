# Runbook: Piper-ParIR → Piper-SeqIR sequence

The sequence is defined in [`configs/training/sequences/piper-ir-2026-10.json`](../../configs/training/sequences/piper-ir-2026-10.json) and the protocol in [piper-ir-protocol.md](piper-ir-protocol.md).

`scripts/training/run_sequence.py` trains the two models one after the other, from scratch, with the
Piper_no_VITS2_cpn setup (2 GPUs DDP, bf16, 1,500 epochs). It runs inside the
training clone on Ubuntu/WSL, in tmux. It does not depend on an interactive
session staying open, and it never shuts the machine down.

## Start

From the training clone root:

```sh
export PY="$HIFI_TRAIN_ROOT/venv/bin/python"     # the venv used for Piper_no_VITS2_cpn
tmux new-session -d -s piper-ir \
  "$PY scripts/training/run_sequence.py start --plan configs/training/sequences/piper-ir-2026-10.json"
```

## What the controller does

1. **Takes the lock.** It takes `training_output/piper-ir-sequence/controller.lock`. A second controller refuses to start.

2. **Stops on a bad run.** If a run is already `failed` or `needs_recovery`, the controller stops. A later run never starts after a failed or interrupted one.

3. **Runs the preflight for each `pending` run.** The result is written to `preflight_<run>_start_<time>.json`. Every check must pass, otherwise nothing starts:
   - the Git tree is clean, and the commit is recorded;
   - `dataset.jsonl`, dataset `config.json` and the split match the hashes of the Config A run;
   - the config `model_id` is right;
   - Python, torch, Lightning and NumPy versions equal the Config A environment;
   - the compiled MAS extension is present (its hash is recorded);
   - no other `hifimobinet.training` process is running;
   - both GPUs are below 2 GiB used;
   - at least 60 GiB of disk is free;
   - the run directory and log are new.

4. **Launches the training from scratch.** The command is `python -m hifimobinet.training.train --model-id … --config … --run-dir … <plan train_args>`.
   - Its stdout and stderr go to `training_output/<run>.log`.
   - `train.py` writes `run_record.json` (commit, dirty flag, config, dataset and split hashes, versions, argv) and Lightning `hparams.yaml` / TensorBoard events.
   - The checkpoints are the top-3 by `val_loss_mel` plus `last.ckpt`.

5. **Validates the run when the process exits.** The run is `completed` only if all checks pass (`validation_<run>.json`):
   - the exit code is 0;
   - the log of this attempt says `max_epochs=1500` reached and contains no traceback;
   - `run_record.json` is consistent (model ID, config/split/dataset hashes, clean tree);
   - there are 1,500 `val_loss_mel` points, and every logged scalar is finite;
   - `last.ckpt` is at epoch 1,499 with global step 1,176,000 (392 × 1,500 × 2);
   - the checkpoint chosen is the one at the minimum `val_loss_mel`, and its epoch and value match.

   For a valid run the controller writes `<run_dir>/selected_checkpoint.json`: file, bytes, SHA-256, epoch, global step, `val_loss_mel` and the rule.

6. **Starts SeqIR only after ParIR is `completed`.** After SeqIR completes, it writes `training_report.json` and `training_report.md` in the state directory.

## Progress

```sh
$PY scripts/training/run_sequence.py status --plan configs/training/sequences/piper-ir-2026-10.json
tail -f training_output/piper-ir-sequence/controller.log
tmux attach -t piper-ir          # detach: Ctrl-b d (or from another shell: tmux detach-client -s piper-ir)
tensorboard --logdir training_output/piper-parir-no-vits2-cpn/lightning_logs
```

`status` prints each run's state, PID and whether it is alive, the start and end times, the exit code, the latest epoch from the log, the selected checkpoint and any failure.

## States and recovery

| State | Meaning | Action |
|---|---|---|
| `pending` | Not started (or its preflight failed: see `last_preflight_failure`) | Fix the reported condition and run `start` again |
| `running` | Its process is alive | Wait |
| `completed` | Validated; never re-run | — |
| `failed` | The process ended but validation failed (see `failure` and `validation_<run>.json`) | Inspect. Do not delete or reset outputs |
| `needs_recovery` | The controller found a `running` entry whose process is gone (controller, tmux or machine stopped) | Inspect `last.ckpt` and the log, then resume explicitly |

To resume, use the repository's full-state path (`train.py --resume last.ckpt`). The two-GPU DDP resume was checked on 2026-10-11:

```sh
tmux new-session -d -s piper-ir \
  "$PY scripts/training/run_sequence.py resume --plan configs/training/sequences/piper-ir-2026-10.json --run parir --note 'reason'"
```

- **What a resume records:**
  - an attempt of kind `resume`, with the note, in `status.json`;
  - a new preflight;
  - `resume_record_<time>.json` in the run directory;
  - the attempt's log byte offset.
- **What happens next:** if the resumed run validates, the controller continues with the next run. The run directory is never emptied and no run is restarted from scratch.
- **Not restored under DDP:** the RNG state, by design (`harness.restore_rng_state`). Data order and noise therefore differ from an uninterrupted run. The training log warns about this; the resume attempt is listed in `status.json`.
- **Re-checking a run:** `validate --run <name>` re-runs the validation of a finished run, for example after inspecting it.

**If a training process is still alive without its controller:** for example, the controller was killed but the trainer continues. `start` then refuses to act. Let the process finish, then run `validate`. Or decide manually; never start a second copy.

## What is recorded and where

| File | Location | Tracked in Git? |
|---|---|---|
| Plan (runs, arguments, expected hashes and versions, budget) | `configs/training/sequences/` | yes |
| Status, controller log, preflights, validations, final report | `training_output/piper-ir-sequence/` | no (ignored) |
| Run records, hparams, TensorBoard events, checkpoints, selected-checkpoint manifest | `training_output/piper-*ir-no-vits2-cpn/` | no |
| Training stdout/stderr | `training_output/piper-*ir-no-vits2-cpn.log` | no |

The run record's commit and clean flag show what the tree contained at launch. They do
not prove which bytes the process loaded: Git-ignored files, such as the compiled MAS
extension, are identified by hash in the preflight.

## Not part of this sequence

The controller does not run Harvard, MOS, ONNX/PTQ or edge benchmarks.
