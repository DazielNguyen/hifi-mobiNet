# Plan: ONNX/PTQ round 2 (flow weight norm folded)

*Tiếng Việt: [plan-onnx-ptq-v2.vi.md](plan-onnx-ptq-v2.vi.md)*

Status: **prepared, not run.** Written 2026-10-22. The author will start the run.

## 1. Why a second round

Round 1 (2026-10-21, `results/piper-ir-onnx-ptq/`) left the flow's weight norm unfolded at
export, which broke the declared recipe for most of the flow (protocol `docs/evaluation-onnx-ptq.md`, section 7):
- 32 of the 40 flow convolutions had their weights quantized at run time, per tensor, as UINT8, instead of the declared static per-channel INT8.
- The INT8 graphs kept 27.8 MB of FP32 flow weights. They are 41–43 MB, against 22–24 MB for the historical INT8 graphs.
- The INT8 RTF includes per-run weight-norm and quantization work.

Round 2 repeats the whole round with only that fixed. Everything else is the round-1 protocol unchanged: models, checkpoints, recipe, scope, calibration set, matched-noise quality evaluation, scorer and timing design.

## 2. What is already prepared

| Change | Where | Check |
|---|---|---|
| The flow's weight norm is folded before export (`fold_flow_weight_norm`: `WN.remove_weight_norm()` on each coupling layer) | `scripts/evaluation/onnx/common.py`, `export.py` (default; `--keep-flow-weight-norm` reproduces round 1) | new gate `flow_weight_norm_fold_changes_only_rounding`: outputs before and after the fold within NRMSE ≤ 1e-5 and max abs ≤ 1e-4 |
| No weight-norm op may stay in the PTQ scope | `export.py` | new gate `no_weight_norm_ops_in_ptq_scope` (no `ReduceL2` under enc_p, dp or flow) |
| Every QLinearConv must have a constant per-channel INT8 weight | `common.py: qlinearconv_weight_kinds`, `quantize.py` | new gate `every_qlinearconv_weight_constant_per_channel_int8` |
| The pipeline needs an explicit, new output folder | `scripts/evaluation/onnx/run_round.sh` (`OUT` is required) | an existing `OUT` would resume that round |
| Unit tests | `tests/test_onnx_ptq.py` | the fold changes only rounding on a small EdgeTTS model; the weight-kind check detects run-time weights |
| Local launcher with the machine paths, outside OneDrive and Git | `<WSL_PROJECT_HOME>/hifimobinet-private/launch_onnx_ptq_v2.sh` | refuses to start if the session or output folders exist, if a training runs, or if the scripts are uncommitted |

## 3. Before starting

1. Commit the prepared changes. The launcher refuses uncommitted `scripts/evaluation`, `tests` or protocol files, so the records name a clean commit.
2. Make sure the machine is idle:
   - no training is running;
   - nothing heavy is open on Windows;
   - Windows sleep is set to never;
   - a WSL keep-alive is running.

   WSL restarted on its own on 2026-10-22 and dropped every tmux session. Without a keep-alive, a WSL shutdown would kill the round.
3. Check disk space: round 1 used 1.2 GB under `training_output/onnx-ptq/`.

## 4. Run

In WSL:

```sh
bash <WSL_PROJECT_HOME>/hifimobinet-private/launch_onnx_ptq_v2.sh
```

The launcher starts the tmux session `onnx-ptq-v2`, which runs `scripts/evaluation/onnx/run_round.sh` with:
- `OUT=training_output/onnx-ptq-v2` in the training clone;
- `RESULTS=<repository>/results/piper-ir-onnx-ptq-v2`.

Expected time, from round 1: about 5 h 15 min.

| Stage | Round-1 duration |
|---|---|
| export and PTQ, 3 models | about 4 min |
| quality synthesis, 6 graphs | about 16 min |
| timing session (idle) | about 30 min |
| scoring, 6 WAV sets | about 4 h 20 min |
| analysis and copy | about 1 min |

**Following the run:**
- `tail -f <training clone>/training_output/onnx-ptq-v2/round.log` shows progress.
- A Windows pop-up (`msg.exe`) reports DONE or FAILED.
- After an interruption, start `run_round.sh` again with the same `OUT`: it resumes after the last finished stage. Do not use the launcher for this, because it refuses an existing folder.

## 5. Acceptance criteria

The round counts only if **every** gate passes. The round-1 gates are listed in the protocol, sections 2–3; the three new gates are in section 2 above. In addition, check the following before writing any result:

| Check | Expected |
|---|---|
| `quantize_record.json → int8_structure.qlinearconv_weight_kinds` | only `constant int8 per-channel`, 109 per model (enc_p 37, dp 32, flow 40) |
| `export_record.json → weight_norm_ops_in_graph` | no entry for enc_p, dp or flow |
| FP32 parity on Harvard-720 | 720 / 720 per model, as in round 1 |
| PyTorch reference WAVs | 720 / 720 equal to the stored evaluations |
| INT8 file size | expected about 20–25 MB (the 27.8 MB of FP32 flow weights becoming INT8). This is an expectation, not a gate: report the value whatever it is |

## 6. After the run

1. **Read, do not assume.** Compare round 2 with round 1 per model:
   - INT8 size;
   - INT8 RTF at 2 and 12 threads;
   - INT8-vs-FP32 ΔUTMOSv2 and ΔWER.
   - In particular, check whether SeqIR's UTMOSv2 loss (−0.036 in round 1) remains.
2. **Write `results/piper-ir-onnx-ptq-v2/README.md` and `README.vi.md`**, in the structure of the round-1 README:
   - section 1 says what changed (only the flow fold);
   - add a short round-1 versus round-2 table.
3. **Add `results/manifest.json` entries** for the new records.
4. **Update `results/README.md` and `README.vi.md`:**
   - "ONNX export and INT8 PTQ": round 2 becomes the main INT8 result. Round 1 stays, labelled superseded, with its defect note.
   - "Seven-model comparison": the deployment rows and the ONNX INT8 file row.
   - "Model size": the ONNX INT8 column.
5. **Update the other documents:**
   - `docs/q2-readiness-seq-par.md` and `.vi.md`: the PTQ claim row in section 3 and the paper-story bullet in section 5.
   - `docs/evaluation-onnx-ptq.md` section 7: record that round 2 applied the correction.
6. **Re-profile only if the INT8 speed changes materially.** Run `profile_graphs.py` on the round-2 graphs; the decoder is unchanged, so the `Sin` finding should stand.
7. **Commit** in separate scopes (results; docs) without attribution. The author pushes.

## 7. Prompt for the session that runs it

> Run ONNX/PTQ round 2 as planned in `hifi-mobiNet/docs/plan-onnx-ptq-v2.md`.
> - First check section 3 (committed scripts, idle machine, WSL keep-alive). Then start `<WSL_PROJECT_HOME>/hifimobinet-private/launch_onnx_ptq_v2.sh` and follow `round.log`.
> - Do not change the recipe, scope, calibration or scorer.
> - If a gate fails, stop and report it; do not relax it.
> - When ALL DONE is logged, check the acceptance criteria of section 5, then do the steps of section 6.
> - Commit only when I ask, without attribution. Discuss in Vietnamese; documents in English with a Vietnamese README.vi.md.
