# Reproduction

All commands run from this folder (`docs/evidence/claim-evidence-2026-10-07/` in a hifi-mobiNet checkout). The scripts take paths as arguments and resolve the repository root from their own location (`--repo-root` overrides it).

## 1. Base checks (standard library only)

```
python scripts/verify_stored_outputs.py --output verify_base.json
```

This mode needs nothing beyond the Python standard library and the files in Git. Section by section, it re-derives:

| Section | What is re-derived |
|---|---|
| package integrity | every copied file against `source-manifest.json` |
| J-C001 | counts from the stored name inventories |
| J-C011 | metadata checkpoint hash = run-record hash; thresholds equal `RUN_CONFIG`; `RUN_CONFIG` written before each model run; stored parity results pass |
| J-C023 | 495/500, 498/500 and the 21 shared sentences, from the historical result JSONs |
| J-C019 | the WER CI and its conversion to percentage points |
| J-C010 | split membership of the candidate (60/60 train) and historical (57/3) sets |
| J-C026 | Piper hash = public LFS oid; model-card blob; phoneme maps |

Expected: 35 checks, 0 failed.

## 2. Full checks (transfer bundle + graphs)

Requirements:
- numpy and onnx (the run was made with numpy 1.26.4 and onnx 1.22.0, Python 3.10.20);
- optionally jiwer, for the second normalization.

```
python scripts/verify_stored_outputs.py \
  --parity-dir    <bundle>/transfer-only/q05-parity \
  --dataset-jsonl <bundle>/transfer-only/dataset/dataset.jsonl \
  --graph-dir     <folder containing the ONNX graphs> \
  --output outputs/verify_report.json
```

- `--parity-dir`: hashes each `parity_case_XX.npz` against the run report and recomputes NRMSE and max|err|.
- `--dataset-jsonl`:
  - re-samples the historical calibration with `random.Random(42)`;
  - recomputes the Harvard/LJSpeech exact, n-gram and phoneme-ID overlap.
- `--graph-dir`:
  - finds graphs by SHA-256 (any layout or file names);
  - counts flow initializers and ops (J-C025);
  - reads the QLinear scope and dtypes (J-C024);
  - compares the INT8 decoder initializers with their FP32 parents (J-C023).
  - Graphs are parsed, never run.

The graphs are not in the transfer ZIP:
- The four Q05 FP32 graphs are public at `https://huggingface.co/DazielNguyen/hifi-mobiNet-inference` (revision `f760e85a45b84c217091acf963837ffc817bad8d`).
- The historical and validation graphs exist only on the workstation.
- `artifact-manifest.json` lists every graph with its SHA-256.

Expected: 85 checks, 0 failed (`outputs/verify_report.json`).

## 3. Checkpoint name inventories (J-C001)

```
python scripts/export_key_inventories.py \
  --baseline-ckpt <baseline best-epoch=1489 checkpoint> \
  --piper-ckpt    <lj-med_1000.ckpt> \
  --hifi-repo     <hifi-mobiNet checkout with its training environment> \
  --output outputs/J-C001_key_inventories.json
```

- Checkpoint names are read from `data.pkl` with `pickletools.genops`; nothing is unpickled.
- The code-side inventories are built in the training environment: torch 2.13.0+cu130, PyTorch Lightning 1.7.7, with MAS built.
- The checkpoints are not in the bundle. `artifact-manifest.json` gives their SHA-256; the Piper checkpoint is public at `rhasspy/piper-checkpoints`.

## 4. Harvard statistics (J-C019)

`repo:scripts/evaluation/original/recompute_B_stats.py` produced `repo:results/prior_audit/harvard_stats_recomputed.json` from the stored per-utterance results (SciPy 1.15.3). It is not re-run by this package.

## What is not reproducible here

- **Q05 export and parity runs:** they need the checkpoints and the export environment; this round did not repeat them.
- **Historical ONNX timings:** they are measurements.
- **The scripts under `scripts/pc/` and `scripts/historical/`:** they are records with redacted host paths and are not runnable as-is.
