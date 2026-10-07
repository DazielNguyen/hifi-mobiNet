# Claim evidence package, 2026-10-07

This package collects the evidence found for 11 journal claims that the claim ledger lists as not fully supported:
J-C001, J-C010 to J-C013, J-C015, J-C019 and J-C023 to J-C026.

It is a **traceability and hand-off package, not a verdict**:
- every claim carries `journal_verification: NOT_CHECKED`;
- the PC session's own labels appear only as `pc_reported_assessment`;
- suggested wording (`proposed_narrowed_claim`, `source-records/pc-package/README.md`) is not evidence of results.

The next step is an independent evidence audit on the manuscript side.

Nothing was trained, run for inference, exported, quantized or benchmarked for this package, and no checkpoint was unpickled or loaded. Checkpoint names were read statically with `pickletools`. ONNX graphs were parsed with `onnx` but never executed. Parity was recomputed from stored outputs.

## Contents

| Path | What it is |
|---|---|
| `claim-evidence-index.json` | Per-claim record: original claim, narrowed wording, PC assessment, evidence paths with locators/type/SHA-256, calculation script and output, limitations, remaining inputs |
| `evidence-map.md` | Readable claim-by-claim map with the key numbers and where each comes from |
| `provenance-and-limitations.md` | Evidence classes, what each source can and cannot show, redactions |
| `reproduction.md` | Commands to re-derive every number from stored outputs |
| `source-manifest.json` | Every copied file: source, destination, bytes, SHA-256 before/after, transformation, evidence type |
| `artifact-manifest.json` | Large inputs kept out of Git (transfer bundle) and artifacts referenced by hash only |
| `source-records/` | Run-time records (Q05), public sources, calibration and overlap records, ONNX inventory extract, search log, PC synthesis |
| `scripts/` | `verify_stored_outputs.py` and `export_key_inventories.py` (portable, parameterized); `pc/`, `q05/`, `historical/` hold scripts copied as records |
| `outputs/` | `verify_report.json` (85 checks), `J-C001_key_inventories.json`, and the PC session outputs under `pc/` |

## Evidence per claim (summary)

| Claim | Strongest evidence here | Kind |
|---|---|---|
| J-C001 | 876/876 parameter/buffer names: baseline checkpoint = released code; 92 names (72 flow attention, 20 duration discriminator) absent from the Piper checkpoint | current check (names only) + author account |
| J-C010 | Historical calibration: 60 rows by `Random(42)`, 57 train / 3 test; train-only 60/60 set exists only as a design | reconstructed from script rule + design record |
| J-C011 | Q05 graphs carry the checkpoint SHA-256 in metadata; 10/10 parity cases per model within thresholds declared before the runs | run-time records + recomputed from stored outputs |
| J-C012 | Not found in the scanned scope; the evaluator found still feeds 22.05 kHz audio to Whisper | AI search log |
| J-C013 | Not found in the scanned scope (no device logs) | AI search log |
| J-C015 | Not found in the scanned scope (no listening study) | AI search log |
| J-C019 | WER SEQ−MRF +0.0149, 95% CI [+0.0019, +0.0277] (fraction); equivalence not established, trade-off | recomputed prior round |
| J-C023 | SEQ vs **baseline**: 495/500 (FP32), 498/500 (INT8); benchmark MRF graph not linked to checkpoint 1386 | historical results + graph identity checks |
| J-C024 | Historical INT8 graphs: QOperator, per-channel INT8 weights, UINT8 activations, 133 convs, nothing under `/dec/`; SEQ adds 40 attention MatMuls | current check of graphs |
| J-C025 | Flow in Q05 graphs: 8,285,568 elements (internal), 7,090,560 (Piper); op counts per graph | current check of graphs |
| J-C026 | 0 exact / 0 6-gram / 0 phoneme-ID overlap with LJSpeech text; Piper checkpoint hash = public LFS oid; model card: trained from scratch on LJ Speech | recomputed + public source |

## Check from stored outputs

From this folder, with only the Python standard library and the files in Git:
```
python scripts/verify_stored_outputs.py --output /tmp/verify_report.json
```
With the transfer bundle and graphs (numpy, onnx, optionally jiwer), see `reproduction.md`. The stored `outputs/verify_report.json` is the full run: 85 checks, 0 failed.

## What cannot be recomputed from this package

- The Q05 export and parity runs themselves: they need the checkpoints and the training environment, and this round did not repeat them.
- The historical ONNX timings: these are measurements, not computations.
- Harvard-720 PyTorch statistics: only the stored per-utterance results are used.
- Piper's training data: only its public model card is available.
- J-C012, J-C013 and J-C015: no records were found to recompute from.

## Handling rules

- Checkpoints, ONNX graphs, WAVs and the training dataset are not in Git.
- The parity arrays and `dataset.jsonl` travel in the transfer ZIP; everything else is referenced by SHA-256 in `artifact-manifest.json`.
- Files that contained machine-specific paths were copied with path placeholders (`<WORKSPACE>/`, `<WSL_PROJECT_HOME>/`, ...). Both hashes are in `source-manifest.json`, and the unmodified originals are only in the private part of the transfer ZIP.
- The source package (`BanhmiTTS/work/claim-evidence-2026-10-07/`) and all research artifacts were left unchanged.
