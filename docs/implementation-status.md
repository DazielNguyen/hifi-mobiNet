# Implementation status

Project: hifi-mobiNet

Scope: independent repository; source repositories and original evidence remain
read-only. The author authorized public GitHub code publication on 2026-10-06.
No new training, export, quantization or benchmark. Functional validation is
separate from historical research results; weights/audio upload remains deferred.

| Milestone | Status |
|---|---|
| Structure and provenance policy | Complete; independent Git repository; public publication authorized |
| Original implementation/configuration import | Complete; source/destination SHA-256 recorded, unmodified import committed first |
| Portable inference and model access | Complete; four identified Q05 ONNX models pass local functional synthesis |
| Evaluation/statistical tools | Complete; fixtures passed; prior Harvard audit reproduced |
| Historical results and experiment manifests | Complete; 20 raw result tables preserved unchanged |
| Streamlit comparison and TTS | Complete locally; twelve original WAVs verified, real UI/TTS checks passed; assets external to Git |
| Native frontend | Built in isolated Windows environment; three stored Harvard ID sequences matched |
| PyTorch model components | All three internal synthesizers construct; original versus relocated decoder outputs exactly match on fixed functional inputs |
| Release/license status | Public code authorized; named license and weight hosting unconfirmed |

Resume by reading this file, `release-gaps.md`, `source-map.json` and the Git
log. Inspect existing changes before continuing. Do not reset, amend or
rewrite history; commit finished milestones separately.

## Performed checks

- Source file identity and destination hashes verified for 121 mappings; no changes
  were made in BanhmiTTS, Piper, manuscript directories or the original evidence.
- Model/config mapping, forbidden model/path inputs, checksum failures, unchanged
  model `forward`/`infer` AST and original PCM16 conversion were tested.
- Evaluation fixtures include known word counts, mean versus ratio-of-sums RTF,
  duplicate/mismatched sentence rejection and a known exact Wilcoxon result.
- Portable prior-audit reproduction matched 279 numeric values. No new TTS dataset,
  Whisper or UTMOS execution was performed.
- Streamlit AppTest covered sentence selection, four audio widgets using a synthetic
  fixture and missing/corrupt assets. Local HTTP root/health endpoints returned 200.
- Two existing workspace checkpoints were hashed without deserialization. The
  selected original Piper file matches its archived/published identity record.
- Full local environment: 20 tests passed. A separate clean clone and minimal
  environment: 15 passed, with two optional modules skipped (PyTorch/frontend
  absent). CLI and result auditing work without access to source repositories.
- Entire reachable Git history and tracked files passed the bounded content scan;
  the release manifest checks file identities. No weights, secrets detected by
  the scan, environments or caches were staged. This is not a security guarantee.
- Native frontend build initially failed with a long temporary path. A short ignored
  build directory fixed it without changing original source. Windows UTF-8 output
  handling was made explicit in the portable statistical script.

## Resolved artifact access and author decisions (2026-10-06)

Ubuntu-24.04/WSL2 contains the four selected checkpoints and Q05 FP32 graphs.
All eight SHA-256 identities match the previous manifests. Four existing graphs
and twelve original Harvard WAVs were copied into ignored local storage.
All four models synthesized a short sentence successfully; Streamlit rendered
three aligned choices and exercised the baseline TTS button. These are functional
checks, not benchmarks, human listening scores or historical runtime reproduction.

The author explicitly selected public code publication in DazielNguyen/hifi-mobiNet,
confirmed sharing permission, and restricted artifact scope to journal-related
selected checkpoints, relevant logs and Harvard audio. The named license and
weight storage provider remain unconfirmed. No weights/audio are uploaded in
this step. See release-decisions.json and artifact-inventory.json.

## Resume

1. Choose the license for original contributions, finish upstream notice/compatibility
   review and document specific model/audio sharing terms.
2. Choose external storage and approved URLs. Selected full training checkpoints,
   Q05 ONNX and all Harvard WAVs total about 4.02 GB before logs. Full TensorBoard
   event files from the three main runs alone total about 42.94 GB; decide separately
   whether those are needed in the artifact release.
3. A fresh clone needs external assets and the native frontend. Use the checksum
   installer and functional smoke commands in setup.md; never substitute other weights.
4. Before deployment, validate the target Linux host, peak loaded-model memory,
   process isolation/timeouts and concurrent-user capacity. No deployment is authorized
   or completed here.

The initial local-only restriction was superseded by explicit authorization for
public repository creation and code push. Training/export/quantization/benchmark
restrictions remain in effect. Original source/evidence files remain unchanged.
