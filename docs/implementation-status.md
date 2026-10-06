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

The public repository `DazielNguyen/hifi-mobiNet` was created on 2026-10-06. The
authorized origin is recorded in release-decisions.json. Publication includes
audited code, documentation, manifests and existing result tables; it excludes
model files, WAVs, environments, native binaries and private artifact locations.

## Local Mac transfer preparation (2026-10-06)

The subsequent author request prohibits any new push, upload, tag, release or
deployment in this preparation. The earlier public code repository remains at its
previous commit; the new preparation commits are local only.

Four exact selected full checkpoints, four Q05 FP32 graphs, 2,880 original Harvard
WAVs, eight original text logs and nine training-written hparams files were copied
from WSL into an external staging directory outside OneDrive/Git. Source and copy
hashes/sizes were verified. Original logs/configs and exact host locators remain
private. Separate redacted copies and later recovered config metadata are labelled.
Seven transfer ZIPs were created and every member reverified; each is below 2 GiB.

The portable importer passed 35 tests on Windows and additional source/destination
symlink fixtures on WSL. Real ONNX/audio/metadata ZIPs were imported into a fresh
directory. All 2,880 WAVs verified through the runtime reader, four model IDs passed
CLI verification, 720 sentence choices appeared in Streamlit, and SEQ synthesized
one short utterance through the UI. No timing or quality metric was collected.
All 121 original source mappings still matched their recorded hashes.

Use docs/mac-setup.md and docs/mac-transfer-preparation.md to resume. Transfer the
prepared code Git bundle as well as ZIPs/checksums so the Mac receives local
unpushed commits. macOS native frontend build, synthesis, human playback and host
resource checks remain NOT CHECKED. License, third-party rights and public artifact
storage remain pending. The preparation is not a published v0.1.0 release.

## Current license decisions (2026-10-06)

The author selected MIT for original owned code and team-owned contents of the three selected checkpoints.
The same team-owned weights in existing ONNX exports use MIT. External Piper retains the upstream MIT declaration.
Generated Harvard WAVs use CC BY 4.0 only for rights the team holds. Original datasets retain their terms.
Harvard sentence text terms remain unresolved. See artifact-licensing.md and artifact-licensing.json.
Earlier sections record the handoff state. This update preserves transferred bytes and does not publish artifacts.

## Hugging Face local preparation (2026-10-06)

The author supplied namespace DazielNguyen. Three local model/dataset/Space candidates now exist outside Git.
Linux amd64 Docker build, native frontend, four-model synthesis, historical audio checks and Streamlit functional checks passed locally.
The runtime requires pinned model/dataset commits and verifies file identities. Historical comparison is off by default.
No HF authentication, repository creation, upload or deployment occurred. Remaining rights and target-host checks are open.
See huggingface-deployment.md, huggingface-validation.json and huggingface-checkpoint.md for the next steps.

## Hugging Face private model upload (2026-10-06)

The author selected model storage first and deferred demo hosting.
The model candidate is uploaded privately at https://huggingface.co/DazielNguyen/hifi-mobiNet.
Its verified current commit is `a4354007630f14b2cd0046a6f6b0dc17dd1f7ac9`. Four checkpoints and four ONNX artifacts retain their original bytes.
All 47 final inventoried files passed remote checks; a Sequential-IR ONNX download passed local hash verification.
See huggingface-model-upload.md and huggingface-model-upload.json for exact observations and verification limits.
Dataset/Space candidates remain local. No public visibility change, GitHub push, tag, paid compute or deployment occurred.
Earlier sections describe historical preparation states. Remaining rights and target-host checks are still open.

## Separate public inference release (2026-10-06)

Four Q05 ONNX graphs, selected configs and notices are public at https://huggingface.co/DazielNguyen/hifi-mobiNet-inference.
Their pinned revision is `8561554ae2ad1936262153ed195a2bc04c747235`. All four anonymous downloads matched original hashes.
The original checkpoint archive remains private. No training log, checkpoint or Harvard material is in the new inference repository.
Source runtime targets now use the public inference repo. Previously prepared Space snapshots remain unchanged and need regeneration before deployment.
See huggingface-inference-release.md/json and model-publication-review.md for observations and limits.
Dataset/Space publication remains deferred. Earlier sections record preparation states.
