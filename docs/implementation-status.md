# Implementation status

Project: hifi-mobiNet

Scope: independent local repository; source repositories and original evidence
are read-only. No new training, export, quantization or benchmark. Functional
validation is separate from historical research results.

| Milestone | Status |
|---|---|
| Structure and provenance policy | Complete; independent Git repository, no remote |
| Original implementation/configuration import | Complete; source/destination SHA-256 recorded, unmodified import committed first |
| Portable inference and model access | Implemented; actual Q05 model inference pending asset access |
| Evaluation/statistical tools | Complete; fixtures passed; prior Harvard audit reproduced |
| Historical results and experiment manifests | Complete; 20 raw result tables preserved unchanged |
| Streamlit comparison and TTS | Implemented; UI/playback-fixture/error-state checks passed; historical audio bytes and real-model TTS pending |
| Native frontend | Built in isolated Windows environment; three stored Harvard ID sequences matched |
| PyTorch model components | All three internal synthesizers construct; original versus relocated decoder outputs exactly match on fixed functional inputs |
| Release/license status | Local preparation only; author decisions and external artifact access remain open |

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

## Pending functional access

The actual Q05 ONNX files and the twelve selected Harvard WAVs are referenced in WSL,
outside the newly stated workspace source scope. Access clarification was requested;
no WSL artifact was read or copied while that clarification remained pending.
Consequently, all four new-code TTS entries remain disabled. The comparison interface
is complete but actual historical listening requires those original WAVs.

This is separate from missing training history and release rights: receiving assets
would enable functional checks, not prove historical quality/causality claims.

## Resume

1. With authorized artifact access, run `scripts/inference/install_local_assets.py`
   for Q05 and Harvard directories; keep weights/audio outside Git as configured.
2. Run `scripts/validation/smoke_models.py --enable-verified-models` with a new output
   directory, then exercise actual historical audio and TTS in Streamlit.
3. Update the model/source/release manifests and this status only for performed checks;
   rerun tests and the history audit, then make a new local commit.
4. Before any remote/push/deployment, resolve author licensing, upstream notices,
   model/audio redistribution, approved storage URLs and target-host validation.

No remote, push, checkpoint upload, publication or deployment is authorized by the
current preparation task. No training/export/quantization/benchmark was performed.
