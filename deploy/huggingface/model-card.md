---
language: en
license: mit
pipeline_tag: text-to-speech
tags:
- onnx
- vits
- hifi-gan
- cpu
---
# HiFi-MobiNet model artifacts

This is a local release candidate for `DazielNguyen/hifi-mobiNet`. Publication is not complete.
The package contains the selected Baseline, Parallel-IR and Sequential-IR checkpoints, plus an external Piper checkpoint.
Four later Q05 FP32 ONNX exports support functional CPU inference.

| Model | Selected epoch | Role |
|---|---|---|
| baseline-resblock2 | 1489 | Internal HiFi-GAN ResBlock2 reference |
| parallel-ir | 1386 | Parallel inverted-residual system |
| sequential-ir | 1442 | Sequential inverted-residual system |
| piper-original | 999, filename 1000 | External Piper LJSpeech reference |

Exact file identities appear in `models/manifest.json`. Current license scope appears in `docs/artifact-licensing.json`.
The preserved model manifest records the earlier handoff state. The separate licensing record supplies later decisions.
MIT covers team-owned checkpoint contents and their equivalent ONNX weights. Piper retains its upstream MIT declaration and attribution.
Third-party content keeps its own terms. Remaining component conditions appear in `THIRD_PARTY_NOTICES.md`.

## Training dataset

The author identifies [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) as the training dataset.
Its [rights statement](https://keithito.com/LJ-Speech-Dataset/#license) identifies public-domain status in the United States for text, audio and annotations.
Attribution: Keith Ito and Linda Johnson (2017), with LibriVox credits.
Dataset terms are separate from checkpoint licenses. This release preparation does not audit every training run.

## Use

Use the [code repository](https://github.com/DazielNguyen/hifi-mobiNet) and its native frontend.
Download the identified ONNX files from `onnx-q05/`. Point `HIFIMOBINET_MODEL_DIR` to that directory.
Run the checksum verification before inference. The demo does not load training checkpoint pickle files or perform automatic exports.
The full checkpoints retain training state. They are archival files, not files that the Space needs at startup.

## Research scope and limits

Historical Harvard/PyTorch comparisons concern system trade-offs. Training recipes and splits differ.
They do not establish decoder-only causality, quality equivalence, human MOS or production readiness.
Q05 ONNX files are engineering artifacts. Historical quality scores do not establish quality on these graphs or on Spaces.
No new training, export, quantization or benchmark forms part of this package.
Piper training-run config is unavailable. Recovered config metadata is distinct from training-written hparams.
Logs in `logs/public/` are publication-intended redacted copies, with a redaction record. Embedded checkpoint metadata remains unchanged.
The source repository contains provenance and recorded historical results. This card does not promise Transformers or hosted Inference API compatibility.
