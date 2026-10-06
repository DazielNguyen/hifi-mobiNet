# Inference artifact notices

This package contains existing ONNX graphs, selected model config metadata and release records.
It excludes training checkpoints, logs, recordings, sentence catalogs and native frontend binaries.

## Team-owned models

The author selected MIT for team-owned weights in Baseline, Parallel-IR and Sequential-IR on 2026-10-06.
The root LICENSE supplies that text, with copyright 2026 DazielNguyen.
This selection does not relicense third-party content or original datasets.

## External Piper

Piper remains an external reference. DazielNguyen does not claim ownership of its original weights.
The [upstream checkpoint repository](https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/main/README.md) declares MIT.
The supplied Piper copyright and MIT text remain in `licenses/Piper-MIT.txt`.
The graph identity appears in `artifacts.json`. This records upstream terms, not a blanket legal clearance.

## Training dataset and code

The author identifies [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) as the training dataset.
Its [rights statement](https://keithito.com/LJ-Speech-Dataset/#license) covers public-domain status in the United States.
Attribution: Keith Ito and Linda Johnson (2017), with LibriVox credits.
No training dataset is distributed in this package. No new MIT or CC0 license applies to that dataset.

The [code repository](https://github.com/DazielNguyen/hifi-mobiNet) retains separate source and dependency notices.
Its native frontend builds a pinned eSpeak NG dependency. That dependency retains its GPL terms.
This ONNX package does not distribute that compiled frontend. Building or redistributing frontend code requires its applicable notices.
Harvard sentence text and generated audio are outside this package and its MIT scope.
