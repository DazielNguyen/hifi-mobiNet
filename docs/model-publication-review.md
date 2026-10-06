# Model publication scope review

Date: 2026-10-06. This is an AI artifact review, not a legal clearance or author research approval.

The author chose a separate public inference repository and kept original checkpoints private.
The planned inference repository is `DazielNguyen/hifi-mobiNet-inference`.
The existing model archive, `DazielNguyen/hifi-mobiNet`, remains private.
The candidate contains four unchanged Q05 ONNX graphs, four selected model configs, notices, checksums and metadata.
It excludes training checkpoints, optimizer state, logs, Harvard text/audio and native frontend binaries.

## Static metadata findings

The reproducible report is [model-artifact-metadata.json](audit/model-artifact-metadata.json).
The script uses ZIP inspection and pickletools opcode parsing. It does not deserialize or execute checkpoint objects.
The report covers the 45-file preparation snapshot plus four pinned current remote metadata files.
Together, these inputs cover all 47 inventoried files at the verified private model commit.
The older card and licensing snapshot differ from the subsequent upload-status commit. Both states remain identifiable.

| Checkpoint | Distinct host-path literals | Token-pattern matches | Email-pattern matches |
|---|---:|---:|---:|
| Baseline | 7 | 0 | 0 |
| Parallel-IR | 6 | 0 | 0 |
| Sequential-IR | 6 | 0 | 0 |
| Piper | 0 | 0 | 0 |

These counts concern literal strings, not resolved object fields or a complete privacy inventory.
All checkpoints contain literals for state_dict, optimizer_states, lr_schedulers, hyper_parameters, callbacks, epoch and global_step.
Literal presence does not establish their complete contents or a training recipe.
The report contains value fingerprints, not the literal host paths.
No checkpoint bytes were rewritten or loaded.

No selected privacy pattern matched the preparation ONNX graphs, configs or logs.
Eight publication-intended logs contained no exact complete Harvard sentence from the supplied catalog after case folding.
This comparison does not exclude partial, normalized, encoded or other copyrighted text.
The report states the remaining scanner limits.

## Distribution boundaries

MIT covers team-owned weights through the author's existing license decision.
The [Piper checkpoint repository](https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/main/README.md) declares MIT.
The inference candidate retains the supplied Piper copyright and MIT text. DazielNguyen does not claim ownership of Piper weights.
The [LJSpeech rights statement](https://keithito.com/LJ-Speech-Dataset/#license) describes public-domain status in the United States.
No training dataset is packaged in the inference candidate.

The [code repository](https://github.com/DazielNguyen/hifi-mobiNet) retains its separate source and dependency notices.
This inference-only scope does not close its source lineage or native frontend distribution reviews.
No compiled frontend is bundled with the ONNX candidate.
Harvard text distribution remains unresolved. The dataset candidate stays local, and historical comparison stays off by default.

## Publication checks

A fresh inference repository must contain only the allowlisted candidate and service-generated metadata.
The checkpoint archive must stay private. Removing checkpoint files from its latest commit would still leave historical bytes.
Before public access, the inference upload needs pinned inventory checks.
After public access, anonymous pinned downloads need size and SHA-256 checks.
Functional synthesis uses a new demonstration sentence. It does not collect timing or quality metrics.
No new training, export or quantization belongs to this operation.
The completed publication receipt will record actual repository revisions and observations.
