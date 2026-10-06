# Public inference release

Date: 2026-10-06. Account: DazielNguyen.

The author chose a separate inference repository and kept original checkpoints private.
The public repository is [hifi-mobiNet-inference](https://huggingface.co/DazielNguyen/hifi-mobiNet-inference).
Its verified commit is `f760e85a45b84c217091acf963837ffc817bad8d`.
The private checkpoint archive remains `DazielNguyen/hifi-mobiNet`.

The release contains 15 inventoried files: four existing Q05 FP32 ONNX graphs, four selected configs, notices and metadata.
Hugging Face also created `.gitattributes`. No checkpoint, training log, Harvard text/audio or frontend binary was uploaded to this repository.
The preparation code was committed as `d7180a2` before upload.
The repository was created privately. Its uploaded inventory passed checks before public access.

## Completed checks

All uploaded files matched their expected sizes and identities.
Remote large files matched server-reported original-file SHA-256 values. Small metadata matched pinned downloaded bytes.
All four ONNX graphs were then downloaded anonymously at the exact public commit.
Each downloaded graph matched its original size and SHA-256 at artifact commit `8561554ae2ad1936262153ed195a2bc04c747235`.
A subsequent metadata commit corrects the complete-package download command. The graph and config bytes remain unchanged.
The corrected CLI command passed an anonymous dry-run. Its 16 planned files match the inventory plus service-generated metadata. The checkpoint archive remained private after publication.
All four downloaded graphs synthesized a new demonstration sentence in the existing Linux amd64 image.
Each output was nonempty mono PCM16 at 22,050 Hz. This was a local functional check under Docker emulation.
Nine bootstrap regression tests passed, including the separate checkpoint archive boundary. Neither check measured latency, quality or hosted capacity.
The [machine-readable receipt](huggingface-inference-release.json) records the inventory and observations.
The [artifact review](model-publication-review.md) records why original checkpoints remain private.

## Download

```sh
python -m pip install huggingface_hub==2.1.1
hf download DazielNguyen/hifi-mobiNet-inference \
  --revision f760e85a45b84c217091acf963837ffc817bad8d \
  \
  --local-dir "$HOME/hifi-mobiNet-inference-assets"
cd "$HOME/hifi-mobiNet-inference-assets"
shasum -a 256 -c SHA256SUMS
export HIFIMOBINET_MODEL_DIR="$HOME/hifi-mobiNet-inference-assets/onnx-q05"
```

The code repository supplies inference and the original native frontend. Follow its setup instructions before synthesis.
The model card contains CLI commands and retained notices.
Public weights do not include an installed frontend or a running hosted demo.

## Remaining scope

MIT applies to team-owned weights. Piper retains upstream terms and attribution.
This inference release does not close every code-lineage or native frontend redistribution review.
The dataset candidate remains local. Harvard text distribution conditions remain unresolved.
Hosting is deferred. No Space, paid plan, tag or GitHub push belongs to this step.
No training, export, quantization, latency benchmark or quality evaluation occurred.
The manuscript and research evidence were not edited.
