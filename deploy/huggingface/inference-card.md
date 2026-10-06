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
# HiFi-MobiNet inference artifacts

This package contains four identified Q05 FP32 ONNX graphs and selected config metadata.
Training checkpoints, optimizer state, training logs and Harvard text/audio are excluded.
The full checkpoint archive remains private. These are existing graphs, not new exports.

| Model ID | Role | ONNX filename |
|---|---|---|
| baseline-resblock2 | HiFi-GAN ResBlock2 reference | baseline_fp32.onnx |
| parallel-ir | Parallel inverted-residual system | mrf_fp32.onnx |
| sequential-ir | Sequential inverted-residual system | seq_fp32.onnx |
| piper-original | External Piper LJSpeech reference | piper_fp32.onnx |

Exact sizes and SHA-256 values appear in `artifacts.json` and `SHA256SUMS`.
MIT covers team-owned weights. Piper retains its upstream MIT declaration and Michael Hansen attribution.
See `INFERENCE-NOTICES.md` and `licenses/Piper-MIT.txt` for the retained scope and notice.
The author identifies LJSpeech as the training dataset. Its publisher states public-domain status in the United States.

## Download and use

Select the full 40-character commit from this repository's commit history.
Set `HF_REVISION` to that commit before the download.

```sh
python -m pip install huggingface_hub==2.1.1
export HF_REVISION='<full-40-character-commit>'
hf download DazielNguyen/hifi-mobiNet-inference \
  --revision "$HF_REVISION" --include 'onnx-q05/*' 'artifacts.json' 'SHA256SUMS' \
  --local-dir "$HOME/hifi-mobiNet-inference-assets"
cd "$HOME/hifi-mobiNet-inference-assets"
shasum -a 256 -c SHA256SUMS
export HIFIMOBINET_MODEL_DIR="$HOME/hifi-mobiNet-inference-assets/onnx-q05"
```

The [code repository](https://github.com/DazielNguyen/hifi-mobiNet) supplies the CLI and original native frontend.
Install its inference dependencies and build `vendor/banhmi-phonemize` before synthesis.
Frontend source and dependency terms remain separate. No frontend binary is packaged here.
After code installation, run:

```sh
python -m hifimobinet.cli verify --model sequential-ir
python -m hifimobinet.cli tts --model sequential-ir \
  --text 'Hello. This is a speech demonstration.' --output outputs/seq-demo.wav
```

Use a new output filename. These graphs produce mono audio at 22,050 Hz through the documented frontend.
The package is not a running demo or a Transformers-compatible model.

## Evidence limits

These Q05 graphs are engineering artifacts. Historical Harvard/PyTorch scores do not establish quality or latency for this ONNX release.
Local functional checks do not establish human MOS, quality equivalence, decoder-only causality or production readiness.
Selected config fields and current file identities do not establish historical runtime source identity.
No new training, export, quantization or benchmark forms part of this release.
