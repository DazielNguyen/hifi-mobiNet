---
title: HiFi-MobiNet Listening Room
emoji: "🔊"
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
license: mit
models:
- DazielNguyen/hifi-mobiNet-inference
short_description: Play the same English sentence from four speech systems
thumbnail: https://huggingface.co/spaces/DazielNguyen/hifi-mobiNet/resolve/04125976a67938eb96d6ddfe144a7721eb2819fb/branding/cover.png
---
# HiFi-MobiNet Listening Room

![HiFi-MobiNet — Listen. Compare. Explore.](https://huggingface.co/spaces/DazielNguyen/hifi-mobiNet/resolve/04125976a67938eb96d6ddfe144a7721eb2819fb/branding/cover.png)

Choose one of three sentences and listen to four fixed model recordings.
This static page does not run live inference or require a paid compute plan.
The same interface is published through GitHub Pages.

The recordings use existing identified Q05 FP32 ONNX graphs. The original full checkpoints remain private.
These are new demo recordings, not the historical Harvard audio or a quality evaluation.
No training, export, quantization or benchmark belongs to this listening page.

Original frontend code for this page uses MIT. The generated WAVs use CC BY 4.0 only for rights held by the project team.
See `audio/LICENSE.md` and `audio/generation-record.json` for scope and provenance.
Model weights and original datasets retain their separate terms.
LJSpeech credits: Keith Ito and Linda Johnson (2017), with LibriVox credits.
Piper remains an external model reference. DazielNguyen does not claim ownership of its original weights.

Code and download instructions: https://github.com/DazielNguyen/hifi-mobiNet
Public ONNX weights: https://huggingface.co/DazielNguyen/hifi-mobiNet-inference

Brand artwork was supplied by the project author. The MIT frontend license does not assign a new license to these images.
