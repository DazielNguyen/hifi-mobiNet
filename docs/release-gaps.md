# Release gaps

- BanhmiTTS has no top-level license. Author licensing of original contributions
  and compatibility with upstream material are unresolved. Do not infer an MIT
  grant from the license of Piper or a phonemizer subdirectory.
- Phonemization involves eSpeak NG (GPL-3.0); bundled binaries/build dependencies
  need a separate distribution review. Preserve all existing notices.
- Third-party VITS/HiFi-GAN/BigVGAN utility attribution needs an author-reviewed
  per-file assessment. Structural similarity alone does not establish lineage.
- Checkpoint, generated-audio and Harvard sentence redistribution conditions
  are not yet fully documented. Local access is not publication permission.
- Existing ONNX/PTQ quality results are historical. They cannot be attached to
  the later Q05 FP32 graphs as if they were evaluated together.
- The historical LJSpeech-500 sets differ between MRF and the other internal
  models. Harvard-720 has common sentences, but selection independence and the
  external Piper training split are not fully established.
- Training recipes differ (activation, MRD, clipping, numerical guards, split).
  System comparisons do not isolate decoder topology or establish equivalence.
- Intel N150 / Raspberry Pi 5 measurements and human listening scores are absent.
- Public model download links for team artifacts are pending. No fake URL or
  implicit substitution is allowed.
- Local access to actual Q05 ONNX and Harvard WAVs is pending clarification of WSL
  scope. The four TTS models remain disabled; no real-model inference pass is claimed.
- Native frontend was built on Windows only. Linux hosting, loaded-model memory,
  concurrent sessions and actual browser listening are not validated.
- The current package exposes inference, component inspection and stored-result
  audits. It does not provide a validated training launcher, ASR/UTMOS rescoring
  installation, export or quantization workflow.
