# Release gaps

- MIT was selected for original repository code/documentation owned by
  DazielNguyen on 2026-10-06. BanhmiTTS source lineage and compatibility with
  upstream material still need per-file assessment. Root MIT does not relicense
  third-party material. Separate MIT decisions cover team-owned checkpoint contents and equivalent ONNX weights.
  See artifact-licensing.md for exact scope.
- Phonemization involves eSpeak NG (GPL-3.0); bundled binaries/build dependencies
  need a separate distribution review. Preserve all existing notices.
- Third-party VITS/HiFi-GAN/BigVGAN utility attribution needs an author-reviewed
  per-file assessment. Structural similarity alone does not establish lineage.
- The author confirmed permission to share code, weights, logs and Harvard audio
  on 2026-10-06. Team checkpoints and equivalent ONNX weights use MIT for team-owned content.
  Generated Harvard WAVs use CC BY 4.0 for rights the team holds.
  Piper retains its upstream MIT declaration. Harvard text terms and remaining upstream conditions still need documentation.
  These decisions are recorded in release-decisions.json and artifact-licensing.json.
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
- WSL artifact access is resolved. All four Q05 ONNX models passed a local functional
  synthesis check; twelve original Harvard WAVs passed checksum and UI checks.
  Assets remain outside Git. Storage provider/public download URLs are unconfirmed.
- Native frontend was built on Windows only. Linux hosting, loaded-model memory,
  concurrent sessions and actual browser listening are not validated.
- The current package exposes inference, component inspection and stored-result
  audits. It does not provide a validated training launcher, ASR/UTMOS rescoring
  installation, export or quantization workflow.
- Local Mac transfer ZIPs and manifests are prepared and integrity-checked.
  This does not resolve distribution rights or imply manuscript acceptance of Q05.
  Mac execution has not been tested. See mac-setup.md and mac-transfer-preparation.md.
- Raw logs/hparams are kept privately; publication-intended copies have separate
  redaction records. The redaction scan is heuristic. Full unchanged training
  checkpoints can embed host paths/state and need distribution review before upload.
