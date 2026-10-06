---
title: HiFi-MobiNet
emoji: "🔊"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
license: mit
short_description: CPU speech synthesis and scoped historical audio comparison
---
# HiFi-MobiNet demo

This is a local candidate for `DazielNguyen/hifi-mobiNet` Spaces. It has not been uploaded or deployed.
The Docker image installs the original native frontend and identified Q05 FP32 ONNX runtime.
Set `HIFIMOBINET_MODEL_REVISION` to the full commit hash of the uploaded model repository.
The startup script downloads four ONNX files and verifies their sizes and SHA-256 values.
Training checkpoints are not downloaded or deserialized by the Space.

Historical comparison is off by default. Its sentence redistribution conditions remain pending confirmation.
After resolving the relevant conditions, set `HIFIMOBINET_ENABLE_HARVARD_AUDIO=1` and `HIFIMOBINET_AUDIO_REVISION` to the dataset commit.
This setting is an operator choice, not a record that rights were verified.
If a private repository supplies assets, store a read-only token in the Space secret `HF_TOKEN`.

See `deploy/huggingface/targets.json` for proposed targets and the pinned audio catalog identity.
See `docs/artifact-licensing.md` and `THIRD_PARTY_NOTICES.md` for license scope.
MIT applies only to original owned code and team-owned weights. Upstream terms remain separate.
Historical generated WAVs use CC BY 4.0 only for rights the team holds. Sentence text is excluded.

This demo does not measure the historical research protocol. CPU hosting performance, memory and concurrency require target-host checks.
The code limits text to 300 characters and uses serial synthesis. Its output-length check occurs after inference, not as a hard timeout.
The image retains native build source under `.local/native-build` and the pinned source URL in the frontend build files.
No production-readiness or human listening claim is made.
