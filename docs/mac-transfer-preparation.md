# Local artifact preparation for Mac

Project: hifi-mobiNet

Status: prepared locally; proposed v0.1.0 is unpublished. No new remote action is authorized in this preparation.

## Package identities

ZIPs preserve original model/WAV bytes (ZIP_STORED). Each is below 2 GiB.

| Archive | Exact bytes | SHA-256 |
|---|---:|---|
| hifi-mobiNet-v0.1.0-checkpoint-baseline.zip | 867105727 | `db0802402121d7c55d34e1febbbac86074c55dadb0d8a71bea1f930a0c6e793e` |
| hifi-mobiNet-v0.1.0-checkpoint-parallel-ir.zip | 864544234 | `9dad8b9bffe8ea8023d7df44d0705558ddafac1412147b92ba630f1c1dbfeda5` |
| hifi-mobiNet-v0.1.0-checkpoint-sequential-ir.zip | 863929612 | `f87b1a336765ee21b52b6d4bf44777629a2fdb874c8960932507912a088aedc3` |
| hifi-mobiNet-v0.1.0-checkpoint-piper.zip | 845955600 | `9e0503f5e13b98bcd590a2bdfa91169a6f0336eb47019e93857a24bc0c1b75c5` |
| hifi-mobiNet-v0.1.0-onnx-q05.zip | 265076377 | `51b1c8bd845840e6c6d40d1e2aba5701159e44c3bcdbf4f8317178c4bdbdc458` |
| hifi-mobiNet-v0.1.0-audio-harvard.zip | 320750826 | `4be5c0d14b1d306378bfd9d151b1a05171a597830e2c0b2e33c0d6d0b65f29d2` |
| hifi-mobiNet-v0.1.0-metadata.zip | 10100264 | `74f2bffe07149b2620cd46727aad216e85bafa2bc4b3cf75223cd8bf559d905d` |

Seven ZIPs total **4037462640 bytes**. The separate code Git bundle and its checksum are generated after committing this documentation, avoiding a self-referential commit hash. Keep the whole transfer directory together.

## Evidence and validation

- Four selected checkpoints (epochs 1489, 1386, 1442 and Piper metadata epoch 999) match the saved checkpoint hashes. No unsafe deserialization; full training state retained.
- Four existing Q05 FP32 graphs match the saved artifact hashes. Engineering artifacts only; no final paper parity/quality acceptance is asserted.
- All 720 WAV names per model, sentence IDs/text mapping, 2,880 hashes, byte sizes and PCM headers match the historical inventory. No normalization.
- Eight original text logs and nine training-written hparams copies are preserved privately. Twenty-one derived log/config files have a redaction record; categories/counts only, never removed values.
- Recovered Q05 checkpoint hparams are explicitly separate from training-written Lightning hparams. Neither timestamp nor a config filename is taken as historical runtime source binding. Piper original training run hparams are unavailable.
- Windows: 35 tests passed. WSL: source and destination symlinks rejected. All archive members passed size/hash checks. Real three-package runtime import succeeded with zero missing runtime assets.
- Four models verified through CLI; all 2,880 imported WAVs verified through the shared runtime reader; Streamlit had 720 choices, four players for indices 0/359/719, and successful SEQ TTS. No timing, quality scores, benchmark or human listening evaluation.
- All 121 original source-map fingerprints remained unchanged. No source repo or original evidence changed.

## Resume

1. Transfer ZIPs, Git bundle, archive index and checksums. Clone the bundle to obtain the unpushed importer/docs, then follow [Mac setup](mac-setup.md).
2. Run checksum/import verification, build the native frontend in a Mac virtual environment, and record actual Mac CLI/playback/TTS results separately.
3. Resolve named license, third-party terms and model storage before any upload. Original checkpoints may retain embedded host metadata; they have not been rewritten.
4. Inspect local commits before any later publication. Do not create a release, tag or push from this preparation automatically.

The private staging report holds exact Windows/WSL paths and the transfer bundle hash. It is intentionally outside Git. The public-candidate manifests use relative paths. See [validation record](mac-transfer-validation.json) for actual Windows/WSL checks.
