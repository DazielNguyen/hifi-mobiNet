---
language: en
license: cc-by-4.0
tags:
- audio
- text-to-speech
- synthetic-speech
---
# HiFi-MobiNet generated Harvard audio

This is a local candidate for `DazielNguyen/hifi-mobiNet-harvard`. It is not a published dataset release.
The package contains 2,880 historical synthetic WAVs: 720 utterances from each of four systems.
The systems are Baseline, Parallel-IR, Sequential-IR and external Piper. Bytes retain their original form without additional normalization.

## License scope

CC BY 4.0 applies only to rights the team holds in the generated WAVs.
See `licenses/Harvard-generated-audio-CC-BY-4.0.md` for the notice and suggested attribution.
Original dataset recordings, Harvard sentence text, model weights and other third-party rights are excluded.
The package includes a sentence catalog for alignment. Its presence does not establish permission to redistribute the text.

The [Columbia University list](https://www.cs.columbia.edu/~hgs/audio/harvard.html) attributes the sentences to IEEE, Appendix C, 1969.
The supplied page does not state an explicit redistribution license. Sentence distribution rights remain **pending confirmation**.
Keep this candidate private until the remaining release conditions are resolved. The repository license field does not cover every package member.

## Identity and interpretation

`manifests/audio-manifest.json` binds sentence IDs, models, file sizes, hashes and PCM headers.
`release-manifest.json` inventories this candidate. It does not replace the original transfer manifest.
These WAVs come from historical PyTorch runs. They are not regenerated ONNX outputs or a controlled human listening study.
Use the [code repository](https://github.com/DazielNguyen/hifi-mobiNet) for comparison and protocol limits.
UTMOSv2 scores are predicted scores, not human MOS. This release does not establish unseen status for every model.
