# Artifact licensing

The author selected these licenses on 2026-10-06. This record covers the exact artifacts in [artifact-licensing.json](artifact-licensing.json).

| Material | License or terms | Scope |
|---|---|---|
| Team Baseline, Parallel-IR and Sequential-IR checkpoints | [MIT](../LICENSE), copyright 2026 DazielNguyen | Team-owned contents. Third-party content retains its terms. |
| Existing Q05 ONNX exports of team models | MIT for the same team-owned weights | An export does not replace upstream notices or license embedded third-party material. |
| External Piper checkpoint | Upstream MIT declaration | Retain upstream attribution. Do not replace upstream ownership with DazielNguyen. |
| Q05 Piper ONNX export | Retain upstream MIT for checkpoint weights | Export provenance and component notices remain separate records. |
| 2,880 generated Harvard WAVs | [CC BY 4.0 notice](../licenses/Harvard-generated-audio-CC-BY-4.0.md) | Only rights the team holds in generated audio, across all four models. |
| Original LJSpeech dataset | Publisher identifies public domain | Retain dataset credits and original terms. Do not assign MIT or CC0. |
| Harvard sentence text | Original terms | Authoritative redistribution terms remain unresolved. No new license is assigned. |
| Logs, configs and other archive members | Their existing terms | A model or audio license does not automatically cover the whole archive. |

## Primary sources

The [Piper checkpoint repository README](https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/main/README.md) declares MIT.
The [selected checkpoint page](https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/main/en/en_US/ljspeech/medium/lj-med_1000.ckpt) supplies its SHA-256.
That SHA-256 matches the local checkpoint identity. This establishes the declared repository license and file identity, not every upstream distribution condition.
The [Piper LJSpeech card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ljspeech/medium/MODEL_CARD) separately describes the dataset.

The [LJSpeech publisher](https://keithito.com/LJ-Speech-Dataset/) states that its text, audio and annotations are public domain.
The publisher explicitly identifies public-domain status in the United States. This statement does not license generated outputs or checkpoints.
The dataset credits Keith Ito, Linda Johnson and LibriVox.

The [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) require appropriate attribution, a license link and an indication of changes.
They permit commercial use. The legal code, linked in the audio notice, defines the license.
All source checks took place on 2026-10-06. The Harvard text terms still need an authoritative source.

## Provenance and release preparation

Original transfer archives, checksums and imported manifests remain unchanged. Their earlier license-pending fields describe the handoff state.
The new JSON record supplies current decisions for identified artifacts. It does not change research results or establish blanket upstream clearance.

For a future artifact release, distribute this record, the relevant license text and attribution with the artifacts.
Use separate model and audio metadata. Do not assign one license to all members of a mixed archive.
Keep the original transfer archives as provenance copies. Any new release package needs a new manifest and checksum.
No push, upload, tag, release or deployment forms part of this license update.
