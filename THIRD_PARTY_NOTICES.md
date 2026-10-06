# Third-party notices and licensing status

No license is granted for the repository as a whole at this stage. Local
consolidation does not resolve rights to redistribute original contributions,
weights, generated audio, sentence collections or dependencies.

| Material | Evidence and retained notice | Status |
|---|---|---|
| Piper reference code | `vendor/piper/`; [Michael Hansen, MIT](licenses/Piper-MIT.txt) copied from the inspected Piper repository | Notice retained; no claim that BanhmiTTS is a Git fork of this revision |
| Banhmi model/utilities | `vendor/banhmi/`; source comments reference Piper/VITS and BigVGAN-related adaptations | Original repository has no root license; authors must approve per-file attribution and license |
| Banhmi phonemizer | `vendor/banhmi-phonemize/NOTICE.md`, supplied MIT text credited to Michael Hansen (2023) | The supplied notice describes an independent implementation with compatible mapping; authors must reconcile scope/ownership before release |
| eSpeak NG dependency | [GPL-3.0 text](licenses/eSpeak-NG-GPL-3.0.txt) from the source package | Not bundled as a binary; distributing a linked phonemizer needs GPL compliance review |
| VITS / HiFi-GAN / BigVGAN / VITS2 | Architectural and source-comment references within imported material | Do not treat citations as licenses. Full upstream attribution review remains open |
| LJSpeech | Existing Piper model card identifies the dataset as public domain | Does not independently establish a license for every checkpoint or generated output |
| Historical Harvard sentences/audio/results | Original Banhmi evidence package | Local research use; redistribution conditions pending author review |

All unchanged imported source retains its original comments. Some comments make
historical performance or novelty claims that this repository does not endorse.
Inactive F0/Vocos-related references are preserved only for provenance; selected
models do not use these branches. See `docs/release-gaps.md`.

Dependency licenses are independent of the license of this repository. A later
release must inventory the actual installed/bundled dependency versions and retain
their required notices. No remote publication or binary redistribution has occurred.
