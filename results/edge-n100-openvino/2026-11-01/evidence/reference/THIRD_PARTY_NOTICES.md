# Third-party notices and licensing status

The author selected MIT on 2026-10-06 for original repository code and
documentation owned by DazielNguyen; see the root LICENSE. This selection does not
relicense third-party material, original datasets or imported evidence.
The separate [artifact licensing record](docs/artifact-licensing.md) covers team-owned checkpoint contents and equivalent ONNX weights under MIT.
Generated Harvard WAVs use CC BY 4.0 only for rights the team holds.
Upstream license and copyright notices remain applicable. Remaining distribution conditions are documented separately.

| Material | Evidence and retained notice | Status |
|---|---|---|
| Piper reference code | `vendor/piper/`; [Michael Hansen, MIT](licenses/Piper-MIT.txt) copied from the inspected Piper repository | Notice retained; no claim that BanhmiTTS is a Git fork of this revision |
| Banhmi model/utilities | `vendor/banhmi/`; source comments reference Piper/VITS and BigVGAN-related adaptations | Original repository has no root license; authors must approve per-file attribution and license |
| Banhmi phonemizer | `vendor/banhmi-phonemize/NOTICE.md`, supplied MIT text credited to Michael Hansen (2023) | The supplied notice describes an independent implementation with compatible mapping; authors must reconcile scope/ownership before release |
| eSpeak NG dependency | [GPL-3.0 text](licenses/eSpeak-NG-GPL-3.0.txt) from the source package | Not bundled as a binary; distributing a linked phonemizer needs GPL compliance review |
| VITS / HiFi-GAN / BigVGAN / VITS2 | Architectural and source-comment references within imported material | Do not treat citations as licenses. Full upstream attribution review remains open |
| External Piper checkpoint | Upstream checkpoint repository declares MIT; file page SHA-256 matches the local manifest | Retain upstream terms and attribution; do not assign DazielNguyen ownership |
| LJSpeech training dataset | [Rights statement](https://keithito.com/LJ-Speech-Dataset/#license) identifies text, audio and annotations as public domain in the United States | Attribution: Keith Ito and Linda Johnson (2017), with LibriVox credits; no new dataset license |
| Generated Harvard WAVs | Author selected [CC BY 4.0](licenses/Harvard-generated-audio-CC-BY-4.0.md) | Covers only rights the team holds in 2,880 generated WAVs; excludes sentence text and third-party rights |
| Harvard sentence text | [Columbia University list](https://www.cs.columbia.edu/~hgs/audio/harvard.html) attributes the source to IEEE | No explicit license on the page; distribution rights pending confirmation; generated audio license does not cover sentence text |
| Historical results | Original evidence package | Existing provenance and terms retained; audio license does not cover result files |

All unchanged imported source retains its original comments. Some comments make
historical performance or novelty claims that this repository does not endorse.
Inactive F0/Vocos-related references are preserved only for provenance; selected
models do not use these branches. See `docs/release-gaps.md`.

Dependency licenses are independent of the license of this repository. An artifact
release must inventory the actual installed/bundled dependency versions and retain
their required notices. No compiled frontend binary, model weight or WAV is included
in Git. Public code publication does not imply blanket licensing or binary clearance.
