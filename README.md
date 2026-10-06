# hifi-mobiNet

An independent local repository organizing the **HiFi-GAN ResBlock2 baseline,
Parallel-IR and Sequential-IR** speech systems developed in BanhmiTTS, with an
external Piper LJSpeech checkpoint. It includes model components, historical
results, manifests, statistical tools and a Streamlit interface.

This is newly organized code, **not a recovered training-time source tree**.
The research concerns system-level quality and computational tradeoffs. Recipe
and split differences prevent decoder-only causal claims. No quality equivalence,
edge-device performance or production readiness is established here.

The author has authorized public code publication at [DazielNguyen/hifi-mobiNet](https://github.com/DazielNguyen/hifi-mobiNet)
and confirmed permission to share journal-related artifacts. Original repository
code and documentation owned by DazielNguyen are licensed under [MIT](LICENSE),
subject to the scope and exclusions below. Artifact licenses are documented in [artifact licensing](docs/artifact-licensing.md).
Four FP32 ONNX graphs are available from the public [inference repository](https://huggingface.co/DazielNguyen/hifi-mobiNet-inference).
Original training checkpoints remain in a private archive. Weights and audio remain outside Git.
Missing local weights are reported explicitly; no different model is silently substituted.

| Model ID | Decoder | Selected epoch | FP32 artifact | Local TTS |
|---|---|---:|---|---|
| `baseline-resblock2` | HiFi-GAN ResBlock2 | 1489 | Q05 SHA-256 verified | Functional pass |
| `parallel-ir` | Parallel IR, kernels 3/5/7, expansion 1 | 1386 | Q05 SHA-256 verified | Functional pass |
| `sequential-ir` | Two sequential IR blocks/stage, kernel 3, expansion 1 | 1442 | Q05 SHA-256 verified | Functional pass |
| `piper-original` | Piper medium ResBlock2 | 999 (filename 1000) | Q05 SHA-256 verified | Functional pass |

Exact configurations, sizes, SHA-256 values, source and sharing status are in
[models/manifest.json](models/manifest.json). Checkpoint identity, ONNX identity
and local functional validation are separate records. Passes were recorded on
2026-10-06 in the prepared local installation; a fresh checkout needs the assets.

## Listen in your browser

Choose a sentence and compare four fixed recordings in the
[Hugging Face listening room](https://huggingface.co/spaces/DazielNguyen/hifi-mobiNet)
or the [GitHub Pages listening page](https://dazielnguyen.github.io/hifi-mobiNet/).
The page contains three new English sentences and 12 WAV samples.
It does not synthesize user-entered text or require a model download.

These samples are demonstrations, not historical Harvard audio or a listening test.
They use CC BY 4.0 only for rights held by the project team.
See [demo provenance and hosting](docs/listening-demo.md) and the
[publication receipt](docs/listening-publication.json) for identities and actual deployment checks.

## Checkout and setup

Clone the public repository into a new directory:

```sh
git clone https://github.com/DazielNguyen/hifi-mobiNet.git
cd hifi-mobiNet
python -m venv .venv
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on Linux, then:

```sh
python -m pip install -e ".[demo,evaluation,test]"
python -m hifimobinet.cli --help
python -m hifimobinet.cli models
```

Comparison UI and stored-result audits do not require PyTorch, a phonemizer or
weights. Windows Python 3.13 is the local validation platform. See
[setup](docs/setup.md) for native frontend and optional architecture checks.

## Models and inference

Four identified FP32 ONNX graphs are public. Follow the pinned download instructions in
[the inference release guide](docs/huggingface-inference-release.md). Original training checkpoints remain private.
See [model access](models/README.md). No checkpoint pickle is loaded or model exported.

After installing the existing identified artifacts, building the original frontend
and passing the functional smoke check described in the setup guide:

```sh
python -m hifimobinet.cli verify --model sequential-ir
python -m hifimobinet.cli tts --model sequential-ir --text "The birch canoe slid on the smooth planks." --output outputs/example.wav
```

Until these prerequisites pass, inference returns an explicit unavailable error.
The CLI accepts manifest IDs only and does not overwrite existing WAVs.

## Historical results and analysis

Twenty original tables are preserved byte for byte, with source locators and hashes.
Both LJSpeech-500 and Harvard-720 are retained with their protocol limits. Read
[results/README.md](results/README.md) before interpreting comparisons.

```sh
python scripts/evaluation/audit_results.py summarize tests/fixtures/metrics.json
python scripts/evaluation/recompute_harvard.py --input-dir results/historical/harvard --output-dir outputs/harvard-statistics
```

The second command recomputes a prior audit from stored scores, without synthesis,
ASR or UTMOS. Its 279 numeric values matched the archived audit in local validation.
See [evaluation](docs/evaluation.md) for settings and denominator/pairing rules.

Historical Harvard means, from PyTorch CPU on the development workstation:

| System | Mean RTF ↓ | UTMOSv2 predicted ↑ | Macro WER ↓ |
|---|---:|---:|---:|
| HiFi-GAN ResBlock2 | 0.04177 | 3.0260 | 0.1476 |
| Parallel-IR | 0.03388 | 3.5680 | 0.1073 |
| Sequential-IR | 0.03081 | 3.5443 | 0.1222 |
| Piper | 0.03916 | 3.4507 | 0.1747 |

These scores do not describe this reorganized code, its ONNX demo or an edge device.
UTMOSv2 is not human MOS. Historical PTQ tables remain exploratory and are excluded
from the demo because of artifact and protocol gaps.

## Streamlit

```sh
python -m streamlit run demo/streamlit_app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

The audio tab selects the same utterance across four models and verifies WAV hashes.
The first three Harvard IDs are catalogued without score-based selection. Actual
historical WAVs are not bundled in Git; install authorized copies as described in
[demo setup](docs/demo.md). Missing audio/model states do not break the page.
Synthetic WAV fixtures are used only in automated tests.

TTS requires successful functional validation and local assets. There is no upload,
custom-path input, PTQ selector or benchmark timing display. See
[deployment preparation](docs/deployment.md) for storage and unmeasured hosting limits.

## Repository map

```text
src/hifimobinet/    Portable inference, registry, evaluation, model components
vendor/            Unchanged reference imports and native frontend source
configs/           Checkpoint-derived JSON and original YAML templates
scripts/           Inference, evaluation and validation entry points
models/            Artifact/checkpoint manifest and access instructions
results/           Historical observations, manifests and prior audit
demo/              Streamlit app and aligned audio catalog
tests/             Functional/fixture checks, not research experiments
docs/              Provenance, checks, status and release gaps
licenses/          Retained third-party license texts
```

## Validation and release status

For the local, unpublished Mac handoff, use [Mac setup](docs/mac-setup.md).
The checksum-verified importer accepts staging directories or ZIP packages and
uses `HIFIMOBINET_ASSET_DIR` for both CLI models and the complete 720-sentence
comparison catalog. See [transfer preparation](docs/mac-transfer-preparation.md)
for archive identities and Windows/WSL validation limits. Mac runtime validation
has not yet been performed.

```sh
python -m pytest -q
python scripts/validation/audit_repository.py
```

Optional PyTorch/native checks skip explicitly when dependencies are absent.
Performed checks and pending work are in [implementation status](docs/implementation-status.md),
[validation record](docs/validation-record.json) and [release gaps](docs/release-gaps.md).
The [source map](docs/source-map.json) and [release manifest](docs/release-manifest.json)
separate imported evidence from new work. Source repositories remain unchanged.

## Attribution and access

The [MIT License](LICENSE), copyright 2026 DazielNguyen, applies to original
repository code and documentation owned by that copyright holder. It does not
replace third-party terms. The separate [artifact licensing record](docs/artifact-licensing.md)
assigns MIT to team-owned checkpoint contents and equivalent ONNX weights.
Generated Harvard WAVs use CC BY 4.0 only for rights the team holds.
Original datasets and imported evidence retain their own terms. Imported and derived upstream material retains its
applicable terms; uncertain provenance remains under review. Piper MIT, the
supplied phonemizer notice and eSpeak NG GPL text are retained. Native frontend
binary distribution needs a separate review. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Hugging Face model storage

Four unchanged Q05 FP32 ONNX graphs are public at [DazielNguyen/hifi-mobiNet-inference](https://huggingface.co/DazielNguyen/hifi-mobiNet-inference).
Use commit `f760e85a45b84c217091acf963837ffc817bad8d` for pinned downloads.
See [the inference release](docs/huggingface-inference-release.md) for download commands, checksums and scope.
Original training checkpoints remain private at `DazielNguyen/hifi-mobiNet`.
The historical dataset and Docker Space candidates remain local.
A separate Static Space hosts the fixed listening demo; historical audio remains off by default.
See [the artifact review](docs/model-publication-review.md) for metadata findings and limits.
