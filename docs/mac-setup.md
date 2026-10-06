# Local Mac artifact handoff (proposed v0.1.0, unpublished)

This handoff prepares an existing research system for local validation on a Mac.
Windows/WSL checks do not establish macOS functionality. No release, upload, tag,
push, deployment, new export or benchmark is part of this preparation.

## Transfer the code and artifacts

Copy the whole prepared `archives/` directory to `~/Downloads/hifi-mobiNet-transfer`.
It contains four checkpoint ZIPs, one ONNX ZIP, one Harvard audio ZIP, one metadata
ZIP, the local code Git bundle, an archive index and `SHA256SUMS`. Keep checksum
files from the trusted handoff together; hashes detect corruption, not authenticity.
Do not transfer the staging `private/` directory for publication. Original logs,
host locators and training hparams remain there locally. Full checkpoint bytes can
still contain training host paths/optimizer state; they were not deserialized or
rewritten. Named license, weight host and public artifact URLs remain pending.

```sh
cd "$HOME/Downloads/hifi-mobiNet-transfer"
shasum -a 256 -c SHA256SUMS
mkdir -p "$HOME/Projects"
git clone "$HOME/Downloads/hifi-mobiNet-transfer/hifi-mobiNet-code-v0.1.0.bundle" "$HOME/Projects/hifi-mobiNet"
cd "$HOME/Projects/hifi-mobiNet"
git status --short
```

The bundle includes local unpushed preparation commits. Cloning the GitHub version
alone may omit these changes. No Windows virtual environment or native binary is
included. Bundle cloning creates a local bundle origin, not a new remote service.
Do not overwrite an existing checkout; use a fresh destination or inspect it first.

## Python and system tools

Use CPython **3.12** in a new virtual environment. The project requires Python
>=3.10; its pinned ONNX Runtime 1.23.2 supplies CPython 3.12 wheels for macOS 13+
on both arm64 and x86_64. This was checked against [official PyPI metadata](https://pypi.org/pypi/onnxruntime/1.23.2/json).
Streamlit remains pinned to [1.50.0](https://pypi.org/project/streamlit/1.50.0/).
Use native arm64 Python on Apple Silicon or native x86_64 on Intel; do not mix
architectures. An older macOS version is outside this prepared wheel path.

The original frontend builds C++17 code and a pinned eSpeak NG source revision.
Install Apple's Command Line Tools if absent (`xcode-select --install`). With an
existing Homebrew installation, `brew install python@3.12 cmake ninja` provides
the interpreter and build tools. These commands are Mac instructions, not actions
performed on the Windows preparation machine. CMake downloads the pinned source
commit `0f65aa301e0d6bae5e172cc74197d32a6182200f`; package dependencies also require
network access. A system eSpeak installation is not silently substituted.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[demo,evaluation,test]"
python -m pip check
python scripts/inference/import_artifacts.py --help
```

CPUExecutionProvider is the existing inference default. CUDA, MPS and Core ML
are not enabled or assumed to work. PyTorch is unnecessary for the ONNX demo.

## Verify and import

```sh
python scripts/inference/import_artifacts.py verify "$HOME/Downloads/hifi-mobiNet-transfer/"*.zip
python scripts/inference/import_artifacts.py import "$HOME/Downloads/hifi-mobiNet-transfer/"*.zip --destination "$HOME/hifi-mobiNet-assets-v0.1.0"
export HIFIMOBINET_ASSET_DIR="$HOME/hifi-mobiNet-assets-v0.1.0"
```

Importing all seven ZIPs preserves checkpoint files as archival data; no pickle
load is performed. The utility validates the external archive checksums, each
member's size/hash and the package manifest before writing. It rejects traversal,
links/special ZIP members, duplicate/case-colliding names and conflicting existing
files. Existing identical files are reused. It reports missing runtime assets;
it never downloads or substitutes a model. Keep several GB free for archives and
their extracted copies. Checkpoints are not needed for inference.

An already transferred full staging directory is also accepted:

```sh
python scripts/inference/import_artifacts.py verify "$HOME/hifi-mobiNet-artifacts-v0.1.0-staging"
python scripts/inference/import_artifacts.py import "$HOME/hifi-mobiNet-artifacts-v0.1.0-staging" --destination "$HOME/hifi-mobiNet-assets-v0.1.0"
```

This imports only manifest-listed public candidate files, excluding `private/`
and archives. `HIFIMOBINET_ASSET_DIR` selects both `onnx-q05/` and the complete
`manifests/audio-manifest.json` (720 sentences, four models). Existing specialized
`HIFIMOBINET_MODEL_DIR`/`HIFIMOBINET_AUDIO_DIR` overrides take precedence; unset
them if they point to an older installation. No operator path input is exposed
inside the demo UI.

## Comparison first, then native frontend and TTS

```sh
python -m streamlit run demo/streamlit_app.py --server.address 127.0.0.1
```

Select a sentence and check that four historical WAV players work. Playback uses
original bytes with no added loudness normalization. No weights/frontend are
required for comparison. Stop Streamlit before building the frontend:

```sh
python -m pip install ./vendor/banhmi-phonemize --config-settings=build-dir=../../.local/b
python -m pytest tests/test_frontend_native.py -q
python -m hifimobinet.cli models
python -m hifimobinet.cli verify --model sequential-ir
python -m hifimobinet.cli tts --model sequential-ir --text "The birch canoe slid on the smooth planks." --output outputs/mac-seq-smoke.wav
python -m streamlit run demo/streamlit_app.py --server.address 127.0.0.1
```

The output filename must not already exist. This is a functional smoke test,
not a timing/quality result. Repeat with other manifest model IDs only as needed.
If the native build fails, retain its log and OS/Python architecture; comparison
remains usable. Do not change text preprocessing or the model to bypass a failure.
Native CMake/dylib behavior has not yet been checked on a Mac.

For tests of the repository's original three-sentence fixture, clear the external
asset variable for that test process: `env -u HIFIMOBINET_ASSET_DIR python -m pytest -q`.
The default repository tests and historical full-catalog smoke checks have distinct
fixtures. Record macOS version, architecture, Python/dependency versions, importer
outcome, frontend tests and observed playback/TTS separately from Windows evidence.

Q05 ONNX files remain **engineering artifacts**. Hashes and short synthesis do not
replace final manuscript parity, quality or benchmark acceptance. No timing from
this demo belongs in the paper. CPU/RAM capacity and concurrency on the target Mac
remain unmeasured. Model loading stays allowlisted and serial by default.
