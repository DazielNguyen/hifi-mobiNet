# Setup and runtime boundaries

Use a dedicated virtual environment. Do not install legacy training requirements
into system Python. Editable installation from a checkout is supported; standalone
wheels do not bundle external results/vendor directories.

```sh
python -m venv .venv
# Activate the environment using your shell's command.
python -m pip install -e ".[demo,evaluation,test]"
```

Validated locally: Windows, Python 3.13.9, NumPy 2.2.6, SciPy 1.15.3, JiWER 4.0.0,
ORT 1.23.2, Streamlit 1.50.0 and optional PyTorch 2.8.0+cpu. These are new validation
versions, not historical training/scoring versions. See `validation-environment.json`.
`HIFIMOBINET_HOME` can point to a checkout when launching from another directory.

## Optional architecture checks

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pytest tests/test_architecture.py -q
```

These construct architectures and compare decoder behavior with reference source;
they do not load checkpoints, train or benchmark. The training-only MAS extension
is preserved as source under `vendor/` and is not built for this inference package.

## Original native frontend

New TTS requires `banhmi_phonemize`; no alternative phonemizer is silently used.
Its build downloads a pinned eSpeak NG source revision, not a model/dataset.
C++ tools are required; the local build used Visual Studio 2022 Desktop C++ tools.

```sh
python -m pip install ./vendor/banhmi-phonemize --config-settings=build-dir=../../.local/b
python -m pytest tests/test_frontend_native.py -q
```

The build path is relative to `vendor/banhmi-phonemize`, resolving to ignored
`.local/b` inside this repo. The shorter path fixed the initial Windows build
failure without changing frontend code. Linux/macOS builds remain unverified.
Local build success does not settle GPL compliance or wheel redistribution rights.

## Installing existing artifacts

```sh
python scripts/inference/install_local_assets.py --help
```

Provide `--q05-root` for an authorized existing Q05 archive and/or `--harvard-root`
for a directory containing `harvard_baseline`, `harvard_mrf`, `harvard_seq` and
`harvard_piper`. Exact host paths are omitted from public docs. The installer checks
the manifest names/sizes/hashes and refuses different existing files. Weights go
to ignored `models/weights/`, WAVs to ignored `.local/audio/`; private source paths
are recorded only under `.local/`. No training/export/quantization occurs.

After installation, explicitly run one short functional utterance per model:

```sh
python scripts/validation/smoke_models.py --enable-verified-models --output-dir outputs/functional-smoke
```

This updates `docs/functional-validation.json` and model status only after checks;
inspect and commit the validation separately. It collects no timing or quality
metrics. A prior pass never enables TTS when the local weights/frontend are missing.
