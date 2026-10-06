# Training setup on Ubuntu / WSL2

Validated on Ubuntu 24.04.4 LTS under WSL2 (distro name `Ubuntu-24.04`) with two
RTX 4070 Ti (12 GiB each), driver 596.36, 62 GiB RAM. Exact observed versions:
[environment-record.json](environment-record.json).

## Storage layout

Clone hifi-mobiNet onto the Linux filesystem (not a Windows drive, not
OneDrive). All training code, data and outputs stay inside that clone; data and
outputs are in Git-ignored directories and never become tracked content:

```text
<hifi-mobiNet clone>/
  src/ vendor/ configs/ scripts/ ...     tracked code and configs
  data/ljspeech-medium/                  staged preprocessed dataset (ignored, ~22 GB)
  data/smoke-ljs-16train-4val/           smoke subset (ignored)
  training_output/<run>/                 checkpoints, TensorBoard logs, run records (ignored)
$HIFI_TRAIN_ROOT/venv/                   training interpreter (outside the clone)
```

| Variable | Meaning |
|---|---|
| `HIFI_TRAIN_ROOT` | Linux-filesystem directory holding `venv/`, `environment.json`, `pip-freeze.txt` |
| `PYTHON310` | Base Python 3.10 interpreter used only to create the venv |

## 1. Clone and create the environment

```sh
git clone <repository-url> <hifi-mobiNet clone>
cd <hifi-mobiNet clone>
bash scripts/training/setup_ubuntu.sh
```

The script creates `$HIFI_TRAIN_ROOT/venv`, installs pip 24.0 and
[requirements/training-linux-cu130.txt](../../requirements/training-linux-cu130.txt)
(torch 2.13.0+cu130, pytorch-lightning 1.7.7, ...), installs the clone in
editable mode without re-resolving dependencies, builds both MAS extensions
(`scripts/training/build_mas.sh`) and writes `environment.json` and
`pip-freeze.txt` into `$HIFI_TRAIN_ROOT`.

Requirements: a C compiler (`build-essential`), an NVIDIA driver with CUDA 13
support, and a Python 3.10 interpreter. Training is supported from an editable
clone only (`vendor/` is not packaged).

Why pip 24.0: pytorch-lightning 1.7.x declares `torch>=1.9.*`, which pip ≥ 24.1
rejects as invalid metadata. Why these versions: they are the versions present
in the existing workstation training interpreter, not the newest releases.

## 2. Check the installation

```sh
PY="$HIFI_TRAIN_ROOT/venv/bin/python"
"$PY" -m pip install pytest==8.4.2   # test tool only
"$PY" -m pytest tests/test_training.py -q
```

All tests must pass. Tests that need Lightning or the compiled MAS skip with an
explicit reason when those are missing, which then means the setup is incomplete.

## 3. Stage the dataset into the clone

```sh
"$PY" scripts/training/stage_dataset.py \
  --source-data-root <directory the original dataset.jsonl paths are relative to> \
  --source-dataset-dir <that directory>/training_output/ljspeech/medium \
  --name ljspeech-medium \
  --expect-dataset-sha256 f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b
```

Copies `dataset.jsonl`, `config.json`, the frame-length cache and the 26,200
waveform/spectrogram tensors it references into `data/ljspeech-medium/`, keeping
the relative layout so `dataset.jsonl` stays byte-identical. Every file is
SHA-256-verified against its source; the source is only read; re-running skips
verified files. No F0 files are copied. The canonical split is checked against
the staged rows. `data/ljspeech-medium/STAGING_RECORD.json` records the result.

Optional ID check (needs the native frontend):
`python scripts/training/verify_phoneme_ids.py --dataset data/ljspeech-medium/training_output/ljspeech/medium/dataset.jsonl`.
On the workstation (Windows frontend build) 13,100/13,100 rows matched.

## 4. Smoke test (short, real training)

```sh
"$PY" -c "
from pathlib import Path
from hifimobinet.training.data import load_split, make_subset_dataset
make_subset_dataset(Path('data/ljspeech-medium/training_output/ljspeech/medium'), Path('data/ljspeech-medium'),
                    load_split(Path('results/manifests/canonical_split.json')), Path('data/smoke-ljs-16train-4val'), 16, 4)"
D=data/smoke-ljs-16train-4val
CUDA_VISIBLE_DEVICES=0 "$PY" -m hifimobinet.training.smoke --model-id Piper_no_VITS2_cpn \
  --config configs/training/piper-no-vits2-cpn.yaml --dataset-dir "$D" --data-root "$D" --split "$D/split.json" \
  --run-dir training_output/smoke-piper-no-vits2-cpn --batch-size 4 --epochs-a 1 --epochs-b 2 --max-updates 26
```

The subset takes the first 16 canonical-train and 4 canonical-validation
utterances; no test utterance is copied. The smoke run writes
`smoke_report.json` (pass / fail / not_checked / not_applicable) and one
PyTorch inference WAV. Run models one at a time. Results:
[training-smoke-report.md](training-smoke-report.md).

## 5. Full training

See [next-run.md](next-run.md). Do not start a long run before the protocol in
[piper-no-vits2-protocol.md](piper-no-vits2-protocol.md) is settled.

## Known setup issues

| Symptom | Root cause | Fix in this repository |
|---|---|---|
| `MisconfigurationException: ... ExponentialLR doesn't follow PyTorch's LRScheduler API` | Lightning 1.7.7 checks `torch.optim.lr_scheduler._LRScheduler`; torch ≥ 2.0 schedulers derive from `LRScheduler` | `lr_scheduler_step` override with Lightning's default body (`training/harness.py`) |
| `UnpicklingError: Weights only load failed ... numpy.core.multiarray._reconstruct` on resume | torch ≥ 2.6 defaults to `weights_only=True`; a NumPy RNG state array was stored in the checkpoint | RNG state stored as tensors/plain values |
| `RuntimeError: cuFFT doesn't support tensor of type: BFloat16` with rhasspy Piper code | rhasspy/piper 73c04d8 computes the generator-output STFT under autocast | The model uses EdgeTTS's training step, which computes mel/STFT outside autocast (bf16 works) |
| `KeyError: 'audio_f0_path'` if EdgeTTS's own dataset class is used | EdgeTTS's `PiperDataset` requires F0 paths; this dataset has none | The harness reads rows like the baseline and passes `f0s=None` (Config A never uses F0) |
| Background install stops when the WSL shell exits | `nohup ... &` inside `wsl -e bash -c` is killed with the session | Run long commands in the foreground of a persistent terminal (or `tmux`) |
| `git clone` refuses: "dubious ownership" | The Windows checkout's `.git` belongs to another Windows account | Use `git -c safe.directory=<path> clone ...` for that command; global Git config left unchanged |
